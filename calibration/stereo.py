#!/usr/bin/env python3
"""
Захват парных фото с логикой из stereo_calibration_manual.py
"""

import sys
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
from datetime import datetime
import json

Gst.init(None)

# Глобальные переменные
capture_interval = 5.0
max_frames = 20
frames_dir = "calibration_data"

# Параметры доски
BOARD_SIZE = (10, 7)
SQUARE_SIZE = 23.5

def ensure_directories():
    dirs = [
        frames_dir,
        os.path.join(frames_dir, "pairs"),
        os.path.join(frames_dir, "pairs", "cam0"),
        os.path.join(frames_dir, "pairs", "cam1")
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    board_info = {
        "board_size": BOARD_SIZE,
        "square_size_mm": SQUARE_SIZE,
        "capture_date": datetime.now().isoformat()
    }
    
    with open(os.path.join(frames_dir, "calibration_info.json"), "w") as f:
        json.dump(board_info, f, indent=2)

def create_dual_pipeline():
    """Pipeline с логикой из stereo_calibration_manual.py - videorate + multifilesink"""
    cam_width = 3840
    cam_height = 2160
    
    # Два независимых pipeline для каждой камеры
    pipeline_str = f"""
        nvarguscamerasrc sensor-id=0 !
        video/x-raw(memory:NVMM),width={cam_width},height={cam_height},framerate=30/1,format=NV12 !
        tee name=t0 !
        queue !
        nvvideoconvert !
        video/x-raw(memory:NVMM),width=960,height=540 !
        nvegltransform !
        nveglglessink window-x=0 window-y=0 window-width=960 window-height=540
        
        t0. !
        queue !
        videorate !
        video/x-raw(memory:NVMM),width={cam_width},height={cam_height},framerate=1/{int(capture_interval)} !
        nvjpegenc quality=100 !
        multifilesink 
            location={frames_dir}/pairs/cam0/pair_%05d.jpg 
            max-files={max_frames}
            post-messages=true
            
        nvarguscamerasrc sensor-id=1 !
        video/x-raw(memory:NVMM),width={cam_width},height={cam_height},framerate=30/1,format=NV12 !
        tee name=t1 !
        queue !
        nvvideoconvert !
        video/x-raw(memory:NVMM),width=960,height=540 !
        nvegltransform !
        nveglglessink window-x=960 window-y=0 window-width=960 window-height=540
        
        t1. !
        queue !
        videorate !
        video/x-raw(memory:NVMM),width={cam_width},height={cam_height},framerate=1/{int(capture_interval)} !
        nvjpegenc quality=100 !
        multifilesink 
            location={frames_dir}/pairs/cam1/pair_%05d.jpg 
            max-files={max_frames}
            post-messages=true
    """
    
    return Gst.parse_launch(pipeline_str)

def main():
    global max_frames, capture_interval
    
    if len(sys.argv) > 1:
        max_frames = int(sys.argv[1])
    if len(sys.argv) > 2:
        capture_interval = float(sys.argv[2])
    
    print("ЗАХВАТ СИНХРОННЫХ ПАР (логика stereo_calibration_manual)")
    print("=" * 50)
    print(f"Доска: {BOARD_SIZE[0]}x{BOARD_SIZE[1]} углов")
    print(f"Количество пар: {max_frames}")
    print(f"Интервал: {capture_interval} сек")
    print("=" * 50)
    
    ensure_directories()
    
    pipeline = create_dual_pipeline()
    
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    loop = GLib.MainLoop()
    
    captured_left = 0
    captured_right = 0
    
    def on_message(bus, message):
        nonlocal captured_left, captured_right
        
        t = message.type
        if t == Gst.MessageType.EOS:
            print(f"\nЗавершено. Захвачено пар: {min(captured_left, captured_right)}")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"\nERROR: {err}")
            loop.quit()
        elif t == Gst.MessageType.ELEMENT:
            if message.get_structure().get_name() == "GstMultiFileSink":
                filename = message.get_structure().get_string("filename")
                
                if "cam0" in filename:
                    captured_left += 1
                    print(f"Левая: кадр {captured_left}")
                elif "cam1" in filename:
                    captured_right += 1
                    print(f"Правая: кадр {captured_right}")
                
                if captured_left == captured_right:
                    print(f"Пара #{captured_left}/{max_frames} сохранена")
                    print("-" * 30)
                
                if captured_left >= max_frames and captured_right >= max_frames:
                    pipeline.send_event(Gst.Event.new_eos())
                    
        elif t == Gst.MessageType.STATE_CHANGED:
            if isinstance(message.src, Gst.Pipeline):
                old, new, pending = message.parse_state_changed()
                if new == Gst.State.PLAYING and old == Gst.State.PAUSED:
                    print("\nPipeline запущен!")
                    print(f"Кадры сохраняются каждые {capture_interval} сек")
                    print("Ctrl+C для остановки\n")
        return True
    
    bus.connect("message", on_message)
    
    print("Запуск...")
    pipeline.set_state(Gst.State.PLAYING)
    
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nОстановка...")
    
    pipeline.set_state(Gst.State.NULL)
    
    print(f"\nРЕЗУЛЬТАТЫ:")
    print(f"Левая камера: {captured_left} фото")
    print(f"Правая камера: {captured_right} фото")
    print(f"Синхронных пар: {min(captured_left, captured_right)}")
    print(f"Файлы в: {frames_dir}/pairs/")
    
    if min(captured_left, captured_right) >= 15:
        print("\nДостаточно для стерео калибровки!")
        print("Запустите: python3 run_full_calibration.py")

if __name__ == "__main__":
    main()