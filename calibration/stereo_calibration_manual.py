#!/usr/bin/env python3
"""
Захват калибровочных фото с одной или двух камер
Доска: 11x8 клеток (10x7 углов), размер клетки 23.5мм
Версия с захватом через valve элемент
"""

import sys
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import time
from datetime import datetime
import json

Gst.init(None)

# Глобальные переменные
capture_mode = 0  # 0 - левая, 1 - правая, 2 - обе
frame_counter = 0
capture_interval = 5.0  # интервал между снимками в секундах
max_frames = 50  # максимальное количество кадров
frames_dir = "calibration_data"
pipeline = None
loop = None
start_time = 0
last_capture_time = 0
valve_left = None
valve_right = None

# Параметры калибровочной доски
BOARD_SIZE = (9, 6)  # количество внутренних углов
SQUARE_SIZE = 25.0  # размер клетки в мм

def ensure_directories():
    """Создание директорий для сохранения"""
    dirs = [
        frames_dir,
        os.path.join(frames_dir, "left"),
        os.path.join(frames_dir, "right"),
        os.path.join(frames_dir, "pairs")
    ]
    
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
    
    # Сохраняем информацию о калибровочной доске
    board_info = {
        "board_size": BOARD_SIZE,
        "square_size_mm": SQUARE_SIZE,
        "cells": (10, 7),
        "capture_date": datetime.now().isoformat(),
        "camera_setup": {
            "angle_between_cameras": 85,
            "tilt_angle": -15,
            "resolution": {"width": 3840, "height": 2160}
        }
    }
    
    with open(os.path.join(frames_dir, "calibration_info.json"), "w") as f:
        json.dump(board_info, f, indent=2)
    
    print(f"📁 Директории созданы в: {frames_dir}/")

def create_single_camera_pipeline(camera_id):
    """Pipeline с valve для управления захватом"""
    cam_width = 3840
    cam_height = 2160
    
    camera_name = "left" if camera_id == 0 else "right"
    
    # Pipeline с tee для разделения потока на preview и capture
    pipeline_str = f"""
        nvarguscamerasrc sensor-id={camera_id} !
        video/x-raw(memory:NVMM),width={cam_width},height={cam_height},framerate=30/1,format=NV12 !
        tee name=t !
        
        queue !
        nvvideoconvert !
        nvegltransform !
        nveglglessink sync=false
        
        t. !
        queue !
        valve name=valve drop=true !
        nvjpegenc !
        multifilesink name=filesink 
            location={frames_dir}/{camera_name}/{camera_name}_%05d.jpg 
            max-files={max_frames}
            post-messages=true
    """
    
    return Gst.parse_launch(pipeline_str)

def create_dual_camera_pipeline():
    """Pipeline для двух камер с независимым управлением"""
    cam_width = 3840
    cam_height = 2160
    
    # Два независимых pipeline для каждой камеры
    pipeline_str = f"""
        nvcompositor name=comp 
            sink_0::xpos=0 
            sink_0::ypos=0 
            sink_0::width={cam_width//2} 
            sink_0::height={cam_height//2}
            sink_1::xpos={cam_width//2} 
            sink_1::ypos=0 
            sink_1::width={cam_width//2} 
            sink_1::height={cam_height//2}
            width={cam_width} 
            height={cam_height//2} !
        
        nvvideoconvert !
        nvegltransform !
        nveglglessink sync=false
        
        nvarguscamerasrc sensor-id=0 !
        video/x-raw(memory:NVMM),width={cam_width},height={cam_height},framerate=30/1,format=NV12 !
        tee name=t0 !
        queue !
        nvvideoconvert !
        comp.sink_0
        
        t0. !
        queue !
        valve name=valve_left drop=true !
        nvjpegenc !
        filesink name=filesink_left location=/dev/null
        
        nvarguscamerasrc sensor-id=1 !
        video/x-raw(memory:NVMM),width={cam_width},height={cam_height},framerate=30/1,format=NV12 !
        tee name=t1 !
        queue !
        nvvideoconvert !
        comp.sink_1
        
        t1. !
        queue !
        valve name=valve_right drop=true !
        nvjpegenc !
        filesink name=filesink_right location=/dev/null
    """
    
    return Gst.parse_launch(pipeline_str)

def capture_frame():
    """Захват одного кадра через valve"""
    global frame_counter, pipeline, valve_left, valve_right
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    
    if capture_mode == 0:  # Левая камера
        valve = pipeline.get_by_name("valve")
        filesink = pipeline.get_by_name("filesink")
        if valve and filesink:
            # Открываем valve на короткое время
            valve.set_property("drop", False)
            GLib.timeout_add(100, lambda: valve.set_property("drop", True))
            return True
            
    elif capture_mode == 1:  # Правая камера
        valve = pipeline.get_by_name("valve")
        filesink = pipeline.get_by_name("filesink")
        if valve and filesink:
            valve.set_property("drop", False)
            GLib.timeout_add(100, lambda: valve.set_property("drop", True))
            return True
            
    else:  # Обе камеры
        if valve_left and valve_right:
            # Устанавливаем имена файлов
            filesink_left = pipeline.get_by_name("filesink_left")
            filesink_right = pipeline.get_by_name("filesink_right")
            
            if filesink_left and filesink_right:
                left_file = f"{frames_dir}/pairs/left_{frame_counter:04d}_{timestamp}.jpg"
                right_file = f"{frames_dir}/pairs/right_{frame_counter:04d}_{timestamp}.jpg"
                
                filesink_left.set_property("location", left_file)
                filesink_right.set_property("location", right_file)
                
                # Открываем оба valve одновременно
                valve_left.set_property("drop", False)
                valve_right.set_property("drop", False)
                
                # Закрываем через 100мс
                GLib.timeout_add(100, lambda: [
                    valve_left.set_property("drop", True),
                    valve_right.set_property("drop", True)
                ])
                return True
    
    return False

def capture_timer_callback():
    """Callback для таймера захвата"""
    global frame_counter, last_capture_time
    
    if frame_counter >= max_frames:
        print(f"\n✅ Захвачено {max_frames} кадров!")
        if loop:
            loop.quit()
        return False
    
    current_time = time.time()
    
    # Проверяем интервал
    if current_time - last_capture_time < capture_interval:
        return True  # Еще рано
    
    # Захватываем кадр
    if capture_frame():
        frame_counter += 1
        last_capture_time = current_time
        
        mode_name = ["левая", "правая", "пара"][capture_mode]
        print(f"\n💾 Кадр #{frame_counter}/{max_frames} ({mode_name}) сохранен")
    
    return True  # Продолжаем

def update_status():
    """Обновление статуса в консоли"""
    global frame_counter
    
    if frame_counter < max_frames:
        time_to_next = capture_interval - (time.time() - last_capture_time)
        if time_to_next > 0:
            print(f"\r📊 Захвачено: {frame_counter}/{max_frames} | След. через: {time_to_next:.0f}с   ", end='', flush=True)
    
    return True

def create_simple_pipeline(camera_id):
    """Самый простой pipeline с периодическим сохранением"""
    cam_width = 3840
    cam_height = 2160
    
    camera_name = "left" if camera_id == 0 else "right"
    
    # Используем videoconvert вместо nvvideoconvert для RGB преобразования
    # Или остаемся с JPEG высокого качества для совместимости
    use_png = False  # PNG требует больше ресурсов на Jetson
    
    if use_png:
        # PNG версия (может быть медленнее)
        pipeline_str = f"""
            nvarguscamerasrc sensor-id={camera_id} !
            video/x-raw(memory:NVMM),width={cam_width},height={cam_height},framerate=30/1,format=NV12 !
            tee name=t !
            
            queue !
            nvvideoconvert !
            video/x-raw(memory:NVMM),width={cam_width},height={cam_height} !
            nvegltransform !
            nveglglessink sync=false
            
            t. !
            queue !
            videorate !
            video/x-raw(memory:NVMM),width={cam_width},height={cam_height},framerate=1/{int(capture_interval)} !
            nvvideoconvert !
            video/x-raw,width={cam_width},height={cam_height},format=I420 !
            videoconvert !
            video/x-raw,format=RGB !
            pngenc compression-level=3 !
            multifilesink 
                location={frames_dir}/{camera_name}/{camera_name}_%05d.png 
                max-files={max_frames}
                post-messages=true
        """
        print(f"🖼️  Формат: PNG (без потерь)")
    else:
        # JPEG версия с максимальным качеством (рекомендуется для Jetson)
        pipeline_str = f"""
            nvarguscamerasrc sensor-id={camera_id} !
            video/x-raw(memory:NVMM),width={cam_width},height={cam_height},framerate=30/1,format=NV12 !
            tee name=t !
            
            queue !
            nvvideoconvert !
            video/x-raw(memory:NVMM),width={cam_width},height={cam_height} !
            nvegltransform !
            nveglglessink sync=false
            
            t. !
            queue !
            videorate !
            video/x-raw(memory:NVMM),width={cam_width},height={cam_height},framerate=1/{int(capture_interval)} !
            nvjpegenc quality=100 !
            multifilesink 
                location={frames_dir}/{camera_name}/{camera_name}_%05d.jpg 
                max-files={max_frames}
                post-messages=true
        """
        print(f"🖼️  Формат: JPEG (качество 100% - практически без потерь)")
    
    print(f"📐 Разрешение захвата: {cam_width}x{cam_height} (4K)")
    print(f"⚠️  Файлы будут ~3-5 MB каждый")
    
    return Gst.parse_launch(pipeline_str)

def main():
    global capture_mode, max_frames, capture_interval, pipeline, loop, start_time, last_capture_time
    global valve_left, valve_right
    
    # Парсинг аргументов
    if len(sys.argv) < 2:
        print("❌ Укажите режим: 0 (левая), 1 (правая), 2 (обе)")
        sys.exit(1)
    
    capture_mode = int(sys.argv[1])
    if capture_mode not in [0, 1, 2]:
        print("❌ Режим должен быть 0, 1 или 2")
        sys.exit(1)
    
    # Опциональные параметры
    if len(sys.argv) > 2:
        max_frames = int(sys.argv[2])
    if len(sys.argv) > 3:
        capture_interval = float(sys.argv[3])
    
    print("📸 ЗАХВАТ КАЛИБРОВОЧНЫХ ФОТО")
    print("=" * 60)
    print(f"🎯 Калибровочная доска: {BOARD_SIZE[0]}x{BOARD_SIZE[1]} углов")
    print(f"📏 Размер клетки: {SQUARE_SIZE} мм")
    print(f"📷 Режим: {['Левая камера', 'Правая камера', 'Обе камеры'][capture_mode]}")
    print(f"🔢 Количество фото: {max_frames}")
    print(f"⏱️  Интервал: {capture_interval} сек")
    print("=" * 60)
    
    # Создаем директории
    ensure_directories()
    
    # Создаем pipeline - используем простой вариант с videorate
    if capture_mode == 2:
        print("📝 Для режима двух камер запустите по очереди:")
        print("   python3 calibration_capture.py 0  # для левой")
        print("   python3 calibration_capture.py 1  # для правой")
        sys.exit(0)
    else:
        camera_id = capture_mode
        pipeline = create_simple_pipeline(camera_id)
    
    # Настройка bus и loop
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    loop = GLib.MainLoop()
    
    captured_count = 0
    
    def on_message(bus, message):
        nonlocal captured_count
        
        t = message.type
        if t == Gst.MessageType.EOS:
            print(f"\n🏁 Завершено. Захвачено {captured_count} кадров")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"\n❌ ERROR: {err}")
            loop.quit()
        elif t == Gst.MessageType.ELEMENT:
            # Сообщение от multifilesink о сохранении файла
            if message.get_structure().get_name() == "GstMultiFileSink":
                captured_count += 1
                filename = message.get_structure().get_string("filename")
                
                # Проверяем разрешение первого сохраненного файла
                if captured_count == 1:
                    import cv2
                    img = cv2.imread(filename)
                    if img is not None:
                        h, w = img.shape[:2]
                        print(f"✅ Подтверждено разрешение: {w}x{h}")
                        if w != 3840 or h != 2160:
                            print(f"⚠️  ВНИМАНИЕ: Разрешение не 4K! Получено {w}x{h}")
                
                print(f"💾 Кадр #{captured_count}/{max_frames} сохранен: {os.path.basename(filename)}")
                
                if captured_count >= max_frames:
                    print(f"\n✅ Захвачено {max_frames} кадров!")
                    pipeline.send_event(Gst.Event.new_eos())
                    
        elif t == Gst.MessageType.STATE_CHANGED:
            if isinstance(message.src, Gst.Pipeline):
                old, new, pending = message.parse_state_changed()
                if new == Gst.State.PLAYING and old == Gst.State.PAUSED:
                    print("\n▶️  Pipeline запущен!")
                    print(f"📸 Кадры будут сохраняться каждые {capture_interval} секунд")
                    print("⏸️  Ctrl+C для остановки\n")
        return True
    
    bus.connect("message", on_message)
    
    # Запуск pipeline
    print("🚀 Запуск pipeline...")
    start_time = time.time()
    pipeline.set_state(Gst.State.PLAYING)
    
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n⏸️  Остановка...")
    
    # Очистка
    pipeline.set_state(Gst.State.NULL)
    
    # Показываем результаты
    print(f"\n📊 РЕЗУЛЬТАТЫ:")
    
    if capture_mode == 0:
        files = [f for f in os.listdir(os.path.join(frames_dir, "left")) if f.endswith(('.jpg', '.png'))]
        print(f"📷 Файлы левой камеры: {len(files)} шт.")
        # Проверяем разрешение последнего файла
        if files:
            import cv2
            last_file = os.path.join(frames_dir, "left", sorted(files)[-1])
            img = cv2.imread(last_file)
            if img is not None:
                h, w = img.shape[:2]
                print(f"📐 Разрешение файлов: {w}x{h}")
                file_size = os.path.getsize(last_file) / (1024*1024)
                print(f"📦 Размер файла: {file_size:.2f} MB")
                print(f"🖼️  Формат: {os.path.splitext(last_file)[1].upper()}")
    else:
        files = [f for f in os.listdir(os.path.join(frames_dir, "right")) if f.endswith(('.jpg', '.png'))]
        print(f"📷 Файлы правой камеры: {len(files)} шт.")
        # Проверяем разрешение последнего файла
        if files:
            import cv2
            last_file = os.path.join(frames_dir, "right", sorted(files)[-1])
            img = cv2.imread(last_file)
            if img is not None:
                h, w = img.shape[:2]
                print(f"📐 Разрешение файлов: {w}x{h}")
                file_size = os.path.getsize(last_file) / (1024*1024)
                print(f"📦 Размер файла: {file_size:.2f} MB")
                print(f"🖼️  Формат: {os.path.splitext(last_file)[1].upper()}")
    
    if files:
        print(f"📁 Сохранено в: {frames_dir}/")
        print("\n💡 Следующие шаги:")
        print("   1. Проверьте качество фото")
        print("   2. Убедитесь, что все углы доски видны")
        print("   3. Запустите калибровку:")
        print("      python3 calibrate_fisheye_cameras.py")

if __name__ == "__main__":
    import cv2  # Импортируем для проверки разрешения
    
    os.environ['GST_PLUGIN_PATH'] = os.getcwd() + ":" + os.environ.get('GST_PLUGIN_PATH', '')
    
    print("\n📌 Использование:")
    print("   python3 calibration_capture.py <режим> [кол-во] [интервал]")
    print("\n   Режимы:")
    print("   0 - только левая камера")
    print("   1 - только правая камера")
    print("   2 - обе камеры (запустите 0 и 1 по очереди)")
    print("\n   Примеры:")
    print("   python3 calibration_capture.py 0        # 50 фото с левой")
    print("   python3 calibration_capture.py 1 30     # 30 фото с правой")
    print("   python3 calibration_capture.py 0 40 3   # 40 фото, интервал 3с\n")
    
    main()