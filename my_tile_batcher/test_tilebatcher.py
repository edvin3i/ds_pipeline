#!/usr/bin/env python3
"""
test_save_tiles.py - Проверка работы батча
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import sys
import ctypes
import os
PANORAMA_PATH = os.path.join(os.path.dirname(__file__), "panorama.jpg")

Gst.init(None)

def test_batch_creation():
    """Простой тест что батч создаётся"""
    print("\n" + "="*60)
    print("ТЕСТ 1: Проверка создания батча")
    print("="*60)
    
    pipeline_str = """
        filesrc location=panorama.jpg !
        jpegdec !
        nvvideoconvert !
        video/x-raw(memory:NVMM),format=RGBA,width=6528,height=1632 !
        nvtilebatcher name=batcher silent=false !
        fakesink name=sink
    """
    
    pipeline = Gst.parse_launch(pipeline_str)
    
    batcher = pipeline.get_by_name('batcher')
    sink = pipeline.get_by_name('sink')
    
    # Счётчик батчей
    batch_count = [0]
    
    def probe_callback(pad, info):
        buffer = info.get_buffer()
        if buffer:
            caps = pad.get_current_caps()
            if caps:
                struct = caps.get_structure(0)
                width = struct.get_int('width')[1]
                height = struct.get_int('height')[1]
                
                # Проверяем batch-size
                batch_size = struct.get_int('batch-size')
                if batch_size[0]:  # Если есть это поле
                    batch_size_val = batch_size[1]
                    print(f"\n✓ Батч #{batch_count[0]}:")
                    print(f"  Размер тайла: {width}x{height}")
                    print(f"  Количество тайлов: {batch_size_val}")
                    print(f"  Buffer size: {buffer.get_size()} bytes")
                    
                    if batch_size_val == 6:
                        print(f"  ✓ УСПЕХ: Батч из 6 тайлов создан!")
                    
                batch_count[0] += 1
                
        return Gst.PadProbeReturn.OK
    
    src_pad = batcher.get_static_pad("src")
    src_pad.add_probe(Gst.PadProbeType.BUFFER, probe_callback)
    
    # Запускаем
    pipeline.set_state(Gst.State.PLAYING)
    
    # Ждём обработки
    bus = pipeline.get_bus()
    msg = bus.timed_pop_filtered(3 * Gst.SECOND, 
                                 Gst.MessageType.ERROR | Gst.MessageType.EOS)
    
    pipeline.set_state(Gst.State.NULL)
    
    if batch_count[0] > 0:
        print(f"\n✓ Тест 1 завершён. Обработано батчей: {batch_count[0]}")
        return True
    else:
        print("\n✗ Батчи не были созданы")
        return False

def test_batch_content():
    """Проверка что батч создается без краша"""
    print("\n" + "="*60)
    print("ТЕСТ 4: Проверка создания батча (без конвертации)")
    print("="*60)
    
    # Просто проверяем что батч создаётся и проходит через pipeline
    pipeline_str = """
        filesrc location=panorama.jpg num-buffers=1 !
        jpegdec !
        nvvideoconvert !
        video/x-raw(memory:NVMM),format=RGBA,width=6528,height=1632 !
        nvtilebatcher name=batcher silent=false !
        fakesink name=sink signal-handoffs=true
    """
    
    success = [False]
    
    def on_handoff(sink, buffer, pad):
        print("  ✓ Батч получен sink без ошибок!")
        
        # Пытаемся получить caps
        caps = pad.get_current_caps()
        if caps:
            struct = caps.get_structure(0)
            width = struct.get_int('width')[1] if struct.get_int('width')[0] else 0
            height = struct.get_int('height')[1] if struct.get_int('height')[0] else 0
            batch_size = struct.get_int('batch-size')
            
            print(f"  Размеры: {width}x{height}")
            if batch_size[0]:
                print(f"  Batch size: {batch_size[1]}")
        
        success[0] = True
    
    try:
        pipeline = Gst.parse_launch(pipeline_str)
        sink = pipeline.get_by_name('sink')
        sink.connect('handoff', on_handoff)
        
        pipeline.set_state(Gst.State.PLAYING)
        
        bus = pipeline.get_bus()
        msg = bus.timed_pop_filtered(3 * Gst.SECOND,
                                     Gst.MessageType.ERROR | Gst.MessageType.EOS)
        
        if msg and msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            print(f"\n✗ Ошибка: {err}")
            return False
        
        pipeline.set_state(Gst.State.NULL)
        
        if success[0]:
            print("\n✓ Батч создан корректно!")
            return True
        else:
            print("\n✗ Батч не был получен")
            return False
            
    except Exception as e:
        print(f"\n✗ Исключение: {e}")
        return False

def extract_tiles_directly():
    """Тест 2: Извлечение тайлов напрямую из буфера nvtilebatcher"""
    print("\n" + "="*60)
    print("ТЕСТ 2: Извлечение тайлов напрямую из буфера nvtilebatcher")
    print("="*60)
    
    import numpy as np
    from PIL import Image
    import traceback
    
    try:
        import pyds
    except ImportError:
        print("✗ pyds не установлен, пропускаем тест")
        return False
    
    # ИСПРАВЛЕНО: добавлены размеры
    pipeline_str = f"""
        filesrc location={PANORAMA_PATH} ! 
        jpegdec ! 
        videoconvert ! 
        video/x-raw,format=RGBA ! 
        nvvideoconvert ! 
        video/x-raw(memory:NVMM),format=RGBA,width=6528,height=1632 ! 
        nvtilebatcher name=batcher ! 
        fakesink name=sink
    """
    
    pipeline = Gst.parse_launch(pipeline_str)
    
    # Счётчик сохранённых тайлов
    saved_tiles = [0]
    
    def buffer_probe(pad, info):
        """Probe для перехвата буфера после nvtilebatcher"""
        import traceback  # ДОБАВЬТЕ ИМПОРТ
        
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        
        try:
            gst_buffer = info.get_buffer()
            if not gst_buffer:
                return Gst.PadProbeReturn.OK
            
            # ПРАВИЛЬНЫЙ способ получения surface
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
            
            # Но для surface нужно сначала замапить буфер
            success, map_info = gst_buffer.map(Gst.MapFlags.READ)
            if not success:
                print("Failed to map buffer")
                return Gst.PadProbeReturn.OK
            
            # Теперь получаем surface из map_info.data
            # map_info.data - это capsule с указателем на NvBufSurface
            surface = pyds.NvBufSurface.cast(hash(map_info.data))
            
            print(f"surface.numFilled: {surface.numFilled}")
            print(f"surface.batchSize: {surface.batchSize}")
            
            # Не забудьте unmap!
            gst_buffer.unmap(map_info)

            for i in range(0):
                # ПРАВИЛЬНЫЙ способ доступа через функцию pyds
                params = pyds.get_nvds_buf_surface(buf_hash, i)
                
                # params - это numpy array (H, W, 4), НО у NvBufSurfaceParams есть другие поля
                # Получаем размеры из первого измерения массива
                height, width = params.shape[:2]
                
                # Для pitch используем стандартное выравнивание
                pitch = ((width * 4 + 255) // 256) * 256
                
                print(f"  Tile {i}: {width}x{height}")
                
                # params уже содержит пиксели! Не нужно дополнительно мапить
                # Просто конвертируем в RGB
                rgb_data = params[:, :, :3]
                
                # Сохраняем
                img = Image.fromarray(rgb_data)
                filename = f"tile_{i}_from_buffer.png"
                img.save(filename)
                
                # Позиция из метаданных
                if batch_meta:
                    l = batch_meta.frame_meta_list
                    while l:
                        frame = pyds.NvDsFrameMeta.cast(l.data)
                        if frame.surface_index == i:
                            print(f"    ✓ [{int(frame.misc_frame_info[0])},{int(frame.misc_frame_info[1])}] -> {filename}")
                            break
                        try:
                            l = l.next
                        except StopIteration:
                            break
                
                saved_tiles[0] += 1
                
        except Exception as e:
            print(f"Ошибка: {e}")
            traceback.print_exc()
        
        GLib.idle_add(lambda: pipeline.set_state(Gst.State.NULL))
        return Gst.PadProbeReturn.DROP
    
    # Добавляем probe на выход nvtilebatcher
    batcher = pipeline.get_by_name('batcher')
    srcpad = batcher.get_static_pad('src')
    srcpad.add_probe(Gst.PadProbeType.BUFFER, buffer_probe)
    
    # Запускаем
    pipeline.set_state(Gst.State.PLAYING)
    
    loop = GLib.MainLoop()
    
    def on_message(bus, message):
        t = message.type
        if t == Gst.MessageType.EOS or t == Gst.MessageType.ERROR:
            loop.quit()
    
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect('message', on_message)
    
    # Таймаут
    GLib.timeout_add_seconds(5, lambda: loop.quit())
    
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    
    pipeline.set_state(Gst.State.NULL)
    
    print(f"\n{'✓' if saved_tiles[0] == 6 else '✗'} Сохранено {saved_tiles[0]}/6 тайлов")
    
    return saved_tiles[0] == 6

def test_batch_to_nvinfer():
    """Тест батча с реальной YOLO моделью"""
    print("\n" + "="*60)
    print("ТЕСТ 3: nvtilebatcher + YOLO11 (детекция мяча)")
    print("="*60)
    
    config_path = "config_infer.txt"
    
    if not os.path.exists(config_path):
        print("⚠ config_infer.txt не найден")
        return False
    
    if not os.path.exists("yolo11s-onlyball_batch_6.engine"):
        print("⚠ yolo11s-onlyball_batch_6.engine не найден")
        return False
    
    pipeline_str = f"""
        filesrc location=panorama.jpg num-buffers=10 !
        jpegdec !
        nvvideoconvert !
        video/x-raw(memory:NVMM),format=RGBA,width=6528,height=1632 !
        nvtilebatcher name=batcher !
        nvinfer config-file-path={config_path} !
        fakesink name=sink
    """
    
    detections = [0]
    
    def probe_callback(pad, info):
        buffer = info.get_buffer()
        if buffer:
            # Получаем метаданные
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
            if batch_meta:
                l_frame = batch_meta.frame_meta_list
                while l_frame is not None:
                    try:
                        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                        l_obj = frame_meta.obj_meta_list
                        
                        while l_obj is not None:
                            try:
                                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                                detections[0] += 1
                                print(f"  Tile {frame_meta.source_id}: "
                                      f"мяч найден (confidence={obj_meta.confidence:.2f})")
                                l_obj = l_obj.next
                            except StopIteration:
                                break
                        
                        l_frame = l_frame.next
                    except StopIteration:
                        break
        
        return Gst.PadProbeReturn.OK
    
    try:
        import pyds
        
        pipeline = Gst.parse_launch(pipeline_str)
        sink = pipeline.get_by_name('sink')
        sink_pad = sink.get_static_pad("sink")
        sink_pad.add_probe(Gst.PadProbeType.BUFFER, probe_callback)
        
        pipeline.set_state(Gst.State.PLAYING)
        
        bus = pipeline.get_bus()
        msg = bus.timed_pop_filtered(30 * Gst.SECOND,
                                     Gst.MessageType.ERROR | Gst.MessageType.EOS)
        
        if msg and msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            print(f"\n✗ Ошибка: {err}")
            return False
        
        pipeline.set_state(Gst.State.NULL)
        
        if detections[0] > 0:
            print(f"\n✓ YOLO работает! Найдено детекций: {detections[0]}")
            return True
        else:
            print("\n⚠ Детекций не найдено (возможно мяча нет на панораме)")
            return True  # Не ошибка
            
    except ImportError:
        print("\n⚠ pyds не установлен, пропускаем проверку детекций")
        # Просто запускаем без подсчёта
        try:
            pipeline = Gst.parse_launch(pipeline_str)
            pipeline.set_state(Gst.State.PLAYING)
            
            bus = pipeline.get_bus()
            msg = bus.timed_pop_filtered(30 * Gst.SECOND,
                                         Gst.MessageType.ERROR | Gst.MessageType.EOS)
            
            if msg and msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                print(f"\n✗ Ошибка: {err}")
                return False
            
            pipeline.set_state(Gst.State.NULL)
            print("\n✓ Pipeline с YOLO отработал без ошибок")
            return True
            
        except Exception as e:
            print(f"\n✗ Исключение: {e}")
            return False
    
    except Exception as e:
        print(f"\n✗ Исключение: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\nТЕСТИРОВАНИЕ NVTILEBATCHER")
    print("="*60)
    
    # Тест 1: Создание батча
    batch_ok = test_batch_creation()
    
    # Тест 2: Извлечение тайлов
    tiles_ok = extract_tiles_directly()

    content_ok = test_batch_content()
    
    # Тест 3: Совместимость с nvinfer
    # nvinfer_ok = test_batch_to_nvinfer()
    
    # Результаты
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ:")
    print("="*60)
    
    # if batch_ok:
    #     print("✓ Батч создаётся корректно")
    # else:
    #     print("✗ Проблема с созданием батча")
    
    # # Проверяем тайлы
    # tiles_found = 0
    # for i in range(6):
    #     if os.path.exists(f"tile_{i}.jpg"):
    #         tiles_found += 1
    
    # if tiles_found == 6:
    #     print(f"✓ Все 6 тайлов извлечены правильно")
    # else:
    #     print(f"⚠ Извлечено {tiles_found} из 6 тайлов")
    
    # if batch_ok and tiles_found == 6:
    #     print("\n🎉 УСПЕХ! Плагин nvtilebatcher работает правильно!")
    #     print("   Батч из 6 тайлов 1024x1024 создаётся корректно")
    #     return 0
    # else:
    #     print("\n⚠ Есть проблемы, см. выше")
    #     return 1

if __name__ == "__main__":
    sys.exit(main())