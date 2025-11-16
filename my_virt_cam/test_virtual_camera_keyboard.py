

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Панорама + Виртуальная камера с ИМИТАЦИЕЙ слежения за мячом
Управление "виртуальным мячом" клавишами W/A/S/Dda
Камера автоматически следит за мячом (auto-follow=true)
"""

import sys
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import threading
import time

# Используем pynput для глобального перехвата клавиатуры (работает независимо от фокуса)
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("⚠️  Библиотека pynput не установлена. Используется fallback режим.")
    print("   Для глобального перехвата клавиш установите: pip3 install pynput")
    import termios
    import tty
    import select

Gst.init(None)

# Константы размеров
PANORAMA_WIDTH = 5700
PANORAMA_HEIGHT = 1900

class VirtualBallController:
    def __init__(self):
        self.pipeline = None
        self.loop = None
        self.virtualcam = None
        self.running = True
        
        # ВИРТУАЛЬНЫЙ МЯЧ - начальная позиция чуть выше центра
        self.ball_x = PANORAMA_WIDTH / 2.0   # 2850 (центр по горизонтали)
        self.ball_y = 950.0                  # Выше центра (было 950) для лучшего обзора поля
        self.ball_radius = 20.0
        self.target_ball_size = 0.055  # Желаемый размер на экране
        
        # Скорость движения мяча (пиксели за шаг)
        self.ball_speed = 50.0
        self.ball_speed_fast = 150.0  # Для Shift+клавиша
        
        # Границы движения (с отступами)
        self.min_x = 100
        self.max_x = PANORAMA_WIDTH - 100
        self.min_y = 100
        self.max_y = PANORAMA_HEIGHT - 100
        
        # Счётчик обновлений
        self.update_count = 0
        
    def create_pipeline(self, left_file, right_file, display_mode="egl"):
        """Создает pipeline с виртуальной камерой в режиме auto-follow"""
        
        pipeline_str = f"""
            filesrc location={left_file} ! 
            qtdemux ! h264parse ! nvv4l2decoder ! 
            nvvideoconvert ! 
            video/x-raw(memory:NVMM),format=RGBA !
            queue max-size-buffers=5 !
            nvstreammux0.sink_0
            
            filesrc location={right_file} ! 
            qtdemux ! h264parse ! nvv4l2decoder ! 
            nvvideoconvert !
            video/x-raw(memory:NVMM),format=RGBA !
            queue max-size-buffers=5 !
            nvstreammux0.sink_1
            
            nvstreammux name=nvstreammux0 
                batch-size=2 
                width=3840 
                height=2160 
                batched-push-timeout=40000 
                live-source=0 !
            
            nvdsstitch
                left-source-id=0
                right-source-id=1
                gpu-id=0
                panorama-width={PANORAMA_WIDTH}
                panorama-height={PANORAMA_HEIGHT} !
            
            queue max-size-buffers=3 !
            
            nvdsvirtualcam name=virtualcam
                panorama-width={PANORAMA_WIDTH}
                panorama-height={PANORAMA_HEIGHT}
                yaw=0
                pitch=15
                roll=0
                fov=68
                auto-follow=true
                smooth-factor=0.15
                ball-x={self.ball_x}
                ball-y={self.ball_y}
                ball-radius={self.ball_radius}
                target-ball-size={self.target_ball_size} !
            
            queue max-size-buffers=3 !
        """
        
        # Вывод
        if display_mode == "egl":
            pipeline_str += "nvegltransform ! nveglglessink sync=false async=false"
        elif display_mode == "x11":
            pipeline_str += "nvvideoconvert ! video/x-raw,format=RGBA ! ximagesink sync=false"
        else:
            pipeline_str += "nvvideoconvert ! video/x-raw,format=RGBA ! autovideosink sync=false"
        
        pipeline = Gst.parse_launch(pipeline_str)
        self.virtualcam = pipeline.get_by_name("virtualcam")
        
        if not self.virtualcam:
            raise RuntimeError("❌ Не удалось найти элемент nvdsvirtualcam!")
        
        print(f"✅ Найден элемент virtualcam: {self.virtualcam}")
        
        return pipeline
    
    def update_ball_position(self):
        """Обновляет позицию виртуального мяча в плагине"""
        if not self.virtualcam:
            return
        
        self.update_count += 1
        
        # ВАЖНО: используем GLib.idle_add для безопасности многопоточности
        def _do_update():
            try:
                if self.virtualcam:
                    self.virtualcam.set_property("ball-x", float(self.ball_x))
                    self.virtualcam.set_property("ball-y", float(self.ball_y))
                    self.virtualcam.set_property("ball-radius", float(self.ball_radius))
                    self.virtualcam.set_property("target-ball-size", float(self.target_ball_size))
            except Exception as e:
                print(f"\n❌ Ошибка обновления: {e}")
            return False
        
        GLib.idle_add(_do_update)
        self.print_status()
    
    def print_status(self):
        """Выводит текущее состояние"""
        # Вычисляем примерные углы камеры для информации
        # (это приблизительно, реальные значения вычисляет плагин)
        lon_min, lon_max = -90.0, 90.0
        lat_min, lat_max = 30.0, -20.0
        
        norm_x = self.ball_x / (PANORAMA_WIDTH - 1)
        approx_yaw = lon_min + norm_x * (lon_max - lon_min)
        
        norm_y = self.ball_y / (PANORAMA_HEIGHT - 1)
        approx_pitch = lat_min - norm_y * (lat_min - lat_max)
        
        print(f"\r⚽ Ball: ({self.ball_x:5.0f}, {self.ball_y:5.0f}) | "
              f"~Yaw: {approx_yaw:6.1f}° | ~Pitch: {approx_pitch:6.1f}° | "
              f"Radius: {self.ball_radius:4.1f}px | "
              f"Updates: {self.update_count}", 
              end='', flush=True)
    
    def move_ball(self, dx, dy, fast=False):
        """Перемещает виртуальный мяч"""
        speed = self.ball_speed_fast if fast else self.ball_speed
        
        self.ball_x += dx * speed
        self.ball_y += dy * speed
        
        # Ограничиваем координаты
        self.ball_x = max(self.min_x, min(self.max_x, self.ball_x))
        self.ball_y = max(self.min_y, min(self.max_y, self.ball_y))
        
        self.update_ball_position()
    
    def reset_ball(self):
        """Сброс мяча в центр"""
        self.ball_x = PANORAMA_WIDTH / 2.0
        self.ball_y = PANORAMA_HEIGHT / 2.0
        self.ball_radius = 20.0
        self.target_ball_size = 0.055
        self.update_ball_position()
        print("\n✅ Мяч возвращён в центр")
        self.print_status()
    
    def change_ball_size(self, delta):
        """Изменяет размер мяча"""
        self.ball_radius = max(5.0, min(50.0, self.ball_radius + delta))
        self.update_ball_position()
    
    def change_zoom(self, delta):
        """Изменяет желаемый размер мяча на экране (зум камеры)"""
        self.target_ball_size = max(0.03, min(0.15, self.target_ball_size + delta))
        self.update_ball_position()
    
    def handle_key(self, key, shift_pressed=False):
        """Обработка клавиш"""
        changed = False
        
        # Движение мяча - W/A/S/D
        if key == 'w':
            self.move_ball(0, -1, shift_pressed)  # Вверх (уменьшаем Y - камера смотрит вверх)
            changed = True
        elif key == 's':
            self.move_ball(0, 1, shift_pressed)   # Вниз (увеличиваем Y - камера смотрит вниз)
            changed = True
        elif key == 'a':
            self.move_ball(-1, 0, shift_pressed)  # Влево
            changed = True
        elif key == 'd':
            self.move_ball(1, 0, shift_pressed)   # Вправо
            changed = True
        
        # Размер мяча - Q/E
        elif key == 'q':
            self.change_ball_size(-2.0)
            changed = True
        elif key == 'e':
            self.change_ball_size(2.0)
            changed = True
        
        # Зум камеры - Z/X
        elif key == 'z':
            self.change_zoom(-0.005)
            changed = True
        elif key == 'x':
            self.change_zoom(0.005)
            changed = True
        
        # Сброс - R
        elif key == 'r':
            self.reset_ball()
            return True
        
        # Выход - ESC/Space
        elif key in ['\x1b', ' ']:
            print("\n\n👋 Выход...")
            self.running = False
            self.loop.quit()
            return False
        
        return True
    
    def keyboard_thread_func_pynput(self):
        """Поток обработки клавиатуры через pynput (глобальный перехват)"""
        print("⌨️  Поток клавиатуры запущен (глобальный перехват - работает независимо от фокуса)")

        def on_press(key):
            """Обработчик нажатия клавиш"""
            try:
                # Обычные символьные клавиши
                if hasattr(key, 'char') and key.char:
                    shift_pressed = key.char.isupper()
                    key_lower = key.char.lower()

                    if not self.handle_key(key_lower, shift_pressed):
                        return False  # Останавливает listener

            except AttributeError:
                # Специальные клавиши
                if key == keyboard.Key.esc:
                    if not self.handle_key('\x1b', False):
                        return False
                elif key == keyboard.Key.space:
                    if not self.handle_key(' ', False):
                        return False

            return True  # Продолжает слушать

        # Запускаем listener
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

        print("\n⌨️  Поток клавиатуры завершён")

    def keyboard_thread_func_fallback(self):
        """Fallback: обработка клавиатуры через termios (требует фокус на терминале)"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)
            print("⌨️  Поток клавиатуры запущен (требуется фокус на терминале)")

            while self.running:
                # Используем select для неблокирующего чтения
                readable, _, _ = select.select([sys.stdin], [], [], 0.05)

                if readable:
                    key = sys.stdin.read(1)

                    # Проверяем Shift (в raw режиме заглавные буквы)
                    shift_pressed = key.isupper()
                    key_lower = key.lower()

                    if not self.handle_key(key_lower, shift_pressed):
                        break

                time.sleep(0.01)

        except Exception as e:
            print(f"\n❌ Ошибка в потоке клавиатуры: {e}")
            import traceback
            traceback.print_exc()

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            print("\n⌨️  Поток клавиатуры завершён")
    
    def run(self, left_file, right_file, display_mode="egl"):
        """Запуск"""
        
        # Проверка файлов
        for f in [left_file, right_file]:
            if not os.path.exists(f):
                print(f"❌ Файл не найден: {f}")
                return False
        
        print("=" * 80)
        print("⚽ ВИРТУАЛЬНАЯ КАМЕРА - ИМИТАЦИЯ СЛЕЖЕНИЯ ЗА МЯЧОМ")
        print("=" * 80)
        print(f"\n📹 Источники:")
        print(f"   • Левый: {left_file}")
        print(f"   • Правый: {right_file}")
        print(f"\n📐 Панорама: {PANORAMA_WIDTH}x{PANORAMA_HEIGHT}")
        print(f"⚽ Начальная позиция мяча: ({self.ball_x:.0f}, {self.ball_y:.0f})")
        print(f"\n🎮 УПРАВЛЕНИЕ ВИРТУАЛЬНЫМ МЯЧОМ:")
        print(f"   W/A/S/D        - движение мяча (медленно)")
        print(f"   Shift+W/A/S/D  - движение мяча (быстро)")
        print(f"   Q/E            - уменьшить/увеличить размер мяча")
        print(f"   Z/X            - уменьшить/увеличить зум камеры")
        print(f"   R              - вернуть мяч в центр")
        print(f"   ESC/Space      - выход")

        if PYNPUT_AVAILABLE:
            print(f"\n✅ Глобальный перехват клавиш активен - работает даже с фокусом на окне!")
        else:
            print(f"\n⚠️  Для работы клавиш держите фокус на терминале (или установите: pip3 install pynput)")

        print(f"\n💡 Камера автоматически следит за мячом (auto-follow=true)")
        print("=" * 80 + "\n")
        
        # Создаем pipeline
        try:
            self.pipeline = self.create_pipeline(left_file, right_file, display_mode)
        except Exception as e:
            print(f"❌ Ошибка создания pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        self.loop = GLib.MainLoop()
        
        def on_message(bus, message):
            t = message.type
            if t == Gst.MessageType.EOS:
                print("\n🏁 Конец потока")
                self.running = False
                self.loop.quit()
            elif t == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                print(f"\n❌ Ошибка GStreamer: {err}")
                if debug:
                    print(f"   Debug: {debug}")
                self.running = False
                self.loop.quit()
            elif t == Gst.MessageType.STATE_CHANGED:
                if isinstance(message.src, Gst.Pipeline):
                    old, new, pending = message.parse_state_changed()
                    if new == Gst.State.PLAYING:
                        print("✅ Pipeline запущен!\n")
                        print("📷 Камера начинает слежение за виртуальным мячом\n")
                        self.print_status()
                        print("\n")
            return True
        
        bus.connect("message", on_message)
        
        # Запуск
        print("⏳ Запуск pipeline...")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        
        if ret == Gst.StateChangeReturn.FAILURE:
            print("❌ Не удалось запустить pipeline")
            return False
        
        # Ждём PLAYING
        state_ret, state, pending = self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
        if state_ret == Gst.StateChangeReturn.FAILURE:
            print("❌ Pipeline не перешёл в состояние PLAYING")
            return False
        
        print(f"✅ Pipeline в состоянии: {state}")

        # Поток клавиатуры - выбираем метод в зависимости от доступности pynput
        if PYNPUT_AVAILABLE:
            keyboard_func = self.keyboard_thread_func_pynput
            print("🎮 Используется глобальный перехват клавиатуры (работает с фокусом на окне)")
        else:
            keyboard_func = self.keyboard_thread_func_fallback
            print("⚠️  Требуется фокус на терминале для управления")

        keyboard_thread = threading.Thread(
            target=keyboard_func,
            daemon=True
        )
        keyboard_thread.start()
        
        # Главный цикл
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\n⏹️  Остановлено (Ctrl+C)")
            self.running = False
        
        # Очистка
        print("\n🧹 Завершение...")
        self.pipeline.set_state(Gst.State.NULL)
        print("✅ Готово!\n")
        
        return True


def main():
    if len(sys.argv) < 3:
        print("Использование: python3 virtual_ball_control.py left.mp4 right.mp4 [display_mode]")
        print("\nРежимы отображения:")
        print("  egl  - через EGL (по умолчанию)")
        print("  x11  - через X11")
        print("  auto - автовыбор")
        print("\nПример:")
        print("  python3 virtual_ball_control.py left.mp4 right.mp4")
        sys.exit(1)
    
    left_file = sys.argv[1]
    right_file = sys.argv[2]
    display_mode = sys.argv[3] if len(sys.argv) > 3 else "egl"
    
    # Настройка путей
    plugin_path = os.path.join(os.getcwd(), "src")
    os.environ['GST_PLUGIN_PATH'] = f"{plugin_path}:{os.environ.get('GST_PLUGIN_PATH', '')}"
    
    # Уменьшаем debug для чистоты
    os.environ['GST_DEBUG'] = '2'
    
    app = VirtualBallController()
    
    try:
        if app.run(left_file, right_file, display_mode):
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()