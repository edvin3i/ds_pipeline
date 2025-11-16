#!/usr/bin/env python3
"""
Автоматический тест зума - плавно меняет FOV и двигает камеру по краям
"""

import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import time

Gst.init(None)

PANORAMA_WIDTH = 5700
PANORAMA_HEIGHT = 1900

# Тестовые позиции (X, Y, название)
TEST_POSITIONS = [
    (2850, 600, "Центр верх"),
    (2850, 1300, "Центр низ"),
    (500, 950, "Левый край"),
    (5200, 950, "Правый край"),
    (500, 600, "Верх лево (угол)"),
    (5200, 600, "Верх право (угол)"),
    (500, 1300, "Низ лево (угол)"),
    (5200, 1300, "Низ право (угол)"),
]

# Тестовые FOV
TEST_FOV = [40.0, 50.0, 60.0, 68.0]

class ZoomTester:
    def __init__(self):
        self.pipeline = None
        self.loop = None
        self.virtualcam = None
        self.test_index = 0
        self.fov_index = 0
        self.position_index = 0

    def create_pipeline(self, left_file, right_file):
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

            nvdsstitcher name=stitcher
                width=5700
                height=1900
                gpu-id=0 !

            nvdsvirtualcam name=vcam
                auto-follow=true
                enable-debug=false !

            queue !
            nvvideoconvert !
            video/x-raw(memory:NVMM),format=RGBA !
            nvegltransform !
            nveglglessink sync=0
        """

        return Gst.parse_launch(pipeline_str)

    def update_position(self):
        """Обновляет позицию мяча и FOV"""
        if self.position_index >= len(TEST_POSITIONS):
            # Переходим к следующему FOV
            self.position_index = 0
            self.fov_index += 1

            if self.fov_index >= len(TEST_FOV):
                # Все тесты завершены
                print("\n✅ Все тесты завершены!")
                self.loop.quit()
                return False

        # Текущий FOV и позиция
        current_fov = TEST_FOV[self.fov_index]
        ball_x, ball_y, pos_name = TEST_POSITIONS[self.position_index]

        print(f"\n📊 Тест {self.fov_index * len(TEST_POSITIONS) + self.position_index + 1}/{len(TEST_FOV) * len(TEST_POSITIONS)}")
        print(f"   FOV={current_fov}° | Позиция: {pos_name} ({ball_x}, {ball_y})")

        # Устанавливаем позицию мяча
        self.virtualcam.set_property('ball-x', float(ball_x))
        self.virtualcam.set_property('ball-y', float(ball_y))
        self.virtualcam.set_property('ball-radius', 50.0)

        # Переходим к следующей позиции
        self.position_index += 1

        # Продолжаем тест через 3 секунды
        GLib.timeout_add(3000, self.update_position)
        return False

    def run(self, left_file, right_file):
        try:
            self.pipeline = self.create_pipeline(left_file, right_file)
        except Exception as e:
            print(f"❌ Ошибка создания pipeline: {e}")
            return False

        # Получаем элемент virtualcam
        self.virtualcam = self.pipeline.get_by_name('vcam')
        if not self.virtualcam:
            print("❌ Не найден элемент nvdsvirtualcam")
            return False

        print("=" * 80)
        print("АВТОМАТИЧЕСКИЙ ТЕСТ ЗУМА И ГРАНИЦ")
        print("=" * 80)
        print(f"\n🎯 Будет протестировано:")
        print(f"   - {len(TEST_FOV)} значений FOV: {TEST_FOV}")
        print(f"   - {len(TEST_POSITIONS)} позиций на панораме")
        print(f"   - Всего: {len(TEST_FOV) * len(TEST_POSITIONS)} комбинаций")
        print(f"\n⏱️  Каждая позиция показывается 3 секунды")
        print(f"   Общее время: ~{len(TEST_FOV) * len(TEST_POSITIONS) * 3} секунд")
        print("\n" + "=" * 80 + "\n")

        # Запускаем pipeline
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("❌ Не удалось запустить pipeline")
            return False

        # Ждём PLAYING
        state_ret, state, pending = self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
        if state_ret == Gst.StateChangeReturn.FAILURE:
            print("❌ Pipeline не перешёл в состояние PLAYING")
            return False

        print(f"✅ Pipeline запущен")

        # Ждём 2 секунды перед началом
        GLib.timeout_add(2000, self.update_position)

        # Главный цикл
        self.loop = GLib.MainLoop()
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\n\n⚠️  Прервано пользователем")

        # Остановка
        self.pipeline.set_state(Gst.State.NULL)

        return True

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Использование: {sys.argv[0]} <left.mp4> <right.mp4>")
        sys.exit(1)

    tester = ZoomTester()
    success = tester.run(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
