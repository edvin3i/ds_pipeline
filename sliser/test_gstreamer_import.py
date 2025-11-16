#!/usr/bin/env python3
"""
Тестовый скрипт для диагностики проблем с импортом GStreamer
"""
import sys
import os

print("=" * 60)
print("ДИАГНОСТИКА ИМПОРТА GSTREAMER")
print("=" * 60)

# 1. Информация о Python
print(f"\n1. Python информация:")
print(f"   Версия: {sys.version}")
print(f"   Исполняемый файл: {sys.executable}")
print(f"   Виртуальное окружение: {'Да' if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 'Нет'}")

# 2. Пути Python
print(f"\n2. Python пути (sys.path):")
for i, path in enumerate(sys.path[:5]):
    print(f"   [{i}] {path}")
print(f"   ... всего путей: {len(sys.path)}")

# 3. Проверка импорта gi
print(f"\n3. Импорт gi:")
try:
    import gi
    print(f"   ✓ gi импортирован успешно")
    print(f"   Путь к gi: {gi.__file__}")
    print(f"   Версия gi: {gi.__version__ if hasattr(gi, '__version__') else 'неизвестно'}")
except ImportError as e:
    print(f"   ✗ Ошибка импорта gi: {e}")
    print(f"   Попробуйте: sudo apt-get install python3-gi")
    sys.exit(1)

# 4. Проверка версий GStreamer
print(f"\n4. Проверка доступных версий GStreamer:")
try:
    available_versions = gi.get_required_version('Gst') if hasattr(gi, 'get_required_version') else None
    print(f"   Требуемая версия: {available_versions if available_versions else 'не установлена'}")
except:
    pass

try:
    gi.require_version('Gst', '1.0')
    print(f"   ✓ GStreamer 1.0 доступен")
except ValueError as e:
    print(f"   ✗ Ошибка версии: {e}")
    print(f"   Попробуйте: sudo apt-get install gir1.2-gstreamer-1.0")

# 5. Импорт Gst из gi.repository
print(f"\n5. Импорт from gi.repository:")
try:
    from gi.repository import Gst
    print(f"   ✓ Gst импортирован успешно")

    # Инициализация GStreamer
    Gst.init(None)
    version = Gst.version()
    print(f"   Версия GStreamer: {version[0]}.{version[1]}.{version[2]}")

except ImportError as e:
    print(f"   ✗ Ошибка импорта Gst: {e}")
    print(f"\n   РЕШЕНИЯ:")
    print(f"   1. Установите GObject Introspection файлы:")
    print(f"      sudo apt-get install gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0")
    print(f"   2. Добавьте системные пути в скрипт:")
    print(f"      sys.path.append('/usr/lib/python3/dist-packages')")
    print(f"   3. Используйте системный Python:")
    print(f"      /usr/bin/python3 script.py")
    sys.exit(1)

try:
    from gi.repository import GLib
    print(f"   ✓ GLib импортирован успешно")
except ImportError as e:
    print(f"   ✗ Ошибка импорта GLib: {e}")

# 6. Проверка переменных окружения
print(f"\n6. Переменные окружения:")
print(f"   LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'не установлена')}")
print(f"   GST_PLUGIN_PATH: {os.environ.get('GST_PLUGIN_PATH', 'не установлена')}")
print(f"   PYTHONPATH: {os.environ.get('PYTHONPATH', 'не установлена')}")

# 7. Проверка GStreamer элементов
print(f"\n7. Проверка GStreamer элементов:")
try:
    # Проверяем наличие основных элементов
    elements_to_check = ['filesrc', 'qtdemux', 'h264parse', 'nvv4l2decoder',
                         'nvvideoconvert', 'nvstreammux', 'nvdsstitch']

    for element in elements_to_check[:3]:  # Проверяем первые 3
        factory = Gst.ElementFactory.find(element)
        if factory:
            print(f"   ✓ {element}: найден")
        else:
            print(f"   ✗ {element}: НЕ найден")

except Exception as e:
    print(f"   Ошибка проверки элементов: {e}")

# 8. Рекомендации
print(f"\n8. ИТОГОВЫЕ РЕКОМЕНДАЦИИ:")
print(f"   Если проблема с 'from gi.repository import Gst':")
print(f"   -----------------------------------------------")
print(f"   В виртуальном окружении:")
print(f"   1. Пересоздайте venv: python3 -m venv myenv --system-site-packages")
print(f"   2. Или добавьте в начало скрипта:")
print(f"      import sys")
print(f"      sys.path.insert(0, '/usr/lib/python3/dist-packages')")
print(f"   3. Или запускайте без venv: /usr/bin/python3 script.py")

print("\n" + "=" * 60)
print("КОНЕЦ ДИАГНОСТИКИ")
print("=" * 60)