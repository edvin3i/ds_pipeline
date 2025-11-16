# Установка зависимостей для panorama_tiles_saver.py

## Проблема
При запуске скрипта возникает ошибка:
```python
ImportError: cannot import name 'Gst' from 'gi.repository'
```

## Решение для разных систем

### Ubuntu/Debian (обычный компьютер)

1. **Установка базовых Python bindings для GStreamer:**
```bash
sudo apt-get update
sudo apt-get install -y \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gtk-3.0 \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0
```

2. **Установка GStreamer и плагинов:**
```bash
sudo apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev
```

3. **Установка OpenCV (для обработки изображений):**
```bash
pip3 install opencv-python numpy
```

### NVIDIA Jetson

1. **Базовые пакеты (уже должны быть в JetPack):**
```bash
sudo apt-get install -y \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0
```

2. **NVIDIA-специфичные пакеты:**
```bash
# Обычно уже установлены с JetPack
sudo apt-get install -y \
    nvidia-jetpack \
    deepstream-6.0
```

### macOS

1. **Через Homebrew:**
```bash
brew install pygobject3 gtk+3 gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly
```

2. **Python пакеты:**
```bash
pip3 install PyGObject opencv-python
```

### Windows

1. **Используйте MSYS2 для установки GStreamer:**
   - Скачайте и установите MSYS2: https://www.msys2.org/
   - В терминале MSYS2:
```bash
pacman -S mingw-w64-x86_64-gtk3 mingw-w64-x86_64-python3 mingw-w64-x86_64-python3-gobject mingw-w64-x86_64-gstreamer mingw-w64-x86_64-gst-plugins-base mingw-w64-x86_64-gst-plugins-good
```

2. **Или используйте предкомпилированные бинарники:**
   - Скачайте GStreamer: https://gstreamer.freedesktop.org/download/
   - Установите Python bindings: `pip install PyGObject`

## Проверка установки

После установки проверьте, что все работает:

```python
python3 -c "from gi.repository import Gst, GLib; print('GStreamer imported successfully')"
```

Если вывод: `GStreamer imported successfully` - все установлено правильно.

## Альтернативное решение (без GStreamer)

Если установка GStreamer проблематична, используйте упрощенную версию скрипта на базе OpenCV:
- `extract_tiles_optimized.py` - работает только с OpenCV, без GStreamer

## Минимальные требования

- Python 3.6+
- GStreamer 1.14+
- OpenCV 4.0+ (для обработки изображений)
- NumPy

## Troubleshooting

### Ошибка: "No module named 'gi'"
```bash
# Ubuntu/Debian
sudo apt-get install python3-gi

# Или через pip
pip3 install PyGObject
```

### Ошибка: "Namespace Gst not available"
```bash
# Установите GObject Introspection файлы
sudo apt-get install gir1.2-gstreamer-1.0
```

### Ошибка с nvdsstitch плагином
Плагин nvdsstitch специфичен для NVIDIA Jetson/DeepStream. На обычном компьютере:
1. Либо компилируйте плагин из исходников
2. Либо используйте версию без nvdsstitch (extract_tiles_optimized.py)

## Контакты для помощи

При проблемах с установкой обратитесь к документации:
- GStreamer: https://gstreamer.freedesktop.org/
- PyGObject: https://pygobject.readthedocs.io/