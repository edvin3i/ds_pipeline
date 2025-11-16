# Синхронизированная запись с двух камер

## Проблема десинхронизации

Камеры стартуют не одновременно из-за:
1. **Последовательного запуска** - между запусками камер была пауза 0.5 секунды
2. **Разного времени инициализации** - IMX678 инициализируется дольше чем IMX477
3. **Отсутствия аппаратной синхронизации** - каждая камера работала независимо

**Результат:** между IMX678 камерами была разница ~11 кадров (367 мс при 30fps)

## Решение

### 1. Hardware Sync (Аппаратная синхронизация)

**Требования:**
- Физически соединить sync пины камер проводом
- Одна камера работает как master, вторая как slave
- IMX678 и IMX477 поддерживают этот режим

**Настройка через V4L2:**
```bash
# Мастер камера (video0)
v4l2-ctl -d /dev/video0 -c operation_mode=0 -c synchronizing_function=1

# Слейв камера (video1)
v4l2-ctl -d /dev/video1 -c operation_mode=1 -c synchronizing_function=2
```

### 2. Программная синхронизация

- **PAUSED → PLAYING метод** - обе камеры сначала инициализируются в PAUSED, затем одновременно переключаются в PLAYING
- **Общие часы (Shared Clock)** - обе камеры используют единое системное время
- **Одновременный запуск** - убрана задержка между запусками

## Использование

### Быстрый старт

```bash
cd soft_record_video

# Тест конфигурации (опционально)
./test_sync.sh

# Запись с синхронизацией
python3 synced_dual_record.py
```

### Расширенные опции

```bash
# С указанием ID камер
python3 synced_dual_record.py --master 0 --slave 1

# H.265 кодек (лучшее сжатие)
python3 synced_dual_record.py --codec h265

# Повышенный битрейт
python3 synced_dual_record.py --bitrate 35

# Sensor mode (0=без HDR для IMX678, по умолчанию)
python3 synced_dual_record.py --sensor-mode 0

# Sensor mode 1 (для HDR или других режимов)
python3 synced_dual_record.py --sensor-mode 1

# Отключить hardware sync (только программная синхронизация)
python3 synced_dual_record.py --no-hardware-sync

# Отключить общие часы (при проблемах)
python3 synced_dual_record.py --no-shared-clock

# Справка
python3 synced_dual_record.py --help
```

### Остановка записи

Нажмите `Ctrl+C` для корректной остановки записи

## Анализ результатов

Скрипт автоматически:
- Измеряет разницу старта камер в миллисекундах
- Показывает разницу в кадрах
- Оценивает качество синхронизации

**Критерии:**
- ✅ Отличная: < 1 кадра разницы
- ✅ Хорошая: < 3 кадров
- ⚠️ Приемлемая: < 5 кадров
- ⚠️ Слабая: >= 5 кадров

## Проверка синхронизации видео

```bash
# Просмотр отдельных файлов
vlc camera_master_*.mp4
vlc camera_slave_*.mp4

# Склейка в один файл (side-by-side)
ffmpeg -i camera_master_*.mp4 -i camera_slave_*.mp4 \
  -filter_complex hstack output.mp4

# Проверка количества кадров
ffprobe -v error -select_streams v:0 \
  -count_packets -show_entries stream=nb_read_packets \
  -of csv=p=0 camera_master_*.mp4

ffprobe -v error -select_streams v:0 \
  -count_packets -show_entries stream=nb_read_packets \
  -of csv=p=0 camera_slave_*.mp4
```

## Устранение проблем

### Hardware sync не работает

1. **Проверьте физическое подключение** sync провода
2. **Проверьте поддержку** камерами:
   ```bash
   v4l2-ctl -d /dev/video0 --list-ctrls | grep -i "operation\|sync"
   ```
3. **Запустите без hardware sync:**
   ```bash
   python3 synced_dual_record.py --no-hardware-sync
   ```

### Камеры определяются в обратном порядке

Укажите ID явно:
```bash
python3 synced_dual_record.py --master 1 --slave 0
```

### Ошибки GStreamer

1. Проверьте доступность камер:
   ```bash
   ls -l /dev/video*
   ```

2. Проверьте permissions:
   ```bash
   sudo usermod -a -G video $USER
   # Перелогиньтесь
   ```

## Файлы

- `synced_dual_record.py` - основной скрипт записи с синхронизацией
- `test_sync.sh` - тест и настройка hardware sync
- `new_dual_record.py` - старая версия (без hardware sync)
- `dual_record.py` - базовая версия

## Технические детали

### V4L2 Controls для синхронизации

| Параметр | Master | Slave |
|----------|--------|-------|
| `operation_mode` | 0 | 1 |
| `synchronizing_function` | 1 | 2 |

### GStreamer параметры

- `do-timestamp=true` - применять stream time к буферам
- `sync=false` в filesink - не синхронизировать запись с часами
- Общий `SystemClock` для обоих пайплайнов
- Одинаковый `base_time` для синхронизации таймлайна

### Метод PAUSED → PLAYING

```
NULL → PAUSED (обе камеры) → ждем инициализации → PLAYING (одновременно)
```

Это обеспечивает:
- Полную инициализацию камер до старта
- Минимальную задержку между запусками
- Синхронизированный старт записи
