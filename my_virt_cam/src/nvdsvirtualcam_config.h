// nvdsvirtualcam_config.h - Конфигурация виртуальной камеры
// Основано на параметрах из Python демо
#ifndef __NVDSVIRTUALCAM_CONFIG_H__
#define __NVDSVIRTUALCAM_CONFIG_H__

namespace NvdsVirtualCamConfig {
    
    // ========== ПАРАМЕТРЫ ВХОДА ==========
    // УДАЛЕНО: INPUT_WIDTH, INPUT_HEIGHT - теперь передаются через properties!
    constexpr int INPUT_FORMAT = 19;    // NVBUF_COLOR_FORMAT_RGBA

    // ========== ПАРАМЕТРЫ ВЫХОДА (виртуальная камера) ==========
    constexpr int DEFAULT_OUTPUT_WIDTH = 1920;
    constexpr int DEFAULT_OUTPUT_HEIGHT = 1080;
    constexpr int OUTPUT_FORMAT = 19; // NVBUF_COLOR_FORMAT_RGBA

    // ========== ПАРАМЕТРЫ ПРОЕКЦИИ ==========
    // Панорама покрывает 180° по горизонтали (~54°) по вертикали
    constexpr float LON_MIN = -90.0f;   // Минимальная долгота
    constexpr float LON_MAX = +90.0f;   // Максимальная долгота
    constexpr float LAT_MIN = -27.0f;   // Минимальная широта
    constexpr float LAT_MAX = +27.0f;   // Максимальная широта

    // УДАЛЕНО: PIXELS_PER_DEGREE_* - не используются, зависели от INPUT_WIDTH/HEIGHT

    // ========== ПАРАМЕТРЫ ВИРТУАЛЬНОЙ КАМЕРЫ ==========
    // Углы по умолчанию (центр панорамы, максимальный обзор)
    constexpr float DEFAULT_YAW = 0.0f;     // Поворот влево-вправо (центр по горизонтали)
    constexpr float DEFAULT_PITCH = 0.0f;   // Наклон вверх-вниз (центр по вертикали, было 5.0°)
    constexpr float DEFAULT_ROLL = 0.0f;    // Наклон вбок
    constexpr float DEFAULT_FOV = 68.0f;    // Стартовый угол обзора (максимальное отдаление = FOV_MAX)

    // Ограничения для углов камеры
    constexpr float YAW_MIN = LON_MIN;      // -90°
    constexpr float YAW_MAX = LON_MAX;      // +90°
    constexpr float PITCH_MIN = LAT_MIN;    // -32° (используется для GStreamer property limits)
    constexpr float PITCH_MAX = LAT_MAX;    // +22° (используется для GStreamer property limits)
    constexpr float ROLL_MIN = -28.0f;      // Максимальный наклон влево
    constexpr float ROLL_MAX = 28.0f;      // Максимальный наклон вправо
    constexpr float FOV_MIN = 55.0f;        // Минимальный угол обзора (было 40°, увеличено для меньшего приближения)
    constexpr float FOV_MAX = 68.0f;        // Максимальный угол обзора

    // ========== ПАРАМЕТРЫ АВТОЗУМА (зависимость FOV от размера мяча) ==========
    constexpr float BALL_RADIUS_MIN = 5.0f;   // Минимальный радиус мяча в пикселях
    constexpr float BALL_RADIUS_MAX = 50.0f;  // Максимальный радиус мяча в пикселях

    // ========== ПАРАМЕТРЫ АВТОСЛЕЖЕНИЯ ==========
    constexpr float SMOOTH_FACTOR_DEFAULT = 0.3f;     // Плавность слежения камеры
    constexpr float ANGLE_CHANGE_THRESHOLD = 0.1f;    // Порог для пересчёта LUT
    constexpr float S_TARGET_DEFAULT = 0.035f;        // Целевой размер объекта на экране
    constexpr float SCREEN_Y_FRACTION = 0.60f;        // Позиция объекта по вертикали
    constexpr int EDGE_MARGIN_PX = 4;                 // Отступ от краев в пикселях
    
    // ========== CUDA ПАРАМЕТРЫ ==========
    constexpr int GPU_ID = 0;
    constexpr int BLOCK_SIZE_X = 32;
    constexpr int BLOCK_SIZE_Y = 16;
    
    // ========== BUFFER POOL ==========
    constexpr int POOL_MIN_BUFFERS = 8;
    constexpr int POOL_MAX_BUFFERS = 10;
    constexpr int POOL_MEMTYPE = 4; // NVBUF_MEM_SURFACE_ARRAY для Jetson
    
    // ========== ФУНКЦИИ-УТИЛИТЫ ==========

    // Расчёт pitch с выравниванием для GPU
    inline int calculatePitch(int width, int bytes_per_pixel = 4) {
        int min_pitch = width * bytes_per_pixel;
        return ((min_pitch + 31) / 32) * 32;  // Выравнивание по 32 байта
    }

    // УДАЛЕНО: все inline функции преобразования координат
    // Они не используются - все преобразования в gstnvdsvirtualcam.cpp
}

#endif // __NVDSVIRTUALCAM_CONFIG_H__