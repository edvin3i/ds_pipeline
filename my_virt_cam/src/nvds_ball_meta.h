// nvds_ball_meta.h - Метаданные для отслеживания мяча
#ifndef __NVDS_BALL_META_H__
#define __NVDS_BALL_META_H__

#include <gst/gst.h>
#include "nvdsmeta.h"

// Тип метаданных для мяча
#define NVDS_BALL_META_TYPE 0x5001  // Уникальный ID

typedef struct {
    // Позиция мяча на панораме (в пикселях)
    gfloat ball_x;       // 0..pano_width
    gfloat ball_y;       // 0..pano_height
    
    // Размер мяча
    gfloat ball_radius;  // Радиус в пикселях на панораме
    
    // Скорость мяча (опционально)
    gfloat velocity_x;
    gfloat velocity_y;
    
    // ID объекта (если используется трекер)
    guint object_id;
    
    // Уверенность детекции (0.0-1.0)
    gfloat confidence;
    
    // Флаги
    gboolean is_tracked;     // Мяч активно отслеживается
    gboolean is_predicted;   // Позиция предсказана (не детектирована)
    
} NvDsBallMeta;

// API функции для работы с метаданными
NvDsBallMeta* nvds_add_ball_meta_to_buffer(GstBuffer *buffer);
NvDsBallMeta* nvds_get_ball_meta_from_buffer(GstBuffer *buffer);

#endif