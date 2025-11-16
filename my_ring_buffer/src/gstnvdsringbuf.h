/*
 * Заголовочный файл для версии с задержкой
 */

#ifndef __GST_NVDS_RINGBUF_H__
#define __GST_NVDS_RINGBUF_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"

G_BEGIN_DECLS

#define GST_TYPE_NVDS_RINGBUF            (gst_nvds_ringbuf_get_type())
#define GST_NVDS_RINGBUF(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVDS_RINGBUF, GstNvdsRingBuf))
#define GST_NVDS_RINGBUF_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_NVDS_RINGBUF, GstNvdsRingBufClass))
#define GST_IS_NVDS_RINGBUF(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_NVDS_RINGBUF))
#define GST_IS_NVDS_RINGBUF_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_NVDS_RINGBUF))

typedef struct _GstNvdsRingBuf      GstNvdsRingBuf;
typedef struct _GstNvdsRingBufClass GstNvdsRingBufClass;

/* Структура для одного слота в кольце */
typedef struct {
  NvBufSurface *surf;      // Пиксели (GPU память)
  GstClockTime pts;        // Presentation timestamp
  GstClockTime dts;        // Decode timestamp
  GstClockTime duration;   // Длительность кадра
} RingSlot;

struct _GstNvdsRingBuf
{
  GstBaseTransform parent;

  /* Свойства */
  guint64 ring_bytes;
  guint min_slots;
  guint chunk;
  gboolean preregister_cuda;

  /* Параметры видео */
  guint width;
  guint height;
  NvBufSurfaceColorFormat nvcolor;
  NvBufSurfaceLayout layout;

  /* Кольцевой буфер */
  GPtrArray *slots;           // Массив RingSlot*
  guint size;                 // Количество слотов
  guint head;                 // Указатель записи
  guint tail;                 // Указатель чтения (для задержки)
  guint accumulated;          // Счётчик накопленных кадров
  gboolean delay_complete;    // Флаг заполнения буфера

  /* Внутреннее состояние */
  guint64 bytes_per_slot;
  gboolean started;
  GRecMutex lock;
};

struct _GstNvdsRingBufClass
{
  GstBaseTransformClass parent_class;
};

GType gst_nvds_ringbuf_get_type (void);

G_END_DECLS

#endif /* __GST_NVDS_RINGBUF_H__ */