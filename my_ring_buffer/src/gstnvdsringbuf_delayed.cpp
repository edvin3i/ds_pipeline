/*
 * Модифицированная версия с реальной задержкой
 */

#include "gstnvdsringbuf.h"
#include <string.h>
#include <cuda_runtime.h>
#include <cudaEGL.h>

GST_DEBUG_CATEGORY_STATIC (gst_nvds_ringbuf_debug);
#define GST_CAT_DEFAULT gst_nvds_ringbuf_debug

/* Свойства */
enum
{
  PROP_0,
  PROP_RING_BYTES,
  PROP_MIN_SLOTS,
  PROP_CHUNK,
  PROP_PREREGISTER_CUDA
};

/* Прототипы */
static void gst_nvds_ringbuf_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_nvds_ringbuf_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_nvds_ringbuf_start (GstBaseTransform * trans);
static gboolean gst_nvds_ringbuf_stop  (GstBaseTransform * trans);

static GstCaps *gst_nvds_ringbuf_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter);

static gboolean gst_nvds_ringbuf_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps);

static GstFlowReturn gst_nvds_ringbuf_transform_ip (GstBaseTransform * trans, GstBuffer * buf);

static void gst_nvds_ringbuf_finalize (GObject * object);

/* Внутренние функции кольца */
static gboolean ring_calc_slot_bytes (GstNvdsRingBuf *self);
static gboolean ring_alloc_pool      (GstNvdsRingBuf *self);
static void     ring_free_pool       (GstNvdsRingBuf *self);
static gboolean ring_push_copy       (GstNvdsRingBuf *self, NvBufSurface *in_surf);
static gboolean ring_pop_copy        (GstNvdsRingBuf *self, NvBufSurface *out_surf);

/* Pad template */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw(memory:NVMM), format=(string){RGBA,NV12}")
);

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw(memory:NVMM), format=(string){RGBA,NV12}")
);

/* Класс */
#define gst_nvds_ringbuf_parent_class parent_class
G_DEFINE_TYPE (GstNvdsRingBuf, gst_nvds_ringbuf, GST_TYPE_BASE_TRANSFORM);

static void
gst_nvds_ringbuf_class_init (GstNvdsRingBufClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (klass);
  GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS (klass);

  gobject_class->set_property = gst_nvds_ringbuf_set_property;
  gobject_class->get_property = gst_nvds_ringbuf_get_property;
  gobject_class->finalize = gst_nvds_ringbuf_finalize;

  base_transform_class->start = gst_nvds_ringbuf_start;
  base_transform_class->stop = gst_nvds_ringbuf_stop;
  base_transform_class->transform_caps = gst_nvds_ringbuf_transform_caps;
  base_transform_class->set_caps = gst_nvds_ringbuf_set_caps;
  base_transform_class->transform_ip = gst_nvds_ringbuf_transform_ip;

  // Мы работаем in-place (изменяем буфер)
  base_transform_class->transform_ip_on_passthrough = FALSE;

  // ВАЖНО: НЕ passthrough! Мы изменяем содержимое
  gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(klass), FALSE);

  /* Свойства */
  g_object_class_install_property (gobject_class, PROP_RING_BYTES,
      g_param_spec_uint64 ("ring-bytes", "Ring buffer bytes",
          "Total ring buffer size in bytes", 0, G_MAXUINT64, 0,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_MIN_SLOTS,
      g_param_spec_uint ("min-slots", "Min slots",
          "Minimum number of slots", 1, 10000, 30,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_CHUNK,
      g_param_spec_uint ("chunk", "Chunk",
          "Chunk batches for allocation", 1, 100, 1,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PREREGISTER_CUDA,
      g_param_spec_boolean ("preregister-cuda", "Preregister CUDA",
          "Pre-register surfaces for CUDA", FALSE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  /* Метаданные элемента */
  gst_element_class_set_static_metadata (gstelement_class,
      "NVDS Ring Buffer", "Buffer",
      "Ring buffer with delay for NVMM surfaces",
      "NVIDIA");

  gst_element_class_add_static_pad_template (gstelement_class, &sink_template);
  gst_element_class_add_static_pad_template (gstelement_class, &src_template);

  GST_DEBUG_CATEGORY_INIT (gst_nvds_ringbuf_debug, "nvdsringbuf", 0,
      "NVDS Ring Buffer");
}

static void
gst_nvds_ringbuf_init (GstNvdsRingBuf * self)
{
  self->ring_bytes = 0;
  self->min_slots = 30;
  self->chunk = 1;
  self->preregister_cuda = FALSE;

  self->width = 0;
  self->height = 0;
  self->nvcolor = NVBUF_COLOR_FORMAT_RGBA;
  self->layout = NVBUF_LAYOUT_PITCH;

  self->slots = NULL;
  self->size = 0;
  self->head = 0;
  self->tail = 0;
  self->accumulated = 0;  // Счётчик накопленных кадров
  self->delay_complete = FALSE;  // Флаг заполнения буфера

  self->started = FALSE;

  g_rec_mutex_init (&self->lock);

  // Важно: отключаем passthrough
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (self), FALSE);
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (self), TRUE);
}

static void
gst_nvds_ringbuf_finalize (GObject * object)
{
  GstNvdsRingBuf *self = GST_NVDS_RINGBUF (object);

  ring_free_pool (self);
  g_rec_mutex_clear (&self->lock);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

static void
gst_nvds_ringbuf_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvdsRingBuf *self = GST_NVDS_RINGBUF (object);

  switch (prop_id) {
    case PROP_RING_BYTES:
      self->ring_bytes = g_value_get_uint64 (value);
      break;
    case PROP_MIN_SLOTS:
      self->min_slots = g_value_get_uint (value);
      break;
    case PROP_CHUNK:
      self->chunk = g_value_get_uint (value);
      break;
    case PROP_PREREGISTER_CUDA:
      self->preregister_cuda = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_nvds_ringbuf_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstNvdsRingBuf *self = GST_NVDS_RINGBUF (object);

  switch (prop_id) {
    case PROP_RING_BYTES:
      g_value_set_uint64 (value, self->ring_bytes);
      break;
    case PROP_MIN_SLOTS:
      g_value_set_uint (value, self->min_slots);
      break;
    case PROP_CHUNK:
      g_value_set_uint (value, self->chunk);
      break;
    case PROP_PREREGISTER_CUDA:
      g_value_set_boolean (value, self->preregister_cuda);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static gboolean
gst_nvds_ringbuf_start (GstBaseTransform * trans)
{
  GstNvdsRingBuf *self = GST_NVDS_RINGBUF (trans);

  // Инициализация CUDA контекста
  cudaError_t cuda_err = cudaSetDevice(0);
  if (cuda_err != cudaSuccess) {
    GST_ERROR_OBJECT (self, "Failed to set CUDA device: %s",
                      cudaGetErrorString(cuda_err));
    return FALSE;
  }

  GST_DEBUG_OBJECT (self, "CUDA device 0 initialized");

  // Сброс счётчиков
  self->accumulated = 0;
  self->delay_complete = FALSE;
  self->head = 0;
  self->tail = 0;

  self->started = TRUE;
  return TRUE;
}

static gboolean
gst_nvds_ringbuf_stop (GstBaseTransform * trans)
{
  GstNvdsRingBuf *self = GST_NVDS_RINGBUF (trans);

  g_rec_mutex_lock (&self->lock);
  ring_free_pool (self);
  g_rec_mutex_unlock (&self->lock);

  self->started = FALSE;
  return TRUE;
}

static GstCaps *
gst_nvds_ringbuf_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  // Пропускаем те же caps без изменений
  GstCaps *ret = gst_caps_copy (caps);
  if (filter) {
    GstCaps *tmp = gst_caps_intersect_full (ret, filter, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref (ret);
    ret = tmp;
  }
  return ret;
}

static gboolean
gst_nvds_ringbuf_set_caps (GstBaseTransform * trans, GstCaps * incaps, GstCaps * outcaps)
{
  GstNvdsRingBuf *self = GST_NVDS_RINGBUF (trans);
  GstStructure *s = gst_caps_get_structure (incaps, 0);

  if (!gst_structure_get_int (s, "width", (gint*)&self->width) ||
      !gst_structure_get_int (s, "height", (gint*)&self->height)) {
    GST_ERROR_OBJECT (self, "Failed to get width/height from caps");
    return FALSE;
  }

  const gchar *format = gst_structure_get_string (s, "format");
  if (!format) {
    GST_ERROR_OBJECT (self, "No format in caps");
    return FALSE;
  }

  if (g_strcmp0 (format, "RGBA") == 0) {
    self->nvcolor = NVBUF_COLOR_FORMAT_RGBA;
  } else if (g_strcmp0 (format, "NV12") == 0) {
    self->nvcolor = NVBUF_COLOR_FORMAT_NV12;
  } else {
    GST_ERROR_OBJECT (self, "Unsupported format: %s", format);
    return FALSE;
  }

  GST_INFO_OBJECT (self, "Set caps: %ux%u format=%s",
                   self->width, self->height, format);

  g_rec_mutex_lock (&self->lock);

  // Вычисляем размер слота и выделяем пул
  if (!ring_calc_slot_bytes (self)) {
    g_rec_mutex_unlock (&self->lock);
    return FALSE;
  }

  if (!ring_alloc_pool (self)) {
    g_rec_mutex_unlock (&self->lock);
    return FALSE;
  }

  GST_INFO_OBJECT (self, "Ring buffer ready: %u slots, %.2f MB total, delay: %.1f seconds",
                   self->size,
                   (double)(self->size * self->bytes_per_slot) / (1024*1024),
                   (double)self->size / 30.0);  // При 30 FPS

  g_rec_mutex_unlock (&self->lock);
  return TRUE;
}

static GstFlowReturn
gst_nvds_ringbuf_transform_ip (GstBaseTransform * trans, GstBuffer * buf)
{
  GstNvdsRingBuf *self = GST_NVDS_RINGBUF (trans);

  if (!self->started || !self->slots) {
    GST_ERROR_OBJECT (self, "Not initialized");
    return GST_FLOW_ERROR;
  }

  // Достаём NvBufSurface из буфера
  GstMapInfo map;
  if (!gst_buffer_map (buf, &map, GST_MAP_READWRITE)) {
    GST_ERROR_OBJECT (self, "Cannot map buffer");
    return GST_FLOW_ERROR;
  }
  NvBufSurface *surf = (NvBufSurface *) map.data;

  if (!surf || surf->batchSize < 1) {
    GST_ERROR_OBJECT (self, "Invalid NvBufSurface");
    gst_buffer_unmap (buf, &map);
    return GST_FLOW_ERROR;
  }

  g_rec_mutex_lock (&self->lock);

  // Копируем входной кадр в кольцо
  if (!ring_push_copy (self, surf)) {
    GST_ERROR_OBJECT (self, "Failed to push to ring");
    g_rec_mutex_unlock (&self->lock);
    gst_buffer_unmap (buf, &map);
    return GST_FLOW_ERROR;
  }

  self->accumulated++;

  // Проверяем, накопили ли достаточно кадров для задержки
  if (!self->delay_complete) {
    if (self->accumulated >= self->size) {
      self->delay_complete = TRUE;
      GST_INFO_OBJECT (self, "🎯 Buffer filled! Starting delayed output after %u frames",
                       self->accumulated);
    } else {
      // Ещё накапливаем - дропаем кадр (не пропускаем дальше)
      g_rec_mutex_unlock (&self->lock);
      gst_buffer_unmap (buf, &map);

      // Каждые 30 кадров сообщаем о прогрессе
      if (self->accumulated % 30 == 0) {
        GST_DEBUG_OBJECT (self, "Accumulating: %u/%u frames (%.1f/%.1f sec)",
                         self->accumulated, self->size,
                         (float)self->accumulated / 30.0,
                         (float)self->size / 30.0);
      }

      return GST_BASE_TRANSFORM_FLOW_DROPPED;  // Важно: дропаем кадр!
    }
  }

  // Буфер заполнен - выдаём старый кадр из хвоста
  if (self->delay_complete) {
    if (!ring_pop_copy (self, surf)) {
      GST_ERROR_OBJECT (self, "Failed to pop from ring");
      g_rec_mutex_unlock (&self->lock);
      gst_buffer_unmap (buf, &map);
      return GST_FLOW_ERROR;
    }

    // Логирование каждые 30 кадров
    if (self->accumulated % 30 == 0) {
      guint delay = self->accumulated - self->size;
      GST_DEBUG_OBJECT (self, "Output frame %u with delay of %u frames (%.1f sec)",
                       delay, self->size, (float)self->size / 30.0);
    }
  }

  g_rec_mutex_unlock (&self->lock);
  gst_buffer_unmap (buf, &map);

  return GST_FLOW_OK;
}

/* ---------------------- Реализация кольца ---------------------- */

static gboolean
ring_calc_slot_bytes (GstNvdsRingBuf *self)
{
  if (self->width == 0 || self->height == 0) {
    GST_WARNING_OBJECT (self, "Invalid dimensions: %ux%u", self->width, self->height);
    return FALSE;
  }

  // Создаём временную поверхность для определения размера
  NvBufSurfaceCreateParams p = {0};
  p.gpuId = 0;
  p.width = self->width;
  p.height = self->height;
  p.colorFormat = self->nvcolor;
  p.layout = self->layout;
  p.memType = NVBUF_MEM_SURFACE_ARRAY;

  NvBufSurface *tmp = NULL;
  if (NvBufSurfaceCreate (&tmp, 1, &p) != 0 || !tmp) {
    GST_ERROR_OBJECT (self, "NvBufSurfaceCreate tmp failed");
    return FALSE;
  }

  guint64 bytes = 0;
  NvBufSurfaceParams *sp = &tmp->surfaceList[0];
  for (guint i = 0; i < sp->planeParams.num_planes; ++i) {
    bytes += (guint64)sp->planeParams.pitch[i] * sp->planeParams.height[i];
  }
  self->bytes_per_slot = bytes;

  NvBufSurfaceDestroy (tmp);
  return TRUE;
}

static gboolean
ring_alloc_pool (GstNvdsRingBuf *self)
{
  ring_free_pool (self);

  // Вычисляем количество слотов
  if (self->ring_bytes > 0 && self->bytes_per_slot > 0) {
    self->size = self->ring_bytes / self->bytes_per_slot;
  }
  if (self->min_slots > 0 && self->size < self->min_slots) {
    self->size = self->min_slots;
  }
  if (self->size == 0) {
    GST_ERROR_OBJECT (self, "Cannot determine ring size");
    return FALSE;
  }

  GST_INFO_OBJECT (self, "Allocating %u slots of %lu bytes each",
                   self->size, self->bytes_per_slot);

  // Создаём массив слотов
  self->slots = g_ptr_array_new ();

  NvBufSurfaceCreateParams p = {0};
  p.gpuId = 0;
  p.width = self->width;
  p.height = self->height;
  p.colorFormat = self->nvcolor;
  p.layout = self->layout;
  p.memType = NVBUF_MEM_SURFACE_ARRAY;

  // Выделяем слоты
  for (guint i = 0; i < self->size; i++) {
    RingSlot *slot = g_new0 (RingSlot, 1);

    if (NvBufSurfaceCreate (&slot->surf, 1, &p) != 0) {
      GST_ERROR_OBJECT (self, "Failed to create surface for slot %u", i);
      g_free (slot);
      ring_free_pool (self);
      return FALSE;
    }

    g_ptr_array_add (self->slots, slot);
  }

  self->head = 0;
  self->tail = 0;
  self->accumulated = 0;
  self->delay_complete = FALSE;

  GST_INFO_OBJECT (self, "✅ Allocated %u slots successfully", self->size);
  return TRUE;
}

static void
ring_free_pool (GstNvdsRingBuf *self)
{
  if (self->slots) {
    for (guint i = 0; i < self->slots->len; i++) {
      RingSlot *slot = (RingSlot*) g_ptr_array_index (self->slots, i);
      if (slot) {
        if (slot->surf) {
          NvBufSurfaceDestroy (slot->surf);
        }
        g_free (slot);
      }
    }
    g_ptr_array_free (self->slots, FALSE);
    self->slots = NULL;
  }
  self->size = 0;
}

static gboolean
ring_push_copy (GstNvdsRingBuf *self, NvBufSurface *in_surf)
{
  if (!self->slots || self->size == 0) return FALSE;

  RingSlot *dst = (RingSlot*) g_ptr_array_index (self->slots, self->head);
  if (!dst || !dst->surf) return FALSE;

  // Копия NVMM->NVMM через NvBufSurfTransform
  NvBufSurfTransformRect src_rect, dst_rect;
  src_rect.top = 0; src_rect.left = 0;
  src_rect.width = self->width; src_rect.height = self->height;
  dst_rect = src_rect;

  NvBufSurfTransformParams params;
  memset (&params, 0, sizeof (params));
  params.src_rect = &src_rect;
  params.dst_rect = &dst_rect;
  params.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
  params.transform_filter = NvBufSurfTransformInter_Default;

  // Оборачиваем входную поверхность
  NvBufSurface src_wrap;
  memcpy (&src_wrap, in_surf, sizeof (NvBufSurface));
  src_wrap.surfaceList = &in_surf->surfaceList[0];
  src_wrap.numFilled = 1;
  src_wrap.batchSize = 1;

  if (NvBufSurfTransform (&src_wrap, dst->surf, &params) != 0) {
    GST_ERROR_OBJECT (self, "NvBufSurfTransform failed");
    return FALSE;
  }

  // Сдвигаем head
  self->head = (self->head + 1) % self->size;
  return TRUE;
}

static gboolean
ring_pop_copy (GstNvdsRingBuf *self, NvBufSurface *out_surf)
{
  if (!self->slots || self->size == 0) return FALSE;

  RingSlot *src = (RingSlot*) g_ptr_array_index (self->slots, self->tail);
  if (!src || !src->surf) return FALSE;

  // Копия из кольца обратно в выходной буфер
  NvBufSurfTransformRect src_rect, dst_rect;
  src_rect.top = 0; src_rect.left = 0;
  src_rect.width = self->width; src_rect.height = self->height;
  dst_rect = src_rect;

  NvBufSurfTransformParams params;
  memset (&params, 0, sizeof (params));
  params.src_rect = &src_rect;
  params.dst_rect = &dst_rect;
  params.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
  params.transform_filter = NvBufSurfTransformInter_Default;

  // Оборачиваем выходную поверхность
  NvBufSurface dst_wrap;
  memcpy (&dst_wrap, out_surf, sizeof (NvBufSurface));
  dst_wrap.surfaceList = &out_surf->surfaceList[0];
  dst_wrap.numFilled = 1;
  dst_wrap.batchSize = 1;

  if (NvBufSurfTransform (src->surf, &dst_wrap, &params) != 0) {
    GST_ERROR_OBJECT (self, "NvBufSurfTransform failed (pop)");
    return FALSE;
  }

  // Сдвигаем tail
  self->tail = (self->tail + 1) % self->size;
  return TRUE;
}

/* ---------------------- Регистрация плагина ---------------------- */

static gboolean plugin_init (GstPlugin * plugin)
{
  return gst_element_register (plugin, "nvdsringbuf", GST_RANK_NONE, GST_TYPE_NVDS_RINGBUF);
}

GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsringbuf,
    "NVMM Ring Buffer with Delay for DeepStream",
    plugin_init,
    "1.0",
    "Proprietary",
    "DeepStream",
    "https://developer.nvidia.com/deepstream"
)