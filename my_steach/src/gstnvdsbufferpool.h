/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file gstnvdsbufferpool.h
 * @brief NVIDIA DeepStream buffer pool for NVMM memory management
 *
 * Provides GStreamer buffer pool implementation for DeepStream plugins.
 * Manages allocation and reuse of NVMM (NVIDIA Multimedia Memory) buffers
 * with support for DeepStream metadata attachment.
 *
 * This is NVIDIA DeepStream SDK code, adapted for nvdsstitch plugin.
 *
 * @author NVIDIA CORPORATION
 * @date 2019-2020
 *
 * @see gstnvdsstitch_allocator.h for custom allocator implementation
 */

#ifndef GSTNVDSBUFFERPOOL_H_
#define GSTNVDSBUFFERPOOL_H_

#include <gst/gst.h>

G_BEGIN_DECLS

/** @brief Opaque structure for DeepStream buffer pool */
typedef struct _GstNvDsBufferPool GstNvDsBufferPool;

/** @brief Opaque structure for DeepStream buffer pool class */
typedef struct _GstNvDsBufferPoolClass GstNvDsBufferPoolClass;

/** @brief Opaque structure for DeepStream buffer pool private data */
typedef struct _GstNvDsBufferPoolPrivate GstNvDsBufferPoolPrivate;

/**
 * @name GObject Type Macros
 * @brief Standard GObject type checking and casting macros
 * @{
 */
#define GST_TYPE_NVDS_BUFFER_POOL      (gst_nvds_buffer_pool_get_type())
#define GST_IS_NVDS_BUFFER_POOL(obj)   (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GST_TYPE_NVDS_BUFFER_POOL))
#define GST_NVDS_BUFFER_POOL(obj)      (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_NVDS_BUFFER_POOL, GstNvDsBufferPool))
#define GST_NVDS_BUFFER_POOL_CAST(obj) ((GstNvDsBufferPool*)(obj))
/** @} */

/**
 * @name Memory and Metadata Identifiers
 * @brief Constants for DeepStream memory type and buffer pool options
 * @{
 */
#define GST_NVDS_MEMORY_TYPE "nvds"  /**< DeepStream NVMM memory type identifier */
#define GST_BUFFER_POOL_OPTION_NVDS_META "GstBufferPoolOptionNvDsMeta"  /**< DeepStream metadata pool option */
/** @} */

/**
 * @brief DeepStream buffer pool structure
 *
 * GStreamer buffer pool subclass for managing NVMM buffers with
 * DeepStream metadata support. Private data contains allocation
 * parameters and buffer management state.
 */
struct _GstNvDsBufferPool
{
  GstBufferPool bufferpool;  /**< Base GstBufferPool instance */

  GstNvDsBufferPoolPrivate *priv;  /**< Private implementation data */
};

/**
 * @brief DeepStream buffer pool class structure
 *
 * Class structure for GstNvDsBufferPool.
 */
struct _GstNvDsBufferPoolClass
{
  GstBufferPoolClass parent_class;  /**< Base GstBufferPoolClass */
};

/**
 * @brief Get GType for DeepStream buffer pool
 *
 * @return GType Type identifier for GstNvDsBufferPool
 */
GType gst_nvds_buffer_pool_get_type (void);

/**
 * @brief Create new DeepStream buffer pool
 *
 * Allocates and initializes a new NVMM buffer pool for DeepStream plugins.
 * Pool parameters (size, format, etc.) are configured via gst_buffer_pool_config_*
 * functions before activation.
 *
 * @return GstBufferPool* New buffer pool instance (caller must unref when done)
 * @retval NULL Allocation failed
 *
 * @note Caller must configure and activate pool before use
 * @note Call gst_object_unref() to destroy pool
 *
 * Example usage:
 * @code
 * GstBufferPool *pool = gst_nvds_buffer_pool_new();
 * GstStructure *config = gst_buffer_pool_get_config(pool);
 * gst_buffer_pool_config_set_params(config, caps, size, min_buffers, max_buffers);
 * gst_buffer_pool_set_config(pool, config);
 * gst_buffer_pool_set_active(pool, TRUE);
 * @endcode
 */
GstBufferPool* gst_nvds_buffer_pool_new (void);

G_END_DECLS

#endif /* GSTNVDSBUFFERPOOL_H_ */
