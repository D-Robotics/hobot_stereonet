/****************************************************************************
*
*    Copyright 2012 - 2022 Vivante Corporation, Santa Clara, California.
*    All Rights Reserved.
*
*    Permission is hereby granted, free of charge, to any person obtaining
*    a copy of this software and associated documentation files (the
*    'Software'), to deal in the Software without restriction, including
*    without limitation the rights to use, copy, modify, merge, publish,
*    distribute, sub license, and/or sell copies of the Software, and to
*    permit persons to whom the Software is furnished to do so, subject
*    to the following conditions:
*
*    The above copyright notice and this permission notice (including the
*    next paragraph) shall be included in all copies or substantial
*    portions of the Software.
*
*    THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
*    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
*    IN NO EVENT SHALL VIVANTE AND/OR ITS SUPPLIERS BE LIABLE FOR ANY
*    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
*    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
*    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

#ifndef _nano2d_h__
#define _nano2d_h__

#if defined(WIN32) && !defined(DDK)
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>

#include "nano2D_types.h"
#include "nano2D_enum.h"
#include "nano2D_base.h"
#include "nano2D_option.h"


#define MAX_UV_PLANE 2
#define MAX_PLANE    3

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n2d_tile_status_buffer
{
    n2d_uintptr_t               handle[MAX_PLANE];
    n2d_uintptr_t               gpu_addr[MAX_PLANE];
    n2d_pointer                 memory[MAX_PLANE];
    n2d_bool_t                  enableFastClear;
    n2d_uint32_t                fastClearValue;
}
n2d_tile_status_buffer_t;

typedef struct n2d_buffer
{
    n2d_int32_t                 width;
    n2d_int32_t                 height;
    n2d_int32_t                 alignedw;
    n2d_int32_t                 alignedh;
    n2d_int32_t                 stride;
    n2d_source_type_t           srcType;
    n2d_buffer_format_t         format;
    n2d_orientation_t           orientation;
    n2d_tiling_t                tiling;
    n2d_cache_mode_t            cacheMode;
    n2d_uintptr_t               handle;
    n2d_pointer                 memory;
    n2d_uintptr_t               gpu;
    n2d_int32_t                 uvstride[MAX_UV_PLANE];
    n2d_uintptr_t               uv_handle[MAX_UV_PLANE];
    n2d_pointer                 uv_memory[MAX_UV_PLANE];
    n2d_uintptr_t               uv_gpu[MAX_UV_PLANE];
    n2d_tile_status_config_t    tile_status_config;
    n2d_tile_status_buffer_t    tile_status_buffer;

#ifdef N2D_SUPPORT_64BIT_ADDRESS
    n2d_uintptr_t               gpuEx;
    n2d_uintptr_t               ts_gpuAddrEx;
#endif
}
n2d_buffer_t;

typedef struct _gcoTLS* gcoTLS;

n2d_error_t n2d_open(
    n2d_void);

n2d_error_t n2d_close(
    n2d_void);

n2d_bool_t n2d_is_feature_support(
    n2d_feature_t feature);

n2d_error_t n2d_switch_device(
    n2d_device_id_t device_id);

n2d_error_t n2d_switch_core(
    n2d_core_id_t core_id);

n2d_error_t n2d_allocate(
    n2d_buffer_t *buffer);

n2d_error_t n2d_free(
    n2d_buffer_t *buffer);

n2d_error_t n2d_map(
    n2d_buffer_t *buffer);

n2d_error_t n2d_unmap(
    n2d_buffer_t *buffer);

n2d_error_t n2d_wrap(
    n2d_user_memory_desc_t* memDesc,
    n2d_uintptr_t* handle);

n2d_error_t n2d_commit(
    n2d_void);

n2d_error_t n2d_commit_ex(
    n2d_bool_t stall);

n2d_error_t n2d_set(
    n2d_state_config_t *config);

n2d_error_t n2d_get(
    n2d_get_state_config_t *config);

n2d_error_t n2d_fill(
    n2d_buffer_t *dest,
    n2d_rectangle_t *rect,
    n2d_color_t color,
    n2d_blend_t mode);

n2d_error_t n2d_blit(
    n2d_buffer_t *dst_buf,
    n2d_rectangle_t *dst_rect,
    n2d_buffer_t *src_buf,
    n2d_rectangle_t *src_rect,
    n2d_blend_t mode);

n2d_error_t n2d_filterblit(
    n2d_buffer_t *dst_buf,
    n2d_rectangle_t *dst_rect,
    n2d_rectangle_t *dst_subrect,
    n2d_buffer_t *src_buf,
    n2d_rectangle_t *src_rect,
    n2d_blend_t mode);

n2d_error_t n2d_multisource_blit(
    n2d_buffer_t *dst_buf,
    n2d_int32_t src_mask);

n2d_error_t n2d_load_palette(
    n2d_uint32_t first_index,
    n2d_uint32_t index_count,
    n2d_pointer color_table,
    n2d_bool_t color_convert);

n2d_error_t  n2d_delogo(
    n2d_buffer_t* src,
    n2d_rectangle_t logo_rect);

n2d_error_t n2d_export_dma_buf(
    n2d_buffer_t* buf,
    n2d_uint32_t flag,
    n2d_export_memory_t* fd);

n2d_error_t n2d_equalizeHist(
    n2d_buffer_t* src,
    n2d_buffer_t* dst);

n2d_error_t n2d_calcHist(
    n2d_buffer_t* src,
    n2d_int32_t* channels,
    n2d_uint32_t* hist,
    n2d_int32_t dims,
    n2d_uint32_t* histSize,
    n2d_float_t** ranges);

n2d_error_t gcTlsConstructor(
    gcoTLS* TLS);

n2d_error_t ExaAllocate(
    n2d_size_t     size,
    n2d_uintptr_t *handle,
    n2d_uint64_t *physAddr,
    n2d_pointer *logicAddr);

n2d_error_t n2d_get_commit(char *commit);

#ifdef __cplusplus
}
#endif

#endif
