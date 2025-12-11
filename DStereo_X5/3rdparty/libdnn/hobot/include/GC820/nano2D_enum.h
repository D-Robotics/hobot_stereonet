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

#ifndef _nano2d_enum_h__
#define _nano2d_enum_h__

#ifdef __cplusplus
extern "C" {
#endif

#include "version.h"

#if (NANO2D_VERSION_MAJOR > 2) && (NANO2D_VERSION_MINOR > 0)
#include "nano2D_chip_enum.h"
#endif

typedef enum n2d_feature
{
    N2D_FEATURE_YUV420_OUTPUT = 1,
    N2D_FEATURE_2D_10BIT_OUTPUT_LINEAR,
    N2D_FEATURE_MAJOR_SUPER_TILE,
    N2D_FEATURE_DEC400_COMPRESSION,
    N2D_FEATURE_ANDROID_ONLY,
    N2D_FEATURE_2D_TILING,
    N2D_FEATURE_2D_MINOR_TILING,
    N2D_FEATURE_DEC_COMPRESSION,
    N2D_FEATURE_2D_MULTI_SOURCE_BLT,
    N2D_FEATURE_BGR_PLANAR,
    N2D_FEATURE_SCALER,
    N2D_FEATURE_2D_ONE_PASS_FILTER,
    N2D_FEATURE_2D_OPF_YUV_OUTPUT,
    N2D_FEATURE_SEPARATE_SRC_DST,
    N2D_FEATURE_2D_ALL_QUAD,
    N2D_FEATURE_2D_POST_FLIP,
    N2D_FEATURE_AXI_FE,

    N2D_FEATURE_CSC_PROGRAMMABLE,
    N2D_FEATURE_DEC400_FC,
    N2D_FEATURE_TRANSPARENCY_MASK,
    N2D_FEATURE_TRANSPARENCY_COLORKEY,

    N2D_FEATURE_NORMALIZATION,
    N2D_FEATURE_NORMALIZATION_QUANTIZATION,

    N2D_FEATURE_HISTOGRAM,
    N2D_FEATURE_BRIGHTNESS_SATURATION,
    N2D_FEATURE_64BIT_ADDRESS,
    N2D_FEATURE_CONTEXT_ID,
    N2D_FEATURE_SECURE_BUFFER,

    N2D_FEATURE_MMU_PAGE_DESCRIPTOR,
    N2D_FEATURE_SECURITY_AHB,
    N2D_FEATURE_FRAME_DONE_INTR,

    N2D_FEATURE_COUNT,
}
n2d_feature_t;

typedef enum n2d_blend
{
    N2D_BLEND_NONE,
    N2D_BLEND_SRC_OVER,
    N2D_BLEND_DST_OVER,
    N2D_BLEND_SRC_IN,
    N2D_BLEND_DST_IN,
    N2D_BLEND_ADDITIVE,
    N2D_BLEND_SUBTRACT,
    N2D_BLEND_CLEAR,
    N2D_BLEND_SRC,
    N2D_BLEND_DST,
    N2D_BLEND_SRC_OUT,
    N2D_BLEND_DST_OUT,
    N2D_BLEND_SRC_ATOP,
    N2D_BLEND_DST_ATOP,
    N2D_BLEND_XOR,
}
n2d_blend_t;

typedef enum n2d_buffer_format
{
    N2D_RGBA8888    = 0x100,     /* A[31:24] B[23:16] G[15:8]  R[7:0] */
    N2D_RGBX8888,                /*          B[23:16] G[15:8]  R[7:0] */
    N2D_R10G10B10A2,             /* A[31:30] B[29:20] G[19:10] R[9:0] */
    N2D_R5G5B5A1,                /* A[15:15] B[14:10] G[9:5]   R[4:0] */
    N2D_R5G5B5X1,                /*          B[14:10] G[9:5]   R[4:0] */
    N2D_RGBA4444,                /* A[15:12] B[11:8]  G[7:4]   R[3:0] */
    N2D_RGBX4444,                /*          B[11:8]  G[7:4]   R[3:0] */
    N2D_RGB565,                  /*          B[15:11] G[10:5]  R[4:0] */
    N2D_RGB888,                  /*          B[23:16] G[15:8]  R[7:0] */
    N2D_RGB888I,                 /*          B[23:16] G[15:8]  R[7:0] */
    N2D_RGB161616I,              /*          B[47:32] G[31:16] R[15:0] */

    N2D_BGRA8888,                /* A[31:24] R[23:16] G[15:8]  B[7:0] */
    N2D_BGRX8888,                /*          R[23:16] G[15:8]  B[7:0] */
    N2D_B10G10R10A2,             /* A[31:30] R[29:20] G[19:10] B[9:0] */
    N2D_B5G5R5A1,                /* A[15:15] R[14:10] G[9:5]   B[4:0] */
    N2D_B5G5R5X1,                /*          R[14:10] G[9:5]   B[4:0] */
    N2D_BGRA4444,                /* A[15:12] R[11:8]  G[7:4]   B[3:0] */
    N2D_BGRX4444,                /*          R[11:8]  G[7:4]   B[3:0] */
    N2D_BGR565,                  /*          R[15:11] G[10:5]  B[4:0] */
    N2D_BGR888,                  /*          R[23:16] G[15:8]  B[7:0] */

    N2D_ABGR8888,                /* R[31:24] G[23:16] B[15:8] A[7:0] */
    N2D_XBGR8888,                /* R[31:24] G[23:16] B[15:8]        */
    N2D_A2B10G10R10,             /* R[31:22] G[21:12] B[2:11] A[1:0] */
    N2D_A1B5G5R5,                /* R[15:11] G[10:6]  B[5:1]  A[0:0] */
    N2D_X1B5G5R5,                /* R[15:11] G[10:6]  B[5:1]         */
    N2D_ABGR4444,                /* R[15:12] G[11:8]  B[7:4]  A[3:0] */
    N2D_XBGR4444,                /* R[15:12] G[11:8]  B[7:4]         */
    N2D_ABGR16161616I,           /* R[63:48] G[47:32] B[31:16] A[15:0] */
    N2D_ABGR8888I,               /* R[31:24] G[23:16] B[15:8]  A[7:0] */

    N2D_ARGB8888,                /* B[31:24] G[23:16] R[15:8] A[7:0] */
    N2D_XRGB8888,                /* B[31:24] G[23:16] R[15:8]        */
    N2D_A2R10G10B10,             /* B[31:22] G[21:12] R[2:11] A[1:0] */
    N2D_A1R5G5B5,                /* B[15:11] G[10:6]  R[5:1]  A[0:0] */
    N2D_X1R5G5B5,                /* B[15:11] G[10:6]  R[5:1]         */
    N2D_ARGB4444,                /* B[15:12] G[11:8]  R[7:4]  A[3:0] */
    N2D_XRGB4444,                /* B[15:12] G[11:8]  R[7:4]         */
    N2D_ARGB16161616I,           /* B[63:48] G[47:32] R[31:16] A[15:0] */
    N2D_ARGB8888I,               /* B[31:24] G[23:16] R[15:8]  A[7:0] */

    N2D_RGB888_PLANAR,           /* BBBB GGGG RRRR*/
    N2D_RGB888I_PLANAR,          /* BBBB GGGG RRRR*/
    N2D_RGB161616I_PLANAR,       /* BBBB GGGG RRRR*/

    N2D_YUYV    = 0x200,
    N2D_UYVY,
    N2D_YV12,
    N2D_I420,
    N2D_NV12,
    N2D_NV21,
    N2D_NV16,
    N2D_NV61,
    N2D_AYUV,                   /* YYYY UUUU VVVV*/

    N2D_GRAY8,                  /* GRAY[7:0] */
    N2D_GRAY8I,                 /* GRAY[7:0] */

    N2D_NV12_10BIT,
    N2D_NV21_10BIT,
    N2D_NV16_10BIT,
    N2D_NV61_10BIT,
    N2D_P010_MSB,
    N2D_P010_LSB,
    N2D_I010,
    N2D_I010_LSB,

    N2D_A8      =0x300,         /* ALPHA[7:0] */

    N2D_INDEX1  = 0x400,        /* INDEX[0:0] */
    N2D_INDEX8,                 /* INDEX[7:0] */

    N2D_RGB161616F = 0x500,     /*          B[47:32] G[31:16] R[15:0] */
    N2D_RGB323232F,             /*          B[95:64] G[63:32] R[31:0] */
    N2D_ARGB16161616F,          /* B[63:48] G[47:32] R[31:16] A[15:0] */
    N2D_ABGR16161616F,          /* R[63:48] G[47:32] B[31:16] A[15:0] */
    N2D_ARGB32323232F,          /* B[127:96] G[95:64] R[63:32] A[31:0] */
    N2D_ABGR32323232F,          /* R[127:96] G[95:64] B[63:32] A[31:0] */
    N2D_RGB161616F_PLANAR,      /* BBBB GGGG RRRR*/
    N2D_RGB323232F_PLANAR,      /* BBBB GGGG RRRR*/

    N2D_GRAY16F,
    N2D_GRAY32F,

    N2D_HSV8 = 0x600,           /*          V[23:16] S[15:8]  H[7:0]  */
    N2D_HSV32,                  /*          V[95:64] S[63:32] H[31:0] */
    N2D_HLS8,                   /*          S[23:16] L[15:8]  H[7:0]  */
    N2D_HLS32,                  /*          S[95:64] L[63:32] H[31:0] */
}
n2d_buffer_format_t;

typedef enum n2d_orientation
{
    N2D_0,
    N2D_90,
    N2D_180,
    N2D_270,
    N2D_FLIP_X,
    N2D_FLIP_Y,
}
n2d_orientation_t;

typedef enum n2d_tiling
{
    N2D_INVALID_TILED       = 0x0,

    N2D_LINEAR              = 0x1,
    N2D_TILED               = 0x2,
    N2D_SUPER_TILED         = 0x4,
    N2D_MINOR_TILED         = 0x8,

    N2D_TILING_SPLIT_BUFFER = 0x10,
    N2D_TILING_X_MAJOR      = 0x20,
    N2D_TILING_Y_MAJOR      = 0x40,
    N2D_TILING_SWAP         = 0x80,

    N2D_MULTI_TILED         = N2D_TILED | N2D_TILING_SPLIT_BUFFER,

    N2D_MULTI_SUPER_TILED   = N2D_SUPER_TILED | N2D_TILING_SPLIT_BUFFER,

    N2D_YMAJOR_SUPER_TILED  = N2D_SUPER_TILED | N2D_TILING_Y_MAJOR,

    N2D_TILED_8X4           = 0x0100,
    N2D_TILED_4X8           = 0x0100 | N2D_TILING_SWAP,
    N2D_TILED_8X8           = 0x0200,
    N2D_TILED_16X4          = 0x0400,
    N2D_TILED_32X4          = 0x0800,
    N2D_TILED_64X4          = 0x1000,

    N2D_TILED_8X8_XMAJOR    = N2D_TILED_8X8 | N2D_TILING_X_MAJOR,
    N2D_TILED_8X8_YMAJOR    = N2D_TILED_8X8 | N2D_TILING_Y_MAJOR,

    N2D_SUPER_TILED_128B     = 0x10000 | N2D_SUPER_TILED,
    N2D_SUPER_TILED_256B     = 0x20000 | N2D_SUPER_TILED,
}
n2d_tiling_t;

typedef enum n2d_cache_mode
{
    N2D_CACHE_AUTO = 0,
    N2D_CACHE_128  = 0x1,
    N2D_CACHE_256  = 0x2,

    N2D_CACHE_ALL  = N2D_CACHE_128|N2D_CACHE_256,
}
n2d_cache_mode_t;

#define N2D_CACHEMODE_DEFAULT    N2D_CACHE_256

typedef enum n2d_tile_status_config
{
    N2D_TSC_DISABLE            = 0,
    N2D_TSC_DEC_COMPRESSED     = 0x00000001,
}
n2d_tile_status_config_t;

typedef enum n2d_memory_mode
{
    N2D_PACKED,
    N2D_PLANAR,
}
n2d_memory_mode_t;

typedef enum n2d_pixel_color_multiply_mode
{
    N2D_COLOR_MULTIPLY_DISABLE = 0,
    N2D_COLOR_MULTIPLY_ENABLE
}
n2d_pixel_color_multiply_mode_t;

typedef enum n2d_global_color_multiply_mode
{
    N2D_GLOBAL_COLOR_MULTIPLY_DISABLE = 0,
    N2D_GLOBAL_COLOR_MULTIPLY_ALPHA,
    N2D_GLOBAL_COLOR_MULTIPLY_COLOR
}
n2d_global_color_multiply_mode_t;

typedef enum n2d_error
{
    N2D_SUCCESS = 0,
    N2D_INVALID_ARGUMENT,
    N2D_OUT_OF_MEMORY,
    N2D_NO_CONTEXT,
    N2D_TIMEOUT,
    N2D_OUT_OF_RESOURCES,
    N2D_GENERIC_IO,
    N2D_NOT_SUPPORTED,
    N2D_ERROR_HEAP_CORRUPTED,
    N2D_NOT_ALIGNED,
    N2D_NOT_FOUND,
    N2D_INTERRUPTED,
    N2D_MEMORY_LEAK,
}
n2d_error_t;

typedef enum n2d_transparency
{
    N2D_OPAQUE = 0,
    N2D_KEYED,
    N2D_MASKED
}
n2d_transparency_t;

typedef enum n2d_global_alpha
{
    N2D_GLOBAL_ALPHA_OFF,
    N2D_GLOBAL_ALPHA_ON,
    N2D_GLOBAL_ALPHA_SCALE
}
n2d_global_alpha_t;

typedef enum n2d_filter_type
{
    N2D_FILTER_SYNC = 0,
    N2D_FILTER_BLUR,
    N2D_FILTER_USER,
    N2D_FILTER_BILINEAR,
    N2D_FILTER_BICUBIC
}
n2d_filter_type_t;

typedef enum n2d_filter_pass_type
{
    N2D_FILTER_HOR_PASS = 0,
    N2D_FILTER_VER_PASS
}
n2d_filter_pass_type_t;

typedef enum n2d_csc_mode
{
    N2D_CSC_BT709               = 0,
    N2D_CSC_BT601               = 0x1,
    N2D_CSC_BT2020              = 0x2,
    N2D_CSC_USER_DEFINED        = 0x4,
    N2D_CSC_USER_DEFINED_CLAMP  = N2D_CSC_USER_DEFINED | 0x1,

    N2D_CSC_SET_FULL_RANGE      = 0x40000000,
    N2D_CSC_DST                 = 0x80000000,
}
n2d_csc_mode_t;

typedef enum n2d_source_type
{
    N2D_SOURCE_DEFAULT = 0,
    N2D_SOURCE_MASKED,
    N2D_SOURCE_MASKED_MONO
}
n2d_source_type_t;

typedef enum n2d_state_value
{
    N2D_CSC_YUV_TO_RGB      = 0x1,
    N2D_CSC_RGB_TO_YUV,
}
n2d_state_value_t;

typedef enum n2d_device_id
{
    N2D_DEVICE_0 = 0,
    N2D_DEVICE_1,
    N2D_DEVICE_2,
    N2D_DEVICE_3,
    N2D_DEVICE_4,
    N2D_DEVICE_5,
    N2D_DEVICE_6,
    N2D_DEVICE_7,
} n2d_device_id_t;

typedef enum n2d_core_id
{
    N2D_CORE_0 = 0,
    N2D_CORE_1,
    N2D_CORE_2,
    N2D_CORE_3,
    N2D_CORE_4,
    N2D_CORE_5,
    N2D_CORE_6,
    N2D_CORE_7,
} n2d_core_id_t;

typedef enum n2d_multicore_mode
{
    N2D_COMBINED,
    N2D_INDEPENDENT
}
n2d_multicore_mode_t;

typedef enum n2d_map_type {
    N2D_MAP_UNKNOWN = 0x00,
    N2D_MAP_TO_GPU = 0x01,
    N2D_MAP_TO_USER = 0x02,
    N2D_MAP_TO_KERNEL = 0x04,
}
n2d_map_type_t;

typedef enum n2d_u8_to_u10_conversion_mode
{
    N2D_ADD_LOWER_BITS,
    N2D_NOT_ADD_LOWER_BITS,
    N2D_NOT_ADD_HIGHER_BITS
}
n2d_u8_to_u10_conversion_mode_t;

typedef enum n2d_state_type
{
    N2D_SET_CSC = 0x0,
    N2D_SET_MASKPACK,

#ifdef N2D_SUPPORT_NORMALIZATION
    N2D_SET_NORMALIZATION,
#endif

    N2D_SET_DITHER,

#ifdef N2D_SUPPORT_HISTOGRAM
    N2D_SET_HISTOGRAM_CALC,
    N2D_SET_HISTOGRAM_EQUAL,
#endif

#ifdef N2D_SUPPORT_BRIGHTNESS
    N2D_SET_BRIGHTNESS,
#endif

#ifdef N2D_SUPPORT_CONTEXT_ID
    N2D_SET_CONTEXT_ID,
#endif

    N2D_SET_FAST_CLEAR,
    N2D_SET_MULTISOURCE_INDEX,
    N2D_SET_CLIP_RECTANGLE,
    N2D_SET_GLOBAL_ALPHA,
    N2D_SET_ALPHABLEND_MODE,
    N2D_SET_MULTISRC_DST_RECTANGLE,
    N2D_SET_ROP,
    N2D_SET_TRANSPARENCY,
    N2D_SET_SRC_COLORKEY,
    N2D_SET_DST_COLORKEY,
    N2D_SET_PIXEL_MULTIPLY_MODE,
    N2D_SET_KERNEL_SIZE,
    N2D_SET_FILTER_TYPE,
    N2D_SET_U8TOU10CVT_MODE,
    N2D_SET_FILTER_USER_DEFINED,
}
n2d_state_type_t;

typedef enum n2d_get_state_type
{
#ifdef N2D_SUPPORT_HISTOGRAM
    N2D_GET_HISTOGRAM_ARRAY_SIZE,
    N2D_GET_HISTOGRAM_ARRAY,
#endif
    N2D_GET_CORE_COUNT,
    N2D_GET_CORE_INDEX,
    N2D_GET_DEVICE_COUNT,
    N2D_GET_DEVICE_INDEX,
}
n2d_get_state_type_t;

typedef enum n2d_normalization_mode
{
    N2D_NORMALIZATION_STD_Z_SCORE,
    N2D_NORMALIZATION_Z_SCORE,
    N2D_NORMALIZATION_MIN_MAX,
}
n2d_normalization_mode_t;

typedef enum n2d_vidmem_pool
{
    N2D_POOL_UNKNOWN = 0,
    N2D_POOL_SYSTEM,
    N2D_POOL_COMMAND,
}
n2d_vidmem_pool_t;

typedef enum n2d_vidmem_type
{
    N2D_TYPE_UNKNOWN = 0,
    N2D_TYPE_COMMAND,
}
n2d_vidmem_type_t;

typedef enum n2d_wrap_memory_type
{
    N2D_WRAP_FROM_DMABUF     = 0x4,
    N2D_WRAP_FROM_USERMEMORY = 0x8,
}
n2d_wrap_memory_type_t;

typedef enum n2d_keep_alpha_value_t
{
    N2D_KEEP_ALPHA_F = 0,
    N2D_KEEP_ALPHA_ZERO,
}
n2d_keep_alpha_value_t;

typedef enum n2d_cache_op {
    N2D_CACHE_CLEAN = 0x01,
    N2D_CACHE_INVALIDATE = 0x02,
    N2D_CACHE_FLUSH = N2D_CACHE_CLEAN | N2D_CACHE_INVALIDATE,
    N2D_CACHE_MEMORY_BARRIER = 0x04
} n2d_cache_op_t;

#ifdef __cplusplus
}
#endif

#endif
