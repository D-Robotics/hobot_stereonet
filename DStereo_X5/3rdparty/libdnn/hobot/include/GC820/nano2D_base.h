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

#ifndef _nano2D_base_h__
#define _nano2D_base_h__

#include "version.h"
#include "nano2D_types.h"
#include "nano2D_enum.h"

#if (NANO2D_VERSION_MAJOR > 2) && (NANO2D_VERSION_MINOR > 0)
#include "nano2D_chip_base.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_UV_PLANE                                2
#define MAX_PLANE                                   3

#define N2D_CSC_PROGRAMMABLE_SIZE                   12
#define N2D_SECURITY_BUFFER_RECORD_SIZE             128

typedef struct n2d_point
{
    n2d_int32_t x;
    n2d_int32_t y;
}
n2d_point_t;

typedef struct n2d_rectangle
{
    n2d_int32_t x;
    n2d_int32_t y;
    n2d_int32_t width;
    n2d_int32_t height;
}
n2d_rectangle_t;

typedef struct n2d_rgb_u32
{
    n2d_uint32_t r;
    n2d_uint32_t g;
    n2d_uint32_t b;
}
n2d_rgb_u32_t;

typedef struct _n2d_dither_info
{
    n2d_bool_t enable;
}
n2d_dither_info_t;

typedef struct n2d_user_memory_desc {
    n2d_uint32_t                    flag;

    n2d_int32_t                     handle;
    n2d_uintptr_t                   dmabuf;

    n2d_uintptr_t                   logical;
    n2d_uintptr_t                   physical;
    n2d_size_t                      size;
}
n2d_user_memory_desc_t;

typedef struct n2d_export_memory {
    n2d_uintptr_t                   fd;
    n2d_uintptr_t                   uv_fd[MAX_UV_PLANE];
    n2d_uintptr_t                   ts_fd[MAX_PLANE];
}
n2d_export_memory_t;

typedef struct n2d_csc_config
{
    n2d_csc_mode_t                  cscMode;
    n2d_state_value_t               userCSCMode;
    n2d_int32_t                     cscTable[N2D_CSC_PROGRAMMABLE_SIZE];
}
n2d_csc_config_t;

typedef struct n2d_maskpack_config
{
    n2d_pointer                     streamBits;
    n2d_uint32_t                    streamStride;
    n2d_uint32_t                    streamWidth;
    n2d_uint32_t                    streamHeight;
    n2d_point_t                     streamOffset;
    n2d_color_t                     fgColor;
    n2d_color_t                     bgColor;
}
n2d_maskpack_config_t;

typedef struct n2d_normalization_info
{
    n2d_bool_t                      enable;
    n2d_keep_alpha_value_t          keepAlphaValue;
    n2d_rgb_u32_t                   meanValue;
    n2d_rgb_u32_t                   minValue;
    n2d_rgb_u32_t                   stdReciprocal;
    n2d_rgb_u32_t                   maxMinReciprocal;
    n2d_uint32_t                    stepReciprocal;
    n2d_normalization_mode_t        normalizationMode;
}
n2d_normalization_info_t;

typedef struct n2d_fastclear_config
{
    n2d_bool_t                        enable;
    n2d_uint32_t                      value;
}
n2d_fastclear_config_t;

typedef n2d_uint32_t n2d_multisource_index_config_t;

typedef struct n2d_global_alpha_config
{
    n2d_global_alpha_t              srcMode;
    n2d_global_alpha_t              dstMode;
    n2d_uint32_t                    srcValue;
    n2d_uint32_t                    dstValue;
}
n2d_global_alpha_config_t;

typedef struct n2d_multisrc_dst_rect_config
{
    struct n2d_buffer*              source;
    n2d_rectangle_t                 srcRect;
    n2d_rectangle_t                 dstRect;
}
n2d_multisrc_dst_rect_config_t;

typedef struct n2d_rop_config
{
    n2d_uint32_t                    fg_rop;
    n2d_uint32_t                    bg_rop;
}
n2d_rop_config_t;

typedef struct n2d_transparency_config
{
    n2d_transparency_t              src;
    n2d_transparency_t              dst;
    n2d_transparency_t              pat;
}
n2d_transparency_config_t;

typedef struct n2d_colorkey_config
{
    n2d_uint32_t                    colorkeyLow;
    n2d_uint32_t                    colorkeyHigh;
}
n2d_colorkey_config_t;

typedef struct n2d_multiply_mode_config
{
    n2d_pixel_color_multiply_mode_t  srcPremult;
    n2d_pixel_color_multiply_mode_t  dstPremult;
    n2d_global_color_multiply_mode_t srcGlobal;
    n2d_pixel_color_multiply_mode_t  dstDemult;
}
n2d_multiply_mode_config_t;

typedef struct n2d_kernel_size_config
{
    n2d_uint8_t horSize;
    n2d_uint8_t verSize;
}
n2d_kernel_size_config_t;

typedef struct n2d_filter_user_defined_config
{
    n2d_bool_t                   horPass;
    n2d_bool_t                   verPass;
    n2d_filter_pass_type_t       passTpye;
    n2d_uint16_t*                kernelArray;
}
n2d_filter_user_defined_config_t;

typedef struct n2d_state_config
{
    n2d_state_type_t      state;
    union _ {
        n2d_csc_config_t         csc;
        n2d_maskpack_config_t    maskPack;
        n2d_dither_info_t        ditherInfo;

#ifdef N2D_SUPPORT_NORMALIZATION

        n2d_normalization_info_t normalizationInfo;
#endif

#ifdef N2D_SUPPORT_BRIGHTNESS

        n2d_histogram_calc_config_t         histogramCalc;
        n2d_histogram_equal_config_t        histogramEqual;
        n2d_brightness_saturation_config_t  brightnessAndSaturation;
#endif

#ifdef N2D_SUPPORT_CONTEXT_ID

        n2d_contextID_config_t              contextID;
#endif
        n2d_fastclear_config_t              fastclear;
        n2d_multisource_index_config_t      multisourceIndex;
        n2d_rectangle_t                     clipRect;
        n2d_global_alpha_config_t           globalAlpha;
        n2d_blend_t                         alphablendMode;
        n2d_multisrc_dst_rect_config_t      multisrcAndDstRect;
        n2d_rop_config_t                    rop;
        n2d_transparency_config_t           transparency;
        n2d_colorkey_config_t               srcColorKey;
        n2d_colorkey_config_t               dstColorKey;
        n2d_multiply_mode_config_t          pixelMultiplyMode;
        n2d_kernel_size_config_t            kernelSize;
        n2d_filter_type_t                   filterType;
        n2d_u8_to_u10_conversion_mode_t     u8Tou10cvtMode;
        n2d_filter_user_defined_config_t    filterUserDefined;
    } config;
}
n2d_state_config_t;

typedef struct n2d_get_state_config
{
    n2d_get_state_type_t      state;
    union __ {
#ifdef N2D_SUPPORT_HISTOGRAM

        n2d_uint32_t              histogramArraySize;
        n2d_histogram_array_t     histogramArray;
#endif
        n2d_uint32_t              coreCount;
        n2d_core_id_t             coreIndex;
        n2d_device_id_t           deviceIndex;
        n2d_uint32_t              deviceCount;
    } config;
}
n2d_get_state_config_t;

#ifdef __cplusplus
}
#endif

#endif
