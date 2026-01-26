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

#ifndef _nano2d_util_h_
#define _nano2d_util_h_

#include "nano2D.h"

#ifdef __cplusplus
extern "C" {
#endif

#define N2D_COLOR_BGRA8(A, R, G, B) \
    ((((A) & 0xFF) << 24) | (((R) & 0xFF) << 16) | (((G) & 0xFF) << 8) | ((B) & 0xFF))

#define N2D_COLOR_B5G6R5(R, G, B) \
    ((((R)&0x1F) << 16) | (((G)&0x3F) << 8) | ((B) & 0x1F))


#pragma pack(1)

typedef struct BMPFILEHEADER
{
    unsigned short bfType;           /* Magic number for file */
    unsigned int   bfSize;           /* Size of file */
    unsigned short bfReserved1;      /* Reserved */
    unsigned short bfReserved2;      /* ... */
    unsigned int   bfOffBits;        /* Offset to bitmap data */
}
BMPFILEHEADER;

/**** BMP file info structure ****/
typedef struct BMPINFOHEADER
{
    unsigned int   biSize;           /* Size of info header */
    int            biWidth;          /* Width of image */
    int            biHeight;         /* Height of image */
    unsigned short biPlanes;         /* Number of color planes */
    unsigned short biBitCount;       /* Number of bits per pixel */
    unsigned int   biCompression;    /* Type of compression to use */
    unsigned int   biSizeImage;      /* Size of image data */
    int            biXPelsPerMeter;  /* X pixels per meter */
    int            biYPelsPerMeter;  /* Y pixels per meter */
    unsigned int   biClrUsed;        /* Number of colors used */
    unsigned int   biClrImportant;   /* Number of important colors */
}
BMPINFOHEADER;

#define BIT_RGB       0             /* No compression - straight BGR data */
#define BIT_RLE8      1             /* 8-bit run-length compression */
#define BIT_RLE4      2             /* 4-bit run-length compression */
#define BIT_BITFIELDS 3             /* RGB bitmap with RGB masks */

typedef struct RGB
{
    unsigned char   rgbBlue;
    unsigned char   rgbGreen;
    unsigned char   rgbRed;
    unsigned char   rgbReserved;
}
RGB;

typedef struct _BMPINFO
{
    BMPINFOHEADER   bmiHeader;
    union _bmpInfo{
    RGB             bmiColors[256];
    unsigned int    mask[3];
    }bmpInfo_u;
}
BMPINFO;

#pragma pack()

n2d_error_t
n2d_util_allocate_buffer(
    n2d_uint32_t width,
    n2d_uint32_t height,
    n2d_buffer_format_t format,
    n2d_orientation_t orientation,
    n2d_tiling_t      tiling,
    n2d_tile_status_config_t tile_status_config,
    n2d_buffer_t *buffer);

unsigned char *
n2d_util_load_dibitmap(
    const char *filename,
    BMPINFO **info);

n2d_error_t
n2d_util_load_buffer_from_file(
    char *name,
    n2d_buffer_t *buffer);

n2d_error_t
n2d_util_load_buffer_from_raw_file(
    char* filename,
    n2d_uint32_t imageWidth,
    n2d_uint32_t imageHeight,
    n2d_buffer_format_t format,
    n2d_tiling_t tiling,
    n2d_buffer_t* buffer);

n2d_error_t
n2d_util_save_buffer_to_file(
    n2d_buffer_t *buffer,
    char *name);

n2d_error_t
n2d_util_save_buffer_to_vimg(
    n2d_buffer_t *buffer,
    char *filename);

n2d_error_t n2d_util_save_array_to_dat(
    n2d_uint32_t *arrar,
    n2d_uint32_t size,
    n2d_string   filename);

n2d_error_t
n2d_util_fb_blit_buffer(
    n2d_buffer_t *buffer);

n2d_error_t
n2d_util_save_rawdata(
    n2d_pointer memory,
    n2d_uint32_t w,
    n2d_uint32_t h,
    float rate ,
    char *prename,
    char *exname);

n2d_error_t
n2d_util_load_buffer_from_comfile(
    n2d_const_string         *Dataname,
    n2d_const_string         *Tname,
    n2d_int32_t              width,
    n2d_int32_t              height,
    n2d_tiling_t             tiling,
    n2d_buffer_format_t      Format,
    n2d_tile_status_config_t TilestatusConfig,
    n2d_buffer_t             *buffer);

#ifdef __cplusplus
}
#endif
#endif
