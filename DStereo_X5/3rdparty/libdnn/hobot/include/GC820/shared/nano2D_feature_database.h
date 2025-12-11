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

#ifndef _nano2D_feature_database_h_
#define _nano2D_feature_databse_h_

typedef struct
{
    /* Chip ID. */
    n2d_uint32_t    chip_id;
    n2d_uint32_t    chip_version;
    n2d_uint32_t    pid;
    n2d_uint32_t    cid;

    n2d_uint32_t    N2D_YUV420_OUTPUT:1;
    n2d_uint32_t    PE2D_LINEAR_YUV420_10BIT:1;
    n2d_uint32_t    PE2D_MAJOR_SUPER_TILE:1;
    n2d_uint32_t    G2D_DEC400EX:1;
    n2d_uint32_t    REG_AndroidOnly:1;
    n2d_uint32_t    REG_OnePass2DFilter:1;
    n2d_uint32_t    REG_DESupertile:1;
    n2d_uint32_t    REG_NewFeatures0:1;
    n2d_uint32_t    REG_DEEnhancements1:1;
    n2d_uint32_t    REG_Compression2D:1;
    n2d_uint32_t    REG_MultiSrcV2:1;
    n2d_uint32_t    RGB_PLANAR:1;
    n2d_uint32_t    REG_NoScaler:1;
    n2d_uint32_t    REG_DualPipeOPF:1;
    n2d_uint32_t    REG_SeperateSRCAndDstCache:1;
    n2d_uint32_t    N2D_FEATURE_AXI_FE:1;
    n2d_uint32_t    N2D_FEATURE_CSC_PROGRAMMABLE:1;
    n2d_uint32_t    N2D_FEATURE_DEC400_FC:1;
    n2d_uint32_t    N2D_FEATURE_MASK:1;
    n2d_uint32_t    N2D_FEATURE_COLORKEY:1;
    n2d_uint32_t    N2D_FEATURE_NORMALIZATION:1;
    n2d_uint32_t    N2D_FEATURE_NORMALIZATION_QUANTIZATION:1;
    n2d_uint32_t    N2D_FEATURE_HISTOGRAM:1;
    n2d_uint32_t    N2D_FEATURE_BRIGHTNESS_SATURATION:1;
    n2d_uint32_t    N2D_FEATURE_64BIT_ADDRESS:1;
    n2d_uint32_t    N2D_FEATURE_CONTEXT_ID:1;
    n2d_uint32_t    N2D_FEATURE_SECURE_BUFFER:1;

    /*for kernel*/
    n2d_uint32_t    N2D_FEATURE_MMU_PAGE_DESCRIPTOR:1;
    n2d_uint32_t    N2D_FEATURE_SECURITY_AHB:1;
    n2d_uint32_t    N2D_FEATURE_FRAME_DONE_INTR:1;
}
n2d_feature_database;

static n2d_feature_database n2d_chip_features[] = {
    /*gc820 0x5630 rc0c*/
    {
        0x820,     /*chip_id*/
        0x5632,    /*chip_version*/
        0x8200,    /*pid*/
        0x218,     /*cid*/
        1,         /*N2D_YUV420_OUTPUT*/
        1,         /*PE2D_LINEAR_YUV420_10BIT*/
        0,         /*PE2D_MAJOR_SUPER_TILE*/
        0,         /*G2D_DEC400EX*/
        1,         /*android only*/
        1,         /*OnePassFilter*/
        1,         /*DE supertile*/
        1,         /*REG_NewFeatures0*/
        1,         /*REG_DEEnhancements1*/
        0,         /*Compression2D*/
        1,         /*REG_MultiSrcV2*/
        1,         /*RGB_PLANAR */
        1,         /*REG_NoScaler*/
        1,         /*DualPipeOPF*/
        1,         /*REG_SeperateSRCAndDstCache*/
        0,         /*AXI-FE*/
        1,         /*CSC*/
        0,         /*FC*/
        0,         /*Mask*/
        0,         /*ColorKey*/
        0,         /*Normalization*/
        0,         /*Quantization*/
        0,         /*Histogram*/
        0,         /*Brightness_Saturation*/
        0,         /*64BitGpuAddress*/
        0,         /*contextID*/
        0,         /*securebuffer*/
        /*kernel*/
        0,         /*MMU PD MODE*/
        0,         /*SECURITY_AHB*/
        0,         /*FRAME_DONE_INTR*/
    },
};

static n2d_feature_database * query_features(
    n2d_uint32_t chip_id,
    n2d_uint32_t chip_version,
    n2d_uint32_t product_id,
    n2d_uint32_t cid
    )
{
    n2d_uint32_t size = sizeof(n2d_chip_features) / sizeof(n2d_chip_features[0]);
    n2d_uint32_t i;
    /* check formal release entries */
    for (i = 0; i < size; ++i)
    {

        if ((n2d_chip_features[i].chip_id == chip_id)
            && (n2d_chip_features[i].chip_version == chip_version)
            && (n2d_chip_features[i].pid == product_id)
            && (n2d_chip_features[i].cid == cid)
           )
        {
            return &n2d_chip_features[i];
        }
    }
    return N2D_NULL;
}

#endif /* _nano2D_feature_database_h_ */
