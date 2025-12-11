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

#ifndef __nano2D_engine_h__
#define __nano2D_engine_h__

#define GC_CHIP_ID_Address                                               0x00020
#define GC_CHIP_REV_Address                                              0x00024
#define GC_PRODUCT_ID_Address                                            0x000A8
#define GC_CHIP_CUSTOMER_Address                                         0x00030
#define GC_CHIP_SPECS_Address                                            0x00048
#define GC_CHIP_TIME_Address                                             0x0002C
#define GC_CHIP_SPECS_NUM_PIXEL_PIPES                                      27:25

#define GC_FEATURES_Address                                              0x0001C
#define GC_MINOR_FEATURES0_Address                                       0x00034
#define GC_MINOR_FEATURES2_NO_INDEX_PATTERN                                28:28
#define GC_FEATURES_PIPE_2D                                                  9:9
#define GC_MINOR_FEATURES4_MULTI_SRC_V2                                    22:22
#define GC_MINOR_FEATURES2_FLUSH_FIXED_2D                                  23:23
#define   GC_MINOR_FEATURES2_NO_INDEX_PATTERN_NONE                           0x0
#define   GC_FEATURES_PIPE_2D_AVAILABLE                                      0x1
#define   GC_MINOR_FEATURES4_MULTI_SRC_V2_AVAILABLE                          0x1
#define   GC_MINOR_FEATURES2_FLUSH_FIXED_2D_NONE                             0x0

#define GCREG_AHBDEC_CONTROL_Address                                     0x00800
#define GCREG_AHBDEC_CONTROL_ResetValue                               0x0201018A
#define GCREG_AHBDEC_CONTROL_DISABLE_COMPRESSION                             1:1
#define GCREG_AHBDEC_CONTROL_EX2_Address                                 0x00808
#define GCREG_AHBDEC_CONTROL_EX2_ResetValue                           0x003FC810
#define GCREG_AHBDEC_CONTROL_EX2_TILE_STATUS_READ_ID                         6:0
#define GCREG_AHBDEC_CONTROL_EX2_TILE_STATUS_WRITE_ID                       13:7
#define   GCREG_AHBDEC_CONTROL_DISABLE_COMPRESSION_ENABLE                    0x0

#define AQ_MEMORY_DEBUG_Address                                          0x00414
#define AQ_MEMORY_DEBUG_MAX_OUTSTANDING_READS                                7:0

#define AQ_INTR_ENBL_Address                                             0x00014
#define AQ_INTR_ACKNOWLEDGE_Address                                      0x00010

#define AQ_CMD_BUFFER_ADDR_Address                                       0x00654
#define AQ_CMD_BUFFER_CTRL_ENABLE                                          16:16
#define AQ_CMD_BUFFER_CTRL_Address                                       0x00658
#define AQ_CMD_BUFFER_CTRL_PREFETCH                                         15:0
#define   AQ_CMD_BUFFER_CTRL_ENABLE_ENABLE                                   0x1

#define AQ_EVENT_PE_SRC                                                      6:6
#define AQ_EVENT_EVENT_ID                                                    4:0
#define   AQ_EVENT_PE_SRC_ENABLE                                             0x1

#define AQEventRegAddrs                                                   0x0E01
#define AQ_FLUSH_Address                                                 0x0380C
#define AQ_FLUSH_PE2D_CACHE                                                  3:3
#define   AQ_FLUSH_PE2D_CACHE_ENABLE                                         0x1

#define gcregStoreCounterRegAddrs                                         0x0E12
#define GCREG_STORE_COUNTER_Address                                      0x03848
#define GCREG_STORE_COUNTER_SOURCE_SH                                      16:16

#define   AQ_COMMAND_NOP_COMMAND_OPCODE_END                                 0x02
#define   AQ_COMMAND_NOP_COMMAND_OPCODE_NOP                                 0x03
#define   AQ_COMMAND_START_DE_COMMAND_OPCODE_START_DE                       0x04
#define   AQ_COMMAND_END_COMMAND_OPCODE_END                                 0x02
#define   AQ_COMMAND_LOAD_STATE_COMMAND_OPCODE_LOAD_STATE                   0x01
#define   AQ_COMMAND_WAIT_COMMAND_OPCODE_WAIT                               0x07
#define   AQ_COMMAND_LINK_COMMAND_OPCODE_LINK                               0x08

#define AQ_COMMAND_START_DE_COMMAND_COUNT                                   15:8
#define AQ_COMMAND_START_DE_COMMAND_DATA_COUNT                             26:16
#define AQ_COMMAND_START_DE_COMMAND_OPCODE                                 31:27
#define AQ_COMMAND_TOP_LEFT_X                                               15:0
#define AQ_COMMAND_BOTTOM_RIGHT_X                                           15:0
#define AQ_COMMAND_TOP_LEFT_Y                                              31:16
#define AQ_COMMAND_BOTTOM_RIGHT_Y                                          31:16
#define AQ_COMMAND_WAIT_COMMAND_DELAY                                       15:0
#define AQ_COMMAND_LINK_COMMAND_PREFETCH                                    15:0
#define AQ_COMMAND_NOP_COMMAND_OPCODE                                      31:27
#define AQ_COMMAND_END_COMMAND_OPCODE                                      31:27
#define AQ_COMMAND_WAIT_COMMAND_OPCODE                                     31:27
#define AQ_COMMAND_LINK_COMMAND_OPCODE                                     31:27
#define AQ_COMMAND_LOAD_STATE_COMMAND_COUNT                                25:16
#define AQ_COMMAND_LOAD_STATE_COMMAND_OPCODE                               31:27
#define AQ_COMMAND_LOAD_STATE_COMMAND_ADDRESS                               15:0

#define AQ_HI_IDLE_Address                                               0x00004
#define AQ_HI_IDLE_IDLE_FE                                                   0:0

#define AQ_PIPE_SELECT_Address                                           0x03800
#define AQ_PIPE_SELECT_PIPE                                                  0:0

#define AQSemaphoreRegAddrs                                               0x0E02
#define AQ_SEMAPHORE_Address                                             0x03808
#define AQ_SEMAPHORE_SOURCE                                                  4:0
#define AQ_SEMAPHORE_DESTINATION                                            12:8
#define AQ_SEMAPHORE_FRONT_END                                             29:28
#define   AQ_SEMAPHORE_FRONT_END_PREFETCH                                    0x3
#define   AQ_SEMAPHORE_DESTINATION_BLT_ENGINE                               0x10

#define STALL_COMMAND_OPCODE                                               31:27
#define STALL_STALL_SOURCE                                                   4:0
#define STALL_STALL_DESTINATION                                             12:8
#define STALL_STALL_FRONT_END                                              29:28
#define STALL_STALL_DESTINATION                                             12:8
#define   STALL_STALL_SOURCE_FRONT_END                                      0x01
#define   STALL_STALL_FRONT_END_PREFETCH                                     0x3
#define   STALL_STALL_DESTINATION_PIXEL_ENGINE                              0x07
#define   STALL_STALL_DESTINATION_BLT_ENGINE                                0x10

#define GCREG_MMUAHB_CONTROL_Address                                     0x00388
#define GCREG_MMUAHB_CONTEXT_PD_ENTRY0_Address                           0x003B4
#define GCREG_MMUAHB_TABLE_ARRAY_BASE_ADDRESS_LOW_Address                0x0038C
#define GCREG_MMUAHB_TABLE_ARRAY_BASE_ADDRESS_HIGH_Address               0x00390
#define GCREG_MMUAHB_TABLE_ARRAY_SIZE_Address                            0x00394
#define GCREG_MMUAHB_SAFE_SECURE_ADDRESS_Address                         0x0039C
#define GCREG_MMUAHB_SAFE_NON_SECURE_ADDRESS_Address                     0x00398
#define GCREG_MMUAHB_SAFE_ADDRESS_EXT_Address                            0x003A0
#define GCREG_MMUAHB_CONTROL_MMU                                             0:0
#define GCREG_MMUAHB_CONTROL_SET_UP_MMU                                      5:5
#define GCREG_MMUAHB_CONTEXT_PD_ENTRY0_PD_ADDRESS                           31:4
#define GCREG_MMUAHB_CONTEXT_PD_ENTRY0_MASTER_TLB_MODE                       1:0
#define GCREG_MMUAHB_CONTEXT_PD_ENTRY0_PD_ADDRESS                           31:4
#define GCREG_MMUAHB_SAFE_ADDRESS_EXT_SAFE_ADDRESS_SECURE                  23:16
#define GCREG_MMUAHB_SAFE_ADDRESS_EXT_MASK_SAFE_ADDRESS_SECURE             31:31
#define GCREG_MMUAHB_SAFE_ADDRESS_EXT_SAFE_ADDRESS_NON_SECURE                7:0
#define GCREG_MMUAHB_SAFE_ADDRESS_EXT_MASK_SAFE_ADDRESS_NON_SECURE         15:15
#define   GCREG_MMUAHB_CONTROL_MMU_ENABLE                                    0x1
#define   GCREG_MMUAHB_CONTROL_SET_UP_MMU_FROM_REG                           0x1
#define   GCREG_MMUAHB_SAFE_ADDRESS_EXT_MASK_SAFE_ADDRESS_SECURE_ENABLED     0x0
#define   GCREG_MMUAHB_CONTEXT_PD_ENTRY0_MASTER_TLB_MODE_BY_PASS_MODE        0x2
#define   GCREG_MMUAHB_CONTEXT_PD_ENTRY0_MASTER_TLB_MODE_FOUR_K_MODE         0x0
#define   GCREG_MMUAHB_SAFE_ADDRESS_EXT_MASK_SAFE_ADDRESS_NON_SECURE_ENABLED 0x0

#define   AQ_PIPE_SELECT_PIPE_PIPE2D                                         0x1
#define   AQ_SEMAPHORE_DESTINATION_PIXEL_ENGINE                             0x07
#define   STALL_COMMAND_OPCODE_STALL                                        0x09
#define   AQ_SEMAPHORE_DESTINATION_PIXEL_ENGINE                             0x07
#define   AQ_SEMAPHORE_SOURCE_FRONT_END                                     0x01

#define GC_TILE_CACHE_FLUSH_Address                                      0x01650
#define GC_TILE_CACHE_FLUSH_FLUSH                                            0:0
#define   GC_TILE_CACHE_FLUSH_FLUSH_ENABLE                                   0x1

#define AQ_MEMORY_ADDRESS_TYPE                                             31:31
#define   AQ_MEMORY_ADDRESS_TYPE_SYSTEM                                      0x0
#define   AQ_MEMORY_ADDRESS_TYPE_VIRTUAL_SYSTEM                              0x1

#define AQ_MEMORY_ADDRESS_ADDRESS                                           30:0
#define AQ_HI_CLOCK_CONTROL_Address                                      0x00000
#define AQ_HI_CLOCK_CONTROL_IDLE2_D                                        17:17
#define AQ_HI_CLOCK_CONTROL_IDLE3_D                                        16:16

#define GCREG_CMD_BUFFER_AHB_CTRL_Address                                0x003A4
#define GCREG_DEYUV2_RGB_COEF0_Address                                   0x01340
#define GCREG_DEYUV2_RGB_COEF0_COEFFICIENT0                                 15:0
#define GCREG_DEYUV2_RGB_COEF0_COEFFICIENT1                                31:16

#define gcregBltGeneralControlRegAddrs                                    0x502E
#define GCREG_BLT_GENERAL_CONTROL_STREAM_CONTROL                             0:0
#define GCREG_BLT_GENERAL_CONTROL_STREAM_CONTROL_UNLOCK                      0x0

#define GCREG_DEC400_READ_CONFIG_COMPRESSION_FORMAT                          7:3
#define GCREG_DEC400_WRITE_CONFIG_COMPRESSION_FORMAT                         7:3
#define GCREG_DEC400_READ_CONFIG_TILE_MODE                                 30:25
#define GCREG_DEC400_WRITE_CONFIG_TILE_MODE                                30:25
#define GCREG_DEC400_FAST_CLEAR_VALUE_Address                            0x19300
#define GCREG_DEC400_READ_CONFIG_COMPRESSION_ENABLE                          0:0
#define GCREG_DEC400_READ_CONFIG_Address                                 0x18880

#define   GCREG_DEC400_READ_CONFIG_COMPRESSION_FORMAT_ARGB8                 0x00
#define   GCREG_DEC400_READ_CONFIG_COMPRESSION_FORMAT_XRGB8                 0x01
#define   GCREG_DEC400_READ_CONFIG_COMPRESSION_FORMAT_AYUV                  0x02
#define   GCREG_DEC400_READ_CONFIG_COMPRESSION_FORMAT_UYVY                  0x03
#define   GCREG_DEC400_READ_CONFIG_COMPRESSION_FORMAT_YUY2                  0x04
#define   GCREG_DEC400_READ_CONFIG_COMPRESSION_FORMAT_YUV_ONLY              0x05
#define   GCREG_DEC400_READ_CONFIG_COMPRESSION_FORMAT_UV_MIX                0x06
#define   GCREG_DEC400_READ_CONFIG_COMPRESSION_FORMAT_ARGB4                 0x07
#define   GCREG_DEC400_READ_CONFIG_COMPRESSION_FORMAT_XRGB4                 0x08
#define   GCREG_DEC400_READ_CONFIG_COMPRESSION_FORMAT_A1RGB5                0x09
#define   GCREG_DEC400_READ_CONFIG_COMPRESSION_FORMAT_X1RGB5                0x0A
#define   GCREG_DEC400_READ_CONFIG_COMPRESSION_FORMAT_R5G6B5                0x0B
#define   GCREG_DEC400_READ_CONFIG_COMPRESSION_FORMAT_A2R10G10B10           0x0F
#define   GCREG_DEC400_READ_CONFIG_COMPRESSION_FORMAT_BAYER                 0x10
#define   GCREG_DEC400_READ_CONFIG_COMPRESSION_FORMAT_COEFFICIENT           0x12
#define   GCREG_DEC400_READ_CONFIG_COMPRESSION_FORMAT_ARGB16                0x13
#define   GCREG_DEC400_READ_CONFIG_COMPRESSION_FORMAT_X2RGB10               0x15

#define   GCREG_DEC400_WRITE_CONFIG_COMPRESSION_FORMAT_ARGB8                0x00
#define   GCREG_DEC400_WRITE_CONFIG_COMPRESSION_FORMAT_XRGB8                0x01
#define   GCREG_DEC400_WRITE_CONFIG_COMPRESSION_FORMAT_AYUV                 0x02
#define   GCREG_DEC400_WRITE_CONFIG_COMPRESSION_FORMAT_UYVY                 0x03
#define   GCREG_DEC400_WRITE_CONFIG_COMPRESSION_FORMAT_YUY2                 0x04
#define   GCREG_DEC400_WRITE_CONFIG_COMPRESSION_FORMAT_YUV_ONLY             0x05
#define   GCREG_DEC400_WRITE_CONFIG_COMPRESSION_FORMAT_UV_MIX               0x06
#define   GCREG_DEC400_WRITE_CONFIG_COMPRESSION_FORMAT_ARGB4                0x07
#define   GCREG_DEC400_WRITE_CONFIG_COMPRESSION_FORMAT_XRGB4                0x08
#define   GCREG_DEC400_WRITE_CONFIG_COMPRESSION_FORMAT_A1RGB5               0x09
#define   GCREG_DEC400_WRITE_CONFIG_COMPRESSION_FORMAT_X1RGB5               0x0A
#define   GCREG_DEC400_WRITE_CONFIG_COMPRESSION_FORMAT_R5G6B5               0x0B
#define   GCREG_DEC400_WRITE_CONFIG_COMPRESSION_FORMAT_A2R10G10B10          0x0F
#define   GCREG_DEC400_WRITE_CONFIG_COMPRESSION_FORMAT_BAYER                0x10
#define   GCREG_DEC400_WRITE_CONFIG_COMPRESSION_FORMAT_COEFFICIENT          0x12
#define   GCREG_DEC400_WRITE_CONFIG_COMPRESSION_FORMAT_ARGB16               0x13
#define   GCREG_DEC400_WRITE_CONFIG_COMPRESSION_FORMAT_X2RGB10              0x15

#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_TILE8X8_XMAJOR                 0x00
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_TILE8X8_YMAJOR                 0x01
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_TILE16X4                       0x02
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_TILE8X4                        0x03
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_TILE4X8                        0x04
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_RASTER16X4                     0x06
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_TILE64X4                       0x07
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_TILE32X4                       0x08
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_RASTER256X1                    0x09
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_RASTER128X1                    0x0A
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_RASTER64X4                     0x0B
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_RASTER256X2                    0x0C
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_RASTER128X2                    0x0D
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_RASTER128X4                    0x0E
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_RASTER64X1                     0x0F
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_TILE16X8                       0x10
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_TILE8X16                       0x11
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_RASTER512X1                    0x12
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_RASTER32X4                     0x13
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_RASTER64X2                     0x14
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_RASTER32X2                     0x15
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_RASTER32X1                     0x16
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_RASTER16X1                     0x17
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_TILE128X4                      0x18
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_TILE256X4                      0x19
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_TILE512X4                      0x1A
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_TILE16X16                      0x1B
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_TILE32X16                      0x1C
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_TILE64X16                      0x1D
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_TILE128X8                      0x1E
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_TILE8X4_S                      0x1F
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_TILE16X4_S                     0x20
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_TILE32X4_S                     0x21
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_TILE16X4_LSB                   0x22
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_TILE32X4_LSB                   0x23
#define   GCREG_DEC400_READ_CONFIG_TILE_MODE_TILE32X8                       0x24

#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_TILE8X8_XMAJOR                0x00
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_TILE8X8_YMAJOR                0x01
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_TILE16X4                      0x02
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_TILE8X4                       0x03
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_TILE4X8                       0x04
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_RASTER16X4                    0x06
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_TILE64X4                      0x07
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_TILE32X4                      0x08
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_RASTER256X1                   0x09
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_RASTER128X1                   0x0A
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_RASTER64X4                    0x0B
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_RASTER256X2                   0x0C
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_RASTER128X2                   0x0D
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_RASTER128X4                   0x0E
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_RASTER64X1                    0x0F
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_TILE16X8                      0x10
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_TILE8X16                      0x11
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_RASTER512X1                   0x12
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_RASTER32X4                    0x13
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_RASTER64X2                    0x14
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_RASTER32X2                    0x15
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_RASTER32X1                    0x16
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_RASTER16X1                    0x17
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_TILE128X4                     0x18
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_TILE256X4                     0x19
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_TILE512X4                     0x1A
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_TILE16X16                     0x1B
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_TILE32X16                     0x1C
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_TILE64X16                     0x1D
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_TILE128X8                     0x1E
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_TILE8X4_S                     0x1F
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_TILE16X4_S                    0x20
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_TILE32X4_S                    0x21
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_TILE16X4_LSB                  0x22
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_TILE32X4_LSB                  0x23
#define   GCREG_DEC400_WRITE_CONFIG_TILE_MODE_TILE32X8                      0x24

#define GCREG_DEC400_READ_CONFIG_COMPRESSION_SIZE                            2:1
#define GCREG_DEC400_READ_CONFIG_COMPRESSION_ALIGN_MODE                    17:16
#define GCREG_DEC400_READ_EX_CONFIG_BIT_DEPTH                              18:16
#define GCREG_DEC400_READ_EX_CONFIG_Address                              0x18900
#define GCREG_DEC400_READ_BUFFER_BASE_Address                            0x18A80
#define GCREG_DEC400_READ_BUFFER_END_Address                             0x18B80
#define GCREG_DEC400_READ_CACHE_BASE_Address                             0x19080
#define GCREG_DEC400_READ_EX_CONFIG_BIT_DEPTH                              18:16
#define GCREG_DEC400_READ_EX_CONFIG_INTEL_P010                             28:28
#define GCREG_DEC400_WRITE_CONFIG_COMPRESSION_ENABLE                         0:0
#define GCREG_DEC400_WRITE_EX_CONFIG_BIT_DEPTH                             18:16
#define GCREG_DEC400_WRITE_EX_CONFIG_Address                             0x18A00
#define GCREG_DEC400_WRITE_CONFIG_COMPRESSION_SIZE                           2:1
#define GCREG_DEC400_WRITE_CONFIG_COMPRESSION_ALIGN_MODE                   17:16
#define GCREG_DEC400_WRITE_BUFFER_BASE_Address                           0x18D80
#define GCREG_DEC400_WRITE_BUFFER_END_Address                            0x18E80
#define GCREG_DEC400_WRITE_CACHE_BASE_Address                            0x19180
#define GCREG_DEC400_WRITE_CONFIG_Address                                0x18980
#define GCREG_DEC400_WRITE_EX_CONFIG_INTEL_P010                            28:28
#define GCREG_DEC400_CONTROL_Address                                     0x18800
#define GCREG_DEC400_CONTROL_FLUSH                                           0:0
#define GCREG_DEC400_CONTROL_ResetValue                               0x0201018A
#define GCREG_DEC400_CONTROL_CONFIGURE_MODE                                26:26
#define GCREG_DEC400_CONTROL_DISABLE_COMPRESSION                             1:1
#define GCREG_DEC400_CONTROL_DISABLE_HW_FLUSH                              16:16
#define GCREG_DEC400_READ_CONFIG_ResetValue                           0x00020000
#define GCREG_DEC400_READ_EX_CONFIG_ResetValue                        0x00000000
#define GCREG_DEC400_READ_BUFFER_BASE_ResetValue                      0xFFFFFFFF
#define GCREG_DEC400_READ_BUFFER_END_ResetValue                       0xFFFFFFFF
#define GCREG_DEC400_READ_CACHE_BASE_ResetValue                       0x00000000
#define GCREG_DEC400_WRITE_CONFIG_ResetValue                          0x00020000
#define GCREG_DEC400_WRITE_EX_CONFIG_ResetValue                       0x00000000
#define GCREG_DEC400_WRITE_BUFFER_BASE_ResetValue                     0xFFFFFFFF
#define GCREG_DEC400_WRITE_BUFFER_END_ResetValue                      0xFFFFFFFF
#define GCREG_DEC400_WRITE_CACHE_BASE_ResetValue                      0x00000000

#define   GCREG_DEC400_READ_CONFIG_COMPRESSION_ENABLE_DISABLE                0x0
#define   GCREG_DEC400_READ_CONFIG_COMPRESSION_ENABLE_ENABLE                 0x1
#define   GCREG_DEC400_READ_CONFIG_COMPRESSION_SIZE_SIZE64_BYTE              0x0
#define   GCREG_DEC400_READ_CONFIG_COMPRESSION_ALIGN_MODE_ALIGN32_BYTE       0x2
#define   GCREG_DEC400_READ_EX_CONFIG_BIT_DEPTH_BIT10                        0x1
#define   GCREG_DEC400_READ_EX_CONFIG_BIT_DEPTH_BIT8                         0x0
#define   GCREG_DEC400_READ_EX_CONFIG_BIT_DEPTH_BIT10                        0x1
#define   GCREG_DEC400_READ_EX_CONFIG_INTEL_P010_ENABLE                      0x1
#define   GCREG_DEC400_CONTROL_FLUSH_ENABLE                                  0x1
#define   GCREG_DEC400_WRITE_CONFIG_COMPRESSION_ENABLE_ENABLE                0x1
#define   GCREG_DEC400_WRITE_EX_CONFIG_BIT_DEPTH_BIT10                       0x1
#define   GCREG_DEC400_WRITE_EX_CONFIG_BIT_DEPTH_BIT8                        0x0
#define   GCREG_DEC400_WRITE_EX_CONFIG_BIT_DEPTH_BIT10                       0x1
#define   GCREG_DEC400_WRITE_EX_CONFIG_BIT_DEPTH_BIT12                       0x2
#define   GCREG_DEC400_WRITE_EX_CONFIG_BIT_DEPTH_BIT14                       0x3
#define   GCREG_DEC400_WRITE_EX_CONFIG_BIT_DEPTH_BIT16                       0x4
#define   GCREG_DEC400_WRITE_EX_CONFIG_INTEL_P010_ENABLE                     0x1
#define   GCREG_DEC400_WRITE_CONFIG_COMPRESSION_SIZE_SIZE64_BYTE             0x0
#define   GCREG_DEC400_WRITE_CONFIG_COMPRESSION_ALIGN_MODE_ALIGN32_BYTE      0x2
#define   GCREG_DEC400_CONTROL_CONFIGURE_MODE_LOAD_STATE                     0x1
#define   GCREG_DEC400_CONTROL_DISABLE_COMPRESSION_DISABLE                   0x1
#define   GCREG_DEC400_CONTROL_DISABLE_COMPRESSION_ENABLE                    0x0
#define   GCREG_DEC400_CONTROL_DISABLE_HW_FLUSH_DISABLE                      0x1

/*******************************************************************************
**                            ~~~~~~~~~~~~~~~~~~~~                            **
**                            Module DrawingEngine                            **
**                            ~~~~~~~~~~~~~~~~~~~~                            **
*******************************************************************************/

#define AQDESrcAddressRegAddrs                                            0x0480
#define AQDE_SRC_ADDRESS_Address                                         0x01200
#define AQDE_SRC_ADDRESS_MSB                                                  15
#define AQDE_SRC_ADDRESS_LSB                                                   0
#define AQDE_SRC_ADDRESS_BLK                                                   0
#define AQDE_SRC_ADDRESS_Count                                                 1
#define AQDE_SRC_ADDRESS_FieldMask                                    0xFFFFFFFF
#define AQDE_SRC_ADDRESS_ReadMask                                     0xFFFFFFFC
#define AQDE_SRC_ADDRESS_WriteMask                                    0xFFFFFFFC
#define AQDE_SRC_ADDRESS_ResetValue                                   0x00000000

#define AQDE_SRC_ADDRESS_TYPE                                              31:31
#define AQDE_SRC_ADDRESS_TYPE_End                                             31
#define AQDE_SRC_ADDRESS_TYPE_Start                                           31
#define AQDE_SRC_ADDRESS_TYPE_Type                                           U01
#define   AQDE_SRC_ADDRESS_TYPE_SYSTEM                                       0x0
#define   AQDE_SRC_ADDRESS_TYPE_VIRTUAL_SYSTEM                               0x1

#define AQDE_SRC_ADDRESS_ADDRESS                                            30:0
#define AQDE_SRC_ADDRESS_ADDRESS_End                                          30
#define AQDE_SRC_ADDRESS_ADDRESS_Start                                         0
#define AQDE_SRC_ADDRESS_ADDRESS_Type                                        U31

#define AQDESrcStrideRegAddrs                                             0x0481
#define AQDE_SRC_STRIDE_Address                                          0x01204
#define AQDE_SRC_STRIDE_MSB                                                   15
#define AQDE_SRC_STRIDE_LSB                                                    0
#define AQDE_SRC_STRIDE_BLK                                                    0
#define AQDE_SRC_STRIDE_Count                                                  1
#define AQDE_SRC_STRIDE_FieldMask                                     0x0003FFFF
#define AQDE_SRC_STRIDE_ReadMask                                      0x0003FFFC
#define AQDE_SRC_STRIDE_WriteMask                                     0x0003FFFC
#define AQDE_SRC_STRIDE_ResetValue                                    0x00000000

#define AQDE_SRC_STRIDE_STRIDE                                              17:0
#define AQDE_SRC_STRIDE_STRIDE_End                                            17
#define AQDE_SRC_STRIDE_STRIDE_Start                                           0
#define AQDE_SRC_STRIDE_STRIDE_Type                                          U18

#define AQDESrcRotationConfigRegAddrs                                     0x0482
#define AQDE_SRC_ROTATION_CONFIG_Address                                 0x01208
#define AQDE_SRC_ROTATION_CONFIG_MSB                                          15
#define AQDE_SRC_ROTATION_CONFIG_LSB                                           0
#define AQDE_SRC_ROTATION_CONFIG_BLK                                           0
#define AQDE_SRC_ROTATION_CONFIG_Count                                         1
#define AQDE_SRC_ROTATION_CONFIG_FieldMask                            0x0001FFFF
#define AQDE_SRC_ROTATION_CONFIG_ReadMask                             0x0001FFFF
#define AQDE_SRC_ROTATION_CONFIG_WriteMask                            0x0001FFFF
#define AQDE_SRC_ROTATION_CONFIG_ResetValue                           0x00000000

#define AQDE_SRC_ROTATION_CONFIG_WIDTH                                      15:0
#define AQDE_SRC_ROTATION_CONFIG_WIDTH_End                                    15
#define AQDE_SRC_ROTATION_CONFIG_WIDTH_Start                                   0
#define AQDE_SRC_ROTATION_CONFIG_WIDTH_Type                                  U16

#define AQDE_SRC_ROTATION_CONFIG_ROTATION                                  16:16
#define AQDE_SRC_ROTATION_CONFIG_ROTATION_End                                 16
#define AQDE_SRC_ROTATION_CONFIG_ROTATION_Start                               16
#define AQDE_SRC_ROTATION_CONFIG_ROTATION_Type                               U01
#define   AQDE_SRC_ROTATION_CONFIG_ROTATION_NORMAL                           0x0
#define   AQDE_SRC_ROTATION_CONFIG_ROTATION_ROTATED                          0x1

#define AQDESrcConfigRegAddrs                                             0x0483
#define AQDE_SRC_CONFIG_Address                                          0x0120C
#define AQDE_SRC_CONFIG_MSB                                                   15
#define AQDE_SRC_CONFIG_LSB                                                    0
#define AQDE_SRC_CONFIG_BLK                                                    0
#define AQDE_SRC_CONFIG_Count                                                  1
#define AQDE_SRC_CONFIG_FieldMask                                     0xFFFFF7FF
#define AQDE_SRC_CONFIG_ReadMask                                      0xFFFFF7FF
#define AQDE_SRC_CONFIG_WriteMask                                     0xFFFFF7FF
#define AQDE_SRC_CONFIG_ResetValue                                    0x00000000

#define AQDE_SRC_CONFIG_ENDIAN_CONTROL                                     31:30
#define AQDE_SRC_CONFIG_ENDIAN_CONTROL_End                                    31
#define AQDE_SRC_CONFIG_ENDIAN_CONTROL_Start                                  30
#define AQDE_SRC_CONFIG_ENDIAN_CONTROL_Type                                  U02
#define   AQDE_SRC_CONFIG_ENDIAN_CONTROL_NO_SWAP                             0x0
#define   AQDE_SRC_CONFIG_ENDIAN_CONTROL_SWAP_WORD                           0x1
#define   AQDE_SRC_CONFIG_ENDIAN_CONTROL_SWAP_DWORD                          0x2
#define   AQDE_SRC_CONFIG_ENDIAN_CONTROL_SWAP_DDWORD                         0x3

#define AQDE_SRC_CONFIG_DISABLE420_L2_CACHE                                29:29
#define AQDE_SRC_CONFIG_DISABLE420_L2_CACHE_End                               29
#define AQDE_SRC_CONFIG_DISABLE420_L2_CACHE_Start                             29
#define AQDE_SRC_CONFIG_DISABLE420_L2_CACHE_Type                             U01
#define   AQDE_SRC_CONFIG_DISABLE420_L2_CACHE_ENABLED                        0x0
#define   AQDE_SRC_CONFIG_DISABLE420_L2_CACHE_DISABLED                       0x1

#define AQDE_SRC_CONFIG_SOURCE_FORMAT                                      28:24
#define AQDE_SRC_CONFIG_SOURCE_FORMAT_End                                     28
#define AQDE_SRC_CONFIG_SOURCE_FORMAT_Start                                   24
#define AQDE_SRC_CONFIG_SOURCE_FORMAT_Type                                   U05
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_X4R4G4B4                            0x00
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_A4R4G4B4                            0x01
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_X1R5G5B5                            0x02
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_A1R5G5B5                            0x03
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_R5G6B5                              0x04
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_X8R8G8B8                            0x05
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_A8R8G8B8                            0x06
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_YUY2                                0x07
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_UYVY                                0x08
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_INDEX8                              0x09
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_MONOCHROME                          0x0A
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_AYUV                                0x0B
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_RGB888_PLANAR                       0x0C
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_RGB888_PACKED                       0x0D
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_EXTENSION                           0x0E
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_YV12                                0x0F
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_A8                                  0x10
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_NV12                                0x11
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_NV16                                0x12
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_RG16                                0x13
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_R8                                  0x14
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_NV12_10BIT                          0x15
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_A2R10G10B10                         0x16
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_NV16_10BIT                          0x17
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_INDEX1                              0x18
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_INDEX2                              0x19
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_INDEX4                              0x1A
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_P010                                0x1B
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_NV12_10BIT_L1                       0x1C
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_NV16_10BIT_L1                       0x1D
#define   AQDE_SRC_CONFIG_SOURCE_FORMAT_I010                                0x1E

#define AQDE_SRC_CONFIG_DE_GAMMA                                           23:23
#define AQDE_SRC_CONFIG_DE_GAMMA_End                                          23
#define AQDE_SRC_CONFIG_DE_GAMMA_Start                                        23
#define AQDE_SRC_CONFIG_DE_GAMMA_Type                                        U01
#define   AQDE_SRC_CONFIG_DE_GAMMA_DISABLED                                  0x0
#define   AQDE_SRC_CONFIG_DE_GAMMA_ENABLED                                   0x1

#define AQDE_SRC_CONFIG_P010_BITS                                          22:22
#define AQDE_SRC_CONFIG_P010_BITS_End                                         22
#define AQDE_SRC_CONFIG_P010_BITS_Start                                       22
#define AQDE_SRC_CONFIG_P010_BITS_Type                                       U01
#define   AQDE_SRC_CONFIG_P010_BITS_P010_MSB                                 0x0
#define   AQDE_SRC_CONFIG_P010_BITS_P010_LSB                                 0x1

#define AQDE_SRC_CONFIG_SWIZZLE                                            21:20
#define AQDE_SRC_CONFIG_SWIZZLE_End                                           21
#define AQDE_SRC_CONFIG_SWIZZLE_Start                                         20
#define AQDE_SRC_CONFIG_SWIZZLE_Type                                         U02
#define   AQDE_SRC_CONFIG_SWIZZLE_ARGB                                       0x0
#define   AQDE_SRC_CONFIG_SWIZZLE_RGBA                                       0x1
#define   AQDE_SRC_CONFIG_SWIZZLE_ABGR                                       0x2
#define   AQDE_SRC_CONFIG_SWIZZLE_BGRA                                       0x3

#define AQDE_SRC_CONFIG_DEC400_TILE_MODE                                   19:17
#define AQDE_SRC_CONFIG_DEC400_TILE_MODE_End                                  19
#define AQDE_SRC_CONFIG_DEC400_TILE_MODE_Start                                17
#define AQDE_SRC_CONFIG_DEC400_TILE_MODE_Type                                U03
#define   AQDE_SRC_CONFIG_DEC400_TILE_MODE_TILE8X8_XMAJOR                    0x0
#define   AQDE_SRC_CONFIG_DEC400_TILE_MODE_TILE8X4                           0x1
#define   AQDE_SRC_CONFIG_DEC400_TILE_MODE_TILE4X8                           0x2
#define   AQDE_SRC_CONFIG_DEC400_TILE_MODE_TILE64X4                          0x3
#define   AQDE_SRC_CONFIG_DEC400_TILE_MODE_TILE32X4                          0x4
#define   AQDE_SRC_CONFIG_DEC400_TILE_MODE_TILE4X4                           0x5

#define AQDE_SRC_CONFIG_COLOR_CONVERT                                      16:16
#define AQDE_SRC_CONFIG_COLOR_CONVERT_End                                     16
#define AQDE_SRC_CONFIG_COLOR_CONVERT_Start                                   16
#define AQDE_SRC_CONFIG_COLOR_CONVERT_Type                                   U01
#define   AQDE_SRC_CONFIG_COLOR_CONVERT_OFF                                  0x0
#define   AQDE_SRC_CONFIG_COLOR_CONVERT_ON                                   0x1

#define AQDE_SRC_CONFIG_MONO_TRANSPARENCY                                  15:15
#define AQDE_SRC_CONFIG_MONO_TRANSPARENCY_End                                 15
#define AQDE_SRC_CONFIG_MONO_TRANSPARENCY_Start                               15
#define AQDE_SRC_CONFIG_MONO_TRANSPARENCY_Type                               U01
#define   AQDE_SRC_CONFIG_MONO_TRANSPARENCY_BACKGROUND                       0x0
#define   AQDE_SRC_CONFIG_MONO_TRANSPARENCY_FOREGROUND                       0x1

#define AQDE_SRC_CONFIG_ENDIAN_CONTROL_EX                                  14:14
#define AQDE_SRC_CONFIG_ENDIAN_CONTROL_EX_End                                 14
#define AQDE_SRC_CONFIG_ENDIAN_CONTROL_EX_Start                               14
#define AQDE_SRC_CONFIG_ENDIAN_CONTROL_EX_Type                               U01
#define   AQDE_SRC_CONFIG_ENDIAN_CONTROL_EX_NO_SWAP                          0x0
#define   AQDE_SRC_CONFIG_ENDIAN_CONTROL_EX_SWAP_BYTE                        0x1

#define AQDE_SRC_CONFIG_PACK                                               13:12
#define AQDE_SRC_CONFIG_PACK_End                                              13
#define AQDE_SRC_CONFIG_PACK_Start                                            12
#define AQDE_SRC_CONFIG_PACK_Type                                            U02
#define   AQDE_SRC_CONFIG_PACK_PACKED8                                       0x0
#define   AQDE_SRC_CONFIG_PACK_PACKED16                                      0x1
#define   AQDE_SRC_CONFIG_PACK_PACKED32                                      0x2
#define   AQDE_SRC_CONFIG_PACK_UNPACKED                                      0x3

#define AQDE_SRC_CONFIG_TILE_MODE                                           10:9
#define AQDE_SRC_CONFIG_TILE_MODE_End                                         10
#define AQDE_SRC_CONFIG_TILE_MODE_Start                                        9
#define AQDE_SRC_CONFIG_TILE_MODE_Type                                       U02
#define   AQDE_SRC_CONFIG_TILE_MODE_TILED4X4                                 0x0
#define   AQDE_SRC_CONFIG_TILE_MODE_SUPER_TILED_XMAJOR                       0x1
#define   AQDE_SRC_CONFIG_TILE_MODE_SUPER_TILED_YMAJOR                       0x2
#define   AQDE_SRC_CONFIG_TILE_MODE_DEC400                                   0x3

#define AQDE_SRC_CONFIG_LOCATION                                             8:8
#define AQDE_SRC_CONFIG_LOCATION_End                                           8
#define AQDE_SRC_CONFIG_LOCATION_Start                                         8
#define AQDE_SRC_CONFIG_LOCATION_Type                                        U01
#define   AQDE_SRC_CONFIG_LOCATION_MEMORY                                    0x0
#define   AQDE_SRC_CONFIG_LOCATION_STREAM                                    0x1

#define AQDE_SRC_CONFIG_TILED                                                7:7
#define AQDE_SRC_CONFIG_TILED_End                                              7
#define AQDE_SRC_CONFIG_TILED_Start                                            7
#define AQDE_SRC_CONFIG_TILED_Type                                           U01
#define   AQDE_SRC_CONFIG_TILED_DISABLED                                     0x0
#define   AQDE_SRC_CONFIG_TILED_ENABLED                                      0x1

#define AQDE_SRC_CONFIG_SRC_RELATIVE                                         6:6
#define AQDE_SRC_CONFIG_SRC_RELATIVE_End                                       6
#define AQDE_SRC_CONFIG_SRC_RELATIVE_Start                                     6
#define AQDE_SRC_CONFIG_SRC_RELATIVE_Type                                    U01
#define   AQDE_SRC_CONFIG_SRC_RELATIVE_ABSOLUTE                              0x0
#define   AQDE_SRC_CONFIG_SRC_RELATIVE_RELATIVE                              0x1

#define AQDE_SRC_CONFIG_TRANSPARENCY                                         5:4
#define AQDE_SRC_CONFIG_TRANSPARENCY_End                                       5
#define AQDE_SRC_CONFIG_TRANSPARENCY_Start                                     4
#define AQDE_SRC_CONFIG_TRANSPARENCY_Type                                    U02
#define   AQDE_SRC_CONFIG_TRANSPARENCY_OPAQUE                                0x0
#define   AQDE_SRC_CONFIG_TRANSPARENCY_TRANSPARENT                           0x1
#define   AQDE_SRC_CONFIG_TRANSPARENCY_MASKED_MASK                           0x2
#define   AQDE_SRC_CONFIG_TRANSPARENCY_MASKED_PATTERN                        0x3

#define AQDE_SRC_CONFIG_FORMAT                                               3:0
#define AQDE_SRC_CONFIG_FORMAT_End                                             3
#define AQDE_SRC_CONFIG_FORMAT_Start                                           0
#define AQDE_SRC_CONFIG_FORMAT_Type                                          U04
#define   AQDE_SRC_CONFIG_FORMAT_X4R4G4B4                                    0x0
#define   AQDE_SRC_CONFIG_FORMAT_A4R4G4B4                                    0x1
#define   AQDE_SRC_CONFIG_FORMAT_X1R5G5B5                                    0x2
#define   AQDE_SRC_CONFIG_FORMAT_A1R5G5B5                                    0x3
#define   AQDE_SRC_CONFIG_FORMAT_R5G6B5                                      0x4
#define   AQDE_SRC_CONFIG_FORMAT_X8R8G8B8                                    0x5
#define   AQDE_SRC_CONFIG_FORMAT_A8R8G8B8                                    0x6
#define   AQDE_SRC_CONFIG_FORMAT_YUY2                                        0x7
#define   AQDE_SRC_CONFIG_FORMAT_UYVY                                        0x8
#define   AQDE_SRC_CONFIG_FORMAT_INDEX8                                      0x9
#define   AQDE_SRC_CONFIG_FORMAT_MONOCHROME                                  0xA
#define   AQDE_SRC_CONFIG_FORMAT_AYUV                                        0xB
#define   AQDE_SRC_CONFIG_FORMAT_RGB888_PLANAR                               0xC
#define   AQDE_SRC_CONFIG_FORMAT_RGB888_PACKED                               0xD
#define   AQDE_SRC_CONFIG_FORMAT_EXTENSION                                   0xE
#define   AQDE_SRC_CONFIG_FORMAT_YV12                                        0xF

#define AQDESrcOriginRegAddrs                                             0x0484
#define AQDE_SRC_ORIGIN_Address                                          0x01210
#define AQDE_SRC_ORIGIN_MSB                                                   15
#define AQDE_SRC_ORIGIN_LSB                                                    0
#define AQDE_SRC_ORIGIN_BLK                                                    0
#define AQDE_SRC_ORIGIN_Count                                                  1
#define AQDE_SRC_ORIGIN_FieldMask                                     0xFFFFFFFF
#define AQDE_SRC_ORIGIN_ReadMask                                      0xFFFFFFFF
#define AQDE_SRC_ORIGIN_WriteMask                                     0xFFFFFFFF
#define AQDE_SRC_ORIGIN_ResetValue                                    0x00000000

#define AQDE_SRC_ORIGIN_Y                                                  31:16
#define AQDE_SRC_ORIGIN_Y_End                                                 31
#define AQDE_SRC_ORIGIN_Y_Start                                               16
#define AQDE_SRC_ORIGIN_Y_Type                                               U16

#define AQDE_SRC_ORIGIN_X                                                   15:0
#define AQDE_SRC_ORIGIN_X_End                                                 15
#define AQDE_SRC_ORIGIN_X_Start                                                0
#define AQDE_SRC_ORIGIN_X_Type                                               U16

#define AQDESrcSizeRegAddrs                                               0x0485
#define AQDE_SRC_SIZE_Address                                            0x01214
#define AQDE_SRC_SIZE_MSB                                                     15
#define AQDE_SRC_SIZE_LSB                                                      0
#define AQDE_SRC_SIZE_BLK                                                      0
#define AQDE_SRC_SIZE_Count                                                    1
#define AQDE_SRC_SIZE_FieldMask                                       0xFFFFFFFF
#define AQDE_SRC_SIZE_ReadMask                                        0xFFFFFFFF
#define AQDE_SRC_SIZE_WriteMask                                       0xFFFFFFFF
#define AQDE_SRC_SIZE_ResetValue                                      0x00000000

#define AQDE_SRC_SIZE_Y                                                    31:16
#define AQDE_SRC_SIZE_Y_End                                                   31
#define AQDE_SRC_SIZE_Y_Start                                                 16
#define AQDE_SRC_SIZE_Y_Type                                                 U16

#define AQDE_SRC_SIZE_X                                                     15:0
#define AQDE_SRC_SIZE_X_End                                                   15
#define AQDE_SRC_SIZE_X_Start                                                  0
#define AQDE_SRC_SIZE_X_Type                                                 U16

#define AQDESrcColorBgRegAddrs                                            0x0486
#define AQDE_SRC_COLOR_BG_Address                                        0x01218
#define AQDE_SRC_COLOR_BG_MSB                                                 15
#define AQDE_SRC_COLOR_BG_LSB                                                  0
#define AQDE_SRC_COLOR_BG_BLK                                                  0
#define AQDE_SRC_COLOR_BG_Count                                                1
#define AQDE_SRC_COLOR_BG_FieldMask                                   0xFFFFFFFF
#define AQDE_SRC_COLOR_BG_ReadMask                                    0xFFFFFFFF
#define AQDE_SRC_COLOR_BG_WriteMask                                   0xFFFFFFFF
#define AQDE_SRC_COLOR_BG_ResetValue                                  0x00000000

#define AQDE_SRC_COLOR_BG_ALPHA                                            31:24
#define AQDE_SRC_COLOR_BG_ALPHA_End                                           31
#define AQDE_SRC_COLOR_BG_ALPHA_Start                                         24
#define AQDE_SRC_COLOR_BG_ALPHA_Type                                         U08

#define AQDE_SRC_COLOR_BG_RED                                              23:16
#define AQDE_SRC_COLOR_BG_RED_End                                             23
#define AQDE_SRC_COLOR_BG_RED_Start                                           16
#define AQDE_SRC_COLOR_BG_RED_Type                                           U08

#define AQDE_SRC_COLOR_BG_GREEN                                             15:8
#define AQDE_SRC_COLOR_BG_GREEN_End                                           15
#define AQDE_SRC_COLOR_BG_GREEN_Start                                          8
#define AQDE_SRC_COLOR_BG_GREEN_Type                                         U08

#define AQDE_SRC_COLOR_BG_BLUE                                               7:0
#define AQDE_SRC_COLOR_BG_BLUE_End                                             7
#define AQDE_SRC_COLOR_BG_BLUE_Start                                           0
#define AQDE_SRC_COLOR_BG_BLUE_Type                                          U08

#define AQDESrcColorFgRegAddrs                                            0x0487
#define AQDE_SRC_COLOR_FG_Address                                        0x0121C
#define AQDE_SRC_COLOR_FG_MSB                                                 15
#define AQDE_SRC_COLOR_FG_LSB                                                  0
#define AQDE_SRC_COLOR_FG_BLK                                                  0
#define AQDE_SRC_COLOR_FG_Count                                                1
#define AQDE_SRC_COLOR_FG_FieldMask                                   0xFFFFFFFF
#define AQDE_SRC_COLOR_FG_ReadMask                                    0xFFFFFFFF
#define AQDE_SRC_COLOR_FG_WriteMask                                   0xFFFFFFFF
#define AQDE_SRC_COLOR_FG_ResetValue                                  0x00000000

#define AQDE_SRC_COLOR_FG_ALPHA                                            31:24
#define AQDE_SRC_COLOR_FG_ALPHA_End                                           31
#define AQDE_SRC_COLOR_FG_ALPHA_Start                                         24
#define AQDE_SRC_COLOR_FG_ALPHA_Type                                         U08

#define AQDE_SRC_COLOR_FG_RED                                              23:16
#define AQDE_SRC_COLOR_FG_RED_End                                             23
#define AQDE_SRC_COLOR_FG_RED_Start                                           16
#define AQDE_SRC_COLOR_FG_RED_Type                                           U08

#define AQDE_SRC_COLOR_FG_GREEN                                             15:8
#define AQDE_SRC_COLOR_FG_GREEN_End                                           15
#define AQDE_SRC_COLOR_FG_GREEN_Start                                          8
#define AQDE_SRC_COLOR_FG_GREEN_Type                                         U08

#define AQDE_SRC_COLOR_FG_BLUE                                               7:0
#define AQDE_SRC_COLOR_FG_BLUE_End                                             7
#define AQDE_SRC_COLOR_FG_BLUE_Start                                           0
#define AQDE_SRC_COLOR_FG_BLUE_Type                                          U08

#define AQDEStretchFactorLowRegAddrs                                      0x0488
#define AQDE_STRETCH_FACTOR_LOW_Address                                  0x01220
#define AQDE_STRETCH_FACTOR_LOW_MSB                                           15
#define AQDE_STRETCH_FACTOR_LOW_LSB                                            0
#define AQDE_STRETCH_FACTOR_LOW_BLK                                            0
#define AQDE_STRETCH_FACTOR_LOW_Count                                          1
#define AQDE_STRETCH_FACTOR_LOW_FieldMask                             0x7FFFFFFF
#define AQDE_STRETCH_FACTOR_LOW_ReadMask                              0x7FFFFFFF
#define AQDE_STRETCH_FACTOR_LOW_WriteMask                             0x7FFFFFFF
#define AQDE_STRETCH_FACTOR_LOW_ResetValue                            0x00000000

#define AQDE_STRETCH_FACTOR_LOW_X                                           30:0
#define AQDE_STRETCH_FACTOR_LOW_X_End                                         30
#define AQDE_STRETCH_FACTOR_LOW_X_Start                                        0
#define AQDE_STRETCH_FACTOR_LOW_X_Type                                       U31

#define AQDEStretchFactorHighRegAddrs                                     0x0489
#define AQDE_STRETCH_FACTOR_HIGH_Address                                 0x01224
#define AQDE_STRETCH_FACTOR_HIGH_MSB                                          15
#define AQDE_STRETCH_FACTOR_HIGH_LSB                                           0
#define AQDE_STRETCH_FACTOR_LOW_HIGH_BLK                                       0
#define AQDE_STRETCH_FACTOR_HIGH_Count                                         1
#define AQDE_STRETCH_FACTOR_HIGH_FieldMask                            0x7FFFFFFF
#define AQDE_STRETCH_FACTOR_HIGH_ReadMask                             0x7FFFFFFF
#define AQDE_STRETCH_FACTOR_HIGH_WriteMask                            0x7FFFFFFF
#define AQDE_STRETCH_FACTOR_HIGH_ResetValue                           0x00000000

#define AQDE_STRETCH_FACTOR_HIGH_Y                                          30:0
#define AQDE_STRETCH_FACTOR_HIGH_Y_End                                        30
#define AQDE_STRETCH_FACTOR_HIGH_Y_Start                                       0
#define AQDE_STRETCH_FACTOR_HIGH_Y_Type                                      U31

#define AQDEDestAddressRegAddrs                                           0x048A
#define AQDE_DEST_ADDRESS_Address                                        0x01228
#define AQDE_DEST_ADDRESS_MSB                                                 15
#define AQDE_DEST_ADDRESS_LSB                                                  0
#define AQDE_DEST_ADDRESS_BLK                                                  0
#define AQDE_DEST_ADDRESS_Count                                                1
#define AQDE_DEST_ADDRESS_FieldMask                                   0xFFFFFFFF
#define AQDE_DEST_ADDRESS_ReadMask                                    0xFFFFFFFC
#define AQDE_DEST_ADDRESS_WriteMask                                   0xFFFFFFFC
#define AQDE_DEST_ADDRESS_ResetValue                                  0x00000000

#define AQDE_DEST_ADDRESS_TYPE                                             31:31
#define AQDE_DEST_ADDRESS_TYPE_End                                            31
#define AQDE_DEST_ADDRESS_TYPE_Start                                          31
#define AQDE_DEST_ADDRESS_TYPE_Type                                          U01
#define   AQDE_DEST_ADDRESS_TYPE_SYSTEM                                      0x0
#define   AQDE_DEST_ADDRESS_TYPE_VIRTUAL_SYSTEM                              0x1

#define AQDE_DEST_ADDRESS_ADDRESS                                           30:0
#define AQDE_DEST_ADDRESS_ADDRESS_End                                         30
#define AQDE_DEST_ADDRESS_ADDRESS_Start                                        0
#define AQDE_DEST_ADDRESS_ADDRESS_Type                                       U31

#define AQDEDestStrideRegAddrs                                            0x048B
#define AQDE_DEST_STRIDE_Address                                         0x0122C
#define AQDE_DEST_STRIDE_MSB                                                  15
#define AQDE_DEST_STRIDE_LSB                                                   0
#define AQDE_DEST_STRIDE_BLK                                                   0
#define AQDE_DEST_STRIDE_Count                                                 1
#define AQDE_DEST_STRIDE_FieldMask                                    0x0003FFFF
#define AQDE_DEST_STRIDE_ReadMask                                     0x0003FFFC
#define AQDE_DEST_STRIDE_WriteMask                                    0x0003FFFC
#define AQDE_DEST_STRIDE_ResetValue                                   0x00000000

#define AQDE_DEST_STRIDE_STRIDE                                             17:0
#define AQDE_DEST_STRIDE_STRIDE_End                                           17
#define AQDE_DEST_STRIDE_STRIDE_Start                                          0
#define AQDE_DEST_STRIDE_STRIDE_Type                                         U18

#define AQDEDestRotationConfigRegAddrs                                    0x048C
#define AQDE_DEST_ROTATION_CONFIG_Address                                0x01230
#define AQDE_DEST_ROTATION_CONFIG_MSB                                         15
#define AQDE_DEST_ROTATION_CONFIG_LSB                                          0
#define AQDE_DEST_ROTATION_CONFIG_BLK                                          0
#define AQDE_DEST_ROTATION_CONFIG_Count                                        1
#define AQDE_DEST_ROTATION_CONFIG_FieldMask                           0x0001FFFF
#define AQDE_DEST_ROTATION_CONFIG_ReadMask                            0x0001FFFF
#define AQDE_DEST_ROTATION_CONFIG_WriteMask                           0x0001FFFF
#define AQDE_DEST_ROTATION_CONFIG_ResetValue                          0x00000000

#define AQDE_DEST_ROTATION_CONFIG_WIDTH                                     15:0
#define AQDE_DEST_ROTATION_CONFIG_WIDTH_End                                   15
#define AQDE_DEST_ROTATION_CONFIG_WIDTH_Start                                  0
#define AQDE_DEST_ROTATION_CONFIG_WIDTH_Type                                 U16

#define AQDE_DEST_ROTATION_CONFIG_ROTATION                                 16:16
#define AQDE_DEST_ROTATION_CONFIG_ROTATION_End                                16
#define AQDE_DEST_ROTATION_CONFIG_ROTATION_Start                              16
#define AQDE_DEST_ROTATION_CONFIG_ROTATION_Type                              U01
#define   AQDE_DEST_ROTATION_CONFIG_ROTATION_NORMAL                          0x0
#define   AQDE_DEST_ROTATION_CONFIG_ROTATION_ROTATED                         0x1

#define AQDEDestConfigRegAddrs                                            0x048D
#define AQDE_DEST_CONFIG_Address                                         0x01234
#define AQDE_DEST_CONFIG_MSB                                                  15
#define AQDE_DEST_CONFIG_LSB                                                   0
#define AQDE_DEST_CONFIG_BLK                                                   0
#define AQDE_DEST_CONFIG_Count                                                 1
#define AQDE_DEST_CONFIG_FieldMask                                    0xFF7BFFFF
#define AQDE_DEST_CONFIG_ReadMask                                     0xFF7BFFFF
#define AQDE_DEST_CONFIG_WriteMask                                    0xFF7BFFFF
#define AQDE_DEST_CONFIG_ResetValue                                   0x00000000

#define AQDE_DEST_CONFIG_MONO_LINE_V2_DISABLED                             31:31
#define AQDE_DEST_CONFIG_MONO_LINE_V2_DISABLED_End                            31
#define AQDE_DEST_CONFIG_MONO_LINE_V2_DISABLED_Start                          31
#define AQDE_DEST_CONFIG_MONO_LINE_V2_DISABLED_Type                          U01
#define   AQDE_DEST_CONFIG_MONO_LINE_V2_DISABLED_DISABLED                    0x0
#define   AQDE_DEST_CONFIG_MONO_LINE_V2_DISABLED_ENABLED                     0x1

#define AQDE_DEST_CONFIG_HIGH_PRECISION_DISABLED                           30:30
#define AQDE_DEST_CONFIG_HIGH_PRECISION_DISABLED_End                          30
#define AQDE_DEST_CONFIG_HIGH_PRECISION_DISABLED_Start                        30
#define AQDE_DEST_CONFIG_HIGH_PRECISION_DISABLED_Type                        U01
#define   AQDE_DEST_CONFIG_HIGH_PRECISION_DISABLED_DISABLED                  0x1
#define   AQDE_DEST_CONFIG_HIGH_PRECISION_DISABLED_ENABLED                   0x0

#define AQDE_DEST_CONFIG_WALKER_V2                                         29:29
#define AQDE_DEST_CONFIG_WALKER_V2_End                                        29
#define AQDE_DEST_CONFIG_WALKER_V2_Start                                      29
#define AQDE_DEST_CONFIG_WALKER_V2_Type                                      U01
#define   AQDE_DEST_CONFIG_WALKER_V2_DISABLED                                0x0
#define   AQDE_DEST_CONFIG_WALKER_V2_ENABLED                                 0x1

#define AQDE_DEST_CONFIG_ALL_QUAD                                          28:28
#define AQDE_DEST_CONFIG_ALL_QUAD_End                                         28
#define AQDE_DEST_CONFIG_ALL_QUAD_Start                                       28
#define AQDE_DEST_CONFIG_ALL_QUAD_Type                                       U01
#define   AQDE_DEST_CONFIG_ALL_QUAD_DISABLED                                 0x0
#define   AQDE_DEST_CONFIG_ALL_QUAD_ENABLED                                  0x1

#define AQDE_DEST_CONFIG_STRETCH_QUAD                                      27:27
#define AQDE_DEST_CONFIG_STRETCH_QUAD_End                                     27
#define AQDE_DEST_CONFIG_STRETCH_QUAD_Start                                   27
#define AQDE_DEST_CONFIG_STRETCH_QUAD_Type                                   U01
#define   AQDE_DEST_CONFIG_STRETCH_QUAD_ENABLED                              0x0
#define   AQDE_DEST_CONFIG_STRETCH_QUAD_DISABLED                             0x1

#define AQDE_DEST_CONFIG_MINOR_TILED                                       26:26
#define AQDE_DEST_CONFIG_MINOR_TILED_End                                      26
#define AQDE_DEST_CONFIG_MINOR_TILED_Start                                    26
#define AQDE_DEST_CONFIG_MINOR_TILED_Type                                    U01
#define   AQDE_DEST_CONFIG_MINOR_TILED_DISABLED                              0x0
#define   AQDE_DEST_CONFIG_MINOR_TILED_ENABLED                               0x1

#define AQDE_DEST_CONFIG_INTER_TILE_PER_FIX                                25:25
#define AQDE_DEST_CONFIG_INTER_TILE_PER_FIX_End                               25
#define AQDE_DEST_CONFIG_INTER_TILE_PER_FIX_Start                             25
#define AQDE_DEST_CONFIG_INTER_TILE_PER_FIX_Type                             U01
#define   AQDE_DEST_CONFIG_INTER_TILE_PER_FIX_DISABLED                       0x1
#define   AQDE_DEST_CONFIG_INTER_TILE_PER_FIX_ENABLED                        0x0

#define AQDE_DEST_CONFIG_GDI_STRE                                          24:24
#define AQDE_DEST_CONFIG_GDI_STRE_End                                         24
#define AQDE_DEST_CONFIG_GDI_STRE_Start                                       24
#define AQDE_DEST_CONFIG_GDI_STRE_Type                                       U01
#define   AQDE_DEST_CONFIG_GDI_STRE_DISABLED                                 0x0
#define   AQDE_DEST_CONFIG_GDI_STRE_ENABLED                                  0x1

#define AQDE_DEST_CONFIG_P010_BITS                                         22:22
#define AQDE_DEST_CONFIG_P010_BITS_End                                        22
#define AQDE_DEST_CONFIG_P010_BITS_Start                                      22
#define AQDE_DEST_CONFIG_P010_BITS_Type                                      U01
#define   AQDE_DEST_CONFIG_P010_BITS_P010_MSB                                0x0
#define   AQDE_DEST_CONFIG_P010_BITS_P010_LSB                                0x1

#define AQDE_DEST_CONFIG_ENDIAN_CONTROL                                    21:20
#define AQDE_DEST_CONFIG_ENDIAN_CONTROL_End                                   21
#define AQDE_DEST_CONFIG_ENDIAN_CONTROL_Start                                 20
#define AQDE_DEST_CONFIG_ENDIAN_CONTROL_Type                                 U02
#define   AQDE_DEST_CONFIG_ENDIAN_CONTROL_NO_SWAP                            0x0
#define   AQDE_DEST_CONFIG_ENDIAN_CONTROL_SWAP_WORD                          0x1
#define   AQDE_DEST_CONFIG_ENDIAN_CONTROL_SWAP_DWORD                         0x2
#define   AQDE_DEST_CONFIG_ENDIAN_CONTROL_SWAP_DDWORD                        0x3

#define AQDE_DEST_CONFIG_ENDIAN_CONTROL_EX                                 19:19
#define AQDE_DEST_CONFIG_ENDIAN_CONTROL_EX_End                                19
#define AQDE_DEST_CONFIG_ENDIAN_CONTROL_EX_Start                              19
#define AQDE_DEST_CONFIG_ENDIAN_CONTROL_EX_Type                              U01
#define   AQDE_DEST_CONFIG_ENDIAN_CONTROL_EX_NO_SWAP                         0x0
#define   AQDE_DEST_CONFIG_ENDIAN_CONTROL_EX_SWAP_BYTE                       0x1

#define AQDE_DEST_CONFIG_SWIZZLE                                           17:16
#define AQDE_DEST_CONFIG_SWIZZLE_End                                          17
#define AQDE_DEST_CONFIG_SWIZZLE_Start                                        16
#define AQDE_DEST_CONFIG_SWIZZLE_Type                                        U02
#define   AQDE_DEST_CONFIG_SWIZZLE_ARGB                                      0x0
#define   AQDE_DEST_CONFIG_SWIZZLE_RGBA                                      0x1
#define   AQDE_DEST_CONFIG_SWIZZLE_ABGR                                      0x2
#define   AQDE_DEST_CONFIG_SWIZZLE_BGRA                                      0x3

#define AQDE_DEST_CONFIG_COMMAND                                           15:12
#define AQDE_DEST_CONFIG_COMMAND_End                                          15
#define AQDE_DEST_CONFIG_COMMAND_Start                                        12
#define AQDE_DEST_CONFIG_COMMAND_Type                                        U04
#define   AQDE_DEST_CONFIG_COMMAND_CLEAR                                     0x0
#define   AQDE_DEST_CONFIG_COMMAND_LINE                                      0x1
#define   AQDE_DEST_CONFIG_COMMAND_BIT_BLT                                   0x2
#define   AQDE_DEST_CONFIG_COMMAND_BIT_BLT_REVERSED                          0x3
#define   AQDE_DEST_CONFIG_COMMAND_STRETCH_BLT                               0x4
#define   AQDE_DEST_CONFIG_COMMAND_HOR_FILTER_BLT                            0x5
#define   AQDE_DEST_CONFIG_COMMAND_VER_FILTER_BLT                            0x6
#define   AQDE_DEST_CONFIG_COMMAND_ONE_PASS_FILTER_BLT                       0x7
#define   AQDE_DEST_CONFIG_COMMAND_MULTI_SOURCE_BLT                          0x8
#define   AQDE_DEST_CONFIG_COMMAND_INVALID_COMMAND                           0xF

#define AQDE_DEST_CONFIG_GAMMA                                             11:11
#define AQDE_DEST_CONFIG_GAMMA_End                                            11
#define AQDE_DEST_CONFIG_GAMMA_Start                                          11
#define AQDE_DEST_CONFIG_GAMMA_Type                                          U01
#define   AQDE_DEST_CONFIG_GAMMA_DISABLED                                    0x0
#define   AQDE_DEST_CONFIG_GAMMA_ENABLED                                     0x1

#define AQDE_DEST_CONFIG_TILE_MODE                                          10:9
#define AQDE_DEST_CONFIG_TILE_MODE_End                                        10
#define AQDE_DEST_CONFIG_TILE_MODE_Start                                       9
#define AQDE_DEST_CONFIG_TILE_MODE_Type                                      U02
#define   AQDE_DEST_CONFIG_TILE_MODE_TILED4X4                                0x0
#define   AQDE_DEST_CONFIG_TILE_MODE_SUPER_TILED_XMAJOR                      0x1
#define   AQDE_DEST_CONFIG_TILE_MODE_SUPER_TILED_YMAJOR                      0x2
#define   AQDE_DEST_CONFIG_TILE_MODE_DEC400                                  0x3

#define AQDE_DEST_CONFIG_TILED                                               8:8
#define AQDE_DEST_CONFIG_TILED_End                                             8
#define AQDE_DEST_CONFIG_TILED_Start                                           8
#define AQDE_DEST_CONFIG_TILED_Type                                          U01
#define   AQDE_DEST_CONFIG_TILED_DISABLED                                    0x0
#define   AQDE_DEST_CONFIG_TILED_ENABLED                                     0x1

#define AQDE_DEST_CONFIG_DEC400_TILE_MODE                                    7:5
#define AQDE_DEST_CONFIG_DEC400_TILE_MODE_End                                  7
#define AQDE_DEST_CONFIG_DEC400_TILE_MODE_Start                                5
#define AQDE_DEST_CONFIG_DEC400_TILE_MODE_Type                               U03
#define   AQDE_DEST_CONFIG_DEC400_TILE_MODE_TILE8X8_XMAJOR                   0x0
#define   AQDE_DEST_CONFIG_DEC400_TILE_MODE_TILE8X4                          0x1
#define   AQDE_DEST_CONFIG_DEC400_TILE_MODE_TILE4X8                          0x2
#define   AQDE_DEST_CONFIG_DEC400_TILE_MODE_TILE64X4                         0x3
#define   AQDE_DEST_CONFIG_DEC400_TILE_MODE_TILE32X4                         0x4
#define   AQDE_DEST_CONFIG_DEC400_TILE_MODE_TILE4X4                          0x5

#define AQDE_DEST_CONFIG_FORMAT                                              4:0
#define AQDE_DEST_CONFIG_FORMAT_End                                            4
#define AQDE_DEST_CONFIG_FORMAT_Start                                          0
#define AQDE_DEST_CONFIG_FORMAT_Type                                         U05
#define   AQDE_DEST_CONFIG_FORMAT_X4R4G4B4                                  0x00
#define   AQDE_DEST_CONFIG_FORMAT_A4R4G4B4                                  0x01
#define   AQDE_DEST_CONFIG_FORMAT_X1R5G5B5                                  0x02
#define   AQDE_DEST_CONFIG_FORMAT_A1R5G5B5                                  0x03
#define   AQDE_DEST_CONFIG_FORMAT_R5G6B5                                    0x04
#define   AQDE_DEST_CONFIG_FORMAT_X8R8G8B8                                  0x05
#define   AQDE_DEST_CONFIG_FORMAT_A8R8G8B8                                  0x06
#define   AQDE_DEST_CONFIG_FORMAT_YUY2                                      0x07
#define   AQDE_DEST_CONFIG_FORMAT_UYVY                                      0x08
#define   AQDE_DEST_CONFIG_FORMAT_INDEX8                                    0x09
#define   AQDE_DEST_CONFIG_FORMAT_MONOCHROME                                0x0A
#define   AQDE_DEST_CONFIG_FORMAT_AYUV                                      0x0B
#define   AQDE_DEST_CONFIG_FORMAT_RGB888_PLANAR                             0x0C
#define   AQDE_DEST_CONFIG_FORMAT_RGB888_PACKED                             0x0D
#define   AQDE_DEST_CONFIG_FORMAT_EXTENSION                                 0x0E
#define   AQDE_DEST_CONFIG_FORMAT_YV12                                      0x0F
#define   AQDE_DEST_CONFIG_FORMAT_A8                                        0x10
#define   AQDE_DEST_CONFIG_FORMAT_NV12                                      0x11
#define   AQDE_DEST_CONFIG_FORMAT_NV16                                      0x12
#define   AQDE_DEST_CONFIG_FORMAT_RG16                                      0x13
#define   AQDE_DEST_CONFIG_FORMAT_R8                                        0x14
#define   AQDE_DEST_CONFIG_FORMAT_NV12_10BIT                                0x15
#define   AQDE_DEST_CONFIG_FORMAT_A2R10G10B10                               0x16
#define   AQDE_DEST_CONFIG_FORMAT_NV16_10BIT                                0x17
#define   AQDE_DEST_CONFIG_FORMAT_INDEX1                                    0x18
#define   AQDE_DEST_CONFIG_FORMAT_INDEX2                                    0x19
#define   AQDE_DEST_CONFIG_FORMAT_INDEX4                                    0x1A
#define   AQDE_DEST_CONFIG_FORMAT_P010                                      0x1B
#define   AQDE_DEST_CONFIG_FORMAT_NV12_10BIT_L1                             0x1C
#define   AQDE_DEST_CONFIG_FORMAT_NV16_10BIT_L1                             0x1D
#define   AQDE_DEST_CONFIG_FORMAT_I010                                      0x1E

#define AQDEPatternAddressRegAddrs                                        0x048E
#define AQDE_PATTERN_ADDRESS_Address                                     0x01238
#define AQDE_PATTERN_ADDRESS_MSB                                              15
#define AQDE_PATTERN_ADDRESS_LSB                                               0
#define AQDE_PATTERN_ADDRESS_BLK                                               0
#define AQDE_PATTERN_ADDRESS_Count                                             1
#define AQDE_PATTERN_ADDRESS_FieldMask                                0xFFFFFFFF
#define AQDE_PATTERN_ADDRESS_ReadMask                                 0xFFFFFFFC
#define AQDE_PATTERN_ADDRESS_WriteMask                                0xFFFFFFFC
#define AQDE_PATTERN_ADDRESS_ResetValue                               0x00000000

#define AQDE_PATTERN_ADDRESS_TYPE                                          31:31
#define AQDE_PATTERN_ADDRESS_TYPE_End                                         31
#define AQDE_PATTERN_ADDRESS_TYPE_Start                                       31
#define AQDE_PATTERN_ADDRESS_TYPE_Type                                       U01
#define   AQDE_PATTERN_ADDRESS_TYPE_SYSTEM                                   0x0
#define   AQDE_PATTERN_ADDRESS_TYPE_VIRTUAL_SYSTEM                           0x1

#define AQDE_PATTERN_ADDRESS_ADDRESS                                        30:0
#define AQDE_PATTERN_ADDRESS_ADDRESS_End                                      30
#define AQDE_PATTERN_ADDRESS_ADDRESS_Start                                     0
#define AQDE_PATTERN_ADDRESS_ADDRESS_Type                                    U31

#define AQDEPatternConfigRegAddrs                                         0x048F
#define AQDE_PATTERN_CONFIG_Address                                      0x0123C
#define AQDE_PATTERN_CONFIG_MSB                                               15
#define AQDE_PATTERN_CONFIG_LSB                                                0
#define AQDE_PATTERN_CONFIG_BLK                                                0
#define AQDE_PATTERN_CONFIG_Count                                              1
#define AQDE_PATTERN_CONFIG_FieldMask                                 0xFF7703FF
#define AQDE_PATTERN_CONFIG_ReadMask                                  0xFF7703FF
#define AQDE_PATTERN_CONFIG_WriteMask                                 0xFF7703FF
#define AQDE_PATTERN_CONFIG_ResetValue                                0x00000000

#define AQDE_PATTERN_CONFIG_ENDIAN_CONTROL                                 31:30
#define AQDE_PATTERN_CONFIG_ENDIAN_CONTROL_End                                31
#define AQDE_PATTERN_CONFIG_ENDIAN_CONTROL_Start                              30
#define AQDE_PATTERN_CONFIG_ENDIAN_CONTROL_Type                              U02
#define   AQDE_PATTERN_CONFIG_ENDIAN_CONTROL_NO_SWAP                         0x0
#define   AQDE_PATTERN_CONFIG_ENDIAN_CONTROL_SWAP_WORD                       0x1
#define   AQDE_PATTERN_CONFIG_ENDIAN_CONTROL_SWAP_DWORD                      0x2
#define   AQDE_PATTERN_CONFIG_ENDIAN_CONTROL_SWAP_DDWORD                     0x3

#define AQDE_PATTERN_CONFIG_ENDIAN_CONTROL_EX                              29:29
#define AQDE_PATTERN_CONFIG_ENDIAN_CONTROL_EX_End                             29
#define AQDE_PATTERN_CONFIG_ENDIAN_CONTROL_EX_Start                           29
#define AQDE_PATTERN_CONFIG_ENDIAN_CONTROL_EX_Type                           U01
#define   AQDE_PATTERN_CONFIG_ENDIAN_CONTROL_EX_NO_SWAP                      0x0
#define   AQDE_PATTERN_CONFIG_ENDIAN_CONTROL_EX_SWAP_BYTE                    0x1

#define AQDE_PATTERN_CONFIG_PATTERN_FORMAT                                 28:24
#define AQDE_PATTERN_CONFIG_PATTERN_FORMAT_End                                28
#define AQDE_PATTERN_CONFIG_PATTERN_FORMAT_Start                              24
#define AQDE_PATTERN_CONFIG_PATTERN_FORMAT_Type                              U05
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_X4R4G4B4                       0x00
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_A4R4G4B4                       0x01
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_X1R5G5B5                       0x02
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_A1R5G5B5                       0x03
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_R5G6B5                         0x04
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_X8R8G8B8                       0x05
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_A8R8G8B8                       0x06
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_YUY2                           0x07
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_UYVY                           0x08
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_INDEX8                         0x09
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_MONOCHROME                     0x0A
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_AYUV                           0x0B
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_RGB888_PLANAR                  0x0C
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_RGB888_PACKED                  0x0D
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_EXTENSION                      0x0E
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_YV12                           0x0F
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_A8                             0x10
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_NV12                           0x11
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_NV16                           0x12
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_RG16                           0x13
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_R8                             0x14
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_NV12_10BIT                     0x15
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_A2R10G10B10                    0x16
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_NV16_10BIT                     0x17
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_INDEX1                         0x18
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_INDEX2                         0x19
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_INDEX4                         0x1A
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_P010                           0x1B
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_NV12_10BIT_L1                  0x1C
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_NV16_10BIT_L1                  0x1D
#define   AQDE_PATTERN_CONFIG_PATTERN_FORMAT_I010                           0x1E

#define AQDE_PATTERN_CONFIG_ORIGIN_Y                                       22:20
#define AQDE_PATTERN_CONFIG_ORIGIN_Y_End                                      22
#define AQDE_PATTERN_CONFIG_ORIGIN_Y_Start                                    20
#define AQDE_PATTERN_CONFIG_ORIGIN_Y_Type                                    U03

#define AQDE_PATTERN_CONFIG_ORIGIN_X                                       18:16
#define AQDE_PATTERN_CONFIG_ORIGIN_X_End                                      18
#define AQDE_PATTERN_CONFIG_ORIGIN_X_Start                                    16
#define AQDE_PATTERN_CONFIG_ORIGIN_X_Type                                    U03

#define AQDE_PATTERN_CONFIG_SWIZZLE                                          9:8
#define AQDE_PATTERN_CONFIG_SWIZZLE_End                                        9
#define AQDE_PATTERN_CONFIG_SWIZZLE_Start                                      8
#define AQDE_PATTERN_CONFIG_SWIZZLE_Type                                     U02
#define   AQDE_PATTERN_CONFIG_SWIZZLE_ARGB                                   0x0
#define   AQDE_PATTERN_CONFIG_SWIZZLE_RGBA                                   0x1
#define   AQDE_PATTERN_CONFIG_SWIZZLE_ABGR                                   0x2
#define   AQDE_PATTERN_CONFIG_SWIZZLE_BGRA                                   0x3

#define AQDE_PATTERN_CONFIG_INIT_TRIGGER                                     7:6
#define AQDE_PATTERN_CONFIG_INIT_TRIGGER_End                                   7
#define AQDE_PATTERN_CONFIG_INIT_TRIGGER_Start                                 6
#define AQDE_PATTERN_CONFIG_INIT_TRIGGER_Type                                U02
#define   AQDE_PATTERN_CONFIG_INIT_TRIGGER_DEFAULT                           0x0
#define   AQDE_PATTERN_CONFIG_INIT_TRIGGER_INIT_CONFIG                       0x1
#define   AQDE_PATTERN_CONFIG_INIT_TRIGGER_INIT_PATTERN                      0x2
#define   AQDE_PATTERN_CONFIG_INIT_TRIGGER_INIT_ALL                          0x3

#define AQDE_PATTERN_CONFIG_COLOR_CONVERT                                    5:5
#define AQDE_PATTERN_CONFIG_COLOR_CONVERT_End                                  5
#define AQDE_PATTERN_CONFIG_COLOR_CONVERT_Start                                5
#define AQDE_PATTERN_CONFIG_COLOR_CONVERT_Type                               U01
#define   AQDE_PATTERN_CONFIG_COLOR_CONVERT_OFF                              0x0
#define   AQDE_PATTERN_CONFIG_COLOR_CONVERT_ON                               0x1

#define AQDE_PATTERN_CONFIG_TYPE                                             4:4
#define AQDE_PATTERN_CONFIG_TYPE_End                                           4
#define AQDE_PATTERN_CONFIG_TYPE_Start                                         4
#define AQDE_PATTERN_CONFIG_TYPE_Type                                        U01
#define   AQDE_PATTERN_CONFIG_TYPE_SOLID_COLOR                               0x0
#define   AQDE_PATTERN_CONFIG_TYPE_PATTERN                                   0x1

#define AQDE_PATTERN_CONFIG_FORMAT                                           3:0
#define AQDE_PATTERN_CONFIG_FORMAT_End                                         3
#define AQDE_PATTERN_CONFIG_FORMAT_Start                                       0
#define AQDE_PATTERN_CONFIG_FORMAT_Type                                      U04
#define   AQDE_PATTERN_CONFIG_FORMAT_X4R4G4B4                                0x0
#define   AQDE_PATTERN_CONFIG_FORMAT_A4R4G4B4                                0x1
#define   AQDE_PATTERN_CONFIG_FORMAT_X1R5G5B5                                0x2
#define   AQDE_PATTERN_CONFIG_FORMAT_A1R5G5B5                                0x3
#define   AQDE_PATTERN_CONFIG_FORMAT_R5G6B5                                  0x4
#define   AQDE_PATTERN_CONFIG_FORMAT_X8R8G8B8                                0x5
#define   AQDE_PATTERN_CONFIG_FORMAT_A8R8G8B8                                0x6
#define   AQDE_PATTERN_CONFIG_FORMAT_YUY2                                    0x7
#define   AQDE_PATTERN_CONFIG_FORMAT_UYVY                                    0x8
#define   AQDE_PATTERN_CONFIG_FORMAT_INDEX8                                  0x9
#define   AQDE_PATTERN_CONFIG_FORMAT_MONOCHROME                              0xA
#define   AQDE_PATTERN_CONFIG_FORMAT_AYUV                                    0xB
#define   AQDE_PATTERN_CONFIG_FORMAT_RGB888_PLANAR                           0xC
#define   AQDE_PATTERN_CONFIG_FORMAT_RGB888_PACKED                           0xD
#define   AQDE_PATTERN_CONFIG_FORMAT_EXTENSION                               0xE
#define   AQDE_PATTERN_CONFIG_FORMAT_YV12                                    0xF

#define AQDEPatternLowRegAddrs                                            0x0490
#define AQDE_PATTERN_LOW_Address                                         0x01240
#define AQDE_PATTERN_LOW_MSB                                                  15
#define AQDE_PATTERN_LOW_LSB                                                   0
#define AQDE_PATTERN_LOW_BLK                                                   0
#define AQDE_PATTERN_LOW_Count                                                 1
#define AQDE_PATTERN_LOW_FieldMask                                    0xFFFFFFFF
#define AQDE_PATTERN_LOW_ReadMask                                     0xFFFFFFFF
#define AQDE_PATTERN_LOW_WriteMask                                    0xFFFFFFFF
#define AQDE_PATTERN_LOW_ResetValue                                   0x00000000

#define AQDE_PATTERN_LOW_ROW0                                                7:0
#define AQDE_PATTERN_LOW_ROW0_End                                              7
#define AQDE_PATTERN_LOW_ROW0_Start                                            0
#define AQDE_PATTERN_LOW_ROW0_Type                                           U08

#define AQDE_PATTERN_LOW_ROW1                                               15:8
#define AQDE_PATTERN_LOW_ROW1_End                                             15
#define AQDE_PATTERN_LOW_ROW1_Start                                            8
#define AQDE_PATTERN_LOW_ROW1_Type                                           U08

#define AQDE_PATTERN_LOW_ROW2                                              23:16
#define AQDE_PATTERN_LOW_ROW2_End                                             23
#define AQDE_PATTERN_LOW_ROW2_Start                                           16
#define AQDE_PATTERN_LOW_ROW2_Type                                           U08

#define AQDE_PATTERN_LOW_ROW3                                              31:24
#define AQDE_PATTERN_LOW_ROW3_End                                             31
#define AQDE_PATTERN_LOW_ROW3_Start                                           24
#define AQDE_PATTERN_LOW_ROW3_Type                                           U08

#define AQDEPatternHighRegAddrs                                           0x0491
#define AQDE_PATTERN_HIGH_Address                                        0x01244
#define AQDE_PATTERN_HIGH_MSB                                                 15
#define AQDE_PATTERN_HIGH_LSB                                                  0
#define AQDE_PATTERN_LOW_HIGH_BLK                                              0
#define AQDE_PATTERN_HIGH_Count                                                1
#define AQDE_PATTERN_HIGH_FieldMask                                   0xFFFFFFFF
#define AQDE_PATTERN_HIGH_ReadMask                                    0xFFFFFFFF
#define AQDE_PATTERN_HIGH_WriteMask                                   0xFFFFFFFF
#define AQDE_PATTERN_HIGH_ResetValue                                  0x00000000

#define AQDE_PATTERN_HIGH_ROW4                                               7:0
#define AQDE_PATTERN_HIGH_ROW4_End                                             7
#define AQDE_PATTERN_HIGH_ROW4_Start                                           0
#define AQDE_PATTERN_HIGH_ROW4_Type                                          U08

#define AQDE_PATTERN_HIGH_ROW5                                              15:8
#define AQDE_PATTERN_HIGH_ROW5_End                                            15
#define AQDE_PATTERN_HIGH_ROW5_Start                                           8
#define AQDE_PATTERN_HIGH_ROW5_Type                                          U08

#define AQDE_PATTERN_HIGH_ROW6                                             23:16
#define AQDE_PATTERN_HIGH_ROW6_End                                            23
#define AQDE_PATTERN_HIGH_ROW6_Start                                          16
#define AQDE_PATTERN_HIGH_ROW6_Type                                          U08

#define AQDE_PATTERN_HIGH_ROW7                                             31:24
#define AQDE_PATTERN_HIGH_ROW7_End                                            31
#define AQDE_PATTERN_HIGH_ROW7_Start                                          24
#define AQDE_PATTERN_HIGH_ROW7_Type                                          U08

#define AQDEPatternMaskLowRegAddrs                                        0x0492
#define AQDE_PATTERN_MASK_LOW_Address                                    0x01248
#define AQDE_PATTERN_MASK_LOW_MSB                                             15
#define AQDE_PATTERN_MASK_LOW_LSB                                              0
#define AQDE_PATTERN_MASK_LOW_BLK                                              0
#define AQDE_PATTERN_MASK_LOW_Count                                            1
#define AQDE_PATTERN_MASK_LOW_FieldMask                               0xFFFFFFFF
#define AQDE_PATTERN_MASK_LOW_ReadMask                                0xFFFFFFFF
#define AQDE_PATTERN_MASK_LOW_WriteMask                               0xFFFFFFFF
#define AQDE_PATTERN_MASK_LOW_ResetValue                              0xFFFFFFFF

#define AQDE_PATTERN_MASK_LOW_ROW0                                           7:0
#define AQDE_PATTERN_MASK_LOW_ROW0_End                                         7
#define AQDE_PATTERN_MASK_LOW_ROW0_Start                                       0
#define AQDE_PATTERN_MASK_LOW_ROW0_Type                                      U08

#define AQDE_PATTERN_MASK_LOW_ROW1                                          15:8
#define AQDE_PATTERN_MASK_LOW_ROW1_End                                        15
#define AQDE_PATTERN_MASK_LOW_ROW1_Start                                       8
#define AQDE_PATTERN_MASK_LOW_ROW1_Type                                      U08

#define AQDE_PATTERN_MASK_LOW_ROW2                                         23:16
#define AQDE_PATTERN_MASK_LOW_ROW2_End                                        23
#define AQDE_PATTERN_MASK_LOW_ROW2_Start                                      16
#define AQDE_PATTERN_MASK_LOW_ROW2_Type                                      U08

#define AQDE_PATTERN_MASK_LOW_ROW3                                         31:24
#define AQDE_PATTERN_MASK_LOW_ROW3_End                                        31
#define AQDE_PATTERN_MASK_LOW_ROW3_Start                                      24
#define AQDE_PATTERN_MASK_LOW_ROW3_Type                                      U08

#define AQDEPatternMaskHighRegAddrs                                       0x0493
#define AQDE_PATTERN_MASK_HIGH_Address                                   0x0124C
#define AQDE_PATTERN_MASK_HIGH_MSB                                            15
#define AQDE_PATTERN_MASK_HIGH_LSB                                             0
#define AQDE_PATTERN_MASK_LOW_HIGH_BLK                                         0
#define AQDE_PATTERN_MASK_HIGH_Count                                           1
#define AQDE_PATTERN_MASK_HIGH_FieldMask                              0xFFFFFFFF
#define AQDE_PATTERN_MASK_HIGH_ReadMask                               0xFFFFFFFF
#define AQDE_PATTERN_MASK_HIGH_WriteMask                              0xFFFFFFFF
#define AQDE_PATTERN_MASK_HIGH_ResetValue                             0xFFFFFFFF

#define AQDE_PATTERN_MASK_HIGH_ROW4                                          7:0
#define AQDE_PATTERN_MASK_HIGH_ROW4_End                                        7
#define AQDE_PATTERN_MASK_HIGH_ROW4_Start                                      0
#define AQDE_PATTERN_MASK_HIGH_ROW4_Type                                     U08

#define AQDE_PATTERN_MASK_HIGH_ROW5                                         15:8
#define AQDE_PATTERN_MASK_HIGH_ROW5_End                                       15
#define AQDE_PATTERN_MASK_HIGH_ROW5_Start                                      8
#define AQDE_PATTERN_MASK_HIGH_ROW5_Type                                     U08

#define AQDE_PATTERN_MASK_HIGH_ROW6                                        23:16
#define AQDE_PATTERN_MASK_HIGH_ROW6_End                                       23
#define AQDE_PATTERN_MASK_HIGH_ROW6_Start                                     16
#define AQDE_PATTERN_MASK_HIGH_ROW6_Type                                     U08

#define AQDE_PATTERN_MASK_HIGH_ROW7                                        31:24
#define AQDE_PATTERN_MASK_HIGH_ROW7_End                                       31
#define AQDE_PATTERN_MASK_HIGH_ROW7_Start                                     24
#define AQDE_PATTERN_MASK_HIGH_ROW7_Type                                     U08

#define AQDEPatternBgColorRegAddrs                                        0x0494
#define AQDE_PATTERN_BG_COLOR_Address                                    0x01250
#define AQDE_PATTERN_BG_COLOR_MSB                                             15
#define AQDE_PATTERN_BG_COLOR_LSB                                              0
#define AQDE_PATTERN_BG_COLOR_BLK                                              0
#define AQDE_PATTERN_BG_COLOR_Count                                            1
#define AQDE_PATTERN_BG_COLOR_FieldMask                               0xFFFFFFFF
#define AQDE_PATTERN_BG_COLOR_ReadMask                                0xFFFFFFFF
#define AQDE_PATTERN_BG_COLOR_WriteMask                               0xFFFFFFFF
#define AQDE_PATTERN_BG_COLOR_ResetValue                              0x00000000

#define AQDE_PATTERN_BG_COLOR_ALPHA                                        31:24
#define AQDE_PATTERN_BG_COLOR_ALPHA_End                                       31
#define AQDE_PATTERN_BG_COLOR_ALPHA_Start                                     24
#define AQDE_PATTERN_BG_COLOR_ALPHA_Type                                     U08

#define AQDE_PATTERN_BG_COLOR_RED                                          23:16
#define AQDE_PATTERN_BG_COLOR_RED_End                                         23
#define AQDE_PATTERN_BG_COLOR_RED_Start                                       16
#define AQDE_PATTERN_BG_COLOR_RED_Type                                       U08

#define AQDE_PATTERN_BG_COLOR_GREEN                                         15:8
#define AQDE_PATTERN_BG_COLOR_GREEN_End                                       15
#define AQDE_PATTERN_BG_COLOR_GREEN_Start                                      8
#define AQDE_PATTERN_BG_COLOR_GREEN_Type                                     U08

#define AQDE_PATTERN_BG_COLOR_BLUE                                           7:0
#define AQDE_PATTERN_BG_COLOR_BLUE_End                                         7
#define AQDE_PATTERN_BG_COLOR_BLUE_Start                                       0
#define AQDE_PATTERN_BG_COLOR_BLUE_Type                                      U08

#define AQDEPatternFgColorRegAddrs                                        0x0495
#define AQDE_PATTERN_FG_COLOR_Address                                    0x01254
#define AQDE_PATTERN_FG_COLOR_MSB                                             15
#define AQDE_PATTERN_FG_COLOR_LSB                                              0
#define AQDE_PATTERN_FG_COLOR_BLK                                              0
#define AQDE_PATTERN_FG_COLOR_Count                                            1
#define AQDE_PATTERN_FG_COLOR_FieldMask                               0xFFFFFFFF
#define AQDE_PATTERN_FG_COLOR_ReadMask                                0xFFFFFFFF
#define AQDE_PATTERN_FG_COLOR_WriteMask                               0xFFFFFFFF
#define AQDE_PATTERN_FG_COLOR_ResetValue                              0x00000000

#define AQDE_PATTERN_FG_COLOR_ALPHA                                        31:24
#define AQDE_PATTERN_FG_COLOR_ALPHA_End                                       31
#define AQDE_PATTERN_FG_COLOR_ALPHA_Start                                     24
#define AQDE_PATTERN_FG_COLOR_ALPHA_Type                                     U08

#define AQDE_PATTERN_FG_COLOR_RED                                          23:16
#define AQDE_PATTERN_FG_COLOR_RED_End                                         23
#define AQDE_PATTERN_FG_COLOR_RED_Start                                       16
#define AQDE_PATTERN_FG_COLOR_RED_Type                                       U08

#define AQDE_PATTERN_FG_COLOR_GREEN                                         15:8
#define AQDE_PATTERN_FG_COLOR_GREEN_End                                       15
#define AQDE_PATTERN_FG_COLOR_GREEN_Start                                      8
#define AQDE_PATTERN_FG_COLOR_GREEN_Type                                     U08

#define AQDE_PATTERN_FG_COLOR_BLUE                                           7:0
#define AQDE_PATTERN_FG_COLOR_BLUE_End                                         7
#define AQDE_PATTERN_FG_COLOR_BLUE_Start                                       0
#define AQDE_PATTERN_FG_COLOR_BLUE_Type                                      U08

#define AQDEFilterBlitRegAddrs                                            0x0496
#define AQDE_FILTER_BLIT_Address                                         0x01258
#define AQDE_FILTER_BLIT_MSB                                                  15
#define AQDE_FILTER_BLIT_LSB                                                   0
#define AQDE_FILTER_BLIT_BLK                                                   0
#define AQDE_FILTER_BLIT_Count                                                 1
#define AQDE_FILTER_BLIT_FieldMask                                    0x000000FF
#define AQDE_FILTER_BLIT_ReadMask                                     0x000000FF
#define AQDE_FILTER_BLIT_WriteMask                                    0x000000FF
#define AQDE_FILTER_BLIT_ResetValue                                   0x00000000

#define AQDE_FILTER_BLIT_IGNORE_LEFT_EDGE                                    7:7
#define AQDE_FILTER_BLIT_IGNORE_LEFT_EDGE_End                                  7
#define AQDE_FILTER_BLIT_IGNORE_LEFT_EDGE_Start                                7
#define AQDE_FILTER_BLIT_IGNORE_LEFT_EDGE_Type                               U01

#define AQDE_FILTER_BLIT_IGNORE_TOP_EDGE                                     6:6
#define AQDE_FILTER_BLIT_IGNORE_TOP_EDGE_End                                   6
#define AQDE_FILTER_BLIT_IGNORE_TOP_EDGE_Start                                 6
#define AQDE_FILTER_BLIT_IGNORE_TOP_EDGE_Type                                U01

#define AQDE_FILTER_BLIT_IGNORE_RIGHT_EDGE                                   5:5
#define AQDE_FILTER_BLIT_IGNORE_RIGHT_EDGE_End                                 5
#define AQDE_FILTER_BLIT_IGNORE_RIGHT_EDGE_Start                               5
#define AQDE_FILTER_BLIT_IGNORE_RIGHT_EDGE_Type                              U01

#define AQDE_FILTER_BLIT_IGNORE_BOTTOM_EDGE                                  4:4
#define AQDE_FILTER_BLIT_IGNORE_BOTTOM_EDGE_End                                4
#define AQDE_FILTER_BLIT_IGNORE_BOTTOM_EDGE_Start                              4
#define AQDE_FILTER_BLIT_IGNORE_BOTTOM_EDGE_Type                             U01

#define AQDE_FILTER_BLIT_KERNEL_SIZE                                         3:0
#define AQDE_FILTER_BLIT_KERNEL_SIZE_End                                       3
#define AQDE_FILTER_BLIT_KERNEL_SIZE_Start                                     0
#define AQDE_FILTER_BLIT_KERNEL_SIZE_Type                                    U04

#define AQDEFilterKernelRegAddrs                                          0x0600
#define AQDE_FILTER_KERNEL_Address                                       0x01800
#define AQDE_FILTER_KERNEL_MSB                                                15
#define AQDE_FILTER_KERNEL_LSB                                                 7
#define AQDE_FILTER_KERNEL_BLK                                                 7
#define AQDE_FILTER_KERNEL_Count                                             128
#define AQDE_FILTER_KERNEL_FieldMask                                  0xFFFFFFFF
#define AQDE_FILTER_KERNEL_ReadMask                                   0xFFFFFFFF
#define AQDE_FILTER_KERNEL_WriteMask                                  0xFFFFFFFF
#define AQDE_FILTER_KERNEL_ResetValue                                 0x00000000

#define AQDE_FILTER_KERNEL_COEFFICIENT0                                     15:0
#define AQDE_FILTER_KERNEL_COEFFICIENT0_End                                   15
#define AQDE_FILTER_KERNEL_COEFFICIENT0_Start                                  0
#define AQDE_FILTER_KERNEL_COEFFICIENT0_Type                                 U16

#define AQDE_FILTER_KERNEL_COEFFICIENT1                                    31:16
#define AQDE_FILTER_KERNEL_COEFFICIENT1_End                                   31
#define AQDE_FILTER_KERNEL_COEFFICIENT1_Start                                 16
#define AQDE_FILTER_KERNEL_COEFFICIENT1_Type                                 U16

#define AQDEIndexColorTableRegAddrs                                       0x0700
#define AQDE_INDEX_COLOR_TABLE_Address                                   0x01C00
#define AQDE_INDEX_COLOR_TABLE_MSB                                            15
#define AQDE_INDEX_COLOR_TABLE_LSB                                             8
#define AQDE_INDEX_COLOR_TABLE_BLK                                             8
#define AQDE_INDEX_COLOR_TABLE_Count                                         256
#define AQDE_INDEX_COLOR_TABLE_FieldMask                              0xFFFFFFFF
#define AQDE_INDEX_COLOR_TABLE_ReadMask                               0xFFFFFFFF
#define AQDE_INDEX_COLOR_TABLE_WriteMask                              0xFFFFFFFF
#define AQDE_INDEX_COLOR_TABLE_ResetValue                             0x00000000

#define AQDE_INDEX_COLOR_TABLE_ALPHA                                       31:24
#define AQDE_INDEX_COLOR_TABLE_ALPHA_End                                      31
#define AQDE_INDEX_COLOR_TABLE_ALPHA_Start                                    24
#define AQDE_INDEX_COLOR_TABLE_ALPHA_Type                                    U08

#define AQDE_INDEX_COLOR_TABLE_RED                                         23:16
#define AQDE_INDEX_COLOR_TABLE_RED_End                                        23
#define AQDE_INDEX_COLOR_TABLE_RED_Start                                      16
#define AQDE_INDEX_COLOR_TABLE_RED_Type                                      U08

#define AQDE_INDEX_COLOR_TABLE_GREEN                                        15:8
#define AQDE_INDEX_COLOR_TABLE_GREEN_End                                      15
#define AQDE_INDEX_COLOR_TABLE_GREEN_Start                                     8
#define AQDE_INDEX_COLOR_TABLE_GREEN_Type                                    U08

#define AQDE_INDEX_COLOR_TABLE_BLUE                                          7:0
#define AQDE_INDEX_COLOR_TABLE_BLUE_End                                        7
#define AQDE_INDEX_COLOR_TABLE_BLUE_Start                                      0
#define AQDE_INDEX_COLOR_TABLE_BLUE_Type                                     U08

#define AQDEIndexColorTable32RegAddrs                                     0x0D00
#define AQDE_INDEX_COLOR_TABLE32_Address                                 0x03400
#define AQDE_INDEX_COLOR_TABLE32_MSB                                          15
#define AQDE_INDEX_COLOR_TABLE32_LSB                                           8
#define AQDE_INDEX_COLOR_TABLE32_BLK                                           8
#define AQDE_INDEX_COLOR_TABLE32_Count                                       256
#define AQDE_INDEX_COLOR_TABLE32_FieldMask                            0xFFFFFFFF
#define AQDE_INDEX_COLOR_TABLE32_ReadMask                             0xFFFFFFFF
#define AQDE_INDEX_COLOR_TABLE32_WriteMask                            0xFFFFFFFF
#define AQDE_INDEX_COLOR_TABLE32_ResetValue                           0x00000000

#define AQDE_INDEX_COLOR_TABLE32_ALPHA                                     31:24
#define AQDE_INDEX_COLOR_TABLE32_ALPHA_End                                    31
#define AQDE_INDEX_COLOR_TABLE32_ALPHA_Start                                  24
#define AQDE_INDEX_COLOR_TABLE32_ALPHA_Type                                  U08

#define AQDE_INDEX_COLOR_TABLE32_RED                                       23:16
#define AQDE_INDEX_COLOR_TABLE32_RED_End                                      23
#define AQDE_INDEX_COLOR_TABLE32_RED_Start                                    16
#define AQDE_INDEX_COLOR_TABLE32_RED_Type                                    U08

#define AQDE_INDEX_COLOR_TABLE32_GREEN                                      15:8
#define AQDE_INDEX_COLOR_TABLE32_GREEN_End                                    15
#define AQDE_INDEX_COLOR_TABLE32_GREEN_Start                                   8
#define AQDE_INDEX_COLOR_TABLE32_GREEN_Type                                  U08

#define AQDE_INDEX_COLOR_TABLE32_BLUE                                        7:0
#define AQDE_INDEX_COLOR_TABLE32_BLUE_End                                      7
#define AQDE_INDEX_COLOR_TABLE32_BLUE_Start                                    0
#define AQDE_INDEX_COLOR_TABLE32_BLUE_Type                                   U08

#define AQDERopRegAddrs                                                   0x0497
#define AQDE_ROP_Address                                                 0x0125C
#define AQDE_ROP_MSB                                                          15
#define AQDE_ROP_LSB                                                           0
#define AQDE_ROP_BLK                                                           0
#define AQDE_ROP_Count                                                         1
#define AQDE_ROP_FieldMask                                            0x0030FFFF
#define AQDE_ROP_ReadMask                                             0x0030FFFF
#define AQDE_ROP_WriteMask                                            0x0030FFFF
#define AQDE_ROP_ResetValue                                           0x00000000

#define AQDE_ROP_TYPE                                                      21:20
#define AQDE_ROP_TYPE_End                                                     21
#define AQDE_ROP_TYPE_Start                                                   20
#define AQDE_ROP_TYPE_Type                                                   U02
#define   AQDE_ROP_TYPE_ROP2_PATTERN                                         0x0
#define   AQDE_ROP_TYPE_ROP2_SOURCE                                          0x1
#define   AQDE_ROP_TYPE_ROP3                                                 0x2
#define   AQDE_ROP_TYPE_ROP4                                                 0x3

#define AQDE_ROP_ROP_BG                                                     15:8
#define AQDE_ROP_ROP_BG_End                                                   15
#define AQDE_ROP_ROP_BG_Start                                                  8
#define AQDE_ROP_ROP_BG_Type                                                 U08

#define AQDE_ROP_ROP_FG                                                      7:0
#define AQDE_ROP_ROP_FG_End                                                    7
#define AQDE_ROP_ROP_FG_Start                                                  0
#define AQDE_ROP_ROP_FG_Type                                                 U08

#define AQDEClipTopLeftRegAddrs                                           0x0498
#define AQDE_CLIP_TOP_LEFT_Address                                       0x01260
#define AQDE_CLIP_TOP_LEFT_MSB                                                15
#define AQDE_CLIP_TOP_LEFT_LSB                                                 0
#define AQDE_CLIP_TOP_LEFT_BLK                                                 0
#define AQDE_CLIP_TOP_LEFT_Count                                               1
#define AQDE_CLIP_TOP_LEFT_FieldMask                                  0x7FFF7FFF
#define AQDE_CLIP_TOP_LEFT_ReadMask                                   0x7FFF7FFF
#define AQDE_CLIP_TOP_LEFT_WriteMask                                  0x7FFF7FFF
#define AQDE_CLIP_TOP_LEFT_ResetValue                                 0x00000000

#define AQDE_CLIP_TOP_LEFT_Y                                               30:16
#define AQDE_CLIP_TOP_LEFT_Y_End                                              30
#define AQDE_CLIP_TOP_LEFT_Y_Start                                            16
#define AQDE_CLIP_TOP_LEFT_Y_Type                                            U15

#define AQDE_CLIP_TOP_LEFT_X                                                14:0
#define AQDE_CLIP_TOP_LEFT_X_End                                              14
#define AQDE_CLIP_TOP_LEFT_X_Start                                             0
#define AQDE_CLIP_TOP_LEFT_X_Type                                            U15

#define AQDEClipBottomRightRegAddrs                                       0x0499
#define AQDE_CLIP_BOTTOM_RIGHT_Address                                   0x01264
#define AQDE_CLIP_BOTTOM_RIGHT_MSB                                            15
#define AQDE_CLIP_BOTTOM_RIGHT_LSB                                             0
#define AQDE_CLIP_BOTTOM_RIGHT_BLK                                             0
#define AQDE_CLIP_BOTTOM_RIGHT_Count                                           1
#define AQDE_CLIP_BOTTOM_RIGHT_FieldMask                              0x7FFF7FFF
#define AQDE_CLIP_BOTTOM_RIGHT_ReadMask                               0x7FFF7FFF
#define AQDE_CLIP_BOTTOM_RIGHT_WriteMask                              0x7FFF7FFF
#define AQDE_CLIP_BOTTOM_RIGHT_ResetValue                             0x00000000

#define AQDE_CLIP_BOTTOM_RIGHT_Y                                           30:16
#define AQDE_CLIP_BOTTOM_RIGHT_Y_End                                          30
#define AQDE_CLIP_BOTTOM_RIGHT_Y_Start                                        16
#define AQDE_CLIP_BOTTOM_RIGHT_Y_Type                                        U15

#define AQDE_CLIP_BOTTOM_RIGHT_X                                            14:0
#define AQDE_CLIP_BOTTOM_RIGHT_X_End                                          14
#define AQDE_CLIP_BOTTOM_RIGHT_X_Start                                         0
#define AQDE_CLIP_BOTTOM_RIGHT_X_Type                                        U15

#define AQDEClearByteMaskRegAddrs                                         0x049A
#define AQDE_CLEAR_BYTE_MASK_Address                                     0x01268
#define AQDE_CLEAR_BYTE_MASK_MSB                                              15
#define AQDE_CLEAR_BYTE_MASK_LSB                                               0
#define AQDE_CLEAR_BYTE_MASK_BLK                                               0
#define AQDE_CLEAR_BYTE_MASK_Count                                             1
#define AQDE_CLEAR_BYTE_MASK_FieldMask                                0x000000FF
#define AQDE_CLEAR_BYTE_MASK_ReadMask                                 0x000000FF
#define AQDE_CLEAR_BYTE_MASK_WriteMask                                0x000000FF
#define AQDE_CLEAR_BYTE_MASK_ResetValue                               0x00000000

#define AQDE_CLEAR_BYTE_MASK_VALUE                                           7:0
#define AQDE_CLEAR_BYTE_MASK_VALUE_End                                         7
#define AQDE_CLEAR_BYTE_MASK_VALUE_Start                                       0
#define AQDE_CLEAR_BYTE_MASK_VALUE_Type                                      U08

#define AQDEClearPixelValueLowRegAddrs                                    0x049C
#define AQDE_CLEAR_PIXEL_VALUE_LOW_Address                               0x01270
#define AQDE_CLEAR_PIXEL_VALUE_LOW_MSB                                        15
#define AQDE_CLEAR_PIXEL_VALUE_LOW_LSB                                         0
#define AQDE_CLEAR_PIXEL_VALUE_LOW_BLK                                         0
#define AQDE_CLEAR_PIXEL_VALUE_LOW_Count                                       1
#define AQDE_CLEAR_PIXEL_VALUE_LOW_FieldMask                          0xFFFFFFFF
#define AQDE_CLEAR_PIXEL_VALUE_LOW_ReadMask                           0xFFFFFFFF
#define AQDE_CLEAR_PIXEL_VALUE_LOW_WriteMask                          0xFFFFFFFF
#define AQDE_CLEAR_PIXEL_VALUE_LOW_ResetValue                         0x00000000

#define AQDE_CLEAR_PIXEL_VALUE_LOW_VALUE                                    31:0
#define AQDE_CLEAR_PIXEL_VALUE_LOW_VALUE_End                                  31
#define AQDE_CLEAR_PIXEL_VALUE_LOW_VALUE_Start                                 0
#define AQDE_CLEAR_PIXEL_VALUE_LOW_VALUE_Type                                U32

#define AQDEClearPixelValueHighRegAddrs                                   0x049D
#define AQDE_CLEAR_PIXEL_VALUE_HIGH_Address                              0x01274
#define AQDE_CLEAR_PIXEL_VALUE_HIGH_MSB                                       15
#define AQDE_CLEAR_PIXEL_VALUE_HIGH_LSB                                        0
#define AQDE_CLEAR_PIXEL_VALUE_LOW_HIGH_BLK                                    0
#define AQDE_CLEAR_PIXEL_VALUE_HIGH_Count                                      1
#define AQDE_CLEAR_PIXEL_VALUE_HIGH_FieldMask                         0xFFFFFFFF
#define AQDE_CLEAR_PIXEL_VALUE_HIGH_ReadMask                          0xFFFFFFFF
#define AQDE_CLEAR_PIXEL_VALUE_HIGH_WriteMask                         0xFFFFFFFF
#define AQDE_CLEAR_PIXEL_VALUE_HIGH_ResetValue                        0x00000000

#define AQDE_CLEAR_PIXEL_VALUE_HIGH_VALUE                                   31:0
#define AQDE_CLEAR_PIXEL_VALUE_HIGH_VALUE_End                                 31
#define AQDE_CLEAR_PIXEL_VALUE_HIGH_VALUE_Start                                0
#define AQDE_CLEAR_PIXEL_VALUE_HIGH_VALUE_Type                               U32

#define AQDEConfigRegAddrs                                                0x049B
#define AQDE_CONFIG_Address                                              0x0126C
#define AQDE_CONFIG_MSB                                                       15
#define AQDE_CONFIG_LSB                                                        0
#define AQDE_CONFIG_BLK                                                        0
#define AQDE_CONFIG_Count                                                      1
#define AQDE_CONFIG_FieldMask                                         0x00370031
#define AQDE_CONFIG_ReadMask                                          0x00370031
#define AQDE_CONFIG_WriteMask                                         0x00370031
#define AQDE_CONFIG_ResetValue                                        0x00000000

#define AQDE_CONFIG_MIRROR_BLT_MODE                                          5:4
#define AQDE_CONFIG_MIRROR_BLT_MODE_End                                        5
#define AQDE_CONFIG_MIRROR_BLT_MODE_Start                                      4
#define AQDE_CONFIG_MIRROR_BLT_MODE_Type                                     U02
#define   AQDE_CONFIG_MIRROR_BLT_MODE_NORMAL                                 0x0
#define   AQDE_CONFIG_MIRROR_BLT_MODE_HMIRROR                                0x1
#define   AQDE_CONFIG_MIRROR_BLT_MODE_VMIRROR                                0x2
#define   AQDE_CONFIG_MIRROR_BLT_MODE_FULL_MIRROR                            0x3

#define AQDE_CONFIG_MIRROR_BLT_ENABLE                                        0:0
#define AQDE_CONFIG_MIRROR_BLT_ENABLE_End                                      0
#define AQDE_CONFIG_MIRROR_BLT_ENABLE_Start                                    0
#define AQDE_CONFIG_MIRROR_BLT_ENABLE_Type                                   U01
#define   AQDE_CONFIG_MIRROR_BLT_ENABLE_OFF                                  0x0
#define   AQDE_CONFIG_MIRROR_BLT_ENABLE_ON                                   0x1

#define AQDE_CONFIG_SOURCE_SELECT                                          18:16
#define AQDE_CONFIG_SOURCE_SELECT_End                                         18
#define AQDE_CONFIG_SOURCE_SELECT_Start                                       16
#define AQDE_CONFIG_SOURCE_SELECT_Type                                       U03

#define AQDE_CONFIG_DESTINATION_SELECT                                     21:20
#define AQDE_CONFIG_DESTINATION_SELECT_End                                    21
#define AQDE_CONFIG_DESTINATION_SELECT_Start                                  20
#define AQDE_CONFIG_DESTINATION_SELECT_Type                                  U02

#define AQDESrcOriginFractionRegAddrs                                     0x049E
#define AQDE_SRC_ORIGIN_FRACTION_Address                                 0x01278
#define AQDE_SRC_ORIGIN_FRACTION_MSB                                          15
#define AQDE_SRC_ORIGIN_FRACTION_LSB                                           0
#define AQDE_SRC_ORIGIN_FRACTION_BLK                                           0
#define AQDE_SRC_ORIGIN_FRACTION_Count                                         1
#define AQDE_SRC_ORIGIN_FRACTION_FieldMask                            0xFFFFFFFF
#define AQDE_SRC_ORIGIN_FRACTION_ReadMask                             0xFFFFFFFF
#define AQDE_SRC_ORIGIN_FRACTION_WriteMask                            0xFFFFFFFF
#define AQDE_SRC_ORIGIN_FRACTION_ResetValue                           0x00000000

#define AQDE_SRC_ORIGIN_FRACTION_Y                                         31:16
#define AQDE_SRC_ORIGIN_FRACTION_Y_End                                        31
#define AQDE_SRC_ORIGIN_FRACTION_Y_Start                                      16
#define AQDE_SRC_ORIGIN_FRACTION_Y_Type                                      U16

#define AQDE_SRC_ORIGIN_FRACTION_X                                          15:0
#define AQDE_SRC_ORIGIN_FRACTION_X_End                                        15
#define AQDE_SRC_ORIGIN_FRACTION_X_Start                                       0
#define AQDE_SRC_ORIGIN_FRACTION_X_Type                                      U16

#define AQDEAlphaControlRegAddrs                                          0x049F
#define AQDE_ALPHA_CONTROL_Address                                       0x0127C
#define AQDE_ALPHA_CONTROL_MSB                                                15
#define AQDE_ALPHA_CONTROL_LSB                                                 0
#define AQDE_ALPHA_CONTROL_BLK                                                 0
#define AQDE_ALPHA_CONTROL_Count                                               1
#define AQDE_ALPHA_CONTROL_FieldMask                                  0xFFFF0001
#define AQDE_ALPHA_CONTROL_ReadMask                                   0xFFFF0001
#define AQDE_ALPHA_CONTROL_WriteMask                                  0xFFFF0001
#define AQDE_ALPHA_CONTROL_ResetValue                                 0x00000000

#define AQDE_ALPHA_CONTROL_ENABLE                                            0:0
#define AQDE_ALPHA_CONTROL_ENABLE_End                                          0
#define AQDE_ALPHA_CONTROL_ENABLE_Start                                        0
#define AQDE_ALPHA_CONTROL_ENABLE_Type                                       U01
#define   AQDE_ALPHA_CONTROL_ENABLE_OFF                                      0x0
#define   AQDE_ALPHA_CONTROL_ENABLE_ON                                       0x1

#define AQDE_ALPHA_CONTROL_SRC_VALUE                                       23:16
#define AQDE_ALPHA_CONTROL_SRC_VALUE_End                                      23
#define AQDE_ALPHA_CONTROL_SRC_VALUE_Start                                    16
#define AQDE_ALPHA_CONTROL_SRC_VALUE_Type                                    U08

#define AQDE_ALPHA_CONTROL_DST_VALUE                                       31:24
#define AQDE_ALPHA_CONTROL_DST_VALUE_End                                      31
#define AQDE_ALPHA_CONTROL_DST_VALUE_Start                                    24
#define AQDE_ALPHA_CONTROL_DST_VALUE_Type                                    U08

#define AQDEAlphaModesRegAddrs                                            0x04A0
#define AQDE_ALPHA_MODES_Address                                         0x01280
#define AQDE_ALPHA_MODES_MSB                                                  15
#define AQDE_ALPHA_MODES_LSB                                                   0
#define AQDE_ALPHA_MODES_BLK                                                   0
#define AQDE_ALPHA_MODES_Count                                                 1
#define AQDE_ALPHA_MODES_FieldMask                                    0xFF113311
#define AQDE_ALPHA_MODES_ReadMask                                     0xFF113311
#define AQDE_ALPHA_MODES_WriteMask                                    0xFF113311
#define AQDE_ALPHA_MODES_ResetValue                                   0x00000000

#define AQDE_ALPHA_MODES_SRC_ALPHA_MODE                                      0:0
#define AQDE_ALPHA_MODES_SRC_ALPHA_MODE_End                                    0
#define AQDE_ALPHA_MODES_SRC_ALPHA_MODE_Start                                  0
#define AQDE_ALPHA_MODES_SRC_ALPHA_MODE_Type                                 U01
#define   AQDE_ALPHA_MODES_SRC_ALPHA_MODE_NORMAL                             0x0
#define   AQDE_ALPHA_MODES_SRC_ALPHA_MODE_INVERSED                           0x1

#define AQDE_ALPHA_MODES_DST_ALPHA_MODE                                      4:4
#define AQDE_ALPHA_MODES_DST_ALPHA_MODE_End                                    4
#define AQDE_ALPHA_MODES_DST_ALPHA_MODE_Start                                  4
#define AQDE_ALPHA_MODES_DST_ALPHA_MODE_Type                                 U01
#define   AQDE_ALPHA_MODES_DST_ALPHA_MODE_NORMAL                             0x0
#define   AQDE_ALPHA_MODES_DST_ALPHA_MODE_INVERSED                           0x1

#define AQDE_ALPHA_MODES_GLOBAL_SRC_ALPHA_MODE                               9:8
#define AQDE_ALPHA_MODES_GLOBAL_SRC_ALPHA_MODE_End                             9
#define AQDE_ALPHA_MODES_GLOBAL_SRC_ALPHA_MODE_Start                           8
#define AQDE_ALPHA_MODES_GLOBAL_SRC_ALPHA_MODE_Type                          U02
#define   AQDE_ALPHA_MODES_GLOBAL_SRC_ALPHA_MODE_NORMAL                      0x0
#define   AQDE_ALPHA_MODES_GLOBAL_SRC_ALPHA_MODE_GLOBAL                      0x1
#define   AQDE_ALPHA_MODES_GLOBAL_SRC_ALPHA_MODE_SCALED                      0x2

#define AQDE_ALPHA_MODES_GLOBAL_DST_ALPHA_MODE                             13:12
#define AQDE_ALPHA_MODES_GLOBAL_DST_ALPHA_MODE_End                            13
#define AQDE_ALPHA_MODES_GLOBAL_DST_ALPHA_MODE_Start                          12
#define AQDE_ALPHA_MODES_GLOBAL_DST_ALPHA_MODE_Type                          U02
#define   AQDE_ALPHA_MODES_GLOBAL_DST_ALPHA_MODE_NORMAL                      0x0
#define   AQDE_ALPHA_MODES_GLOBAL_DST_ALPHA_MODE_GLOBAL                      0x1
#define   AQDE_ALPHA_MODES_GLOBAL_DST_ALPHA_MODE_SCALED                      0x2

#define AQDE_ALPHA_MODES_SRC_COLOR_MODE                                    16:16
#define AQDE_ALPHA_MODES_SRC_COLOR_MODE_End                                   16
#define AQDE_ALPHA_MODES_SRC_COLOR_MODE_Start                                 16
#define AQDE_ALPHA_MODES_SRC_COLOR_MODE_Type                                 U01
#define   AQDE_ALPHA_MODES_SRC_COLOR_MODE_NORMAL                             0x0
#define   AQDE_ALPHA_MODES_SRC_COLOR_MODE_MULTIPLY                           0x1

#define AQDE_ALPHA_MODES_DST_COLOR_MODE                                    20:20
#define AQDE_ALPHA_MODES_DST_COLOR_MODE_End                                   20
#define AQDE_ALPHA_MODES_DST_COLOR_MODE_Start                                 20
#define AQDE_ALPHA_MODES_DST_COLOR_MODE_Type                                 U01
#define   AQDE_ALPHA_MODES_DST_COLOR_MODE_NORMAL                             0x0
#define   AQDE_ALPHA_MODES_DST_COLOR_MODE_MULTIPLY                           0x1

#define AQDE_ALPHA_MODES_SRC_BLENDING_MODE                                 26:24
#define AQDE_ALPHA_MODES_SRC_BLENDING_MODE_End                                26
#define AQDE_ALPHA_MODES_SRC_BLENDING_MODE_Start                              24
#define AQDE_ALPHA_MODES_SRC_BLENDING_MODE_Type                              U03
#define   AQDE_ALPHA_MODES_SRC_BLENDING_MODE_ZERO                            0x0
#define   AQDE_ALPHA_MODES_SRC_BLENDING_MODE_ONE                             0x1
#define   AQDE_ALPHA_MODES_SRC_BLENDING_MODE_NORMAL                          0x2
#define   AQDE_ALPHA_MODES_SRC_BLENDING_MODE_INVERSED                        0x3
#define   AQDE_ALPHA_MODES_SRC_BLENDING_MODE_COLOR                           0x4
#define   AQDE_ALPHA_MODES_SRC_BLENDING_MODE_COLOR_INVERSED                  0x5
#define   AQDE_ALPHA_MODES_SRC_BLENDING_MODE_SATURATED_ALPHA                 0x6
#define   AQDE_ALPHA_MODES_SRC_BLENDING_MODE_SATURATED_DEST_ALPHA            0x7

#define AQDE_ALPHA_MODES_SRC_ALPHA_FACTOR                                  27:27
#define AQDE_ALPHA_MODES_SRC_ALPHA_FACTOR_End                                 27
#define AQDE_ALPHA_MODES_SRC_ALPHA_FACTOR_Start                               27
#define AQDE_ALPHA_MODES_SRC_ALPHA_FACTOR_Type                               U01
#define   AQDE_ALPHA_MODES_SRC_ALPHA_FACTOR_DISABLED                         0x0
#define   AQDE_ALPHA_MODES_SRC_ALPHA_FACTOR_ENABLED                          0x1

#define AQDE_ALPHA_MODES_DST_BLENDING_MODE                                 30:28
#define AQDE_ALPHA_MODES_DST_BLENDING_MODE_End                                30
#define AQDE_ALPHA_MODES_DST_BLENDING_MODE_Start                              28
#define AQDE_ALPHA_MODES_DST_BLENDING_MODE_Type                              U03
#define   AQDE_ALPHA_MODES_DST_BLENDING_MODE_ZERO                            0x0
#define   AQDE_ALPHA_MODES_DST_BLENDING_MODE_ONE                             0x1
#define   AQDE_ALPHA_MODES_DST_BLENDING_MODE_NORMAL                          0x2
#define   AQDE_ALPHA_MODES_DST_BLENDING_MODE_INVERSED                        0x3
#define   AQDE_ALPHA_MODES_DST_BLENDING_MODE_COLOR                           0x4
#define   AQDE_ALPHA_MODES_DST_BLENDING_MODE_COLOR_INVERSED                  0x5
#define   AQDE_ALPHA_MODES_DST_BLENDING_MODE_SATURATED_ALPHA                 0x6
#define   AQDE_ALPHA_MODES_DST_BLENDING_MODE_SATURATED_DEST_ALPHA            0x7

#define AQDE_ALPHA_MODES_DST_ALPHA_FACTOR                                  31:31
#define AQDE_ALPHA_MODES_DST_ALPHA_FACTOR_End                                 31
#define AQDE_ALPHA_MODES_DST_ALPHA_FACTOR_Start                               31
#define AQDE_ALPHA_MODES_DST_ALPHA_FACTOR_Type                               U01
#define   AQDE_ALPHA_MODES_DST_ALPHA_FACTOR_DISABLED                         0x0
#define   AQDE_ALPHA_MODES_DST_ALPHA_FACTOR_ENABLED                          0x1

#define UPlaneAddressRegAddrs                                             0x04A1
#define UPLANE_ADDRESS_Address                                           0x01284
#define UPLANE_ADDRESS_MSB                                                    15
#define UPLANE_ADDRESS_LSB                                                     0
#define UPLANE_ADDRESS_BLK                                                     0
#define UPLANE_ADDRESS_Count                                                   1
#define UPLANE_ADDRESS_FieldMask                                      0xFFFFFFFF
#define UPLANE_ADDRESS_ReadMask                                       0xFFFFFFFC
#define UPLANE_ADDRESS_WriteMask                                      0xFFFFFFFC
#define UPLANE_ADDRESS_ResetValue                                     0x00000000

#define UPLANE_ADDRESS_TYPE                                                31:31
#define UPLANE_ADDRESS_TYPE_End                                               31
#define UPLANE_ADDRESS_TYPE_Start                                             31
#define UPLANE_ADDRESS_TYPE_Type                                             U01
#define   UPLANE_ADDRESS_TYPE_SYSTEM                                         0x0
#define   UPLANE_ADDRESS_TYPE_VIRTUAL_SYSTEM                                 0x1

#define UPLANE_ADDRESS_ADDRESS                                              30:0
#define UPLANE_ADDRESS_ADDRESS_End                                            30
#define UPLANE_ADDRESS_ADDRESS_Start                                           0
#define UPLANE_ADDRESS_ADDRESS_Type                                          U31

#define UPlaneStrideRegAddrs                                              0x04A2
#define UPLANE_STRIDE_Address                                            0x01288
#define UPLANE_STRIDE_MSB                                                     15
#define UPLANE_STRIDE_LSB                                                      0
#define UPLANE_STRIDE_BLK                                                      0
#define UPLANE_STRIDE_Count                                                    1
#define UPLANE_STRIDE_FieldMask                                       0x0003FFFF
#define UPLANE_STRIDE_ReadMask                                        0x0003FFFC
#define UPLANE_STRIDE_WriteMask                                       0x0003FFFC
#define UPLANE_STRIDE_ResetValue                                      0x00000000

#define UPLANE_STRIDE_STRIDE                                                17:0
#define UPLANE_STRIDE_STRIDE_End                                              17
#define UPLANE_STRIDE_STRIDE_Start                                             0
#define UPLANE_STRIDE_STRIDE_Type                                            U18

#define VPlaneAddressRegAddrs                                             0x04A3
#define VPLANE_ADDRESS_Address                                           0x0128C
#define VPLANE_ADDRESS_MSB                                                    15
#define VPLANE_ADDRESS_LSB                                                     0
#define VPLANE_ADDRESS_BLK                                                     0
#define VPLANE_ADDRESS_Count                                                   1
#define VPLANE_ADDRESS_FieldMask                                      0xFFFFFFFF
#define VPLANE_ADDRESS_ReadMask                                       0xFFFFFFFC
#define VPLANE_ADDRESS_WriteMask                                      0xFFFFFFFC
#define VPLANE_ADDRESS_ResetValue                                     0x00000000

#define VPLANE_ADDRESS_TYPE                                                31:31
#define VPLANE_ADDRESS_TYPE_End                                               31
#define VPLANE_ADDRESS_TYPE_Start                                             31
#define VPLANE_ADDRESS_TYPE_Type                                             U01
#define   VPLANE_ADDRESS_TYPE_SYSTEM                                         0x0
#define   VPLANE_ADDRESS_TYPE_VIRTUAL_SYSTEM                                 0x1

#define VPLANE_ADDRESS_ADDRESS                                              30:0
#define VPLANE_ADDRESS_ADDRESS_End                                            30
#define VPLANE_ADDRESS_ADDRESS_Start                                           0
#define VPLANE_ADDRESS_ADDRESS_Type                                          U31

#define VPlaneStrideRegAddrs                                              0x04A4
#define VPLANE_STRIDE_Address                                            0x01290
#define VPLANE_STRIDE_MSB                                                     15
#define VPLANE_STRIDE_LSB                                                      0
#define VPLANE_STRIDE_BLK                                                      0
#define VPLANE_STRIDE_Count                                                    1
#define VPLANE_STRIDE_FieldMask                                       0x0003FFFF
#define VPLANE_STRIDE_ReadMask                                        0x0003FFFC
#define VPLANE_STRIDE_WriteMask                                       0x0003FFFC
#define VPLANE_STRIDE_ResetValue                                      0x00000000

#define VPLANE_STRIDE_STRIDE                                                17:0
#define VPLANE_STRIDE_STRIDE_End                                              17
#define VPLANE_STRIDE_STRIDE_Start                                             0
#define VPLANE_STRIDE_STRIDE_Type                                            U18

#define AQVRSourceImageLowRegAddrs                                        0x04A6
#define AQVR_SOURCE_IMAGE_LOW_Address                                    0x01298
#define AQVR_SOURCE_IMAGE_LOW_MSB                                             15
#define AQVR_SOURCE_IMAGE_LOW_LSB                                              0
#define AQVR_SOURCE_IMAGE_LOW_BLK                                              0
#define AQVR_SOURCE_IMAGE_LOW_Count                                            1
#define AQVR_SOURCE_IMAGE_LOW_FieldMask                               0xFFFFFFFF
#define AQVR_SOURCE_IMAGE_LOW_ReadMask                                0xFFFFFFFF
#define AQVR_SOURCE_IMAGE_LOW_WriteMask                               0xFFFFFFFF
#define AQVR_SOURCE_IMAGE_LOW_ResetValue                              0x00000000

#define AQVR_SOURCE_IMAGE_LOW_LEFT                                          15:0
#define AQVR_SOURCE_IMAGE_LOW_LEFT_End                                        15
#define AQVR_SOURCE_IMAGE_LOW_LEFT_Start                                       0
#define AQVR_SOURCE_IMAGE_LOW_LEFT_Type                                      U16

#define AQVR_SOURCE_IMAGE_LOW_TOP                                          31:16
#define AQVR_SOURCE_IMAGE_LOW_TOP_End                                         31
#define AQVR_SOURCE_IMAGE_LOW_TOP_Start                                       16
#define AQVR_SOURCE_IMAGE_LOW_TOP_Type                                       U16

#define AQVRSourceImageHighRegAddrs                                       0x04A7
#define AQVR_SOURCE_IMAGE_HIGH_Address                                   0x0129C
#define AQVR_SOURCE_IMAGE_HIGH_MSB                                            15
#define AQVR_SOURCE_IMAGE_HIGH_LSB                                             0
#define AQVR_SOURCE_IMAGE_LOW_HIGH_BLK                                         0
#define AQVR_SOURCE_IMAGE_HIGH_Count                                           1
#define AQVR_SOURCE_IMAGE_HIGH_FieldMask                              0xFFFFFFFF
#define AQVR_SOURCE_IMAGE_HIGH_ReadMask                               0xFFFFFFFF
#define AQVR_SOURCE_IMAGE_HIGH_WriteMask                              0xFFFFFFFF
#define AQVR_SOURCE_IMAGE_HIGH_ResetValue                             0x00000000

#define AQVR_SOURCE_IMAGE_HIGH_RIGHT                                        15:0
#define AQVR_SOURCE_IMAGE_HIGH_RIGHT_End                                      15
#define AQVR_SOURCE_IMAGE_HIGH_RIGHT_Start                                     0
#define AQVR_SOURCE_IMAGE_HIGH_RIGHT_Type                                    U16

#define AQVR_SOURCE_IMAGE_HIGH_BOTTOM                                      31:16
#define AQVR_SOURCE_IMAGE_HIGH_BOTTOM_End                                     31
#define AQVR_SOURCE_IMAGE_HIGH_BOTTOM_Start                                   16
#define AQVR_SOURCE_IMAGE_HIGH_BOTTOM_Type                                   U16

#define AQVRSourceOriginLowRegAddrs                                       0x04A8
#define AQVR_SOURCE_ORIGIN_LOW_Address                                   0x012A0
#define AQVR_SOURCE_ORIGIN_LOW_MSB                                            15
#define AQVR_SOURCE_ORIGIN_LOW_LSB                                             0
#define AQVR_SOURCE_ORIGIN_LOW_BLK                                             0
#define AQVR_SOURCE_ORIGIN_LOW_Count                                           1
#define AQVR_SOURCE_ORIGIN_LOW_FieldMask                              0xFFFFFFFF
#define AQVR_SOURCE_ORIGIN_LOW_ReadMask                               0xFFFFFFFF
#define AQVR_SOURCE_ORIGIN_LOW_WriteMask                              0xFFFFFFFF
#define AQVR_SOURCE_ORIGIN_LOW_ResetValue                             0x00000000

#define AQVR_SOURCE_ORIGIN_LOW_X                                            31:0
#define AQVR_SOURCE_ORIGIN_LOW_X_End                                          31
#define AQVR_SOURCE_ORIGIN_LOW_X_Start                                         0
#define AQVR_SOURCE_ORIGIN_LOW_X_Type                                        U32

#define AQVRSourceOriginHighRegAddrs                                      0x04A9
#define AQVR_SOURCE_ORIGIN_HIGH_Address                                  0x012A4
#define AQVR_SOURCE_ORIGIN_HIGH_MSB                                           15
#define AQVR_SOURCE_ORIGIN_HIGH_LSB                                            0
#define AQVR_SOURCE_ORIGIN_LOW_HIGH_BLK                                        0
#define AQVR_SOURCE_ORIGIN_HIGH_Count                                          1
#define AQVR_SOURCE_ORIGIN_HIGH_FieldMask                             0xFFFFFFFF
#define AQVR_SOURCE_ORIGIN_HIGH_ReadMask                              0xFFFFFFFF
#define AQVR_SOURCE_ORIGIN_HIGH_WriteMask                             0xFFFFFFFF
#define AQVR_SOURCE_ORIGIN_HIGH_ResetValue                            0x00000000

#define AQVR_SOURCE_ORIGIN_HIGH_Y                                           31:0
#define AQVR_SOURCE_ORIGIN_HIGH_Y_End                                         31
#define AQVR_SOURCE_ORIGIN_HIGH_Y_Start                                        0
#define AQVR_SOURCE_ORIGIN_HIGH_Y_Type                                       U32

#define AQVRTargetWindowLowRegAddrs                                       0x04AA
#define AQVR_TARGET_WINDOW_LOW_Address                                   0x012A8
#define AQVR_TARGET_WINDOW_LOW_MSB                                            15
#define AQVR_TARGET_WINDOW_LOW_LSB                                             0
#define AQVR_TARGET_WINDOW_LOW_BLK                                             0
#define AQVR_TARGET_WINDOW_LOW_Count                                           1
#define AQVR_TARGET_WINDOW_LOW_FieldMask                              0xFFFFFFFF
#define AQVR_TARGET_WINDOW_LOW_ReadMask                               0xFFFFFFFF
#define AQVR_TARGET_WINDOW_LOW_WriteMask                              0xFFFFFFFF
#define AQVR_TARGET_WINDOW_LOW_ResetValue                             0x00000000

#define AQVR_TARGET_WINDOW_LOW_LEFT                                         15:0
#define AQVR_TARGET_WINDOW_LOW_LEFT_End                                       15
#define AQVR_TARGET_WINDOW_LOW_LEFT_Start                                      0
#define AQVR_TARGET_WINDOW_LOW_LEFT_Type                                     U16

#define AQVR_TARGET_WINDOW_LOW_TOP                                         31:16
#define AQVR_TARGET_WINDOW_LOW_TOP_End                                        31
#define AQVR_TARGET_WINDOW_LOW_TOP_Start                                      16
#define AQVR_TARGET_WINDOW_LOW_TOP_Type                                      U16

#define AQVRTargetWindowHighRegAddrs                                      0x04AB
#define AQVR_TARGET_WINDOW_HIGH_Address                                  0x012AC
#define AQVR_TARGET_WINDOW_HIGH_MSB                                           15
#define AQVR_TARGET_WINDOW_HIGH_LSB                                            0
#define AQVR_TARGET_WINDOW_LOW_HIGH_BLK                                        0
#define AQVR_TARGET_WINDOW_HIGH_Count                                          1
#define AQVR_TARGET_WINDOW_HIGH_FieldMask                             0xFFFFFFFF
#define AQVR_TARGET_WINDOW_HIGH_ReadMask                              0xFFFFFFFF
#define AQVR_TARGET_WINDOW_HIGH_WriteMask                             0xFFFFFFFF
#define AQVR_TARGET_WINDOW_HIGH_ResetValue                            0x00000000

#define AQVR_TARGET_WINDOW_HIGH_RIGHT                                       15:0
#define AQVR_TARGET_WINDOW_HIGH_RIGHT_End                                     15
#define AQVR_TARGET_WINDOW_HIGH_RIGHT_Start                                    0
#define AQVR_TARGET_WINDOW_HIGH_RIGHT_Type                                   U16

#define AQVR_TARGET_WINDOW_HIGH_BOTTOM                                     31:16
#define AQVR_TARGET_WINDOW_HIGH_BOTTOM_End                                    31
#define AQVR_TARGET_WINDOW_HIGH_BOTTOM_Start                                  16
#define AQVR_TARGET_WINDOW_HIGH_BOTTOM_Type                                  U16

#define AQVRConfigRegAddrs                                                0x04A5
#define AQVR_CONFIG_Address                                              0x01294
#define AQVR_CONFIG_MSB                                                       15
#define AQVR_CONFIG_LSB                                                        0
#define AQVR_CONFIG_BLK                                                        0
#define AQVR_CONFIG_Count                                                      1
#define AQVR_CONFIG_FieldMask                                         0x0000000B
#define AQVR_CONFIG_ReadMask                                          0x0000000B
#define AQVR_CONFIG_WriteMask                                         0x0000000B
#define AQVR_CONFIG_ResetValue                                        0x00000000

#define AQVR_CONFIG_START                                                    1:0
#define AQVR_CONFIG_START_End                                                  1
#define AQVR_CONFIG_START_Start                                                0
#define AQVR_CONFIG_START_Type                                               U02
#define   AQVR_CONFIG_START_HORIZONTAL_BLIT                                  0x0
#define   AQVR_CONFIG_START_VERTICAL_BLIT                                    0x1
#define   AQVR_CONFIG_START_ONE_PASS_BLIT                                    0x2

#define AQVR_CONFIG_MASK_START                                               3:3
#define AQVR_CONFIG_MASK_START_End                                             3
#define AQVR_CONFIG_MASK_START_Start                                           3
#define AQVR_CONFIG_MASK_START_Type                                          U01
#define   AQVR_CONFIG_MASK_START_ENABLED                                     0x0
#define   AQVR_CONFIG_MASK_START_MASKED                                      0x1

#define AQPEConfigRegAddrs                                                0x04AC
#define AQPE_CONFIG_Address                                              0x012B0
#define AQPE_CONFIG_MSB                                                       15
#define AQPE_CONFIG_LSB                                                        0
#define AQPE_CONFIG_BLK                                                        0
#define AQPE_CONFIG_Count                                                      1
#define AQPE_CONFIG_FieldMask                                         0x000007FB
#define AQPE_CONFIG_ReadMask                                          0x000007FB
#define AQPE_CONFIG_WriteMask                                         0x000007FB
#define AQPE_CONFIG_ResetValue                                        0x00000000

#define AQPE_CONFIG_DESTINATION_FETCH                                        1:0
#define AQPE_CONFIG_DESTINATION_FETCH_End                                      1
#define AQPE_CONFIG_DESTINATION_FETCH_Start                                    0
#define AQPE_CONFIG_DESTINATION_FETCH_Type                                   U02
#define   AQPE_CONFIG_DESTINATION_FETCH_DISABLE                              0x0
#define   AQPE_CONFIG_DESTINATION_FETCH_DEFAULT                              0x1
#define   AQPE_CONFIG_DESTINATION_FETCH_ALWAYS                               0x2

#define AQPE_CONFIG_MASK_DESTINATION_FETCH                                   3:3
#define AQPE_CONFIG_MASK_DESTINATION_FETCH_End                                 3
#define AQPE_CONFIG_MASK_DESTINATION_FETCH_Start                               3
#define AQPE_CONFIG_MASK_DESTINATION_FETCH_Type                              U01
#define   AQPE_CONFIG_MASK_DESTINATION_FETCH_ENABLED                         0x0
#define   AQPE_CONFIG_MASK_DESTINATION_FETCH_MASKED                          0x1

#define AQPE_CONFIG_DST_FETCH_ENABLE                                         4:4
#define AQPE_CONFIG_DST_FETCH_ENABLE_End                                       4
#define AQPE_CONFIG_DST_FETCH_ENABLE_Start                                     4
#define AQPE_CONFIG_DST_FETCH_ENABLE_Type                                    U01
#define   AQPE_CONFIG_DST_FETCH_ENABLE_ENABLED                               0x0
#define   AQPE_CONFIG_DST_FETCH_ENABLE_DISABLED                              0x1

#define AQPE_CONFIG_MASK_DST_FETCH_ENABLE                                    5:5
#define AQPE_CONFIG_MASK_DST_FETCH_ENABLE_End                                  5
#define AQPE_CONFIG_MASK_DST_FETCH_ENABLE_Start                                5
#define AQPE_CONFIG_MASK_DST_FETCH_ENABLE_Type                               U01
#define   AQPE_CONFIG_MASK_DST_FETCH_ENABLE_ENABLED                          0x0
#define   AQPE_CONFIG_MASK_DST_FETCH_ENABLE_MASKED                           0x1

#define AQPE_CONFIG_PIPE                                                     7:6
#define AQPE_CONFIG_PIPE_End                                                   7
#define AQPE_CONFIG_PIPE_Start                                                 6
#define AQPE_CONFIG_PIPE_Type                                                U02
#define   AQPE_CONFIG_PIPE_AUTO                                              0x0
#define   AQPE_CONFIG_PIPE_BIT8                                              0x1
#define   AQPE_CONFIG_PIPE_BIT10                                             0x2

#define AQPE_CONFIG_MASK_PIPE                                                8:8
#define AQPE_CONFIG_MASK_PIPE_End                                              8
#define AQPE_CONFIG_MASK_PIPE_Start                                            8
#define AQPE_CONFIG_MASK_PIPE_Type                                           U01
#define   AQPE_CONFIG_MASK_PIPE_ENABLED                                      0x0
#define   AQPE_CONFIG_MASK_PIPE_MASKED                                       0x1

#define AQPE_CONFIG_MULTI_ID_DISABLED                                        9:9
#define AQPE_CONFIG_MULTI_ID_DISABLED_End                                      9
#define AQPE_CONFIG_MULTI_ID_DISABLED_Start                                    9
#define AQPE_CONFIG_MULTI_ID_DISABLED_Type                                   U01
#define   AQPE_CONFIG_MULTI_ID_DISABLED_DISABLED                             0x0
#define   AQPE_CONFIG_MULTI_ID_DISABLED_ENABLED                              0x1

#define AQPE_CONFIG_MASK_MULTI_ID_DISABLED                                 10:10
#define AQPE_CONFIG_MASK_MULTI_ID_DISABLED_End                                10
#define AQPE_CONFIG_MASK_MULTI_ID_DISABLED_Start                              10
#define AQPE_CONFIG_MASK_MULTI_ID_DISABLED_Type                              U01
#define   AQPE_CONFIG_MASK_MULTI_ID_DISABLED_ENABLED                         0x0
#define   AQPE_CONFIG_MASK_MULTI_ID_DISABLED_MASKED                          0x1

#define AQDEDstRotationHeightRegAddrs                                     0x04AD
#define AQDE_DST_ROTATION_HEIGHT_Address                                 0x012B4
#define AQDE_DST_ROTATION_HEIGHT_MSB                                          15
#define AQDE_DST_ROTATION_HEIGHT_LSB                                           0
#define AQDE_DST_ROTATION_HEIGHT_BLK                                           0
#define AQDE_DST_ROTATION_HEIGHT_Count                                         1
#define AQDE_DST_ROTATION_HEIGHT_FieldMask                            0x0000FFFF
#define AQDE_DST_ROTATION_HEIGHT_ReadMask                             0x0000FFFF
#define AQDE_DST_ROTATION_HEIGHT_WriteMask                            0x0000FFFF
#define AQDE_DST_ROTATION_HEIGHT_ResetValue                           0x00000000

#define AQDE_DST_ROTATION_HEIGHT_HEIGHT                                     15:0
#define AQDE_DST_ROTATION_HEIGHT_HEIGHT_End                                   15
#define AQDE_DST_ROTATION_HEIGHT_HEIGHT_Start                                  0
#define AQDE_DST_ROTATION_HEIGHT_HEIGHT_Type                                 U16

#define AQDESrcRotationHeightRegAddrs                                     0x04AE
#define AQDE_SRC_ROTATION_HEIGHT_Address                                 0x012B8
#define AQDE_SRC_ROTATION_HEIGHT_MSB                                          15
#define AQDE_SRC_ROTATION_HEIGHT_LSB                                           0
#define AQDE_SRC_ROTATION_HEIGHT_BLK                                           0
#define AQDE_SRC_ROTATION_HEIGHT_Count                                         1
#define AQDE_SRC_ROTATION_HEIGHT_FieldMask                            0x0000FFFF
#define AQDE_SRC_ROTATION_HEIGHT_ReadMask                             0x0000FFFF
#define AQDE_SRC_ROTATION_HEIGHT_WriteMask                            0x0000FFFF
#define AQDE_SRC_ROTATION_HEIGHT_ResetValue                           0x00000000

#define AQDE_SRC_ROTATION_HEIGHT_HEIGHT                                     15:0
#define AQDE_SRC_ROTATION_HEIGHT_HEIGHT_End                                   15
#define AQDE_SRC_ROTATION_HEIGHT_HEIGHT_Start                                  0
#define AQDE_SRC_ROTATION_HEIGHT_HEIGHT_Type                                 U16

#define AQDERotAngleRegAddrs                                              0x04AF
#define AQDE_ROT_ANGLE_Address                                           0x012BC
#define AQDE_ROT_ANGLE_MSB                                                    15
#define AQDE_ROT_ANGLE_LSB                                                     0
#define AQDE_ROT_ANGLE_BLK                                                     0
#define AQDE_ROT_ANGLE_Count                                                   1
#define AQDE_ROT_ANGLE_FieldMask                                      0x03FBB33F
#define AQDE_ROT_ANGLE_ReadMask                                       0x03FBB33F
#define AQDE_ROT_ANGLE_WriteMask                                      0x03FBB33F
#define AQDE_ROT_ANGLE_ResetValue                                     0x00000000

#define AQDE_ROT_ANGLE_SRC                                                   2:0
#define AQDE_ROT_ANGLE_SRC_End                                                 2
#define AQDE_ROT_ANGLE_SRC_Start                                               0
#define AQDE_ROT_ANGLE_SRC_Type                                              U03
#define   AQDE_ROT_ANGLE_SRC_ROT0                                            0x0
#define   AQDE_ROT_ANGLE_SRC_FLIP_X                                          0x1
#define   AQDE_ROT_ANGLE_SRC_FLIP_Y                                          0x2
#define   AQDE_ROT_ANGLE_SRC_ROT90                                           0x4
#define   AQDE_ROT_ANGLE_SRC_ROT180                                          0x5
#define   AQDE_ROT_ANGLE_SRC_ROT270                                          0x6

#define AQDE_ROT_ANGLE_DST                                                   5:3
#define AQDE_ROT_ANGLE_DST_End                                                 5
#define AQDE_ROT_ANGLE_DST_Start                                               3
#define AQDE_ROT_ANGLE_DST_Type                                              U03
#define   AQDE_ROT_ANGLE_DST_ROT0                                            0x0
#define   AQDE_ROT_ANGLE_DST_FLIP_X                                          0x1
#define   AQDE_ROT_ANGLE_DST_FLIP_Y                                          0x2
#define   AQDE_ROT_ANGLE_DST_ROT90                                           0x4
#define   AQDE_ROT_ANGLE_DST_ROT180                                          0x5
#define   AQDE_ROT_ANGLE_DST_ROT270                                          0x6

#define AQDE_ROT_ANGLE_MASK_SRC                                              8:8
#define AQDE_ROT_ANGLE_MASK_SRC_End                                            8
#define AQDE_ROT_ANGLE_MASK_SRC_Start                                          8
#define AQDE_ROT_ANGLE_MASK_SRC_Type                                         U01
#define   AQDE_ROT_ANGLE_MASK_SRC_ENABLED                                    0x0
#define   AQDE_ROT_ANGLE_MASK_SRC_MASKED                                     0x1

#define AQDE_ROT_ANGLE_MASK_DST                                              9:9
#define AQDE_ROT_ANGLE_MASK_DST_End                                            9
#define AQDE_ROT_ANGLE_MASK_DST_Start                                          9
#define AQDE_ROT_ANGLE_MASK_DST_Type                                         U01
#define   AQDE_ROT_ANGLE_MASK_DST_ENABLED                                    0x0
#define   AQDE_ROT_ANGLE_MASK_DST_MASKED                                     0x1

#define AQDE_ROT_ANGLE_SRC_MIRROR                                          13:12
#define AQDE_ROT_ANGLE_SRC_MIRROR_End                                         13
#define AQDE_ROT_ANGLE_SRC_MIRROR_Start                                       12
#define AQDE_ROT_ANGLE_SRC_MIRROR_Type                                       U02
#define   AQDE_ROT_ANGLE_SRC_MIRROR_NONE                                     0x0
#define   AQDE_ROT_ANGLE_SRC_MIRROR_MIRROR_X                                 0x1
#define   AQDE_ROT_ANGLE_SRC_MIRROR_MIRROR_Y                                 0x2
#define   AQDE_ROT_ANGLE_SRC_MIRROR_MIRROR_XY                                0x3

#define AQDE_ROT_ANGLE_MASK_SRC_MIRROR                                     15:15
#define AQDE_ROT_ANGLE_MASK_SRC_MIRROR_End                                    15
#define AQDE_ROT_ANGLE_MASK_SRC_MIRROR_Start                                  15
#define AQDE_ROT_ANGLE_MASK_SRC_MIRROR_Type                                  U01
#define   AQDE_ROT_ANGLE_MASK_SRC_MIRROR_ENABLED                             0x0
#define   AQDE_ROT_ANGLE_MASK_SRC_MIRROR_MASKED                              0x1

#define AQDE_ROT_ANGLE_DST_MIRROR                                          17:16
#define AQDE_ROT_ANGLE_DST_MIRROR_End                                         17
#define AQDE_ROT_ANGLE_DST_MIRROR_Start                                       16
#define AQDE_ROT_ANGLE_DST_MIRROR_Type                                       U02
#define   AQDE_ROT_ANGLE_DST_MIRROR_NONE                                     0x0
#define   AQDE_ROT_ANGLE_DST_MIRROR_MIRROR_X                                 0x1
#define   AQDE_ROT_ANGLE_DST_MIRROR_MIRROR_Y                                 0x2
#define   AQDE_ROT_ANGLE_DST_MIRROR_MIRROR_XY                                0x3

#define AQDE_ROT_ANGLE_MASK_DST_MIRROR                                     19:19
#define AQDE_ROT_ANGLE_MASK_DST_MIRROR_End                                    19
#define AQDE_ROT_ANGLE_MASK_DST_MIRROR_Start                                  19
#define AQDE_ROT_ANGLE_MASK_DST_MIRROR_Type                                  U01
#define   AQDE_ROT_ANGLE_MASK_DST_MIRROR_ENABLED                             0x0
#define   AQDE_ROT_ANGLE_MASK_DST_MIRROR_MASKED                              0x1

#define AQDE_ROT_ANGLE_SRC_FLIP                                            21:20
#define AQDE_ROT_ANGLE_SRC_FLIP_End                                           21
#define AQDE_ROT_ANGLE_SRC_FLIP_Start                                         20
#define AQDE_ROT_ANGLE_SRC_FLIP_Type                                         U02
#define   AQDE_ROT_ANGLE_SRC_FLIP_NONE                                       0x0
#define   AQDE_ROT_ANGLE_SRC_FLIP_FLIP_X                                     0x1
#define   AQDE_ROT_ANGLE_SRC_FLIP_FLIP_Y                                     0x2
#define   AQDE_ROT_ANGLE_SRC_FLIP_FLIP_XY                                    0x3

#define AQDE_ROT_ANGLE_MASK_SRC_FLIP                                       22:22
#define AQDE_ROT_ANGLE_MASK_SRC_FLIP_End                                      22
#define AQDE_ROT_ANGLE_MASK_SRC_FLIP_Start                                    22
#define AQDE_ROT_ANGLE_MASK_SRC_FLIP_Type                                    U01
#define   AQDE_ROT_ANGLE_MASK_SRC_FLIP_ENABLED                               0x0
#define   AQDE_ROT_ANGLE_MASK_SRC_FLIP_MASKED                                0x1

#define AQDE_ROT_ANGLE_DST_FLIP                                            24:23
#define AQDE_ROT_ANGLE_DST_FLIP_End                                           24
#define AQDE_ROT_ANGLE_DST_FLIP_Start                                         23
#define AQDE_ROT_ANGLE_DST_FLIP_Type                                         U02
#define   AQDE_ROT_ANGLE_DST_FLIP_NONE                                       0x0
#define   AQDE_ROT_ANGLE_DST_FLIP_FLIP_X                                     0x1
#define   AQDE_ROT_ANGLE_DST_FLIP_FLIP_Y                                     0x2
#define   AQDE_ROT_ANGLE_DST_FLIP_FLIP_XY                                    0x3

#define AQDE_ROT_ANGLE_MASK_DST_FLIP                                       25:25
#define AQDE_ROT_ANGLE_MASK_DST_FLIP_End                                      25
#define AQDE_ROT_ANGLE_MASK_DST_FLIP_Start                                    25
#define AQDE_ROT_ANGLE_MASK_DST_FLIP_Type                                    U01
#define   AQDE_ROT_ANGLE_MASK_DST_FLIP_ENABLED                               0x0
#define   AQDE_ROT_ANGLE_MASK_DST_FLIP_MASKED                                0x1

#define AQDEClearPixelValue32RegAddrs                                     0x04B0
#define AQDE_CLEAR_PIXEL_VALUE32_Address                                 0x012C0
#define AQDE_CLEAR_PIXEL_VALUE32_MSB                                          15
#define AQDE_CLEAR_PIXEL_VALUE32_LSB                                           0
#define AQDE_CLEAR_PIXEL_VALUE32_BLK                                           0
#define AQDE_CLEAR_PIXEL_VALUE32_Count                                         1
#define AQDE_CLEAR_PIXEL_VALUE32_FieldMask                            0xFFFFFFFF
#define AQDE_CLEAR_PIXEL_VALUE32_ReadMask                             0xFFFFFFFF
#define AQDE_CLEAR_PIXEL_VALUE32_WriteMask                            0xFFFFFFFF
#define AQDE_CLEAR_PIXEL_VALUE32_ResetValue                           0x00000000

#define AQDE_CLEAR_PIXEL_VALUE32_ALPHA                                     31:24
#define AQDE_CLEAR_PIXEL_VALUE32_ALPHA_End                                    31
#define AQDE_CLEAR_PIXEL_VALUE32_ALPHA_Start                                  24
#define AQDE_CLEAR_PIXEL_VALUE32_ALPHA_Type                                  U08

#define AQDE_CLEAR_PIXEL_VALUE32_RED                                       23:16
#define AQDE_CLEAR_PIXEL_VALUE32_RED_End                                      23
#define AQDE_CLEAR_PIXEL_VALUE32_RED_Start                                    16
#define AQDE_CLEAR_PIXEL_VALUE32_RED_Type                                    U08

#define AQDE_CLEAR_PIXEL_VALUE32_GREEN                                      15:8
#define AQDE_CLEAR_PIXEL_VALUE32_GREEN_End                                    15
#define AQDE_CLEAR_PIXEL_VALUE32_GREEN_Start                                   8
#define AQDE_CLEAR_PIXEL_VALUE32_GREEN_Type                                  U08

#define AQDE_CLEAR_PIXEL_VALUE32_BLUE                                        7:0
#define AQDE_CLEAR_PIXEL_VALUE32_BLUE_End                                      7
#define AQDE_CLEAR_PIXEL_VALUE32_BLUE_Start                                    0
#define AQDE_CLEAR_PIXEL_VALUE32_BLUE_Type                                   U08

#define AQDEDestColorKeyRegAddrs                                          0x04B1
#define AQDE_DEST_COLOR_KEY_Address                                      0x012C4
#define AQDE_DEST_COLOR_KEY_MSB                                               15
#define AQDE_DEST_COLOR_KEY_LSB                                                0
#define AQDE_DEST_COLOR_KEY_BLK                                                0
#define AQDE_DEST_COLOR_KEY_Count                                              1
#define AQDE_DEST_COLOR_KEY_FieldMask                                 0xFFFFFFFF
#define AQDE_DEST_COLOR_KEY_ReadMask                                  0xFFFFFFFF
#define AQDE_DEST_COLOR_KEY_WriteMask                                 0xFFFFFFFF
#define AQDE_DEST_COLOR_KEY_ResetValue                                0x00000000

#define AQDE_DEST_COLOR_KEY_ALPHA                                          31:24
#define AQDE_DEST_COLOR_KEY_ALPHA_End                                         31
#define AQDE_DEST_COLOR_KEY_ALPHA_Start                                       24
#define AQDE_DEST_COLOR_KEY_ALPHA_Type                                       U08

#define AQDE_DEST_COLOR_KEY_RED                                            23:16
#define AQDE_DEST_COLOR_KEY_RED_End                                           23
#define AQDE_DEST_COLOR_KEY_RED_Start                                         16
#define AQDE_DEST_COLOR_KEY_RED_Type                                         U08

#define AQDE_DEST_COLOR_KEY_GREEN                                           15:8
#define AQDE_DEST_COLOR_KEY_GREEN_End                                         15
#define AQDE_DEST_COLOR_KEY_GREEN_Start                                        8
#define AQDE_DEST_COLOR_KEY_GREEN_Type                                       U08

#define AQDE_DEST_COLOR_KEY_BLUE                                             7:0
#define AQDE_DEST_COLOR_KEY_BLUE_End                                           7
#define AQDE_DEST_COLOR_KEY_BLUE_Start                                         0
#define AQDE_DEST_COLOR_KEY_BLUE_Type                                        U08

#define AQDEGlobalSrcColorRegAddrs                                        0x04B2
#define AQDE_GLOBAL_SRC_COLOR_Address                                    0x012C8
#define AQDE_GLOBAL_SRC_COLOR_MSB                                             15
#define AQDE_GLOBAL_SRC_COLOR_LSB                                              0
#define AQDE_GLOBAL_SRC_COLOR_BLK                                              0
#define AQDE_GLOBAL_SRC_COLOR_Count                                            1
#define AQDE_GLOBAL_SRC_COLOR_FieldMask                               0xFFFFFFFF
#define AQDE_GLOBAL_SRC_COLOR_ReadMask                                0xFFFFFFFF
#define AQDE_GLOBAL_SRC_COLOR_WriteMask                               0xFFFFFFFF
#define AQDE_GLOBAL_SRC_COLOR_ResetValue                              0x00000000

#define AQDE_GLOBAL_SRC_COLOR_ALPHA                                        31:24
#define AQDE_GLOBAL_SRC_COLOR_ALPHA_End                                       31
#define AQDE_GLOBAL_SRC_COLOR_ALPHA_Start                                     24
#define AQDE_GLOBAL_SRC_COLOR_ALPHA_Type                                     U08

#define AQDE_GLOBAL_SRC_COLOR_RED                                          23:16
#define AQDE_GLOBAL_SRC_COLOR_RED_End                                         23
#define AQDE_GLOBAL_SRC_COLOR_RED_Start                                       16
#define AQDE_GLOBAL_SRC_COLOR_RED_Type                                       U08

#define AQDE_GLOBAL_SRC_COLOR_GREEN                                         15:8
#define AQDE_GLOBAL_SRC_COLOR_GREEN_End                                       15
#define AQDE_GLOBAL_SRC_COLOR_GREEN_Start                                      8
#define AQDE_GLOBAL_SRC_COLOR_GREEN_Type                                     U08

#define AQDE_GLOBAL_SRC_COLOR_BLUE                                           7:0
#define AQDE_GLOBAL_SRC_COLOR_BLUE_End                                         7
#define AQDE_GLOBAL_SRC_COLOR_BLUE_Start                                       0
#define AQDE_GLOBAL_SRC_COLOR_BLUE_Type                                      U08

#define AQDEGlobalDestColorRegAddrs                                       0x04B3
#define AQDE_GLOBAL_DEST_COLOR_Address                                   0x012CC
#define AQDE_GLOBAL_DEST_COLOR_MSB                                            15
#define AQDE_GLOBAL_DEST_COLOR_LSB                                             0
#define AQDE_GLOBAL_DEST_COLOR_BLK                                             0
#define AQDE_GLOBAL_DEST_COLOR_Count                                           1
#define AQDE_GLOBAL_DEST_COLOR_FieldMask                              0xFFFFFFFF
#define AQDE_GLOBAL_DEST_COLOR_ReadMask                               0xFFFFFFFF
#define AQDE_GLOBAL_DEST_COLOR_WriteMask                              0xFFFFFFFF
#define AQDE_GLOBAL_DEST_COLOR_ResetValue                             0x00000000

#define AQDE_GLOBAL_DEST_COLOR_ALPHA                                       31:24
#define AQDE_GLOBAL_DEST_COLOR_ALPHA_End                                      31
#define AQDE_GLOBAL_DEST_COLOR_ALPHA_Start                                    24
#define AQDE_GLOBAL_DEST_COLOR_ALPHA_Type                                    U08

#define AQDE_GLOBAL_DEST_COLOR_RED                                         23:16
#define AQDE_GLOBAL_DEST_COLOR_RED_End                                        23
#define AQDE_GLOBAL_DEST_COLOR_RED_Start                                      16
#define AQDE_GLOBAL_DEST_COLOR_RED_Type                                      U08

#define AQDE_GLOBAL_DEST_COLOR_GREEN                                        15:8
#define AQDE_GLOBAL_DEST_COLOR_GREEN_End                                      15
#define AQDE_GLOBAL_DEST_COLOR_GREEN_Start                                     8
#define AQDE_GLOBAL_DEST_COLOR_GREEN_Type                                    U08

#define AQDE_GLOBAL_DEST_COLOR_BLUE                                          7:0
#define AQDE_GLOBAL_DEST_COLOR_BLUE_End                                        7
#define AQDE_GLOBAL_DEST_COLOR_BLUE_Start                                      0
#define AQDE_GLOBAL_DEST_COLOR_BLUE_Type                                     U08

#define AQDEColorMultiplyModesRegAddrs                                    0x04B4
#define AQDE_COLOR_MULTIPLY_MODES_Address                                0x012D0
#define AQDE_COLOR_MULTIPLY_MODES_MSB                                         15
#define AQDE_COLOR_MULTIPLY_MODES_LSB                                          0
#define AQDE_COLOR_MULTIPLY_MODES_BLK                                          0
#define AQDE_COLOR_MULTIPLY_MODES_Count                                        1
#define AQDE_COLOR_MULTIPLY_MODES_FieldMask                           0x00300311
#define AQDE_COLOR_MULTIPLY_MODES_ReadMask                            0x00300311
#define AQDE_COLOR_MULTIPLY_MODES_WriteMask                           0x00300311
#define AQDE_COLOR_MULTIPLY_MODES_ResetValue                          0x00000000

#define AQDE_COLOR_MULTIPLY_MODES_SRC_PREMULTIPLY                            0:0
#define AQDE_COLOR_MULTIPLY_MODES_SRC_PREMULTIPLY_End                          0
#define AQDE_COLOR_MULTIPLY_MODES_SRC_PREMULTIPLY_Start                        0
#define AQDE_COLOR_MULTIPLY_MODES_SRC_PREMULTIPLY_Type                       U01
#define   AQDE_COLOR_MULTIPLY_MODES_SRC_PREMULTIPLY_DISABLE                  0x0
#define   AQDE_COLOR_MULTIPLY_MODES_SRC_PREMULTIPLY_ENABLE                   0x1

#define AQDE_COLOR_MULTIPLY_MODES_DST_PREMULTIPLY                            4:4
#define AQDE_COLOR_MULTIPLY_MODES_DST_PREMULTIPLY_End                          4
#define AQDE_COLOR_MULTIPLY_MODES_DST_PREMULTIPLY_Start                        4
#define AQDE_COLOR_MULTIPLY_MODES_DST_PREMULTIPLY_Type                       U01
#define   AQDE_COLOR_MULTIPLY_MODES_DST_PREMULTIPLY_DISABLE                  0x0
#define   AQDE_COLOR_MULTIPLY_MODES_DST_PREMULTIPLY_ENABLE                   0x1

#define AQDE_COLOR_MULTIPLY_MODES_SRC_GLOBAL_PREMULTIPLY                     9:8
#define AQDE_COLOR_MULTIPLY_MODES_SRC_GLOBAL_PREMULTIPLY_End                   9
#define AQDE_COLOR_MULTIPLY_MODES_SRC_GLOBAL_PREMULTIPLY_Start                 8
#define AQDE_COLOR_MULTIPLY_MODES_SRC_GLOBAL_PREMULTIPLY_Type                U02
#define   AQDE_COLOR_MULTIPLY_MODES_SRC_GLOBAL_PREMULTIPLY_DISABLE           0x0
#define   AQDE_COLOR_MULTIPLY_MODES_SRC_GLOBAL_PREMULTIPLY_ALPHA             0x1
#define   AQDE_COLOR_MULTIPLY_MODES_SRC_GLOBAL_PREMULTIPLY_COLOR             0x2

#define AQDE_COLOR_MULTIPLY_MODES_DST_DEMULTIPLY                           20:20
#define AQDE_COLOR_MULTIPLY_MODES_DST_DEMULTIPLY_End                          20
#define AQDE_COLOR_MULTIPLY_MODES_DST_DEMULTIPLY_Start                        20
#define AQDE_COLOR_MULTIPLY_MODES_DST_DEMULTIPLY_Type                        U01
#define   AQDE_COLOR_MULTIPLY_MODES_DST_DEMULTIPLY_DISABLE                   0x0
#define   AQDE_COLOR_MULTIPLY_MODES_DST_DEMULTIPLY_ENABLE                    0x1

#define AQDE_COLOR_MULTIPLY_MODES_DST_DEMULTIPLY_ZERO                      21:21
#define AQDE_COLOR_MULTIPLY_MODES_DST_DEMULTIPLY_ZERO_End                     21
#define AQDE_COLOR_MULTIPLY_MODES_DST_DEMULTIPLY_ZERO_Start                   21
#define AQDE_COLOR_MULTIPLY_MODES_DST_DEMULTIPLY_ZERO_Type                   U01
#define   AQDE_COLOR_MULTIPLY_MODES_DST_DEMULTIPLY_ZERO_DISABLE              0x0
#define   AQDE_COLOR_MULTIPLY_MODES_DST_DEMULTIPLY_ZERO_ENABLE               0x1

#define AQPETransparencyRegAddrs                                          0x04B5
#define AQPE_TRANSPARENCY_Address                                        0x012D4
#define AQPE_TRANSPARENCY_MSB                                                 15
#define AQPE_TRANSPARENCY_LSB                                                  0
#define AQPE_TRANSPARENCY_BLK                                                  0
#define AQPE_TRANSPARENCY_Count                                                1
#define AQPE_TRANSPARENCY_FieldMask                                   0xB3331333
#define AQPE_TRANSPARENCY_ReadMask                                    0xB3331333
#define AQPE_TRANSPARENCY_WriteMask                                   0xB3331333
#define AQPE_TRANSPARENCY_ResetValue                                  0x00000000

#define AQPE_TRANSPARENCY_SOURCE                                             1:0
#define AQPE_TRANSPARENCY_SOURCE_End                                           1
#define AQPE_TRANSPARENCY_SOURCE_Start                                         0
#define AQPE_TRANSPARENCY_SOURCE_Type                                        U02
#define   AQPE_TRANSPARENCY_SOURCE_OPAQUE                                    0x0
#define   AQPE_TRANSPARENCY_SOURCE_MASK                                      0x1
#define   AQPE_TRANSPARENCY_SOURCE_KEY                                       0x2

#define AQPE_TRANSPARENCY_PATTERN                                            5:4
#define AQPE_TRANSPARENCY_PATTERN_End                                          5
#define AQPE_TRANSPARENCY_PATTERN_Start                                        4
#define AQPE_TRANSPARENCY_PATTERN_Type                                       U02
#define   AQPE_TRANSPARENCY_PATTERN_OPAQUE                                   0x0
#define   AQPE_TRANSPARENCY_PATTERN_MASK                                     0x1
#define   AQPE_TRANSPARENCY_PATTERN_KEY                                      0x2

#define AQPE_TRANSPARENCY_DESTINATION                                        9:8
#define AQPE_TRANSPARENCY_DESTINATION_End                                      9
#define AQPE_TRANSPARENCY_DESTINATION_Start                                    8
#define AQPE_TRANSPARENCY_DESTINATION_Type                                   U02
#define   AQPE_TRANSPARENCY_DESTINATION_OPAQUE                               0x0
#define   AQPE_TRANSPARENCY_DESTINATION_MASK                                 0x1
#define   AQPE_TRANSPARENCY_DESTINATION_KEY                                  0x2

#define AQPE_TRANSPARENCY_MASK_TRANSPARENCY                                12:12
#define AQPE_TRANSPARENCY_MASK_TRANSPARENCY_End                               12
#define AQPE_TRANSPARENCY_MASK_TRANSPARENCY_Start                             12
#define AQPE_TRANSPARENCY_MASK_TRANSPARENCY_Type                             U01
#define   AQPE_TRANSPARENCY_MASK_TRANSPARENCY_ENABLED                        0x0
#define   AQPE_TRANSPARENCY_MASK_TRANSPARENCY_MASKED                         0x1

#define AQPE_TRANSPARENCY_USE_SRC_OVERRIDE                                 17:16
#define AQPE_TRANSPARENCY_USE_SRC_OVERRIDE_End                                17
#define AQPE_TRANSPARENCY_USE_SRC_OVERRIDE_Start                              16
#define AQPE_TRANSPARENCY_USE_SRC_OVERRIDE_Type                              U02
#define   AQPE_TRANSPARENCY_USE_SRC_OVERRIDE_DEFAULT                         0x0
#define   AQPE_TRANSPARENCY_USE_SRC_OVERRIDE_USE_ENABLE                      0x1
#define   AQPE_TRANSPARENCY_USE_SRC_OVERRIDE_USE_DISABLE                     0x2

#define AQPE_TRANSPARENCY_USE_PAT_OVERRIDE                                 21:20
#define AQPE_TRANSPARENCY_USE_PAT_OVERRIDE_End                                21
#define AQPE_TRANSPARENCY_USE_PAT_OVERRIDE_Start                              20
#define AQPE_TRANSPARENCY_USE_PAT_OVERRIDE_Type                              U02
#define   AQPE_TRANSPARENCY_USE_PAT_OVERRIDE_DEFAULT                         0x0
#define   AQPE_TRANSPARENCY_USE_PAT_OVERRIDE_USE_ENABLE                      0x1
#define   AQPE_TRANSPARENCY_USE_PAT_OVERRIDE_USE_DISABLE                     0x2

#define AQPE_TRANSPARENCY_USE_DST_OVERRIDE                                 25:24
#define AQPE_TRANSPARENCY_USE_DST_OVERRIDE_End                                25
#define AQPE_TRANSPARENCY_USE_DST_OVERRIDE_Start                              24
#define AQPE_TRANSPARENCY_USE_DST_OVERRIDE_Type                              U02
#define   AQPE_TRANSPARENCY_USE_DST_OVERRIDE_DEFAULT                         0x0
#define   AQPE_TRANSPARENCY_USE_DST_OVERRIDE_USE_ENABLE                      0x1
#define   AQPE_TRANSPARENCY_USE_DST_OVERRIDE_USE_DISABLE                     0x2

#define AQPE_TRANSPARENCY_MASK_RESOURCE_OVERRIDE                           28:28
#define AQPE_TRANSPARENCY_MASK_RESOURCE_OVERRIDE_End                          28
#define AQPE_TRANSPARENCY_MASK_RESOURCE_OVERRIDE_Start                        28
#define AQPE_TRANSPARENCY_MASK_RESOURCE_OVERRIDE_Type                        U01
#define   AQPE_TRANSPARENCY_MASK_RESOURCE_OVERRIDE_ENABLED                   0x0
#define   AQPE_TRANSPARENCY_MASK_RESOURCE_OVERRIDE_MASKED                    0x1

#define AQPE_TRANSPARENCY_DFB_COLOR_KEY                                    29:29
#define AQPE_TRANSPARENCY_DFB_COLOR_KEY_End                                   29
#define AQPE_TRANSPARENCY_DFB_COLOR_KEY_Start                                 29
#define AQPE_TRANSPARENCY_DFB_COLOR_KEY_Type                                 U01
#define   AQPE_TRANSPARENCY_DFB_COLOR_KEY_DISABLED                           0x0
#define   AQPE_TRANSPARENCY_DFB_COLOR_KEY_ENABLED                            0x1

#define AQPE_TRANSPARENCY_MASK_DFB_COLOR_KEY                               31:31
#define AQPE_TRANSPARENCY_MASK_DFB_COLOR_KEY_End                              31
#define AQPE_TRANSPARENCY_MASK_DFB_COLOR_KEY_Start                            31
#define AQPE_TRANSPARENCY_MASK_DFB_COLOR_KEY_Type                            U01
#define   AQPE_TRANSPARENCY_MASK_DFB_COLOR_KEY_ENABLED                       0x0
#define   AQPE_TRANSPARENCY_MASK_DFB_COLOR_KEY_MASKED                        0x1

#define AQPEControlRegAddrs                                               0x04B6
#define AQPE_CONTROL_Address                                             0x012D8
#define AQPE_CONTROL_MSB                                                      15
#define AQPE_CONTROL_LSB                                                       0
#define AQPE_CONTROL_BLK                                                       0
#define AQPE_CONTROL_Count                                                     1
#define AQPE_CONTROL_FieldMask                                        0x0009999B
#define AQPE_CONTROL_ReadMask                                         0x0009999B
#define AQPE_CONTROL_WriteMask                                        0x0009999B
#define AQPE_CONTROL_ResetValue                                       0x00000000

#define AQPE_CONTROL_YUV                                                     1:0
#define AQPE_CONTROL_YUV_End                                                   1
#define AQPE_CONTROL_YUV_Start                                                 0
#define AQPE_CONTROL_YUV_Type                                                U02
#define   AQPE_CONTROL_YUV_601                                               0x0
#define   AQPE_CONTROL_YUV_709                                               0x1
#define   AQPE_CONTROL_YUV_UserDefined                                       0x2
#define   AQPE_CONTROL_YUV_2020                                              0x3

#define AQPE_CONTROL_MASK_YUV                                                3:3
#define AQPE_CONTROL_MASK_YUV_End                                              3
#define AQPE_CONTROL_MASK_YUV_Start                                            3
#define AQPE_CONTROL_MASK_YUV_Type                                           U01
#define   AQPE_CONTROL_MASK_YUV_ENABLED                                      0x0
#define   AQPE_CONTROL_MASK_YUV_MASKED                                       0x1

#define AQPE_CONTROL_UV_SWIZZLE                                              4:4
#define AQPE_CONTROL_UV_SWIZZLE_End                                            4
#define AQPE_CONTROL_UV_SWIZZLE_Start                                          4
#define AQPE_CONTROL_UV_SWIZZLE_Type                                         U01
#define   AQPE_CONTROL_UV_SWIZZLE_UV                                         0x0
#define   AQPE_CONTROL_UV_SWIZZLE_VU                                         0x1

#define AQPE_CONTROL_MASK_UV_SWIZZLE                                         7:7
#define AQPE_CONTROL_MASK_UV_SWIZZLE_End                                       7
#define AQPE_CONTROL_MASK_UV_SWIZZLE_Start                                     7
#define AQPE_CONTROL_MASK_UV_SWIZZLE_Type                                    U01
#define   AQPE_CONTROL_MASK_UV_SWIZZLE_ENABLED                               0x0
#define   AQPE_CONTROL_MASK_UV_SWIZZLE_MASKED                                0x1

#define AQPE_CONTROL_YUVRGB                                                  8:8
#define AQPE_CONTROL_YUVRGB_End                                                8
#define AQPE_CONTROL_YUVRGB_Start                                              8
#define AQPE_CONTROL_YUVRGB_Type                                             U01
#define   AQPE_CONTROL_YUVRGB_DISABLED                                       0x0
#define   AQPE_CONTROL_YUVRGB_ENABLED                                        0x1

#define AQPE_CONTROL_MASK_YUVRGB                                           11:11
#define AQPE_CONTROL_MASK_YUVRGB_End                                          11
#define AQPE_CONTROL_MASK_YUVRGB_Start                                        11
#define AQPE_CONTROL_MASK_YUVRGB_Type                                        U01
#define   AQPE_CONTROL_MASK_YUVRGB_ENABLED                                   0x0
#define   AQPE_CONTROL_MASK_YUVRGB_MASKED                                    0x1

#define AQPE_CONTROL_YUV_CLAMP                                             12:12
#define AQPE_CONTROL_YUV_CLAMP_End                                            12
#define AQPE_CONTROL_YUV_CLAMP_Start                                          12
#define AQPE_CONTROL_YUV_CLAMP_Type                                          U01
#define   AQPE_CONTROL_YUV_CLAMP_DISABLED                                    0x0
#define   AQPE_CONTROL_YUV_CLAMP_ENABLED                                     0x1

#define AQPE_CONTROL_MASK_YUV_CLAMP                                        15:15
#define AQPE_CONTROL_MASK_YUV_CLAMP_End                                       15
#define AQPE_CONTROL_MASK_YUV_CLAMP_Start                                     15
#define AQPE_CONTROL_MASK_YUV_CLAMP_Type                                     U01
#define   AQPE_CONTROL_MASK_YUV_CLAMP_ENABLED                                0x0
#define   AQPE_CONTROL_MASK_YUV_CLAMP_MASKED                                 0x1

#define AQPE_CONTROL_GRAY_YUVRGB                                           16:16
#define AQPE_CONTROL_GRAY_YUVRGB_End                                          16
#define AQPE_CONTROL_GRAY_YUVRGB_Start                                        16
#define AQPE_CONTROL_GRAY_YUVRGB_Type                                        U01
#define   AQPE_CONTROL_GRAY_YUVRGB_DISABLED                                  0x0
#define   AQPE_CONTROL_GRAY_YUVRGB_ENABLED                                   0x1

#define AQPE_CONTROL_MASK_GRAY_YUVRGB                                      19:19
#define AQPE_CONTROL_MASK_GRAY_YUVRGB_End                                     19
#define AQPE_CONTROL_MASK_GRAY_YUVRGB_Start                                   19
#define AQPE_CONTROL_MASK_GRAY_YUVRGB_Type                                   U01
#define   AQPE_CONTROL_MASK_GRAY_YUVRGB_ENABLED                              0x0
#define   AQPE_CONTROL_MASK_GRAY_YUVRGB_MASKED                               0x1

#define AQDESrcColorKeyHighRegAddrs                                       0x04B7
#define AQDE_SRC_COLOR_KEY_HIGH_Address                                  0x012DC
#define AQDE_SRC_COLOR_KEY_HIGH_MSB                                           15
#define AQDE_SRC_COLOR_KEY_HIGH_LSB                                            0
#define AQDE_SRC_COLOR_KEY_HIGH_BLK                                            0
#define AQDE_SRC_COLOR_KEY_HIGH_Count                                          1
#define AQDE_SRC_COLOR_KEY_HIGH_FieldMask                             0xFFFFFFFF
#define AQDE_SRC_COLOR_KEY_HIGH_ReadMask                              0xFFFFFFFF
#define AQDE_SRC_COLOR_KEY_HIGH_WriteMask                             0xFFFFFFFF
#define AQDE_SRC_COLOR_KEY_HIGH_ResetValue                            0x00000000

#define AQDE_SRC_COLOR_KEY_HIGH_ALPHA                                      31:24
#define AQDE_SRC_COLOR_KEY_HIGH_ALPHA_End                                     31
#define AQDE_SRC_COLOR_KEY_HIGH_ALPHA_Start                                   24
#define AQDE_SRC_COLOR_KEY_HIGH_ALPHA_Type                                   U08

#define AQDE_SRC_COLOR_KEY_HIGH_RED                                        23:16
#define AQDE_SRC_COLOR_KEY_HIGH_RED_End                                       23
#define AQDE_SRC_COLOR_KEY_HIGH_RED_Start                                     16
#define AQDE_SRC_COLOR_KEY_HIGH_RED_Type                                     U08

#define AQDE_SRC_COLOR_KEY_HIGH_GREEN                                       15:8
#define AQDE_SRC_COLOR_KEY_HIGH_GREEN_End                                     15
#define AQDE_SRC_COLOR_KEY_HIGH_GREEN_Start                                    8
#define AQDE_SRC_COLOR_KEY_HIGH_GREEN_Type                                   U08

#define AQDE_SRC_COLOR_KEY_HIGH_BLUE                                         7:0
#define AQDE_SRC_COLOR_KEY_HIGH_BLUE_End                                       7
#define AQDE_SRC_COLOR_KEY_HIGH_BLUE_Start                                     0
#define AQDE_SRC_COLOR_KEY_HIGH_BLUE_Type                                    U08

#define AQDEDestColorKeyHighRegAddrs                                      0x04B8
#define AQDE_DEST_COLOR_KEY_HIGH_Address                                 0x012E0
#define AQDE_DEST_COLOR_KEY_HIGH_MSB                                          15
#define AQDE_DEST_COLOR_KEY_HIGH_LSB                                           0
#define AQDE_DEST_COLOR_KEY_HIGH_BLK                                           0
#define AQDE_DEST_COLOR_KEY_HIGH_Count                                         1
#define AQDE_DEST_COLOR_KEY_HIGH_FieldMask                            0xFFFFFFFF
#define AQDE_DEST_COLOR_KEY_HIGH_ReadMask                             0xFFFFFFFF
#define AQDE_DEST_COLOR_KEY_HIGH_WriteMask                            0xFFFFFFFF
#define AQDE_DEST_COLOR_KEY_HIGH_ResetValue                           0x00000000

#define AQDE_DEST_COLOR_KEY_HIGH_ALPHA                                     31:24
#define AQDE_DEST_COLOR_KEY_HIGH_ALPHA_End                                    31
#define AQDE_DEST_COLOR_KEY_HIGH_ALPHA_Start                                  24
#define AQDE_DEST_COLOR_KEY_HIGH_ALPHA_Type                                  U08

#define AQDE_DEST_COLOR_KEY_HIGH_RED                                       23:16
#define AQDE_DEST_COLOR_KEY_HIGH_RED_End                                      23
#define AQDE_DEST_COLOR_KEY_HIGH_RED_Start                                    16
#define AQDE_DEST_COLOR_KEY_HIGH_RED_Type                                    U08

#define AQDE_DEST_COLOR_KEY_HIGH_GREEN                                      15:8
#define AQDE_DEST_COLOR_KEY_HIGH_GREEN_End                                    15
#define AQDE_DEST_COLOR_KEY_HIGH_GREEN_Start                                   8
#define AQDE_DEST_COLOR_KEY_HIGH_GREEN_Type                                  U08

#define AQDE_DEST_COLOR_KEY_HIGH_BLUE                                        7:0
#define AQDE_DEST_COLOR_KEY_HIGH_BLUE_End                                      7
#define AQDE_DEST_COLOR_KEY_HIGH_BLUE_Start                                    0
#define AQDE_DEST_COLOR_KEY_HIGH_BLUE_Type                                   U08

#define AQVRConfigExRegAddrs                                              0x04B9
#define AQVR_CONFIG_EX_Address                                           0x012E4
#define AQVR_CONFIG_EX_MSB                                                    15
#define AQVR_CONFIG_EX_LSB                                                     0
#define AQVR_CONFIG_EX_BLK                                                     0
#define AQVR_CONFIG_EX_Count                                                   1
#define AQVR_CONFIG_EX_FieldMask                                      0x000FFFFB
#define AQVR_CONFIG_EX_ReadMask                                       0x000FFFFB
#define AQVR_CONFIG_EX_WriteMask                                      0x000FFFFB
#define AQVR_CONFIG_EX_ResetValue                                     0x00000000

#define AQVR_CONFIG_EX_VERTICAL_LINE_WIDTH                                   1:0
#define AQVR_CONFIG_EX_VERTICAL_LINE_WIDTH_End                                 1
#define AQVR_CONFIG_EX_VERTICAL_LINE_WIDTH_Start                               0
#define AQVR_CONFIG_EX_VERTICAL_LINE_WIDTH_Type                              U02
#define   AQVR_CONFIG_EX_VERTICAL_LINE_WIDTH_AUTO                            0x0
#define   AQVR_CONFIG_EX_VERTICAL_LINE_WIDTH_PIXELS16                        0x1
#define   AQVR_CONFIG_EX_VERTICAL_LINE_WIDTH_PIXELS32                        0x2

#define AQVR_CONFIG_EX_MASK_VERTICAL_LINE_WIDTH                              3:3
#define AQVR_CONFIG_EX_MASK_VERTICAL_LINE_WIDTH_End                            3
#define AQVR_CONFIG_EX_MASK_VERTICAL_LINE_WIDTH_Start                          3
#define AQVR_CONFIG_EX_MASK_VERTICAL_LINE_WIDTH_Type                         U01
#define   AQVR_CONFIG_EX_MASK_VERTICAL_LINE_WIDTH_ENABLED                    0x0
#define   AQVR_CONFIG_EX_MASK_VERTICAL_LINE_WIDTH_MASKED                     0x1

#define AQVR_CONFIG_EX_FILTER_TAP                                            7:4
#define AQVR_CONFIG_EX_FILTER_TAP_End                                          7
#define AQVR_CONFIG_EX_FILTER_TAP_Start                                        4
#define AQVR_CONFIG_EX_FILTER_TAP_Type                                       U04

#define AQVR_CONFIG_EX_MASK_FILTER_TAP                                       8:8
#define AQVR_CONFIG_EX_MASK_FILTER_TAP_End                                     8
#define AQVR_CONFIG_EX_MASK_FILTER_TAP_Start                                   8
#define AQVR_CONFIG_EX_MASK_FILTER_TAP_Type                                  U01
#define   AQVR_CONFIG_EX_MASK_FILTER_TAP_ENABLED                             0x0
#define   AQVR_CONFIG_EX_MASK_FILTER_TAP_MASKED                              0x1

#define AQVR_CONFIG_EX_DISABLE_QUAD_OPF                                      9:9
#define AQVR_CONFIG_EX_DISABLE_QUAD_OPF_End                                    9
#define AQVR_CONFIG_EX_DISABLE_QUAD_OPF_Start                                  9
#define AQVR_CONFIG_EX_DISABLE_QUAD_OPF_Type                                 U01
#define   AQVR_CONFIG_EX_DISABLE_QUAD_OPF_DISABLED                           0x0
#define   AQVR_CONFIG_EX_DISABLE_QUAD_OPF_ENABLED                            0x1

#define AQVR_CONFIG_EX_MASK_DISABLE_QUAD_OPF                               10:10
#define AQVR_CONFIG_EX_MASK_DISABLE_QUAD_OPF_End                              10
#define AQVR_CONFIG_EX_MASK_DISABLE_QUAD_OPF_Start                            10
#define AQVR_CONFIG_EX_MASK_DISABLE_QUAD_OPF_Type                            U01
#define   AQVR_CONFIG_EX_MASK_DISABLE_QUAD_OPF_ENABLED                       0x0
#define   AQVR_CONFIG_EX_MASK_DISABLE_QUAD_OPF_MASKED                        0x1

#define AQVR_CONFIG_EX_DISABLE_DUAL_PIPE_OPF                               11:11
#define AQVR_CONFIG_EX_DISABLE_DUAL_PIPE_OPF_End                              11
#define AQVR_CONFIG_EX_DISABLE_DUAL_PIPE_OPF_Start                            11
#define AQVR_CONFIG_EX_DISABLE_DUAL_PIPE_OPF_Type                            U01
#define   AQVR_CONFIG_EX_DISABLE_DUAL_PIPE_OPF_DISABLED                      0x0
#define   AQVR_CONFIG_EX_DISABLE_DUAL_PIPE_OPF_ENABLED                       0x1

#define AQVR_CONFIG_EX_MASK_DISABLE_DUAL_PIPE_OPF                          12:12
#define AQVR_CONFIG_EX_MASK_DISABLE_DUAL_PIPE_OPF_End                         12
#define AQVR_CONFIG_EX_MASK_DISABLE_DUAL_PIPE_OPF_Start                       12
#define AQVR_CONFIG_EX_MASK_DISABLE_DUAL_PIPE_OPF_Type                       U01
#define   AQVR_CONFIG_EX_MASK_DISABLE_DUAL_PIPE_OPF_ENABLED                  0x0
#define   AQVR_CONFIG_EX_MASK_DISABLE_DUAL_PIPE_OPF_MASKED                   0x1

#define AQVR_CONFIG_EX_ENABLE_HORIZONTAL_FILTER_TAP                        13:13
#define AQVR_CONFIG_EX_ENABLE_HORIZONTAL_FILTER_TAP_End                       13
#define AQVR_CONFIG_EX_ENABLE_HORIZONTAL_FILTER_TAP_Start                     13
#define AQVR_CONFIG_EX_ENABLE_HORIZONTAL_FILTER_TAP_Type                     U01
#define   AQVR_CONFIG_EX_ENABLE_HORIZONTAL_FILTER_TAP_DISABLED               0x0
#define   AQVR_CONFIG_EX_ENABLE_HORIZONTAL_FILTER_TAP_ENABLED                0x1

#define AQVR_CONFIG_EX_MASK_ENABLE_HORIZONTAL_FILTER_TAP                   14:14
#define AQVR_CONFIG_EX_MASK_ENABLE_HORIZONTAL_FILTER_TAP_End                  14
#define AQVR_CONFIG_EX_MASK_ENABLE_HORIZONTAL_FILTER_TAP_Start                14
#define AQVR_CONFIG_EX_MASK_ENABLE_HORIZONTAL_FILTER_TAP_Type                U01
#define   AQVR_CONFIG_EX_MASK_ENABLE_HORIZONTAL_FILTER_TAP_ENABLED           0x0
#define   AQVR_CONFIG_EX_MASK_ENABLE_HORIZONTAL_FILTER_TAP_MASKED            0x1

#define AQVR_CONFIG_EX_HORIZONTAL_FILTER_TAP                               18:15
#define AQVR_CONFIG_EX_HORIZONTAL_FILTER_TAP_End                              18
#define AQVR_CONFIG_EX_HORIZONTAL_FILTER_TAP_Start                            15
#define AQVR_CONFIG_EX_HORIZONTAL_FILTER_TAP_Type                            U04

#define AQVR_CONFIG_EX_MASK_HORIZONTAL_FILTER_TAP                          19:19
#define AQVR_CONFIG_EX_MASK_HORIZONTAL_FILTER_TAP_End                         19
#define AQVR_CONFIG_EX_MASK_HORIZONTAL_FILTER_TAP_Start                       19
#define AQVR_CONFIG_EX_MASK_HORIZONTAL_FILTER_TAP_Type                       U01
#define   AQVR_CONFIG_EX_MASK_HORIZONTAL_FILTER_TAP_ENABLED                  0x0
#define   AQVR_CONFIG_EX_MASK_HORIZONTAL_FILTER_TAP_MASKED                   0x1

#define AQPEDitherLowRegAddrs                                             0x04BA
#define AQPE_DITHER_LOW_Address                                          0x012E8
#define AQPE_DITHER_LOW_MSB                                                   15
#define AQPE_DITHER_LOW_LSB                                                    0
#define AQPE_DITHER_LOW_BLK                                                    0
#define AQPE_DITHER_LOW_Count                                                  1
#define AQPE_DITHER_LOW_FieldMask                                     0xFFFFFFFF
#define AQPE_DITHER_LOW_ReadMask                                      0xFFFFFFFF
#define AQPE_DITHER_LOW_WriteMask                                     0xFFFFFFFF
#define AQPE_DITHER_LOW_ResetValue                                    0xFFFFFFFF

#define AQPE_DITHER_LOW_PIXEL_X0_Y0                                          3:0
#define AQPE_DITHER_LOW_PIXEL_X0_Y0_End                                        3
#define AQPE_DITHER_LOW_PIXEL_X0_Y0_Start                                      0
#define AQPE_DITHER_LOW_PIXEL_X0_Y0_Type                                     U04

#define AQPE_DITHER_LOW_PIXEL_X1_Y0                                          7:4
#define AQPE_DITHER_LOW_PIXEL_X1_Y0_End                                        7
#define AQPE_DITHER_LOW_PIXEL_X1_Y0_Start                                      4
#define AQPE_DITHER_LOW_PIXEL_X1_Y0_Type                                     U04

#define AQPE_DITHER_LOW_PIXEL_X2_Y0                                         11:8
#define AQPE_DITHER_LOW_PIXEL_X2_Y0_End                                       11
#define AQPE_DITHER_LOW_PIXEL_X2_Y0_Start                                      8
#define AQPE_DITHER_LOW_PIXEL_X2_Y0_Type                                     U04

#define AQPE_DITHER_LOW_PIXEL_X3_Y0                                        15:12
#define AQPE_DITHER_LOW_PIXEL_X3_Y0_End                                       15
#define AQPE_DITHER_LOW_PIXEL_X3_Y0_Start                                     12
#define AQPE_DITHER_LOW_PIXEL_X3_Y0_Type                                     U04

#define AQPE_DITHER_LOW_PIXEL_X0_Y1                                        19:16
#define AQPE_DITHER_LOW_PIXEL_X0_Y1_End                                       19
#define AQPE_DITHER_LOW_PIXEL_X0_Y1_Start                                     16
#define AQPE_DITHER_LOW_PIXEL_X0_Y1_Type                                     U04

#define AQPE_DITHER_LOW_PIXEL_X1_Y1                                        23:20
#define AQPE_DITHER_LOW_PIXEL_X1_Y1_End                                       23
#define AQPE_DITHER_LOW_PIXEL_X1_Y1_Start                                     20
#define AQPE_DITHER_LOW_PIXEL_X1_Y1_Type                                     U04

#define AQPE_DITHER_LOW_PIXEL_X2_Y1                                        27:24
#define AQPE_DITHER_LOW_PIXEL_X2_Y1_End                                       27
#define AQPE_DITHER_LOW_PIXEL_X2_Y1_Start                                     24
#define AQPE_DITHER_LOW_PIXEL_X2_Y1_Type                                     U04

#define AQPE_DITHER_LOW_PIXEL_X3_Y1                                        31:28
#define AQPE_DITHER_LOW_PIXEL_X3_Y1_End                                       31
#define AQPE_DITHER_LOW_PIXEL_X3_Y1_Start                                     28
#define AQPE_DITHER_LOW_PIXEL_X3_Y1_Type                                     U04

#define AQPEDitherHighRegAddrs                                            0x04BB
#define AQPE_DITHER_HIGH_Address                                         0x012EC
#define AQPE_DITHER_HIGH_MSB                                                  15
#define AQPE_DITHER_HIGH_LSB                                                   0
#define AQPE_DITHER_LOW_HIGH_BLK                                               0
#define AQPE_DITHER_HIGH_Count                                                 1
#define AQPE_DITHER_HIGH_FieldMask                                    0xFFFFFFFF
#define AQPE_DITHER_HIGH_ReadMask                                     0xFFFFFFFF
#define AQPE_DITHER_HIGH_WriteMask                                    0xFFFFFFFF
#define AQPE_DITHER_HIGH_ResetValue                                   0xFFFFFFFF

#define AQPE_DITHER_HIGH_PIXEL_X0_Y2                                         3:0
#define AQPE_DITHER_HIGH_PIXEL_X0_Y2_End                                       3
#define AQPE_DITHER_HIGH_PIXEL_X0_Y2_Start                                     0
#define AQPE_DITHER_HIGH_PIXEL_X0_Y2_Type                                    U04

#define AQPE_DITHER_HIGH_PIXEL_X1_Y2                                         7:4
#define AQPE_DITHER_HIGH_PIXEL_X1_Y2_End                                       7
#define AQPE_DITHER_HIGH_PIXEL_X1_Y2_Start                                     4
#define AQPE_DITHER_HIGH_PIXEL_X1_Y2_Type                                    U04

#define AQPE_DITHER_HIGH_PIXEL_X2_Y2                                        11:8
#define AQPE_DITHER_HIGH_PIXEL_X2_Y2_End                                      11
#define AQPE_DITHER_HIGH_PIXEL_X2_Y2_Start                                     8
#define AQPE_DITHER_HIGH_PIXEL_X2_Y2_Type                                    U04

#define AQPE_DITHER_HIGH_PIXEL_X3_Y2                                       15:12
#define AQPE_DITHER_HIGH_PIXEL_X3_Y2_End                                      15
#define AQPE_DITHER_HIGH_PIXEL_X3_Y2_Start                                    12
#define AQPE_DITHER_HIGH_PIXEL_X3_Y2_Type                                    U04

#define AQPE_DITHER_HIGH_PIXEL_X0_Y3                                       19:16
#define AQPE_DITHER_HIGH_PIXEL_X0_Y3_End                                      19
#define AQPE_DITHER_HIGH_PIXEL_X0_Y3_Start                                    16
#define AQPE_DITHER_HIGH_PIXEL_X0_Y3_Type                                    U04

#define AQPE_DITHER_HIGH_PIXEL_X1_Y3                                       23:20
#define AQPE_DITHER_HIGH_PIXEL_X1_Y3_End                                      23
#define AQPE_DITHER_HIGH_PIXEL_X1_Y3_Start                                    20
#define AQPE_DITHER_HIGH_PIXEL_X1_Y3_Type                                    U04

#define AQPE_DITHER_HIGH_PIXEL_X2_Y3                                       27:24
#define AQPE_DITHER_HIGH_PIXEL_X2_Y3_End                                      27
#define AQPE_DITHER_HIGH_PIXEL_X2_Y3_Start                                    24
#define AQPE_DITHER_HIGH_PIXEL_X2_Y3_Type                                    U04

#define AQPE_DITHER_HIGH_PIXEL_X3_Y3                                       31:28
#define AQPE_DITHER_HIGH_PIXEL_X3_Y3_End                                      31
#define AQPE_DITHER_HIGH_PIXEL_X3_Y3_Start                                    28
#define AQPE_DITHER_HIGH_PIXEL_X3_Y3_Type                                    U04

#define AQDEHoriFilterKernelRegAddrs                                      0x0A00
#define AQDE_HORI_FILTER_KERNEL_Address                                  0x02800
#define AQDE_HORI_FILTER_KERNEL_MSB                                           15
#define AQDE_HORI_FILTER_KERNEL_LSB                                            7
#define AQDE_HORI_FILTER_KERNEL_BLK                                            7
#define AQDE_HORI_FILTER_KERNEL_Count                                        128
#define AQDE_HORI_FILTER_KERNEL_FieldMask                             0xFFFFFFFF
#define AQDE_HORI_FILTER_KERNEL_ReadMask                              0xFFFFFFFF
#define AQDE_HORI_FILTER_KERNEL_WriteMask                             0xFFFFFFFF
#define AQDE_HORI_FILTER_KERNEL_ResetValue                            0x00000000

#define AQDE_HORI_FILTER_KERNEL_COEFFICIENT0                                15:0
#define AQDE_HORI_FILTER_KERNEL_COEFFICIENT0_End                              15
#define AQDE_HORI_FILTER_KERNEL_COEFFICIENT0_Start                             0
#define AQDE_HORI_FILTER_KERNEL_COEFFICIENT0_Type                            U16

#define AQDE_HORI_FILTER_KERNEL_COEFFICIENT1                               31:16
#define AQDE_HORI_FILTER_KERNEL_COEFFICIENT1_End                              31
#define AQDE_HORI_FILTER_KERNEL_COEFFICIENT1_Start                            16
#define AQDE_HORI_FILTER_KERNEL_COEFFICIENT1_Type                            U16

#define AQDEVertiFilterKernelRegAddrs                                     0x0A80
#define AQDE_VERTI_FILTER_KERNEL_Address                                 0x02A00
#define AQDE_VERTI_FILTER_KERNEL_MSB                                          15
#define AQDE_VERTI_FILTER_KERNEL_LSB                                           7
#define AQDE_VERTI_FILTER_KERNEL_BLK                                           7
#define AQDE_VERTI_FILTER_KERNEL_Count                                       128
#define AQDE_VERTI_FILTER_KERNEL_FieldMask                            0xFFFFFFFF
#define AQDE_VERTI_FILTER_KERNEL_ReadMask                             0xFFFFFFFF
#define AQDE_VERTI_FILTER_KERNEL_WriteMask                            0xFFFFFFFF
#define AQDE_VERTI_FILTER_KERNEL_ResetValue                           0x00000000

#define AQDE_VERTI_FILTER_KERNEL_COEFFICIENT0                               15:0
#define AQDE_VERTI_FILTER_KERNEL_COEFFICIENT0_End                             15
#define AQDE_VERTI_FILTER_KERNEL_COEFFICIENT0_Start                            0
#define AQDE_VERTI_FILTER_KERNEL_COEFFICIENT0_Type                           U16

#define AQDE_VERTI_FILTER_KERNEL_COEFFICIENT1                              31:16
#define AQDE_VERTI_FILTER_KERNEL_COEFFICIENT1_End                             31
#define AQDE_VERTI_FILTER_KERNEL_COEFFICIENT1_Start                           16
#define AQDE_VERTI_FILTER_KERNEL_COEFFICIENT1_Type                           U16

#define AQBWConfigRegAddrs                                                0x04BC
#define AQBW_CONFIG_Address                                              0x012F0
#define AQBW_CONFIG_MSB                                                       15
#define AQBW_CONFIG_LSB                                                        0
#define AQBW_CONFIG_BLK                                                        0
#define AQBW_CONFIG_Count                                                      1
#define AQBW_CONFIG_FieldMask                                         0x00099999
#define AQBW_CONFIG_ReadMask                                          0x00099999
#define AQBW_CONFIG_WriteMask                                         0x00099999
#define AQBW_CONFIG_ResetValue                                        0x00000000

#define AQBW_CONFIG_BLOCK_CONFIG                                             0:0
#define AQBW_CONFIG_BLOCK_CONFIG_End                                           0
#define AQBW_CONFIG_BLOCK_CONFIG_Start                                         0
#define AQBW_CONFIG_BLOCK_CONFIG_Type                                        U01
#define   AQBW_CONFIG_BLOCK_CONFIG_AUTO                                      0x0
#define   AQBW_CONFIG_BLOCK_CONFIG_CUSTOMIZE                                 0x1

#define AQBW_CONFIG_MASK_BLOCK_CONFIG                                        3:3
#define AQBW_CONFIG_MASK_BLOCK_CONFIG_End                                      3
#define AQBW_CONFIG_MASK_BLOCK_CONFIG_Start                                    3
#define AQBW_CONFIG_MASK_BLOCK_CONFIG_Type                                   U01
#define   AQBW_CONFIG_MASK_BLOCK_CONFIG_ENABLED                              0x0
#define   AQBW_CONFIG_MASK_BLOCK_CONFIG_MASKED                               0x1

#define AQBW_CONFIG_BLOCK_WALK_DIRECTION                                     4:4
#define AQBW_CONFIG_BLOCK_WALK_DIRECTION_End                                   4
#define AQBW_CONFIG_BLOCK_WALK_DIRECTION_Start                                 4
#define AQBW_CONFIG_BLOCK_WALK_DIRECTION_Type                                U01
#define   AQBW_CONFIG_BLOCK_WALK_DIRECTION_RIGHT_BOTTOM                      0x0
#define   AQBW_CONFIG_BLOCK_WALK_DIRECTION_BOTTOM_RIGHT                      0x1

#define AQBW_CONFIG_MASK_BLOCK_WALK_DIRECTION                                7:7
#define AQBW_CONFIG_MASK_BLOCK_WALK_DIRECTION_End                              7
#define AQBW_CONFIG_MASK_BLOCK_WALK_DIRECTION_Start                            7
#define AQBW_CONFIG_MASK_BLOCK_WALK_DIRECTION_Type                           U01
#define   AQBW_CONFIG_MASK_BLOCK_WALK_DIRECTION_ENABLED                      0x0
#define   AQBW_CONFIG_MASK_BLOCK_WALK_DIRECTION_MASKED                       0x1

#define AQBW_CONFIG_TILE_WALK_DIRECTION                                      8:8
#define AQBW_CONFIG_TILE_WALK_DIRECTION_End                                    8
#define AQBW_CONFIG_TILE_WALK_DIRECTION_Start                                  8
#define AQBW_CONFIG_TILE_WALK_DIRECTION_Type                                 U01
#define   AQBW_CONFIG_TILE_WALK_DIRECTION_RIGHT_BOTTOM                       0x0
#define   AQBW_CONFIG_TILE_WALK_DIRECTION_BOTTOM_RIGHT                       0x1

#define AQBW_CONFIG_MASK_TILE_WALK_DIRECTION                               11:11
#define AQBW_CONFIG_MASK_TILE_WALK_DIRECTION_End                              11
#define AQBW_CONFIG_MASK_TILE_WALK_DIRECTION_Start                            11
#define AQBW_CONFIG_MASK_TILE_WALK_DIRECTION_Type                            U01
#define   AQBW_CONFIG_MASK_TILE_WALK_DIRECTION_ENABLED                       0x0
#define   AQBW_CONFIG_MASK_TILE_WALK_DIRECTION_MASKED                        0x1

#define AQBW_CONFIG_PIXEL_WALK_DIRECTION                                   12:12
#define AQBW_CONFIG_PIXEL_WALK_DIRECTION_End                                  12
#define AQBW_CONFIG_PIXEL_WALK_DIRECTION_Start                                12
#define AQBW_CONFIG_PIXEL_WALK_DIRECTION_Type                                U01
#define   AQBW_CONFIG_PIXEL_WALK_DIRECTION_RIGHT_BOTTOM                      0x0
#define   AQBW_CONFIG_PIXEL_WALK_DIRECTION_BOTTOM_RIGHT                      0x1

#define AQBW_CONFIG_MASK_PIXEL_WALK_DIRECTION                              15:15
#define AQBW_CONFIG_MASK_PIXEL_WALK_DIRECTION_End                             15
#define AQBW_CONFIG_MASK_PIXEL_WALK_DIRECTION_Start                           15
#define AQBW_CONFIG_MASK_PIXEL_WALK_DIRECTION_Type                           U01
#define   AQBW_CONFIG_MASK_PIXEL_WALK_DIRECTION_ENABLED                      0x0
#define   AQBW_CONFIG_MASK_PIXEL_WALK_DIRECTION_MASKED                       0x1

#define AQBW_CONFIG_LINEAR_ROTATION_OPT                                    16:16
#define AQBW_CONFIG_LINEAR_ROTATION_OPT_End                                   16
#define AQBW_CONFIG_LINEAR_ROTATION_OPT_Start                                 16
#define AQBW_CONFIG_LINEAR_ROTATION_OPT_Type                                 U01
#define   AQBW_CONFIG_LINEAR_ROTATION_OPT_DISABLED                           0x0
#define   AQBW_CONFIG_LINEAR_ROTATION_OPT_ENABLED                            0x1

#define AQBW_CONFIG_MASK_LINEAR_ROTATION_OPT                               19:19
#define AQBW_CONFIG_MASK_LINEAR_ROTATION_OPT_End                              19
#define AQBW_CONFIG_MASK_LINEAR_ROTATION_OPT_Start                            19
#define AQBW_CONFIG_MASK_LINEAR_ROTATION_OPT_Type                            U01
#define   AQBW_CONFIG_MASK_LINEAR_ROTATION_OPT_ENABLED                       0x0
#define   AQBW_CONFIG_MASK_LINEAR_ROTATION_OPT_MASKED                        0x1

#define AQBWBlockSizeRegAddrs                                             0x04BD
#define AQBW_BLOCK_SIZE_Address                                          0x012F4
#define AQBW_BLOCK_SIZE_MSB                                                   15
#define AQBW_BLOCK_SIZE_LSB                                                    0
#define AQBW_BLOCK_SIZE_BLK                                                    0
#define AQBW_BLOCK_SIZE_Count                                                  1
#define AQBW_BLOCK_SIZE_FieldMask                                     0xFFFFFFFF
#define AQBW_BLOCK_SIZE_ReadMask                                      0xFFFFFFFF
#define AQBW_BLOCK_SIZE_WriteMask                                     0xFFFFFFFF
#define AQBW_BLOCK_SIZE_ResetValue                                    0x00000000

#define AQBW_BLOCK_SIZE_WIDTH                                               15:0
#define AQBW_BLOCK_SIZE_WIDTH_End                                             15
#define AQBW_BLOCK_SIZE_WIDTH_Start                                            0
#define AQBW_BLOCK_SIZE_WIDTH_Type                                           U16

#define AQBW_BLOCK_SIZE_HEIGHT                                             31:16
#define AQBW_BLOCK_SIZE_HEIGHT_End                                            31
#define AQBW_BLOCK_SIZE_HEIGHT_Start                                          16
#define AQBW_BLOCK_SIZE_HEIGHT_Type                                          U16

#define AQBWTileSizeRegAddrs                                              0x04BE
#define AQBW_TILE_SIZE_Address                                           0x012F8
#define AQBW_TILE_SIZE_MSB                                                    15
#define AQBW_TILE_SIZE_LSB                                                     0
#define AQBW_TILE_SIZE_BLK                                                     0
#define AQBW_TILE_SIZE_Count                                                   1
#define AQBW_TILE_SIZE_FieldMask                                      0xFFFFFFFF
#define AQBW_TILE_SIZE_ReadMask                                       0xFFFFFFFF
#define AQBW_TILE_SIZE_WriteMask                                      0xFFFFFFFF
#define AQBW_TILE_SIZE_ResetValue                                     0x00000000

#define AQBW_TILE_SIZE_WIDTH                                                15:0
#define AQBW_TILE_SIZE_WIDTH_End                                              15
#define AQBW_TILE_SIZE_WIDTH_Start                                             0
#define AQBW_TILE_SIZE_WIDTH_Type                                            U16

#define AQBW_TILE_SIZE_HEIGHT                                              31:16
#define AQBW_TILE_SIZE_HEIGHT_End                                             31
#define AQBW_TILE_SIZE_HEIGHT_Start                                           16
#define AQBW_TILE_SIZE_HEIGHT_Type                                           U16

#define AQBWBlockMaskRegAddrs                                             0x04BF
#define AQBW_BLOCK_MASK_Address                                          0x012FC
#define AQBW_BLOCK_MASK_MSB                                                   15
#define AQBW_BLOCK_MASK_LSB                                                    0
#define AQBW_BLOCK_MASK_BLK                                                    0
#define AQBW_BLOCK_MASK_Count                                                  1
#define AQBW_BLOCK_MASK_FieldMask                                     0xFFFFFFFF
#define AQBW_BLOCK_MASK_ReadMask                                      0xFFFFFFFF
#define AQBW_BLOCK_MASK_WriteMask                                     0xFFFFFFFF
#define AQBW_BLOCK_MASK_ResetValue                                    0x00000000

#define AQBW_BLOCK_MASK_HORIZONTAL                                          15:0
#define AQBW_BLOCK_MASK_HORIZONTAL_End                                        15
#define AQBW_BLOCK_MASK_HORIZONTAL_Start                                       0
#define AQBW_BLOCK_MASK_HORIZONTAL_Type                                      U16

#define AQBW_BLOCK_MASK_VERTICAL                                           31:16
#define AQBW_BLOCK_MASK_VERTICAL_End                                          31
#define AQBW_BLOCK_MASK_VERTICAL_Start                                        16
#define AQBW_BLOCK_MASK_VERTICAL_Type                                        U16

#define AQDESrcExConfigRegAddrs                                           0x04C0
#define AQDE_SRC_EX_CONFIG_Address                                       0x01300
#define AQDE_SRC_EX_CONFIG_MSB                                                15
#define AQDE_SRC_EX_CONFIG_LSB                                                 0
#define AQDE_SRC_EX_CONFIG_BLK                                                 0
#define AQDE_SRC_EX_CONFIG_Count                                               1
#define AQDE_SRC_EX_CONFIG_FieldMask                                  0x0001F109
#define AQDE_SRC_EX_CONFIG_ReadMask                                   0x0001F109
#define AQDE_SRC_EX_CONFIG_WriteMask                                  0x0001F109
#define AQDE_SRC_EX_CONFIG_ResetValue                                 0x00000000

#define AQDE_SRC_EX_CONFIG_MULTI_TILED                                       0:0
#define AQDE_SRC_EX_CONFIG_MULTI_TILED_End                                     0
#define AQDE_SRC_EX_CONFIG_MULTI_TILED_Start                                   0
#define AQDE_SRC_EX_CONFIG_MULTI_TILED_Type                                  U01
#define   AQDE_SRC_EX_CONFIG_MULTI_TILED_DISABLED                            0x0
#define   AQDE_SRC_EX_CONFIG_MULTI_TILED_ENABLED                             0x1

#define AQDE_SRC_EX_CONFIG_SUPER_TILED                                       3:3
#define AQDE_SRC_EX_CONFIG_SUPER_TILED_End                                     3
#define AQDE_SRC_EX_CONFIG_SUPER_TILED_Start                                   3
#define AQDE_SRC_EX_CONFIG_SUPER_TILED_Type                                  U01
#define   AQDE_SRC_EX_CONFIG_SUPER_TILED_DISABLED                            0x0
#define   AQDE_SRC_EX_CONFIG_SUPER_TILED_ENABLED                             0x1

#define AQDE_SRC_EX_CONFIG_MINOR_TILED                                       8:8
#define AQDE_SRC_EX_CONFIG_MINOR_TILED_End                                     8
#define AQDE_SRC_EX_CONFIG_MINOR_TILED_Start                                   8
#define AQDE_SRC_EX_CONFIG_MINOR_TILED_Type                                  U01
#define   AQDE_SRC_EX_CONFIG_MINOR_TILED_DISABLED                            0x0
#define   AQDE_SRC_EX_CONFIG_MINOR_TILED_ENABLED                             0x1

#define AQDE_SRC_EX_CONFIG_CACHE_MODE                                      12:12
#define AQDE_SRC_EX_CONFIG_CACHE_MODE_End                                     12
#define AQDE_SRC_EX_CONFIG_CACHE_MODE_Start                                   12
#define AQDE_SRC_EX_CONFIG_CACHE_MODE_Type                                   U01
#define   AQDE_SRC_EX_CONFIG_CACHE_MODE_DISABLED                             0x0
#define   AQDE_SRC_EX_CONFIG_CACHE_MODE_ENABLED                              0x1

#define AQDE_SRC_EX_CONFIG_DC_COMPRESSION                                  13:13
#define AQDE_SRC_EX_CONFIG_DC_COMPRESSION_End                                 13
#define AQDE_SRC_EX_CONFIG_DC_COMPRESSION_Start                               13
#define AQDE_SRC_EX_CONFIG_DC_COMPRESSION_Type                               U01
#define   AQDE_SRC_EX_CONFIG_DC_COMPRESSION_DISABLED                         0x0
#define   AQDE_SRC_EX_CONFIG_DC_COMPRESSION_ENABLED                          0x1

#define AQDE_SRC_EX_CONFIG_MASK_DC_COMPRESSION                             14:14
#define AQDE_SRC_EX_CONFIG_MASK_DC_COMPRESSION_End                            14
#define AQDE_SRC_EX_CONFIG_MASK_DC_COMPRESSION_Start                          14
#define AQDE_SRC_EX_CONFIG_MASK_DC_COMPRESSION_Type                          U01
#define   AQDE_SRC_EX_CONFIG_MASK_DC_COMPRESSION_ENABLED                     0x0
#define   AQDE_SRC_EX_CONFIG_MASK_DC_COMPRESSION_MASKED                      0x1

#define AQDE_SRC_EX_CONFIG_DC_COMPRESSION_ENABLE_ALPHA                     15:15
#define AQDE_SRC_EX_CONFIG_DC_COMPRESSION_ENABLE_ALPHA_End                    15
#define AQDE_SRC_EX_CONFIG_DC_COMPRESSION_ENABLE_ALPHA_Start                  15
#define AQDE_SRC_EX_CONFIG_DC_COMPRESSION_ENABLE_ALPHA_Type                  U01
#define   AQDE_SRC_EX_CONFIG_DC_COMPRESSION_ENABLE_ALPHA_DISABLED            0x0
#define   AQDE_SRC_EX_CONFIG_DC_COMPRESSION_ENABLE_ALPHA_ENABLED             0x1

#define AQDE_SRC_EX_CONFIG_MASK_DC_COMPRESSION_ENABLE_ALPHA                16:16
#define AQDE_SRC_EX_CONFIG_MASK_DC_COMPRESSION_ENABLE_ALPHA_End               16
#define AQDE_SRC_EX_CONFIG_MASK_DC_COMPRESSION_ENABLE_ALPHA_Start             16
#define AQDE_SRC_EX_CONFIG_MASK_DC_COMPRESSION_ENABLE_ALPHA_Type             U01
#define   AQDE_SRC_EX_CONFIG_MASK_DC_COMPRESSION_ENABLE_ALPHA_ENABLED        0x0
#define   AQDE_SRC_EX_CONFIG_MASK_DC_COMPRESSION_ENABLE_ALPHA_MASKED         0x1

#define AQDESrcExAddressRegAddrs                                          0x04C1
#define AQDE_SRC_EX_ADDRESS_Address                                      0x01304
#define AQDE_SRC_EX_ADDRESS_MSB                                               15
#define AQDE_SRC_EX_ADDRESS_LSB                                                0
#define AQDE_SRC_EX_ADDRESS_BLK                                                0
#define AQDE_SRC_EX_ADDRESS_Count                                              1
#define AQDE_SRC_EX_ADDRESS_FieldMask                                 0xFFFFFFFF
#define AQDE_SRC_EX_ADDRESS_ReadMask                                  0xFFFFFFFC
#define AQDE_SRC_EX_ADDRESS_WriteMask                                 0xFFFFFFFC
#define AQDE_SRC_EX_ADDRESS_ResetValue                                0x00000000

#define AQDE_SRC_EX_ADDRESS_TYPE                                           31:31
#define AQDE_SRC_EX_ADDRESS_TYPE_End                                          31
#define AQDE_SRC_EX_ADDRESS_TYPE_Start                                        31
#define AQDE_SRC_EX_ADDRESS_TYPE_Type                                        U01
#define   AQDE_SRC_EX_ADDRESS_TYPE_SYSTEM                                    0x0
#define   AQDE_SRC_EX_ADDRESS_TYPE_VIRTUAL_SYSTEM                            0x1

#define AQDE_SRC_EX_ADDRESS_ADDRESS                                         30:0
#define AQDE_SRC_EX_ADDRESS_ADDRESS_End                                       30
#define AQDE_SRC_EX_ADDRESS_ADDRESS_Start                                      0
#define AQDE_SRC_EX_ADDRESS_ADDRESS_Type                                     U31

#define gcregDEMultiSourceRegAddrs                                        0x04C2
#define GCREG_DE_MULTI_SOURCE_Address                                    0x01308
#define GCREG_DE_MULTI_SOURCE_MSB                                             15
#define GCREG_DE_MULTI_SOURCE_LSB                                              0
#define GCREG_DE_MULTI_SOURCE_BLK                                              0
#define GCREG_DE_MULTI_SOURCE_Count                                            1
#define GCREG_DE_MULTI_SOURCE_FieldMask                               0x00770707
#define GCREG_DE_MULTI_SOURCE_ReadMask                                0x00770707
#define GCREG_DE_MULTI_SOURCE_WriteMask                               0x00770707
#define GCREG_DE_MULTI_SOURCE_ResetValue                              0x00000000

#define GCREG_DE_MULTI_SOURCE_MAX_SOURCE                                     2:0
#define GCREG_DE_MULTI_SOURCE_MAX_SOURCE_End                                   2
#define GCREG_DE_MULTI_SOURCE_MAX_SOURCE_Start                                 0
#define GCREG_DE_MULTI_SOURCE_MAX_SOURCE_Type                                U03

#define GCREG_DE_MULTI_SOURCE_HORIZONTAL_BLOCK                              10:8
#define GCREG_DE_MULTI_SOURCE_HORIZONTAL_BLOCK_End                            10
#define GCREG_DE_MULTI_SOURCE_HORIZONTAL_BLOCK_Start                           8
#define GCREG_DE_MULTI_SOURCE_HORIZONTAL_BLOCK_Type                          U03
#define   GCREG_DE_MULTI_SOURCE_HORIZONTAL_BLOCK_PIXEL16                     0x0
#define   GCREG_DE_MULTI_SOURCE_HORIZONTAL_BLOCK_PIXEL32                     0x1
#define   GCREG_DE_MULTI_SOURCE_HORIZONTAL_BLOCK_PIXEL64                     0x2
#define   GCREG_DE_MULTI_SOURCE_HORIZONTAL_BLOCK_PIXEL128                    0x3
#define   GCREG_DE_MULTI_SOURCE_HORIZONTAL_BLOCK_PIXEL256                    0x4
#define   GCREG_DE_MULTI_SOURCE_HORIZONTAL_BLOCK_PIXEL512                    0x5
#define   GCREG_DE_MULTI_SOURCE_HORIZONTAL_BLOCK_PIXEL8                      0x6
#define   GCREG_DE_MULTI_SOURCE_HORIZONTAL_BLOCK_PIXEL4                      0x7

#define GCREG_DE_MULTI_SOURCE_VERTICAL_BLOCK                               18:16
#define GCREG_DE_MULTI_SOURCE_VERTICAL_BLOCK_End                              18
#define GCREG_DE_MULTI_SOURCE_VERTICAL_BLOCK_Start                            16
#define GCREG_DE_MULTI_SOURCE_VERTICAL_BLOCK_Type                            U03
#define   GCREG_DE_MULTI_SOURCE_VERTICAL_BLOCK_LINE1                         0x0
#define   GCREG_DE_MULTI_SOURCE_VERTICAL_BLOCK_LINE2                         0x1
#define   GCREG_DE_MULTI_SOURCE_VERTICAL_BLOCK_LINE4                         0x2
#define   GCREG_DE_MULTI_SOURCE_VERTICAL_BLOCK_LINE8                         0x3
#define   GCREG_DE_MULTI_SOURCE_VERTICAL_BLOCK_LINE16                        0x4
#define   GCREG_DE_MULTI_SOURCE_VERTICAL_BLOCK_LINE32                        0x5
#define   GCREG_DE_MULTI_SOURCE_VERTICAL_BLOCK_LINE64                        0x6
#define   GCREG_DE_MULTI_SOURCE_VERTICAL_BLOCK_LINE128                       0x7

#define GCREG_DE_MULTI_SOURCE_OPTIMIZATION                                 20:20
#define GCREG_DE_MULTI_SOURCE_OPTIMIZATION_End                                20
#define GCREG_DE_MULTI_SOURCE_OPTIMIZATION_Start                              20
#define GCREG_DE_MULTI_SOURCE_OPTIMIZATION_Type                              U01
#define   GCREG_DE_MULTI_SOURCE_OPTIMIZATION_DISABLED                        0x1
#define   GCREG_DE_MULTI_SOURCE_OPTIMIZATION_ENABLED                         0x0

#define GCREG_DE_MULTI_SOURCE_BILINEAR_FILTER                              21:21
#define GCREG_DE_MULTI_SOURCE_BILINEAR_FILTER_End                             21
#define GCREG_DE_MULTI_SOURCE_BILINEAR_FILTER_Start                           21
#define GCREG_DE_MULTI_SOURCE_BILINEAR_FILTER_Type                           U01
#define   GCREG_DE_MULTI_SOURCE_BILINEAR_FILTER_DISABLED                     0x0
#define   GCREG_DE_MULTI_SOURCE_BILINEAR_FILTER_ENABLED                      0x1

#define GCREG_DE_MULTI_SOURCE_BIQUAD_CACHE_MODE                            22:22
#define GCREG_DE_MULTI_SOURCE_BIQUAD_CACHE_MODE_End                           22
#define GCREG_DE_MULTI_SOURCE_BIQUAD_CACHE_MODE_Start                         22
#define GCREG_DE_MULTI_SOURCE_BIQUAD_CACHE_MODE_Type                         U01
#define   GCREG_DE_MULTI_SOURCE_BIQUAD_CACHE_MODE_DISABLED                   0x1
#define   GCREG_DE_MULTI_SOURCE_BIQUAD_CACHE_MODE_ENABLED                    0x0

#define gcregDEYUVConversionRegAddrs                                      0x04C3
#define GCREG_DEYUV_CONVERSION_Address                                   0x0130C
#define GCREG_DEYUV_CONVERSION_MSB                                            15
#define GCREG_DEYUV_CONVERSION_LSB                                             0
#define GCREG_DEYUV_CONVERSION_BLK                                             0
#define GCREG_DEYUV_CONVERSION_Count                                           1
#define GCREG_DEYUV_CONVERSION_FieldMask                              0xFFFFFFFF
#define GCREG_DEYUV_CONVERSION_ReadMask                               0xFFFFFFFF
#define GCREG_DEYUV_CONVERSION_WriteMask                              0xFFFFFFFF
#define GCREG_DEYUV_CONVERSION_ResetValue                             0x00000000

#define GCREG_DEYUV_CONVERSION_ENABLE                                        1:0
#define GCREG_DEYUV_CONVERSION_ENABLE_End                                      1
#define GCREG_DEYUV_CONVERSION_ENABLE_Start                                    0
#define GCREG_DEYUV_CONVERSION_ENABLE_Type                                   U02

#define   GCREG_DEYUV_CONVERSION_ENABLE_OFF                                  0x0

#define   GCREG_DEYUV_CONVERSION_ENABLE_PLANE1                               0x1

#define   GCREG_DEYUV_CONVERSION_ENABLE_PLANE2                               0x2

#define   GCREG_DEYUV_CONVERSION_ENABLE_PLANE3                               0x3

#define GCREG_DEYUV_CONVERSION_PLANE1_COUNT                                  3:2
#define GCREG_DEYUV_CONVERSION_PLANE1_COUNT_End                                3
#define GCREG_DEYUV_CONVERSION_PLANE1_COUNT_Start                              2
#define GCREG_DEYUV_CONVERSION_PLANE1_COUNT_Type                             U02

#define GCREG_DEYUV_CONVERSION_PLANE2_COUNT                                  5:4
#define GCREG_DEYUV_CONVERSION_PLANE2_COUNT_End                                5
#define GCREG_DEYUV_CONVERSION_PLANE2_COUNT_Start                              4
#define GCREG_DEYUV_CONVERSION_PLANE2_COUNT_Type                             U02

#define GCREG_DEYUV_CONVERSION_PLANE3_COUNT                                  7:6
#define GCREG_DEYUV_CONVERSION_PLANE3_COUNT_End                                7
#define GCREG_DEYUV_CONVERSION_PLANE3_COUNT_Start                              6
#define GCREG_DEYUV_CONVERSION_PLANE3_COUNT_Type                             U02

#define GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_B                              9:8
#define GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_B_End                            9
#define GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_B_Start                          8
#define GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_B_Type                         U02
#define   GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_B_BLUE                       0x2
#define   GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_B_GREEN                      0x1
#define   GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_B_RED                        0x0
#define   GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_B_ALPHA                      0x3

#define GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_G                            11:10
#define GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_G_End                           11
#define GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_G_Start                         10
#define GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_G_Type                         U02
#define   GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_G_BLUE                       0x2
#define   GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_G_GREEN                      0x1
#define   GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_G_RED                        0x0
#define   GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_G_ALPHA                      0x3

#define GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_R                            13:12
#define GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_R_End                           13
#define GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_R_Start                         12
#define GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_R_Type                         U02
#define   GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_R_BLUE                       0x2
#define   GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_R_GREEN                      0x1
#define   GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_R_RED                        0x0
#define   GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_R_ALPHA                      0x3

#define GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_A                            15:14
#define GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_A_End                           15
#define GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_A_Start                         14
#define GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_A_Type                         U02
#define   GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_A_BLUE                       0x2
#define   GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_A_GREEN                      0x1
#define   GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_A_RED                        0x0
#define   GCREG_DEYUV_CONVERSION_PLANE1_SWIZZLE_A_ALPHA                      0x3

#define GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_B                            17:16
#define GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_B_End                           17
#define GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_B_Start                         16
#define GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_B_Type                         U02
#define   GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_B_BLUE                       0x2
#define   GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_B_GREEN                      0x1
#define   GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_B_RED                        0x0
#define   GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_B_ALPHA                      0x3

#define GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_G                            19:18
#define GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_G_End                           19
#define GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_G_Start                         18
#define GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_G_Type                         U02
#define   GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_G_BLUE                       0x2
#define   GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_G_GREEN                      0x1
#define   GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_G_RED                        0x0
#define   GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_G_ALPHA                      0x3

#define GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_R                            21:20
#define GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_R_End                           21
#define GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_R_Start                         20
#define GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_R_Type                         U02
#define   GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_R_BLUE                       0x2
#define   GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_R_GREEN                      0x1
#define   GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_R_RED                        0x0
#define   GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_R_ALPHA                      0x3

#define GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_A                            23:22
#define GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_A_End                           23
#define GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_A_Start                         22
#define GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_A_Type                         U02
#define   GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_A_BLUE                       0x2
#define   GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_A_GREEN                      0x1
#define   GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_A_RED                        0x0
#define   GCREG_DEYUV_CONVERSION_PLANE2_SWIZZLE_A_ALPHA                      0x3

#define GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_B                            25:24
#define GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_B_End                           25
#define GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_B_Start                         24
#define GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_B_Type                         U02
#define   GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_B_BLUE                       0x2
#define   GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_B_GREEN                      0x1
#define   GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_B_RED                        0x0
#define   GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_B_ALPHA                      0x3

#define GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_G                            27:26
#define GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_G_End                           27
#define GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_G_Start                         26
#define GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_G_Type                         U02
#define   GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_G_BLUE                       0x2
#define   GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_G_GREEN                      0x1
#define   GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_G_RED                        0x0
#define   GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_G_ALPHA                      0x3

#define GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_R                            29:28
#define GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_R_End                           29
#define GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_R_Start                         28
#define GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_R_Type                         U02
#define   GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_R_BLUE                       0x2
#define   GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_R_GREEN                      0x1
#define   GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_R_RED                        0x0
#define   GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_R_ALPHA                      0x3

#define GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_A                            31:30
#define GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_A_End                           31
#define GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_A_Start                         30
#define GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_A_Type                         U02
#define   GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_A_BLUE                       0x2
#define   GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_A_GREEN                      0x1
#define   GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_A_RED                        0x0
#define   GCREG_DEYUV_CONVERSION_PLANE3_SWIZZLE_A_ALPHA                      0x3

#define gcregDEPlane2AddressRegAddrs                                      0x04C4
#define GCREG_DE_PLANE2_ADDRESS_Address                                  0x01310
#define GCREG_DE_PLANE2_ADDRESS_MSB                                           15
#define GCREG_DE_PLANE2_ADDRESS_LSB                                            0
#define GCREG_DE_PLANE2_ADDRESS_BLK                                            0
#define GCREG_DE_PLANE2_ADDRESS_Count                                          1
#define GCREG_DE_PLANE2_ADDRESS_FieldMask                             0xFFFFFFFF
#define GCREG_DE_PLANE2_ADDRESS_ReadMask                              0xFFFFFFFC
#define GCREG_DE_PLANE2_ADDRESS_WriteMask                             0xFFFFFFFC
#define GCREG_DE_PLANE2_ADDRESS_ResetValue                            0x00000000

#define GCREG_DE_PLANE2_ADDRESS_TYPE                                       31:31
#define GCREG_DE_PLANE2_ADDRESS_TYPE_End                                      31
#define GCREG_DE_PLANE2_ADDRESS_TYPE_Start                                    31
#define GCREG_DE_PLANE2_ADDRESS_TYPE_Type                                    U01
#define   GCREG_DE_PLANE2_ADDRESS_TYPE_SYSTEM                                0x0
#define   GCREG_DE_PLANE2_ADDRESS_TYPE_VIRTUAL_SYSTEM                        0x1

#define GCREG_DE_PLANE2_ADDRESS_ADDRESS                                     30:0
#define GCREG_DE_PLANE2_ADDRESS_ADDRESS_End                                   30
#define GCREG_DE_PLANE2_ADDRESS_ADDRESS_Start                                  0
#define GCREG_DE_PLANE2_ADDRESS_ADDRESS_Type                                 U31

#define gcregDEPlane2StrideRegAddrs                                       0x04C5
#define GCREG_DE_PLANE2_STRIDE_Address                                   0x01314
#define GCREG_DE_PLANE2_STRIDE_MSB                                            15
#define GCREG_DE_PLANE2_STRIDE_LSB                                             0
#define GCREG_DE_PLANE2_STRIDE_BLK                                             0
#define GCREG_DE_PLANE2_STRIDE_Count                                           1
#define GCREG_DE_PLANE2_STRIDE_FieldMask                              0x0003FFFF
#define GCREG_DE_PLANE2_STRIDE_ReadMask                               0x0003FFFC
#define GCREG_DE_PLANE2_STRIDE_WriteMask                              0x0003FFFC
#define GCREG_DE_PLANE2_STRIDE_ResetValue                             0x00000000

#define GCREG_DE_PLANE2_STRIDE_STRIDE                                       17:0
#define GCREG_DE_PLANE2_STRIDE_STRIDE_End                                     17
#define GCREG_DE_PLANE2_STRIDE_STRIDE_Start                                    0
#define GCREG_DE_PLANE2_STRIDE_STRIDE_Type                                   U18

#define gcregDEPlane3AddressRegAddrs                                      0x04C6
#define GCREG_DE_PLANE3_ADDRESS_Address                                  0x01318
#define GCREG_DE_PLANE3_ADDRESS_MSB                                           15
#define GCREG_DE_PLANE3_ADDRESS_LSB                                            0
#define GCREG_DE_PLANE3_ADDRESS_BLK                                            0
#define GCREG_DE_PLANE3_ADDRESS_Count                                          1
#define GCREG_DE_PLANE3_ADDRESS_FieldMask                             0xFFFFFFFF
#define GCREG_DE_PLANE3_ADDRESS_ReadMask                              0xFFFFFFFC
#define GCREG_DE_PLANE3_ADDRESS_WriteMask                             0xFFFFFFFC
#define GCREG_DE_PLANE3_ADDRESS_ResetValue                            0x00000000

#define GCREG_DE_PLANE3_ADDRESS_TYPE                                       31:31
#define GCREG_DE_PLANE3_ADDRESS_TYPE_End                                      31
#define GCREG_DE_PLANE3_ADDRESS_TYPE_Start                                    31
#define GCREG_DE_PLANE3_ADDRESS_TYPE_Type                                    U01
#define   GCREG_DE_PLANE3_ADDRESS_TYPE_SYSTEM                                0x0
#define   GCREG_DE_PLANE3_ADDRESS_TYPE_VIRTUAL_SYSTEM                        0x1

#define GCREG_DE_PLANE3_ADDRESS_ADDRESS                                     30:0
#define GCREG_DE_PLANE3_ADDRESS_ADDRESS_End                                   30
#define GCREG_DE_PLANE3_ADDRESS_ADDRESS_Start                                  0
#define GCREG_DE_PLANE3_ADDRESS_ADDRESS_Type                                 U31

#define gcregDEPlane3StrideRegAddrs                                       0x04C7
#define GCREG_DE_PLANE3_STRIDE_Address                                   0x0131C
#define GCREG_DE_PLANE3_STRIDE_MSB                                            15
#define GCREG_DE_PLANE3_STRIDE_LSB                                             0
#define GCREG_DE_PLANE3_STRIDE_BLK                                             0
#define GCREG_DE_PLANE3_STRIDE_Count                                           1
#define GCREG_DE_PLANE3_STRIDE_FieldMask                              0x0003FFFF
#define GCREG_DE_PLANE3_STRIDE_ReadMask                               0x0003FFFC
#define GCREG_DE_PLANE3_STRIDE_WriteMask                              0x0003FFFC
#define GCREG_DE_PLANE3_STRIDE_ResetValue                             0x00000000

#define GCREG_DE_PLANE3_STRIDE_STRIDE                                       17:0
#define GCREG_DE_PLANE3_STRIDE_STRIDE_End                                     17
#define GCREG_DE_PLANE3_STRIDE_STRIDE_Start                                    0
#define GCREG_DE_PLANE3_STRIDE_STRIDE_Type                                   U18

#define gcregDEStallDERegAddrs                                            0x04C8
#define GCREG_DE_STALL_DE_Address                                        0x01320
#define GCREG_DE_STALL_DE_MSB                                                 15
#define GCREG_DE_STALL_DE_LSB                                                  0
#define GCREG_DE_STALL_DE_BLK                                                  0
#define GCREG_DE_STALL_DE_Count                                                1
#define GCREG_DE_STALL_DE_FieldMask                                   0x00000001
#define GCREG_DE_STALL_DE_ReadMask                                    0x00000001
#define GCREG_DE_STALL_DE_WriteMask                                   0x00000001
#define GCREG_DE_STALL_DE_ResetValue                                  0x00000000

#define GCREG_DE_STALL_DE_ENABLE                                             0:0
#define GCREG_DE_STALL_DE_ENABLE_End                                           0
#define GCREG_DE_STALL_DE_ENABLE_Start                                         0
#define GCREG_DE_STALL_DE_ENABLE_Type                                        U01
#define   GCREG_DE_STALL_DE_ENABLE_DISABLED                                  0x0
#define   GCREG_DE_STALL_DE_ENABLE_ENABLED                                   0x1

#define gcregDEPatternColorRegAddrs                                       0x04CC
#define GCREG_DE_PATTERN_COLOR_Address                                   0x01330
#define GCREG_DE_PATTERN_COLOR_MSB                                            15
#define GCREG_DE_PATTERN_COLOR_LSB                                             2
#define GCREG_DE_PATTERN_COLOR_BLK                                             2
#define GCREG_DE_PATTERN_COLOR_Count                                           4
#define GCREG_DE_PATTERN_COLOR_FieldMask                              0xFFFFFFFF
#define GCREG_DE_PATTERN_COLOR_ReadMask                               0xFFFFFFFF
#define GCREG_DE_PATTERN_COLOR_WriteMask                              0xFFFFFFFF
#define GCREG_DE_PATTERN_COLOR_ResetValue                             0x00000000

#define GCREG_DE_PATTERN_COLOR_ALPHA                                       31:24
#define GCREG_DE_PATTERN_COLOR_ALPHA_End                                      31
#define GCREG_DE_PATTERN_COLOR_ALPHA_Start                                    24
#define GCREG_DE_PATTERN_COLOR_ALPHA_Type                                    U08

#define GCREG_DE_PATTERN_COLOR_RED                                         23:16
#define GCREG_DE_PATTERN_COLOR_RED_End                                        23
#define GCREG_DE_PATTERN_COLOR_RED_Start                                      16
#define GCREG_DE_PATTERN_COLOR_RED_Type                                      U08

#define GCREG_DE_PATTERN_COLOR_GREEN                                        15:8
#define GCREG_DE_PATTERN_COLOR_GREEN_End                                      15
#define GCREG_DE_PATTERN_COLOR_GREEN_Start                                     8
#define GCREG_DE_PATTERN_COLOR_GREEN_Type                                    U08

#define GCREG_DE_PATTERN_COLOR_BLUE                                          7:0
#define GCREG_DE_PATTERN_COLOR_BLUE_End                                        7
#define GCREG_DE_PATTERN_COLOR_BLUE_Start                                      0
#define GCREG_DE_PATTERN_COLOR_BLUE_Type                                     U08

#define gcregDEBlockSizeRegAddrs                                          0x04C9
#define GCREG_DE_BLOCK_SIZE_Address                                      0x01324
#define GCREG_DE_BLOCK_SIZE_MSB                                               15
#define GCREG_DE_BLOCK_SIZE_LSB                                                0
#define GCREG_DE_BLOCK_SIZE_BLK                                                0
#define GCREG_DE_BLOCK_SIZE_Count                                              1
#define GCREG_DE_BLOCK_SIZE_FieldMask                                 0xFFFFFFFF
#define GCREG_DE_BLOCK_SIZE_ReadMask                                  0xFFFFFFFF
#define GCREG_DE_BLOCK_SIZE_WriteMask                                 0xFFFFFFFF
#define GCREG_DE_BLOCK_SIZE_ResetValue                                0x00000000

#define GCREG_DE_BLOCK_SIZE_WIDTH_BIT_CNT                                   15:0
#define GCREG_DE_BLOCK_SIZE_WIDTH_BIT_CNT_End                                 15
#define GCREG_DE_BLOCK_SIZE_WIDTH_BIT_CNT_Start                                0
#define GCREG_DE_BLOCK_SIZE_WIDTH_BIT_CNT_Type                               U16

#define GCREG_DE_BLOCK_SIZE_HEIGHT_BIT_CNT                                 31:16
#define GCREG_DE_BLOCK_SIZE_HEIGHT_BIT_CNT_End                                31
#define GCREG_DE_BLOCK_SIZE_HEIGHT_BIT_CNT_Start                              16
#define GCREG_DE_BLOCK_SIZE_HEIGHT_BIT_CNT_Type                              U16

#define gcregDEGeneralConfigRegAddrs                                      0x04CA
#define GCREG_DE_GENERAL_CONFIG_Address                                  0x01328
#define GCREG_DE_GENERAL_CONFIG_MSB                                           15
#define GCREG_DE_GENERAL_CONFIG_LSB                                            0
#define GCREG_DE_GENERAL_CONFIG_BLK                                            0
#define GCREG_DE_GENERAL_CONFIG_Count                                          1
#define GCREG_DE_GENERAL_CONFIG_FieldMask                             0x079FFBFB
#define GCREG_DE_GENERAL_CONFIG_ReadMask                              0x079FFBFB
#define GCREG_DE_GENERAL_CONFIG_WriteMask                             0x079FFBFB
#define GCREG_DE_GENERAL_CONFIG_ResetValue                            0x00000012

#define GCREG_DE_GENERAL_CONFIG_SUPERTILE_VERSION                            1:0
#define GCREG_DE_GENERAL_CONFIG_SUPERTILE_VERSION_End                          1
#define GCREG_DE_GENERAL_CONFIG_SUPERTILE_VERSION_Start                        0
#define GCREG_DE_GENERAL_CONFIG_SUPERTILE_VERSION_Type                       U02
#define   GCREG_DE_GENERAL_CONFIG_SUPERTILE_VERSION_SUPER_TILE_V1            0x0
#define   GCREG_DE_GENERAL_CONFIG_SUPERTILE_VERSION_SUPER_TILE_V2            0x1
#define   GCREG_DE_GENERAL_CONFIG_SUPERTILE_VERSION_SUPER_TILE_V3            0x2

#define GCREG_DE_GENERAL_CONFIG_MASK_SUPERTILE_VERSION                       3:3
#define GCREG_DE_GENERAL_CONFIG_MASK_SUPERTILE_VERSION_End                     3
#define GCREG_DE_GENERAL_CONFIG_MASK_SUPERTILE_VERSION_Start                   3
#define GCREG_DE_GENERAL_CONFIG_MASK_SUPERTILE_VERSION_Type                  U01
#define   GCREG_DE_GENERAL_CONFIG_MASK_SUPERTILE_VERSION_ENABLED             0x0
#define   GCREG_DE_GENERAL_CONFIG_MASK_SUPERTILE_VERSION_MASKED              0x1

#define GCREG_DE_GENERAL_CONFIG_BY_PASS_DC_COMPRESSION                       4:4
#define GCREG_DE_GENERAL_CONFIG_BY_PASS_DC_COMPRESSION_End                     4
#define GCREG_DE_GENERAL_CONFIG_BY_PASS_DC_COMPRESSION_Start                   4
#define GCREG_DE_GENERAL_CONFIG_BY_PASS_DC_COMPRESSION_Type                  U01
#define   GCREG_DE_GENERAL_CONFIG_BY_PASS_DC_COMPRESSION_DISABLED            0x0
#define   GCREG_DE_GENERAL_CONFIG_BY_PASS_DC_COMPRESSION_ENABLED             0x1

#define GCREG_DE_GENERAL_CONFIG_MASK_BY_PASS_DC_COMPRESSION                  5:5
#define GCREG_DE_GENERAL_CONFIG_MASK_BY_PASS_DC_COMPRESSION_End                5
#define GCREG_DE_GENERAL_CONFIG_MASK_BY_PASS_DC_COMPRESSION_Start              5
#define GCREG_DE_GENERAL_CONFIG_MASK_BY_PASS_DC_COMPRESSION_Type             U01
#define   GCREG_DE_GENERAL_CONFIG_MASK_BY_PASS_DC_COMPRESSION_ENABLED        0x0
#define   GCREG_DE_GENERAL_CONFIG_MASK_BY_PASS_DC_COMPRESSION_MASKED         0x1

#define GCREG_DE_GENERAL_CONFIG_DISABLE_MULTI_SRC_V1DOT5                     6:6
#define GCREG_DE_GENERAL_CONFIG_DISABLE_MULTI_SRC_V1DOT5_End                   6
#define GCREG_DE_GENERAL_CONFIG_DISABLE_MULTI_SRC_V1DOT5_Start                 6
#define GCREG_DE_GENERAL_CONFIG_DISABLE_MULTI_SRC_V1DOT5_Type                U01
#define   GCREG_DE_GENERAL_CONFIG_DISABLE_MULTI_SRC_V1DOT5_DISABLED          0x0
#define   GCREG_DE_GENERAL_CONFIG_DISABLE_MULTI_SRC_V1DOT5_ENABLED           0x1

#define GCREG_DE_GENERAL_CONFIG_MASK_DISABLE_MULTI_SRC_V1DOT5                7:7
#define GCREG_DE_GENERAL_CONFIG_MASK_DISABLE_MULTI_SRC_V1DOT5_End              7
#define GCREG_DE_GENERAL_CONFIG_MASK_DISABLE_MULTI_SRC_V1DOT5_Start            7
#define GCREG_DE_GENERAL_CONFIG_MASK_DISABLE_MULTI_SRC_V1DOT5_Type           U01
#define   GCREG_DE_GENERAL_CONFIG_MASK_DISABLE_MULTI_SRC_V1DOT5_ENABLED      0x0
#define   GCREG_DE_GENERAL_CONFIG_MASK_DISABLE_MULTI_SRC_V1DOT5_MASKED       0x1

#define GCREG_DE_GENERAL_CONFIG_ADDRESS_OVER_LAP                             8:8
#define GCREG_DE_GENERAL_CONFIG_ADDRESS_OVER_LAP_End                           8
#define GCREG_DE_GENERAL_CONFIG_ADDRESS_OVER_LAP_Start                         8
#define GCREG_DE_GENERAL_CONFIG_ADDRESS_OVER_LAP_Type                        U01
#define   GCREG_DE_GENERAL_CONFIG_ADDRESS_OVER_LAP_DISABLED                  0x0
#define   GCREG_DE_GENERAL_CONFIG_ADDRESS_OVER_LAP_ENABLED                   0x1

#define GCREG_DE_GENERAL_CONFIG_MASK_ADDRESS_OVER_LAP                        9:9
#define GCREG_DE_GENERAL_CONFIG_MASK_ADDRESS_OVER_LAP_End                      9
#define GCREG_DE_GENERAL_CONFIG_MASK_ADDRESS_OVER_LAP_Start                    9
#define GCREG_DE_GENERAL_CONFIG_MASK_ADDRESS_OVER_LAP_Type                   U01
#define   GCREG_DE_GENERAL_CONFIG_MASK_ADDRESS_OVER_LAP_ENABLED              0x0
#define   GCREG_DE_GENERAL_CONFIG_MASK_ADDRESS_OVER_LAP_MASKED               0x1

#define GCREG_DE_GENERAL_CONFIG_MASK_PE_CACHE_MODE                         11:11
#define GCREG_DE_GENERAL_CONFIG_MASK_PE_CACHE_MODE_End                        11
#define GCREG_DE_GENERAL_CONFIG_MASK_PE_CACHE_MODE_Start                      11
#define GCREG_DE_GENERAL_CONFIG_MASK_PE_CACHE_MODE_Type                      U01
#define   GCREG_DE_GENERAL_CONFIG_MASK_PE_CACHE_MODE_ENABLED                 0x0
#define   GCREG_DE_GENERAL_CONFIG_MASK_PE_CACHE_MODE_MASKED                  0x1

#define GCREG_DE_GENERAL_CONFIG_PE_CACHE_MODE                              13:12
#define GCREG_DE_GENERAL_CONFIG_PE_CACHE_MODE_End                             13
#define GCREG_DE_GENERAL_CONFIG_PE_CACHE_MODE_Start                           12
#define GCREG_DE_GENERAL_CONFIG_PE_CACHE_MODE_Type                           U02
#define   GCREG_DE_GENERAL_CONFIG_PE_CACHE_MODE_Size64B                      0x0
#define   GCREG_DE_GENERAL_CONFIG_PE_CACHE_MODE_Size256B                     0x1
#define   GCREG_DE_GENERAL_CONFIG_PE_CACHE_MODE_Size128B                     0x2
#define   GCREG_DE_GENERAL_CONFIG_PE_CACHE_MODE_QUAD64_B_64_B                0x3

#define GCREG_DE_GENERAL_CONFIG_MASK_PE_SRC_CACHE_MODE                     14:14
#define GCREG_DE_GENERAL_CONFIG_MASK_PE_SRC_CACHE_MODE_End                    14
#define GCREG_DE_GENERAL_CONFIG_MASK_PE_SRC_CACHE_MODE_Start                  14
#define GCREG_DE_GENERAL_CONFIG_MASK_PE_SRC_CACHE_MODE_Type                  U01
#define   GCREG_DE_GENERAL_CONFIG_MASK_PE_SRC_CACHE_MODE_ENABLED             0x0
#define   GCREG_DE_GENERAL_CONFIG_MASK_PE_SRC_CACHE_MODE_MASKED              0x1

#define GCREG_DE_GENERAL_CONFIG_PE_SRC_CACHE_MODE                          16:15
#define GCREG_DE_GENERAL_CONFIG_PE_SRC_CACHE_MODE_End                         16
#define GCREG_DE_GENERAL_CONFIG_PE_SRC_CACHE_MODE_Start                       15
#define GCREG_DE_GENERAL_CONFIG_PE_SRC_CACHE_MODE_Type                       U02
#define   GCREG_DE_GENERAL_CONFIG_PE_SRC_CACHE_MODE_Size64B                  0x0
#define   GCREG_DE_GENERAL_CONFIG_PE_SRC_CACHE_MODE_Size256B                 0x1
#define   GCREG_DE_GENERAL_CONFIG_PE_SRC_CACHE_MODE_Size128B                 0x2
#define   GCREG_DE_GENERAL_CONFIG_PE_SRC_CACHE_MODE_QUAD64_B_64_B            0x3

#define GCREG_DE_GENERAL_CONFIG_MASK_PE_CACHE_MODE2                        17:17
#define GCREG_DE_GENERAL_CONFIG_MASK_PE_CACHE_MODE2_End                       17
#define GCREG_DE_GENERAL_CONFIG_MASK_PE_CACHE_MODE2_Start                     17
#define GCREG_DE_GENERAL_CONFIG_MASK_PE_CACHE_MODE2_Type                     U01
#define   GCREG_DE_GENERAL_CONFIG_MASK_PE_CACHE_MODE2_ENABLED                0x0
#define   GCREG_DE_GENERAL_CONFIG_MASK_PE_CACHE_MODE2_MASKED                 0x1

#define GCREG_DE_GENERAL_CONFIG_PE_CACHE_MODE2                             20:18
#define GCREG_DE_GENERAL_CONFIG_PE_CACHE_MODE2_End                            20
#define GCREG_DE_GENERAL_CONFIG_PE_CACHE_MODE2_Start                          18
#define GCREG_DE_GENERAL_CONFIG_PE_CACHE_MODE2_Type                          U03
#define   GCREG_DE_GENERAL_CONFIG_PE_CACHE_MODE2_AUTO                        0x0
#define   GCREG_DE_GENERAL_CONFIG_PE_CACHE_MODE2_Size64B                     0x1
#define   GCREG_DE_GENERAL_CONFIG_PE_CACHE_MODE2_Size128B                    0x2
#define   GCREG_DE_GENERAL_CONFIG_PE_CACHE_MODE2_Size256B                    0x3
#define   GCREG_DE_GENERAL_CONFIG_PE_CACHE_MODE2_Quad64B_64B                 0x4

#define GCREG_DE_GENERAL_CONFIG_MASK_PE_SRC_CACHE_MODE2                    23:23
#define GCREG_DE_GENERAL_CONFIG_MASK_PE_SRC_CACHE_MODE2_End                   23
#define GCREG_DE_GENERAL_CONFIG_MASK_PE_SRC_CACHE_MODE2_Start                 23
#define GCREG_DE_GENERAL_CONFIG_MASK_PE_SRC_CACHE_MODE2_Type                 U01
#define   GCREG_DE_GENERAL_CONFIG_MASK_PE_SRC_CACHE_MODE2_ENABLED            0x0
#define   GCREG_DE_GENERAL_CONFIG_MASK_PE_SRC_CACHE_MODE2_MASKED             0x1

#define GCREG_DE_GENERAL_CONFIG_PE_SRC_CACHE_MODE2                         26:24
#define GCREG_DE_GENERAL_CONFIG_PE_SRC_CACHE_MODE2_End                        26
#define GCREG_DE_GENERAL_CONFIG_PE_SRC_CACHE_MODE2_Start                      24
#define GCREG_DE_GENERAL_CONFIG_PE_SRC_CACHE_MODE2_Type                      U03
#define   GCREG_DE_GENERAL_CONFIG_PE_SRC_CACHE_MODE2_AUTO                    0x0
#define   GCREG_DE_GENERAL_CONFIG_PE_SRC_CACHE_MODE2_Size64B                 0x1
#define   GCREG_DE_GENERAL_CONFIG_PE_SRC_CACHE_MODE2_Size128B                0x2
#define   GCREG_DE_GENERAL_CONFIG_PE_SRC_CACHE_MODE2_Size256B                0x3
#define   GCREG_DE_GENERAL_CONFIG_PE_SRC_CACHE_MODE2_Quad64B_64B             0x4

#define gcregDestConfigExRegAddrs                                         0x04CB
#define GCREG_DEST_CONFIG_EX_Address                                     0x0132C
#define GCREG_DEST_CONFIG_EX_MSB                                              15
#define GCREG_DEST_CONFIG_EX_LSB                                               0
#define GCREG_DEST_CONFIG_EX_BLK                                               0
#define GCREG_DEST_CONFIG_EX_Count                                             1
#define GCREG_DEST_CONFIG_EX_FieldMask                                0xF7FF909B
#define GCREG_DEST_CONFIG_EX_ReadMask                                 0xF7FF909B
#define GCREG_DEST_CONFIG_EX_WriteMask                                0xF7FF909B
#define GCREG_DEST_CONFIG_EX_ResetValue                               0x04000000

#define GCREG_DEST_CONFIG_EX_YUV                                             1:0
#define GCREG_DEST_CONFIG_EX_YUV_End                                           1
#define GCREG_DEST_CONFIG_EX_YUV_Start                                         0
#define GCREG_DEST_CONFIG_EX_YUV_Type                                        U02
#define   GCREG_DEST_CONFIG_EX_YUV_601                                       0x0
#define   GCREG_DEST_CONFIG_EX_YUV_709                                       0x1
#define   GCREG_DEST_CONFIG_EX_YUV_UserDefined                               0x2
#define   GCREG_DEST_CONFIG_EX_YUV_2020                                      0x3

#define GCREG_DEST_CONFIG_EX_MASK_YUV                                        3:3
#define GCREG_DEST_CONFIG_EX_MASK_YUV_End                                      3
#define GCREG_DEST_CONFIG_EX_MASK_YUV_Start                                    3
#define GCREG_DEST_CONFIG_EX_MASK_YUV_Type                                   U01
#define   GCREG_DEST_CONFIG_EX_MASK_YUV_ENABLED                              0x0
#define   GCREG_DEST_CONFIG_EX_MASK_YUV_MASKED                               0x1

#define GCREG_DEST_CONFIG_EX_UV_SWIZZLE                                      4:4
#define GCREG_DEST_CONFIG_EX_UV_SWIZZLE_End                                    4
#define GCREG_DEST_CONFIG_EX_UV_SWIZZLE_Start                                  4
#define GCREG_DEST_CONFIG_EX_UV_SWIZZLE_Type                                 U01
#define   GCREG_DEST_CONFIG_EX_UV_SWIZZLE_UV                                 0x0
#define   GCREG_DEST_CONFIG_EX_UV_SWIZZLE_VU                                 0x1

#define GCREG_DEST_CONFIG_EX_MASK_UV_SWIZZLE                                 7:7
#define GCREG_DEST_CONFIG_EX_MASK_UV_SWIZZLE_End                               7
#define GCREG_DEST_CONFIG_EX_MASK_UV_SWIZZLE_Start                             7
#define GCREG_DEST_CONFIG_EX_MASK_UV_SWIZZLE_Type                            U01
#define   GCREG_DEST_CONFIG_EX_MASK_UV_SWIZZLE_ENABLED                       0x0
#define   GCREG_DEST_CONFIG_EX_MASK_UV_SWIZZLE_MASKED                        0x1

#define GCREG_DEST_CONFIG_EX_YUV_CLAMP                                     12:12
#define GCREG_DEST_CONFIG_EX_YUV_CLAMP_End                                    12
#define GCREG_DEST_CONFIG_EX_YUV_CLAMP_Start                                  12
#define GCREG_DEST_CONFIG_EX_YUV_CLAMP_Type                                  U01
#define   GCREG_DEST_CONFIG_EX_YUV_CLAMP_DISABLED                            0x0
#define   GCREG_DEST_CONFIG_EX_YUV_CLAMP_ENABLED                             0x1

#define GCREG_DEST_CONFIG_EX_MASK_YUV_CLAMP                                15:15
#define GCREG_DEST_CONFIG_EX_MASK_YUV_CLAMP_End                               15
#define GCREG_DEST_CONFIG_EX_MASK_YUV_CLAMP_Start                             15
#define GCREG_DEST_CONFIG_EX_MASK_YUV_CLAMP_Type                             U01
#define   GCREG_DEST_CONFIG_EX_MASK_YUV_CLAMP_ENABLED                        0x0
#define   GCREG_DEST_CONFIG_EX_MASK_YUV_CLAMP_MASKED                         0x1

#define GCREG_DEST_CONFIG_EX_DC_COMPRESSION                                16:16
#define GCREG_DEST_CONFIG_EX_DC_COMPRESSION_End                               16
#define GCREG_DEST_CONFIG_EX_DC_COMPRESSION_Start                             16
#define GCREG_DEST_CONFIG_EX_DC_COMPRESSION_Type                             U01
#define   GCREG_DEST_CONFIG_EX_DC_COMPRESSION_DISABLED                       0x0
#define   GCREG_DEST_CONFIG_EX_DC_COMPRESSION_ENABLED                        0x1

#define GCREG_DEST_CONFIG_EX_MASK_DC_COMPRESSION                           17:17
#define GCREG_DEST_CONFIG_EX_MASK_DC_COMPRESSION_End                          17
#define GCREG_DEST_CONFIG_EX_MASK_DC_COMPRESSION_Start                        17
#define GCREG_DEST_CONFIG_EX_MASK_DC_COMPRESSION_Type                        U01
#define   GCREG_DEST_CONFIG_EX_MASK_DC_COMPRESSION_ENABLED                   0x0
#define   GCREG_DEST_CONFIG_EX_MASK_DC_COMPRESSION_MASKED                    0x1

#define GCREG_DEST_CONFIG_EX_DC_COMPRESSION_ENABLE_ALPHA                   18:18
#define GCREG_DEST_CONFIG_EX_DC_COMPRESSION_ENABLE_ALPHA_End                  18
#define GCREG_DEST_CONFIG_EX_DC_COMPRESSION_ENABLE_ALPHA_Start                18
#define GCREG_DEST_CONFIG_EX_DC_COMPRESSION_ENABLE_ALPHA_Type                U01
#define   GCREG_DEST_CONFIG_EX_DC_COMPRESSION_ENABLE_ALPHA_DISABLED          0x0
#define   GCREG_DEST_CONFIG_EX_DC_COMPRESSION_ENABLE_ALPHA_ENABLED           0x1

#define GCREG_DEST_CONFIG_EX_MASK_DC_COMPRESSION_ENABLE_ALPHA              19:19
#define GCREG_DEST_CONFIG_EX_MASK_DC_COMPRESSION_ENABLE_ALPHA_End             19
#define GCREG_DEST_CONFIG_EX_MASK_DC_COMPRESSION_ENABLE_ALPHA_Start           19
#define GCREG_DEST_CONFIG_EX_MASK_DC_COMPRESSION_ENABLE_ALPHA_Type           U01
#define   GCREG_DEST_CONFIG_EX_MASK_DC_COMPRESSION_ENABLE_ALPHA_ENABLED      0x0
#define   GCREG_DEST_CONFIG_EX_MASK_DC_COMPRESSION_ENABLE_ALPHA_MASKED       0x1

#define GCREG_DEST_CONFIG_EX_EXTRA_FORMAT                                  23:20
#define GCREG_DEST_CONFIG_EX_EXTRA_FORMAT_End                                 23
#define GCREG_DEST_CONFIG_EX_EXTRA_FORMAT_Start                               20
#define GCREG_DEST_CONFIG_EX_EXTRA_FORMAT_Type                               U04
#define   GCREG_DEST_CONFIG_EX_EXTRA_FORMAT_RGB_F16F16F16_PLANAR             0x0
#define   GCREG_DEST_CONFIG_EX_EXTRA_FORMAT_RGB_F16F16F16_PACKED             0x1
#define   GCREG_DEST_CONFIG_EX_EXTRA_FORMAT_RGB_F32F32F32_PLANAR             0x2
#define   GCREG_DEST_CONFIG_EX_EXTRA_FORMAT_RGB_F32F32F32_PACKED             0x3
#define   GCREG_DEST_CONFIG_EX_EXTRA_FORMAT_GRAY_I8                          0x4
#define   GCREG_DEST_CONFIG_EX_EXTRA_FORMAT_GRAY_F16                         0x5
#define   GCREG_DEST_CONFIG_EX_EXTRA_FORMAT_GRAY_F32                         0x6
#define   GCREG_DEST_CONFIG_EX_EXTRA_FORMAT_INT8_PLANAR                      0x7
#define   GCREG_DEST_CONFIG_EX_EXTRA_FORMAT_INT8_PACKED                      0x8
#define   GCREG_DEST_CONFIG_EX_EXTRA_FORMAT_INT16_PLANAR                     0x9
#define   GCREG_DEST_CONFIG_EX_EXTRA_FORMAT_INT16_PACKED                     0xA
#define   GCREG_DEST_CONFIG_EX_EXTRA_FORMAT_GRAY_U8                          0xB

#define GCREG_DEST_CONFIG_EX_CONVERSION_CONFIG                             25:24
#define GCREG_DEST_CONFIG_EX_CONVERSION_CONFIG_End                            25
#define GCREG_DEST_CONFIG_EX_CONVERSION_CONFIG_Start                          24
#define GCREG_DEST_CONFIG_EX_CONVERSION_CONFIG_Type                          U02
#define   GCREG_DEST_CONFIG_EX_CONVERSION_CONFIG_ADD_LOWER_BITS              0x0
#define   GCREG_DEST_CONFIG_EX_CONVERSION_CONFIG_NOT_ADD_LOWER_BITS          0x1
#define   GCREG_DEST_CONFIG_EX_CONVERSION_CONFIG_NOT_ADD_HIGHER_BITS         0x2

#define GCREG_DEST_CONFIG_EX_ROUNDING_TO_NEAREST                           26:26
#define GCREG_DEST_CONFIG_EX_ROUNDING_TO_NEAREST_End                          26
#define GCREG_DEST_CONFIG_EX_ROUNDING_TO_NEAREST_Start                        26
#define GCREG_DEST_CONFIG_EX_ROUNDING_TO_NEAREST_Type                        U01
#define   GCREG_DEST_CONFIG_EX_ROUNDING_TO_NEAREST_DISABLE                   0x0
#define   GCREG_DEST_CONFIG_EX_ROUNDING_TO_NEAREST_ENABLE                    0x1

#define GCREG_DEST_CONFIG_EX_NORMALIZATION                                 28:28
#define GCREG_DEST_CONFIG_EX_NORMALIZATION_End                                28
#define GCREG_DEST_CONFIG_EX_NORMALIZATION_Start                              28
#define GCREG_DEST_CONFIG_EX_NORMALIZATION_Type                              U01
#define   GCREG_DEST_CONFIG_EX_NORMALIZATION_DISABLE                         0x0
#define   GCREG_DEST_CONFIG_EX_NORMALIZATION_ENABLE                          0x1

#define GCREG_DEST_CONFIG_EX_NORMALIZATION_WORK_MODE                       30:29
#define GCREG_DEST_CONFIG_EX_NORMALIZATION_WORK_MODE_End                      30
#define GCREG_DEST_CONFIG_EX_NORMALIZATION_WORK_MODE_Start                    29
#define GCREG_DEST_CONFIG_EX_NORMALIZATION_WORK_MODE_Type                    U02
#define   GCREG_DEST_CONFIG_EX_NORMALIZATION_WORK_MODE_MODE1                 0x0
#define   GCREG_DEST_CONFIG_EX_NORMALIZATION_WORK_MODE_MODE2                 0x1

#define GCREG_DEST_CONFIG_EX_RGB_TO_YUV                                    31:31
#define GCREG_DEST_CONFIG_EX_RGB_TO_YUV_End                                   31
#define GCREG_DEST_CONFIG_EX_RGB_TO_YUV_Start                                 31
#define GCREG_DEST_CONFIG_EX_RGB_TO_YUV_Type                                 U01
#define   GCREG_DEST_CONFIG_EX_RGB_TO_YUV_DISABLE                            0x0
#define   GCREG_DEST_CONFIG_EX_RGB_TO_YUV_ENABLE                             0x1

#define gcregDEYUV2RGBCoef0RegAddrs                                       0x04D0
#define GCREG_DEYUV2_RGB_COEF0_Address                                   0x01340
#define GCREG_DEYUV2_RGB_COEF0_MSB                                            15
#define GCREG_DEYUV2_RGB_COEF0_LSB                                             0
#define GCREG_DEYUV2_RGB_COEF0_BLK                                             0
#define GCREG_DEYUV2_RGB_COEF0_Count                                           1
#define GCREG_DEYUV2_RGB_COEF0_FieldMask                              0xFFFFFFFF
#define GCREG_DEYUV2_RGB_COEF0_ReadMask                               0xFFFFFFFF
#define GCREG_DEYUV2_RGB_COEF0_WriteMask                              0xFFFFFFFF
#define GCREG_DEYUV2_RGB_COEF0_ResetValue                             0x00000000

#define GCREG_DEYUV2_RGB_COEF0_COEFFICIENT0                                 15:0
#define GCREG_DEYUV2_RGB_COEF0_COEFFICIENT0_End                               15
#define GCREG_DEYUV2_RGB_COEF0_COEFFICIENT0_Start                              0
#define GCREG_DEYUV2_RGB_COEF0_COEFFICIENT0_Type                             U16

#define GCREG_DEYUV2_RGB_COEF0_COEFFICIENT1                                31:16
#define GCREG_DEYUV2_RGB_COEF0_COEFFICIENT1_End                               31
#define GCREG_DEYUV2_RGB_COEF0_COEFFICIENT1_Start                             16
#define GCREG_DEYUV2_RGB_COEF0_COEFFICIENT1_Type                             U16

#define gcregDEYUV2RGBCoef1RegAddrs                                       0x04D1
#define GCREG_DEYUV2_RGB_COEF1_Address                                   0x01344
#define GCREG_DEYUV2_RGB_COEF1_MSB                                            15
#define GCREG_DEYUV2_RGB_COEF1_LSB                                             0
#define GCREG_DEYUV2_RGB_COEF1_BLK                                             0
#define GCREG_DEYUV2_RGB_COEF1_Count                                           1
#define GCREG_DEYUV2_RGB_COEF1_FieldMask                              0xFFFFFFFF
#define GCREG_DEYUV2_RGB_COEF1_ReadMask                               0xFFFFFFFF
#define GCREG_DEYUV2_RGB_COEF1_WriteMask                              0xFFFFFFFF
#define GCREG_DEYUV2_RGB_COEF1_ResetValue                             0x00000000

#define GCREG_DEYUV2_RGB_COEF1_COEFFICIENT0                                 15:0
#define GCREG_DEYUV2_RGB_COEF1_COEFFICIENT0_End                               15
#define GCREG_DEYUV2_RGB_COEF1_COEFFICIENT0_Start                              0
#define GCREG_DEYUV2_RGB_COEF1_COEFFICIENT0_Type                             U16

#define GCREG_DEYUV2_RGB_COEF1_COEFFICIENT1                                31:16
#define GCREG_DEYUV2_RGB_COEF1_COEFFICIENT1_End                               31
#define GCREG_DEYUV2_RGB_COEF1_COEFFICIENT1_Start                             16
#define GCREG_DEYUV2_RGB_COEF1_COEFFICIENT1_Type                             U16

#define gcregDEYUV2RGBCoef2RegAddrs                                       0x04D2
#define GCREG_DEYUV2_RGB_COEF2_Address                                   0x01348
#define GCREG_DEYUV2_RGB_COEF2_MSB                                            15
#define GCREG_DEYUV2_RGB_COEF2_LSB                                             0
#define GCREG_DEYUV2_RGB_COEF2_BLK                                             0
#define GCREG_DEYUV2_RGB_COEF2_Count                                           1
#define GCREG_DEYUV2_RGB_COEF2_FieldMask                              0xFFFFFFFF
#define GCREG_DEYUV2_RGB_COEF2_ReadMask                               0xFFFFFFFF
#define GCREG_DEYUV2_RGB_COEF2_WriteMask                              0xFFFFFFFF
#define GCREG_DEYUV2_RGB_COEF2_ResetValue                             0x00000000

#define GCREG_DEYUV2_RGB_COEF2_COEFFICIENT0                                 15:0
#define GCREG_DEYUV2_RGB_COEF2_COEFFICIENT0_End                               15
#define GCREG_DEYUV2_RGB_COEF2_COEFFICIENT0_Start                              0
#define GCREG_DEYUV2_RGB_COEF2_COEFFICIENT0_Type                             U16

#define GCREG_DEYUV2_RGB_COEF2_COEFFICIENT1                                31:16
#define GCREG_DEYUV2_RGB_COEF2_COEFFICIENT1_End                               31
#define GCREG_DEYUV2_RGB_COEF2_COEFFICIENT1_Start                             16
#define GCREG_DEYUV2_RGB_COEF2_COEFFICIENT1_Type                             U16

#define gcregDEYUV2RGBCoef3RegAddrs                                       0x04D3
#define GCREG_DEYUV2_RGB_COEF3_Address                                   0x0134C
#define GCREG_DEYUV2_RGB_COEF3_MSB                                            15
#define GCREG_DEYUV2_RGB_COEF3_LSB                                             0
#define GCREG_DEYUV2_RGB_COEF3_BLK                                             0
#define GCREG_DEYUV2_RGB_COEF3_Count                                           1
#define GCREG_DEYUV2_RGB_COEF3_FieldMask                              0xFFFFFFFF
#define GCREG_DEYUV2_RGB_COEF3_ReadMask                               0xFFFFFFFF
#define GCREG_DEYUV2_RGB_COEF3_WriteMask                              0xFFFFFFFF
#define GCREG_DEYUV2_RGB_COEF3_ResetValue                             0x00000000

#define GCREG_DEYUV2_RGB_COEF3_COEFFICIENT0                                 15:0
#define GCREG_DEYUV2_RGB_COEF3_COEFFICIENT0_End                               15
#define GCREG_DEYUV2_RGB_COEF3_COEFFICIENT0_Start                              0
#define GCREG_DEYUV2_RGB_COEF3_COEFFICIENT0_Type                             U16

#define GCREG_DEYUV2_RGB_COEF3_COEFFICIENT1                                31:16
#define GCREG_DEYUV2_RGB_COEF3_COEFFICIENT1_End                               31
#define GCREG_DEYUV2_RGB_COEF3_COEFFICIENT1_Start                             16
#define GCREG_DEYUV2_RGB_COEF3_COEFFICIENT1_Type                             U16

#define gcregDEYUV2RGBCoef4RegAddrs                                       0x04D4
#define GCREG_DEYUV2_RGB_COEF4_Address                                   0x01350
#define GCREG_DEYUV2_RGB_COEF4_MSB                                            15
#define GCREG_DEYUV2_RGB_COEF4_LSB                                             0
#define GCREG_DEYUV2_RGB_COEF4_BLK                                             0
#define GCREG_DEYUV2_RGB_COEF4_Count                                           1
#define GCREG_DEYUV2_RGB_COEF4_FieldMask                              0xFFFFFFFF
#define GCREG_DEYUV2_RGB_COEF4_ReadMask                               0xFFFFFFFF
#define GCREG_DEYUV2_RGB_COEF4_WriteMask                              0xFFFFFFFF
#define GCREG_DEYUV2_RGB_COEF4_ResetValue                             0x00000000

#define GCREG_DEYUV2_RGB_COEF4_COEFFICIENT0                                 15:0
#define GCREG_DEYUV2_RGB_COEF4_COEFFICIENT0_End                               15
#define GCREG_DEYUV2_RGB_COEF4_COEFFICIENT0_Start                              0
#define GCREG_DEYUV2_RGB_COEF4_COEFFICIENT0_Type                             U16

#define GCREG_DEYUV2_RGB_COEF4_COEFFICIENT1                                31:16
#define GCREG_DEYUV2_RGB_COEF4_COEFFICIENT1_End                               31
#define GCREG_DEYUV2_RGB_COEF4_COEFFICIENT1_Start                             16
#define GCREG_DEYUV2_RGB_COEF4_COEFFICIENT1_Type                             U16

#define gcregDEYUV2RGBCoefD0RegAddrs                                      0x04D5
#define GCREG_DEYUV2_RGB_COEF_D0_Address                                 0x01354
#define GCREG_DEYUV2_RGB_COEF_D0_MSB                                          15
#define GCREG_DEYUV2_RGB_COEF_D0_LSB                                           0
#define GCREG_DEYUV2_RGB_COEF_D0_BLK                                           0
#define GCREG_DEYUV2_RGB_COEF_D0_Count                                         1
#define GCREG_DEYUV2_RGB_COEF_D0_FieldMask                            0x01FFFFFF
#define GCREG_DEYUV2_RGB_COEF_D0_ReadMask                             0x01FFFFFF
#define GCREG_DEYUV2_RGB_COEF_D0_WriteMask                            0x01FFFFFF
#define GCREG_DEYUV2_RGB_COEF_D0_ResetValue                           0x00000000

#define GCREG_DEYUV2_RGB_COEF_D0_COEFFICIENT                                24:0
#define GCREG_DEYUV2_RGB_COEF_D0_COEFFICIENT_End                              24
#define GCREG_DEYUV2_RGB_COEF_D0_COEFFICIENT_Start                             0
#define GCREG_DEYUV2_RGB_COEF_D0_COEFFICIENT_Type                            U25

#define gcregDEYUV2RGBCoefD1RegAddrs                                      0x04D6
#define GCREG_DEYUV2_RGB_COEF_D1_Address                                 0x01358
#define GCREG_DEYUV2_RGB_COEF_D1_MSB                                          15
#define GCREG_DEYUV2_RGB_COEF_D1_LSB                                           0
#define GCREG_DEYUV2_RGB_COEF_D1_BLK                                           0
#define GCREG_DEYUV2_RGB_COEF_D1_Count                                         1
#define GCREG_DEYUV2_RGB_COEF_D1_FieldMask                            0x01FFFFFF
#define GCREG_DEYUV2_RGB_COEF_D1_ReadMask                             0x01FFFFFF
#define GCREG_DEYUV2_RGB_COEF_D1_WriteMask                            0x01FFFFFF
#define GCREG_DEYUV2_RGB_COEF_D1_ResetValue                           0x00000000

#define GCREG_DEYUV2_RGB_COEF_D1_COEFFICIENT                                24:0
#define GCREG_DEYUV2_RGB_COEF_D1_COEFFICIENT_End                              24
#define GCREG_DEYUV2_RGB_COEF_D1_COEFFICIENT_Start                             0
#define GCREG_DEYUV2_RGB_COEF_D1_COEFFICIENT_Type                            U25

#define gcregDEYUV2RGBCoefD2RegAddrs                                      0x04D7
#define GCREG_DEYUV2_RGB_COEF_D2_Address                                 0x0135C
#define GCREG_DEYUV2_RGB_COEF_D2_MSB                                          15
#define GCREG_DEYUV2_RGB_COEF_D2_LSB                                           0
#define GCREG_DEYUV2_RGB_COEF_D2_BLK                                           0
#define GCREG_DEYUV2_RGB_COEF_D2_Count                                         1
#define GCREG_DEYUV2_RGB_COEF_D2_FieldMask                            0x01FFFFFF
#define GCREG_DEYUV2_RGB_COEF_D2_ReadMask                             0x01FFFFFF
#define GCREG_DEYUV2_RGB_COEF_D2_WriteMask                            0x01FFFFFF
#define GCREG_DEYUV2_RGB_COEF_D2_ResetValue                           0x00000000

#define GCREG_DEYUV2_RGB_COEF_D2_COEFFICIENT                                24:0
#define GCREG_DEYUV2_RGB_COEF_D2_COEFFICIENT_End                              24
#define GCREG_DEYUV2_RGB_COEF_D2_COEFFICIENT_Start                             0
#define GCREG_DEYUV2_RGB_COEF_D2_COEFFICIENT_Type                            U25

#define gcregDERGB2YUVCoef0RegAddrs                                       0x04D8
#define GCREG_DERGB2_YUV_COEF0_Address                                   0x01360
#define GCREG_DERGB2_YUV_COEF0_MSB                                            15
#define GCREG_DERGB2_YUV_COEF0_LSB                                             0
#define GCREG_DERGB2_YUV_COEF0_BLK                                             0
#define GCREG_DERGB2_YUV_COEF0_Count                                           1
#define GCREG_DERGB2_YUV_COEF0_FieldMask                              0xFFFFFFFF
#define GCREG_DERGB2_YUV_COEF0_ReadMask                               0xFFFFFFFF
#define GCREG_DERGB2_YUV_COEF0_WriteMask                              0xFFFFFFFF
#define GCREG_DERGB2_YUV_COEF0_ResetValue                             0x00000000

#define GCREG_DERGB2_YUV_COEF0_COEFFICIENT0                                 15:0
#define GCREG_DERGB2_YUV_COEF0_COEFFICIENT0_End                               15
#define GCREG_DERGB2_YUV_COEF0_COEFFICIENT0_Start                              0
#define GCREG_DERGB2_YUV_COEF0_COEFFICIENT0_Type                             U16

#define GCREG_DERGB2_YUV_COEF0_COEFFICIENT1                                31:16
#define GCREG_DERGB2_YUV_COEF0_COEFFICIENT1_End                               31
#define GCREG_DERGB2_YUV_COEF0_COEFFICIENT1_Start                             16
#define GCREG_DERGB2_YUV_COEF0_COEFFICIENT1_Type                             U16

#define gcregDERGB2YUVCoef1RegAddrs                                       0x04D9
#define GCREG_DERGB2_YUV_COEF1_Address                                   0x01364
#define GCREG_DERGB2_YUV_COEF1_MSB                                            15
#define GCREG_DERGB2_YUV_COEF1_LSB                                             0
#define GCREG_DERGB2_YUV_COEF1_BLK                                             0
#define GCREG_DERGB2_YUV_COEF1_Count                                           1
#define GCREG_DERGB2_YUV_COEF1_FieldMask                              0xFFFFFFFF
#define GCREG_DERGB2_YUV_COEF1_ReadMask                               0xFFFFFFFF
#define GCREG_DERGB2_YUV_COEF1_WriteMask                              0xFFFFFFFF
#define GCREG_DERGB2_YUV_COEF1_ResetValue                             0x00000000

#define GCREG_DERGB2_YUV_COEF1_COEFFICIENT0                                 15:0
#define GCREG_DERGB2_YUV_COEF1_COEFFICIENT0_End                               15
#define GCREG_DERGB2_YUV_COEF1_COEFFICIENT0_Start                              0
#define GCREG_DERGB2_YUV_COEF1_COEFFICIENT0_Type                             U16

#define GCREG_DERGB2_YUV_COEF1_COEFFICIENT1                                31:16
#define GCREG_DERGB2_YUV_COEF1_COEFFICIENT1_End                               31
#define GCREG_DERGB2_YUV_COEF1_COEFFICIENT1_Start                             16
#define GCREG_DERGB2_YUV_COEF1_COEFFICIENT1_Type                             U16

#define gcregDERGB2YUVCoef2RegAddrs                                       0x04DA
#define GCREG_DERGB2_YUV_COEF2_Address                                   0x01368
#define GCREG_DERGB2_YUV_COEF2_MSB                                            15
#define GCREG_DERGB2_YUV_COEF2_LSB                                             0
#define GCREG_DERGB2_YUV_COEF2_BLK                                             0
#define GCREG_DERGB2_YUV_COEF2_Count                                           1
#define GCREG_DERGB2_YUV_COEF2_FieldMask                              0xFFFFFFFF
#define GCREG_DERGB2_YUV_COEF2_ReadMask                               0xFFFFFFFF
#define GCREG_DERGB2_YUV_COEF2_WriteMask                              0xFFFFFFFF
#define GCREG_DERGB2_YUV_COEF2_ResetValue                             0x00000000

#define GCREG_DERGB2_YUV_COEF2_COEFFICIENT0                                 15:0
#define GCREG_DERGB2_YUV_COEF2_COEFFICIENT0_End                               15
#define GCREG_DERGB2_YUV_COEF2_COEFFICIENT0_Start                              0
#define GCREG_DERGB2_YUV_COEF2_COEFFICIENT0_Type                             U16

#define GCREG_DERGB2_YUV_COEF2_COEFFICIENT1                                31:16
#define GCREG_DERGB2_YUV_COEF2_COEFFICIENT1_End                               31
#define GCREG_DERGB2_YUV_COEF2_COEFFICIENT1_Start                             16
#define GCREG_DERGB2_YUV_COEF2_COEFFICIENT1_Type                             U16

#define gcregDERGB2YUVCoef3RegAddrs                                       0x04DB
#define GCREG_DERGB2_YUV_COEF3_Address                                   0x0136C
#define GCREG_DERGB2_YUV_COEF3_MSB                                            15
#define GCREG_DERGB2_YUV_COEF3_LSB                                             0
#define GCREG_DERGB2_YUV_COEF3_BLK                                             0
#define GCREG_DERGB2_YUV_COEF3_Count                                           1
#define GCREG_DERGB2_YUV_COEF3_FieldMask                              0xFFFFFFFF
#define GCREG_DERGB2_YUV_COEF3_ReadMask                               0xFFFFFFFF
#define GCREG_DERGB2_YUV_COEF3_WriteMask                              0xFFFFFFFF
#define GCREG_DERGB2_YUV_COEF3_ResetValue                             0x00000000

#define GCREG_DERGB2_YUV_COEF3_COEFFICIENT0                                 15:0
#define GCREG_DERGB2_YUV_COEF3_COEFFICIENT0_End                               15
#define GCREG_DERGB2_YUV_COEF3_COEFFICIENT0_Start                              0
#define GCREG_DERGB2_YUV_COEF3_COEFFICIENT0_Type                             U16

#define GCREG_DERGB2_YUV_COEF3_COEFFICIENT1                                31:16
#define GCREG_DERGB2_YUV_COEF3_COEFFICIENT1_End                               31
#define GCREG_DERGB2_YUV_COEF3_COEFFICIENT1_Start                             16
#define GCREG_DERGB2_YUV_COEF3_COEFFICIENT1_Type                             U16

#define gcregDERGB2YUVCoef4RegAddrs                                       0x04DC
#define GCREG_DERGB2_YUV_COEF4_Address                                   0x01370
#define GCREG_DERGB2_YUV_COEF4_MSB                                            15
#define GCREG_DERGB2_YUV_COEF4_LSB                                             0
#define GCREG_DERGB2_YUV_COEF4_BLK                                             0
#define GCREG_DERGB2_YUV_COEF4_Count                                           1
#define GCREG_DERGB2_YUV_COEF4_FieldMask                              0xFFFFFFFF
#define GCREG_DERGB2_YUV_COEF4_ReadMask                               0xFFFFFFFF
#define GCREG_DERGB2_YUV_COEF4_WriteMask                              0xFFFFFFFF
#define GCREG_DERGB2_YUV_COEF4_ResetValue                             0x00000000

#define GCREG_DERGB2_YUV_COEF4_COEFFICIENT0                                 15:0
#define GCREG_DERGB2_YUV_COEF4_COEFFICIENT0_End                               15
#define GCREG_DERGB2_YUV_COEF4_COEFFICIENT0_Start                              0
#define GCREG_DERGB2_YUV_COEF4_COEFFICIENT0_Type                             U16

#define GCREG_DERGB2_YUV_COEF4_COEFFICIENT1                                31:16
#define GCREG_DERGB2_YUV_COEF4_COEFFICIENT1_End                               31
#define GCREG_DERGB2_YUV_COEF4_COEFFICIENT1_Start                             16
#define GCREG_DERGB2_YUV_COEF4_COEFFICIENT1_Type                             U16

#define gcregDERGB2YUVCoefD0RegAddrs                                      0x04DD
#define GCREG_DERGB2_YUV_COEF_D0_Address                                 0x01374
#define GCREG_DERGB2_YUV_COEF_D0_MSB                                          15
#define GCREG_DERGB2_YUV_COEF_D0_LSB                                           0
#define GCREG_DERGB2_YUV_COEF_D0_BLK                                           0
#define GCREG_DERGB2_YUV_COEF_D0_Count                                         1
#define GCREG_DERGB2_YUV_COEF_D0_FieldMask                            0x01FFFFFF
#define GCREG_DERGB2_YUV_COEF_D0_ReadMask                             0x01FFFFFF
#define GCREG_DERGB2_YUV_COEF_D0_WriteMask                            0x01FFFFFF
#define GCREG_DERGB2_YUV_COEF_D0_ResetValue                           0x00000000

#define GCREG_DERGB2_YUV_COEF_D0_COEFFICIENT                                24:0
#define GCREG_DERGB2_YUV_COEF_D0_COEFFICIENT_End                              24
#define GCREG_DERGB2_YUV_COEF_D0_COEFFICIENT_Start                             0
#define GCREG_DERGB2_YUV_COEF_D0_COEFFICIENT_Type                            U25

#define gcregDERGB2YUVCoefD1RegAddrs                                      0x04DE
#define GCREG_DERGB2_YUV_COEF_D1_Address                                 0x01378
#define GCREG_DERGB2_YUV_COEF_D1_MSB                                          15
#define GCREG_DERGB2_YUV_COEF_D1_LSB                                           0
#define GCREG_DERGB2_YUV_COEF_D1_BLK                                           0
#define GCREG_DERGB2_YUV_COEF_D1_Count                                         1
#define GCREG_DERGB2_YUV_COEF_D1_FieldMask                            0x01FFFFFF
#define GCREG_DERGB2_YUV_COEF_D1_ReadMask                             0x01FFFFFF
#define GCREG_DERGB2_YUV_COEF_D1_WriteMask                            0x01FFFFFF
#define GCREG_DERGB2_YUV_COEF_D1_ResetValue                           0x00000000

#define GCREG_DERGB2_YUV_COEF_D1_COEFFICIENT                                24:0
#define GCREG_DERGB2_YUV_COEF_D1_COEFFICIENT_End                              24
#define GCREG_DERGB2_YUV_COEF_D1_COEFFICIENT_Start                             0
#define GCREG_DERGB2_YUV_COEF_D1_COEFFICIENT_Type                            U25

#define gcregDERGB2YUVCoefD2RegAddrs                                      0x04DF
#define GCREG_DERGB2_YUV_COEF_D2_Address                                 0x0137C
#define GCREG_DERGB2_YUV_COEF_D2_MSB                                          15
#define GCREG_DERGB2_YUV_COEF_D2_LSB                                           0
#define GCREG_DERGB2_YUV_COEF_D2_BLK                                           0
#define GCREG_DERGB2_YUV_COEF_D2_Count                                         1
#define GCREG_DERGB2_YUV_COEF_D2_FieldMask                            0x01FFFFFF
#define GCREG_DERGB2_YUV_COEF_D2_ReadMask                             0x01FFFFFF
#define GCREG_DERGB2_YUV_COEF_D2_WriteMask                            0x01FFFFFF
#define GCREG_DERGB2_YUV_COEF_D2_ResetValue                           0x00000000

#define GCREG_DERGB2_YUV_COEF_D2_COEFFICIENT                                24:0
#define GCREG_DERGB2_YUV_COEF_D2_COEFFICIENT_End                              24
#define GCREG_DERGB2_YUV_COEF_D2_COEFFICIENT_Start                             0
#define GCREG_DERGB2_YUV_COEF_D2_COEFFICIENT_Type                            U25

#define gcregDEWalkerConfigRegAddrs                                       0x04E0
#define GCREG_DE_WALKER_CONFIG_Address                                   0x01380
#define GCREG_DE_WALKER_CONFIG_MSB                                            15
#define GCREG_DE_WALKER_CONFIG_LSB                                             0
#define GCREG_DE_WALKER_CONFIG_BLK                                             0
#define GCREG_DE_WALKER_CONFIG_Count                                           1
#define GCREG_DE_WALKER_CONFIG_FieldMask                              0x0000001F
#define GCREG_DE_WALKER_CONFIG_ReadMask                               0x0000001F
#define GCREG_DE_WALKER_CONFIG_WriteMask                              0x0000001F
#define GCREG_DE_WALKER_CONFIG_ResetValue                             0x00000000

#define GCREG_DE_WALKER_CONFIG_DISABLE_MULTI_SRC_V2                          0:0
#define GCREG_DE_WALKER_CONFIG_DISABLE_MULTI_SRC_V2_End                        0
#define GCREG_DE_WALKER_CONFIG_DISABLE_MULTI_SRC_V2_Start                      0
#define GCREG_DE_WALKER_CONFIG_DISABLE_MULTI_SRC_V2_Type                     U01
#define   GCREG_DE_WALKER_CONFIG_DISABLE_MULTI_SRC_V2_DISABLED               0x0
#define   GCREG_DE_WALKER_CONFIG_DISABLE_MULTI_SRC_V2_ENABLED                0x1

#define GCREG_DE_WALKER_CONFIG_MASK_DISABLE_MULTI_SRC_V2                     1:1
#define GCREG_DE_WALKER_CONFIG_MASK_DISABLE_MULTI_SRC_V2_End                   1
#define GCREG_DE_WALKER_CONFIG_MASK_DISABLE_MULTI_SRC_V2_Start                 1
#define GCREG_DE_WALKER_CONFIG_MASK_DISABLE_MULTI_SRC_V2_Type                U01
#define   GCREG_DE_WALKER_CONFIG_MASK_DISABLE_MULTI_SRC_V2_ENABLED           0x0
#define   GCREG_DE_WALKER_CONFIG_MASK_DISABLE_MULTI_SRC_V2_MASKED            0x1

#define GCREG_DE_WALKER_CONFIG_HORIZONTAL_ALIGN                              3:2
#define GCREG_DE_WALKER_CONFIG_HORIZONTAL_ALIGN_End                            3
#define GCREG_DE_WALKER_CONFIG_HORIZONTAL_ALIGN_Start                          2
#define GCREG_DE_WALKER_CONFIG_HORIZONTAL_ALIGN_Type                         U02
#define   GCREG_DE_WALKER_CONFIG_HORIZONTAL_ALIGN_DISABLED                   0x0
#define   GCREG_DE_WALKER_CONFIG_HORIZONTAL_ALIGN_ALIGN16                    0x1
#define   GCREG_DE_WALKER_CONFIG_HORIZONTAL_ALIGN_ALIGN32                    0x2
#define   GCREG_DE_WALKER_CONFIG_HORIZONTAL_ALIGN_ALIGN64                    0x3

#define GCREG_DE_WALKER_CONFIG_MASK_HORIZONTAL_ALIGN                         4:4
#define GCREG_DE_WALKER_CONFIG_MASK_HORIZONTAL_ALIGN_End                       4
#define GCREG_DE_WALKER_CONFIG_MASK_HORIZONTAL_ALIGN_Start                     4
#define GCREG_DE_WALKER_CONFIG_MASK_HORIZONTAL_ALIGN_Type                    U01
#define   GCREG_DE_WALKER_CONFIG_MASK_HORIZONTAL_ALIGN_ENABLED               0x0
#define   GCREG_DE_WALKER_CONFIG_MASK_HORIZONTAL_ALIGN_MASKED                0x1

#define gcregDEDstTileStatusAddressRegAddrs                               0x04E1
#define GCREG_DE_DST_TILE_STATUS_ADDRESS_Address                         0x01384
#define GCREG_DE_DST_TILE_STATUS_ADDRESS_MSB                                  15
#define GCREG_DE_DST_TILE_STATUS_ADDRESS_LSB                                   0
#define GCREG_DE_DST_TILE_STATUS_ADDRESS_BLK                                   0
#define GCREG_DE_DST_TILE_STATUS_ADDRESS_Count                                 1
#define GCREG_DE_DST_TILE_STATUS_ADDRESS_FieldMask                    0xFFFFFFFF
#define GCREG_DE_DST_TILE_STATUS_ADDRESS_ReadMask                     0xFFFFFFFC
#define GCREG_DE_DST_TILE_STATUS_ADDRESS_WriteMask                    0xFFFFFFFC
#define GCREG_DE_DST_TILE_STATUS_ADDRESS_ResetValue                   0x00000000

#define GCREG_DE_DST_TILE_STATUS_ADDRESS_TYPE                              31:31
#define GCREG_DE_DST_TILE_STATUS_ADDRESS_TYPE_End                             31
#define GCREG_DE_DST_TILE_STATUS_ADDRESS_TYPE_Start                           31
#define GCREG_DE_DST_TILE_STATUS_ADDRESS_TYPE_Type                           U01
#define   GCREG_DE_DST_TILE_STATUS_ADDRESS_TYPE_SYSTEM                       0x0
#define   GCREG_DE_DST_TILE_STATUS_ADDRESS_TYPE_VIRTUAL_SYSTEM               0x1

#define GCREG_DE_DST_TILE_STATUS_ADDRESS_ADDRESS                            30:0
#define GCREG_DE_DST_TILE_STATUS_ADDRESS_ADDRESS_End                          30
#define GCREG_DE_DST_TILE_STATUS_ADDRESS_ADDRESS_Start                         0
#define GCREG_DE_DST_TILE_STATUS_ADDRESS_ADDRESS_Type                        U31

#define gcregDEReorderControlRegAddrs                                     0x04E2
#define GCREG_DE_REORDER_CONTROL_Address                                 0x01388
#define GCREG_DE_REORDER_CONTROL_MSB                                          15
#define GCREG_DE_REORDER_CONTROL_LSB                                           0
#define GCREG_DE_REORDER_CONTROL_BLK                                           0
#define GCREG_DE_REORDER_CONTROL_Count                                         1
#define GCREG_DE_REORDER_CONTROL_FieldMask                            0x0000007F
#define GCREG_DE_REORDER_CONTROL_ReadMask                             0x0000007F
#define GCREG_DE_REORDER_CONTROL_WriteMask                            0x0000007F
#define GCREG_DE_REORDER_CONTROL_ResetValue                           0x0000001C

#define GCREG_DE_REORDER_CONTROL_COUNTER                                     5:0
#define GCREG_DE_REORDER_CONTROL_COUNTER_End                                   5
#define GCREG_DE_REORDER_CONTROL_COUNTER_Start                                 0
#define GCREG_DE_REORDER_CONTROL_COUNTER_Type                                U06

#define GCREG_DE_REORDER_CONTROL_MASK_COUNTER                                6:6
#define GCREG_DE_REORDER_CONTROL_MASK_COUNTER_End                              6
#define GCREG_DE_REORDER_CONTROL_MASK_COUNTER_Start                            6
#define GCREG_DE_REORDER_CONTROL_MASK_COUNTER_Type                           U01
#define   GCREG_DE_REORDER_CONTROL_MASK_COUNTER_ENABLED                      0x0
#define   GCREG_DE_REORDER_CONTROL_MASK_COUNTER_MASKED                       0x1

#define gcreg3rdPartyCompressionDstConfigRegAddrs                         0x04E3
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_Address                    0x0138C
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_MSB                             15
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_LSB                              0
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_BLK                              0
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_Count                            1
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FieldMask               0x000003FF
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_ReadMask                0x000003FF
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_WriteMask               0x000003FF
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_ResetValue              0x00000000

#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_TPC                            0:0
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_TPC_End                          0
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_TPC_Start                        0
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_TPC_Type                       U01
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_TPC_DISABLED                 0x0
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_TPC_ENABLED                  0x1

#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_MASK_TPC                       1:1
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_MASK_TPC_End                     1
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_MASK_TPC_Start                   1
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_MASK_TPC_Type                  U01
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_MASK_TPC_ENABLED             0x0
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_MASK_TPC_MASKED              0x1

#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_BY_PASS_TPC                    2:2
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_BY_PASS_TPC_End                  2
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_BY_PASS_TPC_Start                2
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_BY_PASS_TPC_Type               U01
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_BY_PASS_TPC_DISABLED         0x0
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_BY_PASS_TPC_ENABLED          0x1

#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_MASK_BY_PASS_TPC               3:3
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_MASK_BY_PASS_TPC_End             3
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_MASK_BY_PASS_TPC_Start           3
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_MASK_BY_PASS_TPC_Type          U01
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_MASK_BY_PASS_TPC_ENABLED     0x0
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_MASK_BY_PASS_TPC_MASKED      0x1

#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT                         8:4
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_End                       8
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_Start                     4
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_Type                    U05
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_X4R4G4B4             0x00
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_A4R4G4B4             0x01
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_X1R5G5B5             0x02
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_A1R5G5B5             0x03
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_R5G6B5               0x04
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_X8R8G8B8             0x05
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_A8R8G8B8             0x06
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_YUY2                 0x07
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_UYVY                 0x08
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_INDEX8               0x09
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_MONOCHROME           0x0A
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_AYUV                 0x0B
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_RGB888_PLANAR        0x0C
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_RGB888_PACKED        0x0D
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_EXTENSION            0x0E
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_YV12                 0x0F
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_A8                   0x10
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_NV12                 0x11
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_NV16                 0x12
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_RG16                 0x13
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_R8                   0x14
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_NV12_10BIT           0x15
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_A2R10G10B10          0x16
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_NV16_10BIT           0x17
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_INDEX1               0x18
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_INDEX2               0x19
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_INDEX4               0x1A
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_P010                 0x1B
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_NV12_10BIT_L1        0x1C
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_NV16_10BIT_L1        0x1D
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_FORMAT_I010                 0x1E

#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_MASK_FORMAT                    9:9
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_MASK_FORMAT_End                  9
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_MASK_FORMAT_Start                9
#define GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_MASK_FORMAT_Type               U01
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_MASK_FORMAT_ENABLED          0x0
#define   GCREG3RD_PARTY_COMPRESSION_DST_CONFIG_MASK_FORMAT_MASKED           0x1

#define gcreg3rdPartyCompressionDstHeaderAddressRegAddrs                  0x04E4
#define GCREG3RD_PARTY_COMPRESSION_DST_HEADER_ADDRESS_Address            0x01390
#define GCREG3RD_PARTY_COMPRESSION_DST_HEADER_ADDRESS_MSB                     15
#define GCREG3RD_PARTY_COMPRESSION_DST_HEADER_ADDRESS_LSB                      0
#define GCREG3RD_PARTY_COMPRESSION_DST_HEADER_ADDRESS_BLK                      0
#define GCREG3RD_PARTY_COMPRESSION_DST_HEADER_ADDRESS_Count                    1
#define GCREG3RD_PARTY_COMPRESSION_DST_HEADER_ADDRESS_FieldMask       0xFFFFFFFF
#define GCREG3RD_PARTY_COMPRESSION_DST_HEADER_ADDRESS_ReadMask        0xFFFFFFFC
#define GCREG3RD_PARTY_COMPRESSION_DST_HEADER_ADDRESS_WriteMask       0xFFFFFFFC
#define GCREG3RD_PARTY_COMPRESSION_DST_HEADER_ADDRESS_ResetValue      0x00000000

#define GCREG3RD_PARTY_COMPRESSION_DST_HEADER_ADDRESS_TYPE                 31:31
#define GCREG3RD_PARTY_COMPRESSION_DST_HEADER_ADDRESS_TYPE_End                31
#define GCREG3RD_PARTY_COMPRESSION_DST_HEADER_ADDRESS_TYPE_Start              31
#define GCREG3RD_PARTY_COMPRESSION_DST_HEADER_ADDRESS_TYPE_Type              U01
#define   GCREG3RD_PARTY_COMPRESSION_DST_HEADER_ADDRESS_TYPE_SYSTEM          0x0
#define   GCREG3RD_PARTY_COMPRESSION_DST_HEADER_ADDRESS_TYPE_VIRTUAL_SYSTEM  0x1

#define GCREG3RD_PARTY_COMPRESSION_DST_HEADER_ADDRESS_ADDRESS               30:0
#define GCREG3RD_PARTY_COMPRESSION_DST_HEADER_ADDRESS_ADDRESS_End             30
#define GCREG3RD_PARTY_COMPRESSION_DST_HEADER_ADDRESS_ADDRESS_Start            0
#define GCREG3RD_PARTY_COMPRESSION_DST_HEADER_ADDRESS_ADDRESS_Type           U31

#define gcreg3rdPartyCompressionDstPayloadAddressRegAddrs                 0x04E5
#define GCREG3RD_PARTY_COMPRESSION_DST_PAYLOAD_ADDRESS_Address           0x01394
#define GCREG3RD_PARTY_COMPRESSION_DST_PAYLOAD_ADDRESS_MSB                    15
#define GCREG3RD_PARTY_COMPRESSION_DST_PAYLOAD_ADDRESS_LSB                     0
#define GCREG3RD_PARTY_COMPRESSION_DST_PAYLOAD_ADDRESS_BLK                     0
#define GCREG3RD_PARTY_COMPRESSION_DST_PAYLOAD_ADDRESS_Count                   1
#define GCREG3RD_PARTY_COMPRESSION_DST_PAYLOAD_ADDRESS_FieldMask      0xFFFFFFFF
#define GCREG3RD_PARTY_COMPRESSION_DST_PAYLOAD_ADDRESS_ReadMask       0xFFFFFFFC
#define GCREG3RD_PARTY_COMPRESSION_DST_PAYLOAD_ADDRESS_WriteMask      0xFFFFFFFC
#define GCREG3RD_PARTY_COMPRESSION_DST_PAYLOAD_ADDRESS_ResetValue     0x00000000

#define GCREG3RD_PARTY_COMPRESSION_DST_PAYLOAD_ADDRESS_TYPE                31:31
#define GCREG3RD_PARTY_COMPRESSION_DST_PAYLOAD_ADDRESS_TYPE_End               31
#define GCREG3RD_PARTY_COMPRESSION_DST_PAYLOAD_ADDRESS_TYPE_Start             31
#define GCREG3RD_PARTY_COMPRESSION_DST_PAYLOAD_ADDRESS_TYPE_Type             U01
#define   GCREG3RD_PARTY_COMPRESSION_DST_PAYLOAD_ADDRESS_TYPE_SYSTEM         0x0
#define   GCREG3RD_PARTY_COMPRESSION_DST_PAYLOAD_ADDRESS_TYPE_VIRTUAL_SYSTEM 0x1

#define GCREG3RD_PARTY_COMPRESSION_DST_PAYLOAD_ADDRESS_ADDRESS              30:0
#define GCREG3RD_PARTY_COMPRESSION_DST_PAYLOAD_ADDRESS_ADDRESS_End            30
#define GCREG3RD_PARTY_COMPRESSION_DST_PAYLOAD_ADDRESS_ADDRESS_Start           0
#define GCREG3RD_PARTY_COMPRESSION_DST_PAYLOAD_ADDRESS_ADDRESS_Type          U31

#define gcregDEDstTileStatusClearRegAddrs                                 0x04E6
#define GCREG_DE_DST_TILE_STATUS_CLEAR_Address                           0x01398
#define GCREG_DE_DST_TILE_STATUS_CLEAR_MSB                                    15
#define GCREG_DE_DST_TILE_STATUS_CLEAR_LSB                                     0
#define GCREG_DE_DST_TILE_STATUS_CLEAR_BLK                                     0
#define GCREG_DE_DST_TILE_STATUS_CLEAR_Count                                   1
#define GCREG_DE_DST_TILE_STATUS_CLEAR_FieldMask                      0xFFFFFFFF
#define GCREG_DE_DST_TILE_STATUS_CLEAR_ReadMask                       0xFFFFFFFF
#define GCREG_DE_DST_TILE_STATUS_CLEAR_WriteMask                      0xFFFFFFFF
#define GCREG_DE_DST_TILE_STATUS_CLEAR_ResetValue                     0x00000000

#define GCREG_DE_DST_TILE_STATUS_CLEAR_VALUE                                31:0
#define GCREG_DE_DST_TILE_STATUS_CLEAR_VALUE_End                              31
#define GCREG_DE_DST_TILE_STATUS_CLEAR_VALUE_Start                             0
#define GCREG_DE_DST_TILE_STATUS_CLEAR_VALUE_Type                            U32

#define gcregDEDstTileStatusConfigRegAddrs                                0x04E7
#define GCREG_DE_DST_TILE_STATUS_CONFIG_Address                          0x0139C
#define GCREG_DE_DST_TILE_STATUS_CONFIG_MSB                                   15
#define GCREG_DE_DST_TILE_STATUS_CONFIG_LSB                                    0
#define GCREG_DE_DST_TILE_STATUS_CONFIG_BLK                                    0
#define GCREG_DE_DST_TILE_STATUS_CONFIG_Count                                  1
#define GCREG_DE_DST_TILE_STATUS_CONFIG_FieldMask                     0x001C00F3
#define GCREG_DE_DST_TILE_STATUS_CONFIG_ReadMask                      0x001C00F3
#define GCREG_DE_DST_TILE_STATUS_CONFIG_WriteMask                     0x001C00F3
#define GCREG_DE_DST_TILE_STATUS_CONFIG_ResetValue                    0x00000000

#define GCREG_DE_DST_TILE_STATUS_CONFIG_CONTROL                              1:0
#define GCREG_DE_DST_TILE_STATUS_CONFIG_CONTROL_End                            1
#define GCREG_DE_DST_TILE_STATUS_CONFIG_CONTROL_Start                          0
#define GCREG_DE_DST_TILE_STATUS_CONFIG_CONTROL_Type                         U02

#define   GCREG_DE_DST_TILE_STATUS_CONFIG_CONTROL_DISABLE                    0x0

#define   GCREG_DE_DST_TILE_STATUS_CONFIG_CONTROL_ENABLE                     0x1

#define   GCREG_DE_DST_TILE_STATUS_CONFIG_CONTROL_COMPRESSED                 0x3

#define GCREG_DE_DST_TILE_STATUS_CONFIG_FORMAT                               7:4
#define GCREG_DE_DST_TILE_STATUS_CONFIG_FORMAT_End                             7
#define GCREG_DE_DST_TILE_STATUS_CONFIG_FORMAT_Start                           4
#define GCREG_DE_DST_TILE_STATUS_CONFIG_FORMAT_Type                          U04
#define   GCREG_DE_DST_TILE_STATUS_CONFIG_FORMAT_ARGB4                       0x0
#define   GCREG_DE_DST_TILE_STATUS_CONFIG_FORMAT_A1RGB5                      0x1
#define   GCREG_DE_DST_TILE_STATUS_CONFIG_FORMAT_R5G6B5                      0x2
#define   GCREG_DE_DST_TILE_STATUS_CONFIG_FORMAT_ARGB8                       0x3
#define   GCREG_DE_DST_TILE_STATUS_CONFIG_FORMAT_RGB8                        0x4
#define   GCREG_DE_DST_TILE_STATUS_CONFIG_FORMAT_Z24S8                       0x5
#define   GCREG_DE_DST_TILE_STATUS_CONFIG_FORMAT_Z24                         0x6
#define   GCREG_DE_DST_TILE_STATUS_CONFIG_FORMAT_VAA16                       0x7
#define   GCREG_DE_DST_TILE_STATUS_CONFIG_FORMAT_Z16                         0x8
#define   GCREG_DE_DST_TILE_STATUS_CONFIG_FORMAT_S8                          0x9

#define GCREG_DE_DST_TILE_STATUS_CONFIG_MSAA_COMPRESSION                   18:18
#define GCREG_DE_DST_TILE_STATUS_CONFIG_MSAA_COMPRESSION_End                  18
#define GCREG_DE_DST_TILE_STATUS_CONFIG_MSAA_COMPRESSION_Start                18
#define GCREG_DE_DST_TILE_STATUS_CONFIG_MSAA_COMPRESSION_Type                U01
#define   GCREG_DE_DST_TILE_STATUS_CONFIG_MSAA_COMPRESSION_DISABLE           0x0
#define   GCREG_DE_DST_TILE_STATUS_CONFIG_MSAA_COMPRESSION_ENABLE            0x1

#define GCREG_DE_DST_TILE_STATUS_CONFIG_TILE_MAJOR_TYPE                    19:19
#define GCREG_DE_DST_TILE_STATUS_CONFIG_TILE_MAJOR_TYPE_End                   19
#define GCREG_DE_DST_TILE_STATUS_CONFIG_TILE_MAJOR_TYPE_Start                 19
#define GCREG_DE_DST_TILE_STATUS_CONFIG_TILE_MAJOR_TYPE_Type                 U01
#define   GCREG_DE_DST_TILE_STATUS_CONFIG_TILE_MAJOR_TYPE_XMAJOR             0x0
#define   GCREG_DE_DST_TILE_STATUS_CONFIG_TILE_MAJOR_TYPE_YMAJOR             0x1

#define GCREG_DE_DST_TILE_STATUS_CONFIG_COMPRESSION_SIZE                   20:20
#define GCREG_DE_DST_TILE_STATUS_CONFIG_COMPRESSION_SIZE_End                  20
#define GCREG_DE_DST_TILE_STATUS_CONFIG_COMPRESSION_SIZE_Start                20
#define GCREG_DE_DST_TILE_STATUS_CONFIG_COMPRESSION_SIZE_Type                U01
#define   GCREG_DE_DST_TILE_STATUS_CONFIG_COMPRESSION_SIZE_Size128B          0x0
#define   GCREG_DE_DST_TILE_STATUS_CONFIG_COMPRESSION_SIZE_Size256B          0x1

#define gcregDEDstPlane0AlignConfigRegAddrs                               0x04E8
#define GCREG_DE_DST_PLANE0_ALIGN_CONFIG_Address                         0x013A0
#define GCREG_DE_DST_PLANE0_ALIGN_CONFIG_MSB                                  15
#define GCREG_DE_DST_PLANE0_ALIGN_CONFIG_LSB                                   0
#define GCREG_DE_DST_PLANE0_ALIGN_CONFIG_BLK                                   0
#define GCREG_DE_DST_PLANE0_ALIGN_CONFIG_Count                                 1
#define GCREG_DE_DST_PLANE0_ALIGN_CONFIG_FieldMask                    0x0001FFFF
#define GCREG_DE_DST_PLANE0_ALIGN_CONFIG_ReadMask                     0x0001FFFF
#define GCREG_DE_DST_PLANE0_ALIGN_CONFIG_WriteMask                    0x0001FFFF
#define GCREG_DE_DST_PLANE0_ALIGN_CONFIG_ResetValue                   0x00000000

#define GCREG_DE_DST_PLANE0_ALIGN_CONFIG_STRIDE_WIDTH                       16:0
#define GCREG_DE_DST_PLANE0_ALIGN_CONFIG_STRIDE_WIDTH_End                     16
#define GCREG_DE_DST_PLANE0_ALIGN_CONFIG_STRIDE_WIDTH_Start                    0
#define GCREG_DE_DST_PLANE0_ALIGN_CONFIG_STRIDE_WIDTH_Type                   U17

#define gcregDEDstPlane1AlignConfigRegAddrs                               0x04E9
#define GCREG_DE_DST_PLANE1_ALIGN_CONFIG_Address                         0x013A4
#define GCREG_DE_DST_PLANE1_ALIGN_CONFIG_MSB                                  15
#define GCREG_DE_DST_PLANE1_ALIGN_CONFIG_LSB                                   0
#define GCREG_DE_DST_PLANE1_ALIGN_CONFIG_BLK                                   0
#define GCREG_DE_DST_PLANE1_ALIGN_CONFIG_Count                                 1
#define GCREG_DE_DST_PLANE1_ALIGN_CONFIG_FieldMask                    0x0001FFFF
#define GCREG_DE_DST_PLANE1_ALIGN_CONFIG_ReadMask                     0x0001FFFF
#define GCREG_DE_DST_PLANE1_ALIGN_CONFIG_WriteMask                    0x0001FFFF
#define GCREG_DE_DST_PLANE1_ALIGN_CONFIG_ResetValue                   0x00000000

#define GCREG_DE_DST_PLANE1_ALIGN_CONFIG_STRIDE_WIDTH                       16:0
#define GCREG_DE_DST_PLANE1_ALIGN_CONFIG_STRIDE_WIDTH_End                     16
#define GCREG_DE_DST_PLANE1_ALIGN_CONFIG_STRIDE_WIDTH_Start                    0
#define GCREG_DE_DST_PLANE1_ALIGN_CONFIG_STRIDE_WIDTH_Type                   U17

#define gcregDEDstPlane2AlignConfigRegAddrs                               0x04EA
#define GCREG_DE_DST_PLANE2_ALIGN_CONFIG_Address                         0x013A8
#define GCREG_DE_DST_PLANE2_ALIGN_CONFIG_MSB                                  15
#define GCREG_DE_DST_PLANE2_ALIGN_CONFIG_LSB                                   0
#define GCREG_DE_DST_PLANE2_ALIGN_CONFIG_BLK                                   0
#define GCREG_DE_DST_PLANE2_ALIGN_CONFIG_Count                                 1
#define GCREG_DE_DST_PLANE2_ALIGN_CONFIG_FieldMask                    0x0001FFFF
#define GCREG_DE_DST_PLANE2_ALIGN_CONFIG_ReadMask                     0x0001FFFF
#define GCREG_DE_DST_PLANE2_ALIGN_CONFIG_WriteMask                    0x0001FFFF
#define GCREG_DE_DST_PLANE2_ALIGN_CONFIG_ResetValue                   0x00000000

#define GCREG_DE_DST_PLANE2_ALIGN_CONFIG_STRIDE_WIDTH                       16:0
#define GCREG_DE_DST_PLANE2_ALIGN_CONFIG_STRIDE_WIDTH_End                     16
#define GCREG_DE_DST_PLANE2_ALIGN_CONFIG_STRIDE_WIDTH_Start                    0
#define GCREG_DE_DST_PLANE2_ALIGN_CONFIG_STRIDE_WIDTH_Type                   U17

#define AQDESrcTileStatusAddressRegAddrs                                  0x04EB
#define AQDE_SRC_TILE_STATUS_ADDRESS_Address                             0x013AC
#define AQDE_SRC_TILE_STATUS_ADDRESS_MSB                                      15
#define AQDE_SRC_TILE_STATUS_ADDRESS_LSB                                       0
#define AQDE_SRC_TILE_STATUS_ADDRESS_BLK                                       0
#define AQDE_SRC_TILE_STATUS_ADDRESS_Count                                     1
#define AQDE_SRC_TILE_STATUS_ADDRESS_FieldMask                        0xFFFFFFFF
#define AQDE_SRC_TILE_STATUS_ADDRESS_ReadMask                         0xFFFFFFFC
#define AQDE_SRC_TILE_STATUS_ADDRESS_WriteMask                        0xFFFFFFFC
#define AQDE_SRC_TILE_STATUS_ADDRESS_ResetValue                       0x00000000

#define AQDE_SRC_TILE_STATUS_ADDRESS_TYPE                                  31:31
#define AQDE_SRC_TILE_STATUS_ADDRESS_TYPE_End                                 31
#define AQDE_SRC_TILE_STATUS_ADDRESS_TYPE_Start                               31
#define AQDE_SRC_TILE_STATUS_ADDRESS_TYPE_Type                               U01
#define   AQDE_SRC_TILE_STATUS_ADDRESS_TYPE_SYSTEM                           0x0
#define   AQDE_SRC_TILE_STATUS_ADDRESS_TYPE_VIRTUAL_SYSTEM                   0x1

#define AQDE_SRC_TILE_STATUS_ADDRESS_ADDRESS                                30:0
#define AQDE_SRC_TILE_STATUS_ADDRESS_ADDRESS_End                              30
#define AQDE_SRC_TILE_STATUS_ADDRESS_ADDRESS_Start                             0
#define AQDE_SRC_TILE_STATUS_ADDRESS_ADDRESS_Type                            U31

#define AQDESrcTileStatusClearRegAddrs                                    0x04EC
#define AQDE_SRC_TILE_STATUS_CLEAR_Address                               0x013B0
#define AQDE_SRC_TILE_STATUS_CLEAR_MSB                                        15
#define AQDE_SRC_TILE_STATUS_CLEAR_LSB                                         0
#define AQDE_SRC_TILE_STATUS_CLEAR_BLK                                         0
#define AQDE_SRC_TILE_STATUS_CLEAR_Count                                       1
#define AQDE_SRC_TILE_STATUS_CLEAR_FieldMask                          0xFFFFFFFF
#define AQDE_SRC_TILE_STATUS_CLEAR_ReadMask                           0xFFFFFFFF
#define AQDE_SRC_TILE_STATUS_CLEAR_WriteMask                          0xFFFFFFFF
#define AQDE_SRC_TILE_STATUS_CLEAR_ResetValue                         0x00000000

#define AQDE_SRC_TILE_STATUS_CLEAR_VALUE                                    31:0
#define AQDE_SRC_TILE_STATUS_CLEAR_VALUE_End                                  31
#define AQDE_SRC_TILE_STATUS_CLEAR_VALUE_Start                                 0
#define AQDE_SRC_TILE_STATUS_CLEAR_VALUE_Type                                U32

#define AQDESrcTileStatusConfigRegAddrs                                   0x04ED
#define AQDE_SRC_TILE_STATUS_CONFIG_Address                              0x013B4
#define AQDE_SRC_TILE_STATUS_CONFIG_MSB                                       15
#define AQDE_SRC_TILE_STATUS_CONFIG_LSB                                        0
#define AQDE_SRC_TILE_STATUS_CONFIG_BLK                                        0
#define AQDE_SRC_TILE_STATUS_CONFIG_Count                                      1
#define AQDE_SRC_TILE_STATUS_CONFIG_FieldMask                         0x0000010C
#define AQDE_SRC_TILE_STATUS_CONFIG_ReadMask                          0x0000010C
#define AQDE_SRC_TILE_STATUS_CONFIG_WriteMask                         0x0000010C
#define AQDE_SRC_TILE_STATUS_CONFIG_ResetValue                        0x00000000

#define AQDE_SRC_TILE_STATUS_CONFIG_FAST_CLEAR_EN_C                          2:2
#define AQDE_SRC_TILE_STATUS_CONFIG_FAST_CLEAR_EN_C_End                        2
#define AQDE_SRC_TILE_STATUS_CONFIG_FAST_CLEAR_EN_C_Start                      2
#define AQDE_SRC_TILE_STATUS_CONFIG_FAST_CLEAR_EN_C_Type                     U01
#define   AQDE_SRC_TILE_STATUS_CONFIG_FAST_CLEAR_EN_C_DISABLE                0x0
#define   AQDE_SRC_TILE_STATUS_CONFIG_FAST_CLEAR_EN_C_ENABLE                 0x1

#define AQDE_SRC_TILE_STATUS_CONFIG_COLOR64                                  3:3
#define AQDE_SRC_TILE_STATUS_CONFIG_COLOR64_End                                3
#define AQDE_SRC_TILE_STATUS_CONFIG_COLOR64_Start                              3
#define AQDE_SRC_TILE_STATUS_CONFIG_COLOR64_Type                             U01
#define   AQDE_SRC_TILE_STATUS_CONFIG_COLOR64_COLOR16OR32                    0x0
#define   AQDE_SRC_TILE_STATUS_CONFIG_COLOR64_COLOR64                        0x1

#define AQDE_SRC_TILE_STATUS_CONFIG_TILE_STATUS4_BIT                         8:8
#define AQDE_SRC_TILE_STATUS_CONFIG_TILE_STATUS4_BIT_End                       8
#define AQDE_SRC_TILE_STATUS_CONFIG_TILE_STATUS4_BIT_Start                     8
#define AQDE_SRC_TILE_STATUS_CONFIG_TILE_STATUS4_BIT_Type                    U01
#define   AQDE_SRC_TILE_STATUS_CONFIG_TILE_STATUS4_BIT_DISABLED              0x0
#define   AQDE_SRC_TILE_STATUS_CONFIG_TILE_STATUS4_BIT_ENABLED               0x1

#define AQDESrcTileStatusClear64RegAddrs                                  0x04EE
#define AQDE_SRC_TILE_STATUS_CLEAR64_Address                             0x013B8
#define AQDE_SRC_TILE_STATUS_CLEAR64_MSB                                      15
#define AQDE_SRC_TILE_STATUS_CLEAR64_LSB                                       0
#define AQDE_SRC_TILE_STATUS_CLEAR64_BLK                                       0
#define AQDE_SRC_TILE_STATUS_CLEAR64_Count                                     1
#define AQDE_SRC_TILE_STATUS_CLEAR64_FieldMask                        0xFFFFFFFF
#define AQDE_SRC_TILE_STATUS_CLEAR64_ReadMask                         0xFFFFFFFF
#define AQDE_SRC_TILE_STATUS_CLEAR64_WriteMask                        0xFFFFFFFF
#define AQDE_SRC_TILE_STATUS_CLEAR64_ResetValue                       0x00000000

#define AQDE_SRC_TILE_STATUS_CLEAR64_VALUE                                  31:0
#define AQDE_SRC_TILE_STATUS_CLEAR64_VALUE_End                                31
#define AQDE_SRC_TILE_STATUS_CLEAR64_VALUE_Start                               0
#define AQDE_SRC_TILE_STATUS_CLEAR64_VALUE_Type                              U32

#define AQDESrcConfigExRegAddrs                                           0x04EF
#define AQDE_SRC_CONFIG_EX_Address                                       0x013BC
#define AQDE_SRC_CONFIG_EX_MSB                                                15
#define AQDE_SRC_CONFIG_EX_LSB                                                 0
#define AQDE_SRC_CONFIG_EX_BLK                                                 0
#define AQDE_SRC_CONFIG_EX_Count                                               1
#define AQDE_SRC_CONFIG_EX_FieldMask                                  0x0000000F
#define AQDE_SRC_CONFIG_EX_ReadMask                                   0x0000000F
#define AQDE_SRC_CONFIG_EX_WriteMask                                  0x0000000F
#define AQDE_SRC_CONFIG_EX_ResetValue                                 0x00000000

#define AQDE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT                               3:0
#define AQDE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_End                             3
#define AQDE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_Start                           0
#define AQDE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_Type                          U04
#define   AQDE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_RGB_F16F16F16_PLANAR        0x0
#define   AQDE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_RGB_F16F16F16_PACKED        0x1
#define   AQDE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_RGB_F32F32F32_PLANAR        0x2
#define   AQDE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_RGB_F32F32F32_PACKED        0x3
#define   AQDE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_GRAY_I8                     0x4
#define   AQDE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_GRAY_F16                    0x5
#define   AQDE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_GRAY_F32                    0x6
#define   AQDE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_INT8_PLANAR                 0x7
#define   AQDE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_INT8_PACKED                 0x8
#define   AQDE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_INT16_PLANAR                0x9
#define   AQDE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_INT16_PACKED                0xA
#define   AQDE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_GRAY_U8                     0xB

#define gcregDEDstScaleFactorRegAddrs                                     0x04F0
#define GCREG_DE_DST_SCALE_FACTOR_Address                                0x013C0
#define GCREG_DE_DST_SCALE_FACTOR_MSB                                         15
#define GCREG_DE_DST_SCALE_FACTOR_LSB                                          0
#define GCREG_DE_DST_SCALE_FACTOR_BLK                                          0
#define GCREG_DE_DST_SCALE_FACTOR_Count                                        1
#define GCREG_DE_DST_SCALE_FACTOR_FieldMask                           0xFFFFFFFF
#define GCREG_DE_DST_SCALE_FACTOR_ReadMask                            0xFFFFFFFF
#define GCREG_DE_DST_SCALE_FACTOR_WriteMask                           0xFFFFFFFF
#define GCREG_DE_DST_SCALE_FACTOR_ResetValue                          0x00000000

#define GCREG_DE_DST_SCALE_FACTOR_VALUE                                     31:0
#define GCREG_DE_DST_SCALE_FACTOR_VALUE_End                                   31
#define GCREG_DE_DST_SCALE_FACTOR_VALUE_Start                                  0
#define GCREG_DE_DST_SCALE_FACTOR_VALUE_Type                                 U32

#define gcregDEDstMeanForChannelRRegAddrs                                 0x04F1
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_R_Address                          0x013C4
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_R_MSB                                   15
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_R_LSB                                    0
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_R_BLK                                    0
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_R_Count                                  1
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_R_FieldMask                     0xFFFFFFFF
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_R_ReadMask                      0xFFFFFFFF
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_R_WriteMask                     0xFFFFFFFF
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_R_ResetValue                    0x00000000

#define GCREG_DE_DST_MEAN_FOR_CHANNEL_R_VALUE                               31:0
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_R_VALUE_End                             31
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_R_VALUE_Start                            0
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_R_VALUE_Type                           U32

#define gcregDEDstMeanForChannelGRegAddrs                                 0x04F2
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_G_Address                          0x013C8
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_G_MSB                                   15
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_G_LSB                                    0
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_G_BLK                                    0
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_G_Count                                  1
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_G_FieldMask                     0xFFFFFFFF
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_G_ReadMask                      0xFFFFFFFF
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_G_WriteMask                     0xFFFFFFFF
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_G_ResetValue                    0x00000000

#define GCREG_DE_DST_MEAN_FOR_CHANNEL_G_VALUE                               31:0
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_G_VALUE_End                             31
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_G_VALUE_Start                            0
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_G_VALUE_Type                           U32

#define gcregDEDstMeanForChannelBRegAddrs                                 0x04F3
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_B_Address                          0x013CC
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_B_MSB                                   15
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_B_LSB                                    0
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_B_BLK                                    0
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_B_Count                                  1
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_B_FieldMask                     0xFFFFFFFF
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_B_ReadMask                      0xFFFFFFFF
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_B_WriteMask                     0xFFFFFFFF
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_B_ResetValue                    0x00000000

#define GCREG_DE_DST_MEAN_FOR_CHANNEL_B_VALUE                               31:0
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_B_VALUE_End                             31
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_B_VALUE_Start                            0
#define GCREG_DE_DST_MEAN_FOR_CHANNEL_B_VALUE_Type                           U32

#define gcregDEDstStdReciprocalForChannelRRegAddrs                        0x04F4
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_R_Address                0x013D0
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_R_MSB                         15
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_R_LSB                          0
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_R_BLK                          0
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_R_Count                        1
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_R_FieldMask           0xFFFFFFFF
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_R_ReadMask            0xFFFFFFFF
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_R_WriteMask           0xFFFFFFFF
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_R_ResetValue          0x00000000

#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_R_VALUE                     31:0
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_R_VALUE_End                   31
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_R_VALUE_Start                  0
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_R_VALUE_Type                 U32

#define gcregDEDstStdReciprocalForChannelGRegAddrs                        0x04F5
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_G_Address                0x013D4
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_G_MSB                         15
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_G_LSB                          0
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_G_BLK                          0
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_G_Count                        1
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_G_FieldMask           0xFFFFFFFF
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_G_ReadMask            0xFFFFFFFF
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_G_WriteMask           0xFFFFFFFF
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_G_ResetValue          0x00000000

#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_G_VALUE                     31:0
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_G_VALUE_End                   31
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_G_VALUE_Start                  0
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_G_VALUE_Type                 U32

#define gcregDEDstStdReciprocalForChannelBRegAddrs                        0x04F6
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_B_Address                0x013D8
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_B_MSB                         15
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_B_LSB                          0
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_B_BLK                          0
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_B_Count                        1
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_B_FieldMask           0xFFFFFFFF
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_B_ReadMask            0xFFFFFFFF
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_B_WriteMask           0xFFFFFFFF
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_B_ResetValue          0x00000000

#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_B_VALUE                     31:0
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_B_VALUE_End                   31
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_B_VALUE_Start                  0
#define GCREG_DE_DST_STD_RECIPROCAL_FOR_CHANNEL_B_VALUE_Type                 U32

#define gcregDEDstNormalizationModeRegAddrs                               0x04F7
#define GCREG_DE_DST_NORMALIZATION_MODE_Address                          0x013DC
#define GCREG_DE_DST_NORMALIZATION_MODE_MSB                                   15
#define GCREG_DE_DST_NORMALIZATION_MODE_LSB                                    0
#define GCREG_DE_DST_NORMALIZATION_MODE_BLK                                    0
#define GCREG_DE_DST_NORMALIZATION_MODE_Count                                  1
#define GCREG_DE_DST_NORMALIZATION_MODE_ResetValue                    0x00000000

#define GCREG_DE_DST_NORMALIZATION_MODE_VALUE                                1:0
#define GCREG_DE_DST_NORMALIZATION_MODE_VALUE_End                              1
#define GCREG_DE_DST_NORMALIZATION_MODE_VALUE_Start                            0
#define GCREG_DE_DST_NORMALIZATION_MODE_VALUE_Type                           U02
#define   GCREG_DE_DST_NORMALIZATION_MODE_VALUE_STANDARD_ZSCORE              0x0
#define   GCREG_DE_DST_NORMALIZATION_MODE_VALUE_ZSCORE                       0x1
#define   GCREG_DE_DST_NORMALIZATION_MODE_VALUE_MIN_MAX                      0x2

#define GCREG_DE_DST_NORMALIZATION_MODE_KEEP_ALPHA                           2:2
#define GCREG_DE_DST_NORMALIZATION_MODE_KEEP_ALPHA_End                         2
#define GCREG_DE_DST_NORMALIZATION_MODE_KEEP_ALPHA_Start                       2
#define GCREG_DE_DST_NORMALIZATION_MODE_KEEP_ALPHA_Type                      U01
#define   GCREG_DE_DST_NORMALIZATION_MODE_KEEP_ALPHA_DISABLE                 0x0
#define   GCREG_DE_DST_NORMALIZATION_MODE_KEEP_ALPHA_ENABLE                  0x1

#define GCREG_DE_DST_NORMALIZATION_MODE_ALPHA_VALUE                          3:3
#define GCREG_DE_DST_NORMALIZATION_MODE_ALPHA_VALUE_End                        3
#define GCREG_DE_DST_NORMALIZATION_MODE_ALPHA_VALUE_Start                      3
#define GCREG_DE_DST_NORMALIZATION_MODE_ALPHA_VALUE_Type                     U01
#define   GCREG_DE_DST_NORMALIZATION_MODE_ALPHA_VALUE_ALL_F                  0x0
#define   GCREG_DE_DST_NORMALIZATION_MODE_ALPHA_VALUE_ALL_ZERO               0x1

#define gcregDEDstQuantizationBypassRegAddrs                              0x04F8
#define GCREG_DE_DST_QUANTIZATION_BYPASS_Address                         0x013E0
#define GCREG_DE_DST_QUANTIZATION_BYPASS_MSB                                  15
#define GCREG_DE_DST_QUANTIZATION_BYPASS_LSB                                   0
#define GCREG_DE_DST_QUANTIZATION_BYPASS_BLK                                   0
#define GCREG_DE_DST_QUANTIZATION_BYPASS_Count                                 1
#define GCREG_DE_DST_QUANTIZATION_BYPASS_FieldMask                    0x00000002
#define GCREG_DE_DST_QUANTIZATION_BYPASS_ReadMask                     0x00000002
#define GCREG_DE_DST_QUANTIZATION_BYPASS_WriteMask                    0x00000002
#define GCREG_DE_DST_QUANTIZATION_BYPASS_ResetValue                   0x00000000

#define GCREG_DE_DST_QUANTIZATION_BYPASS_VALUE                               1:1
#define GCREG_DE_DST_QUANTIZATION_BYPASS_VALUE_End                             1
#define GCREG_DE_DST_QUANTIZATION_BYPASS_VALUE_Start                           1
#define GCREG_DE_DST_QUANTIZATION_BYPASS_VALUE_Type                          U01
#define   GCREG_DE_DST_QUANTIZATION_BYPASS_VALUE_NOT_BYPASS                  0x0
#define   GCREG_DE_DST_QUANTIZATION_BYPASS_VALUE_BYPASS_QUANTIZATION         0x1

#define gcregDEDstStepReciprocalRegAddrs                                  0x04F9
#define GCREG_DE_DST_STEP_RECIPROCAL_Address                             0x013E4
#define GCREG_DE_DST_STEP_RECIPROCAL_MSB                                      15
#define GCREG_DE_DST_STEP_RECIPROCAL_LSB                                       0
#define GCREG_DE_DST_STEP_RECIPROCAL_BLK                                       0
#define GCREG_DE_DST_STEP_RECIPROCAL_Count                                     1
#define GCREG_DE_DST_STEP_RECIPROCAL_FieldMask                        0xFFFFFFFF
#define GCREG_DE_DST_STEP_RECIPROCAL_ReadMask                         0xFFFFFFFF
#define GCREG_DE_DST_STEP_RECIPROCAL_WriteMask                        0xFFFFFFFF
#define GCREG_DE_DST_STEP_RECIPROCAL_ResetValue                       0x00000000

#define GCREG_DE_DST_STEP_RECIPROCAL_VALUE                                  31:0
#define GCREG_DE_DST_STEP_RECIPROCAL_VALUE_End                                31
#define GCREG_DE_DST_STEP_RECIPROCAL_VALUE_Start                               0
#define GCREG_DE_DST_STEP_RECIPROCAL_VALUE_Type                              U32

#define gcregDEDstMinValueForChannelRRegAddrs                             0x04FA
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_R_Address                     0x013E8
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_R_MSB                              15
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_R_LSB                               0
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_R_BLK                               0
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_R_Count                             1
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_R_FieldMask                0xFFFFFFFF
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_R_ReadMask                 0xFFFFFFFF
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_R_WriteMask                0xFFFFFFFF
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_R_ResetValue               0x00000000

#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_R_VALUE                          31:0
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_R_VALUE_End                        31
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_R_VALUE_Start                       0
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_R_VALUE_Type                      U32

#define gcregDEDstMinValueForChannelGRegAddrs                             0x04FB
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_G_Address                     0x013EC
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_G_MSB                              15
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_G_LSB                               0
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_G_BLK                               0
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_G_Count                             1
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_G_FieldMask                0xFFFFFFFF
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_G_ReadMask                 0xFFFFFFFF
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_G_WriteMask                0xFFFFFFFF
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_G_ResetValue               0x00000000

#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_G_VALUE                          31:0
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_G_VALUE_End                        31
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_G_VALUE_Start                       0
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_G_VALUE_Type                      U32

#define gcregDEDstMinValueForChannelBRegAddrs                             0x04FC
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_B_Address                     0x013F0
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_B_MSB                              15
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_B_LSB                               0
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_B_BLK                               0
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_B_Count                             1
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_B_FieldMask                0xFFFFFFFF
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_B_ReadMask                 0xFFFFFFFF
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_B_WriteMask                0xFFFFFFFF
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_B_ResetValue               0x00000000

#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_B_VALUE                          31:0
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_B_VALUE_End                        31
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_B_VALUE_Start                       0
#define GCREG_DE_DST_MIN_VALUE_FOR_CHANNEL_B_VALUE_Type                      U32

#define gcregDEDstMaxMinReciprocalForChannelRRegAddrs                     0x04FD
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_R_Address            0x013F4
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_R_MSB                     15
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_R_LSB                      0
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_R_BLK                      0
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_R_Count                    1
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_R_FieldMask       0xFFFFFFFF
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_R_ReadMask        0xFFFFFFFF
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_R_WriteMask       0xFFFFFFFF
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_R_ResetValue      0x00000000

#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_R_VALUE                 31:0
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_R_VALUE_End               31
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_R_VALUE_Start              0
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_R_VALUE_Type             U32

#define gcregDEDstMaxMinReciprocalForChannelGRegAddrs                     0x04FE
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_G_Address            0x013F8
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_G_MSB                     15
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_G_LSB                      0
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_G_BLK                      0
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_G_Count                    1
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_G_FieldMask       0xFFFFFFFF
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_G_ReadMask        0xFFFFFFFF
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_G_WriteMask       0xFFFFFFFF
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_G_ResetValue      0x00000000

#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_G_VALUE                 31:0
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_G_VALUE_End               31
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_G_VALUE_Start              0
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_G_VALUE_Type             U32

#define gcregDEDstMaxMinReciprocalForChannelBRegAddrs                     0x04FF
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_B_Address            0x013FC
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_B_MSB                     15
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_B_LSB                      0
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_B_BLK                      0
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_B_Count                    1
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_B_FieldMask       0xFFFFFFFF
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_B_ReadMask        0xFFFFFFFF
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_B_WriteMask       0xFFFFFFFF
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_B_ResetValue      0x00000000

#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_B_VALUE                 31:0
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_B_VALUE_End               31
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_B_VALUE_Start              0
#define GCREG_DE_DST_MAX_MIN_RECIPROCAL_FOR_CHANNEL_B_VALUE_Type             U32

#define gcregDESrcAddressRegAddrs                                         0x4A00
#define GCREG_DE_SRC_ADDRESS_Address                                     0x12800
#define GCREG_DE_SRC_ADDRESS_MSB                                              15
#define GCREG_DE_SRC_ADDRESS_LSB                                               2
#define GCREG_DE_SRC_ADDRESS_BLK                                               0
#define GCREG_DE_SRC_ADDRESS_Count                                             4
#define GCREG_DE_SRC_ADDRESS_FieldMask                                0xFFFFFFFF
#define GCREG_DE_SRC_ADDRESS_ReadMask                                 0xFFFFFFFC
#define GCREG_DE_SRC_ADDRESS_WriteMask                                0xFFFFFFFC
#define GCREG_DE_SRC_ADDRESS_ResetValue                               0x00000000

#define GCREG_DE_SRC_ADDRESS_TYPE                                          31:31
#define GCREG_DE_SRC_ADDRESS_TYPE_End                                         31
#define GCREG_DE_SRC_ADDRESS_TYPE_Start                                       31
#define GCREG_DE_SRC_ADDRESS_TYPE_Type                                       U01
#define   GCREG_DE_SRC_ADDRESS_TYPE_SYSTEM                                   0x0
#define   GCREG_DE_SRC_ADDRESS_TYPE_VIRTUAL_SYSTEM                           0x1

#define GCREG_DE_SRC_ADDRESS_ADDRESS                                        30:0
#define GCREG_DE_SRC_ADDRESS_ADDRESS_End                                      30
#define GCREG_DE_SRC_ADDRESS_ADDRESS_Start                                     0
#define GCREG_DE_SRC_ADDRESS_ADDRESS_Type                                    U31

#define gcregDESrcStrideRegAddrs                                          0x4A04
#define GCREG_DE_SRC_STRIDE_Address                                      0x12810
#define GCREG_DE_SRC_STRIDE_MSB                                               15
#define GCREG_DE_SRC_STRIDE_LSB                                                2
#define GCREG_DE_SRC_STRIDE_BLK                                                0
#define GCREG_DE_SRC_STRIDE_Count                                              4
#define GCREG_DE_SRC_STRIDE_FieldMask                                 0x0003FFFF
#define GCREG_DE_SRC_STRIDE_ReadMask                                  0x0003FFFC
#define GCREG_DE_SRC_STRIDE_WriteMask                                 0x0003FFFC
#define GCREG_DE_SRC_STRIDE_ResetValue                                0x00000000

#define GCREG_DE_SRC_STRIDE_STRIDE                                          17:0
#define GCREG_DE_SRC_STRIDE_STRIDE_End                                        17
#define GCREG_DE_SRC_STRIDE_STRIDE_Start                                       0
#define GCREG_DE_SRC_STRIDE_STRIDE_Type                                      U18

#define gcregDESrcRotationConfigRegAddrs                                  0x4A08
#define GCREG_DE_SRC_ROTATION_CONFIG_Address                             0x12820
#define GCREG_DE_SRC_ROTATION_CONFIG_MSB                                      15
#define GCREG_DE_SRC_ROTATION_CONFIG_LSB                                       2
#define GCREG_DE_SRC_ROTATION_CONFIG_BLK                                       0
#define GCREG_DE_SRC_ROTATION_CONFIG_Count                                     4
#define GCREG_DE_SRC_ROTATION_CONFIG_FieldMask                        0x0001FFFF
#define GCREG_DE_SRC_ROTATION_CONFIG_ReadMask                         0x0001FFFF
#define GCREG_DE_SRC_ROTATION_CONFIG_WriteMask                        0x0001FFFF
#define GCREG_DE_SRC_ROTATION_CONFIG_ResetValue                       0x00000000

#define GCREG_DE_SRC_ROTATION_CONFIG_WIDTH                                  15:0
#define GCREG_DE_SRC_ROTATION_CONFIG_WIDTH_End                                15
#define GCREG_DE_SRC_ROTATION_CONFIG_WIDTH_Start                               0
#define GCREG_DE_SRC_ROTATION_CONFIG_WIDTH_Type                              U16

#define GCREG_DE_SRC_ROTATION_CONFIG_ROTATION                              16:16
#define GCREG_DE_SRC_ROTATION_CONFIG_ROTATION_End                             16
#define GCREG_DE_SRC_ROTATION_CONFIG_ROTATION_Start                           16
#define GCREG_DE_SRC_ROTATION_CONFIG_ROTATION_Type                           U01
#define   GCREG_DE_SRC_ROTATION_CONFIG_ROTATION_NORMAL                       0x0
#define   GCREG_DE_SRC_ROTATION_CONFIG_ROTATION_ROTATED                      0x1

#define gcregDESrcConfigRegAddrs                                          0x4A0C
#define GCREG_DE_SRC_CONFIG_Address                                      0x12830
#define GCREG_DE_SRC_CONFIG_MSB                                               15
#define GCREG_DE_SRC_CONFIG_LSB                                                2
#define GCREG_DE_SRC_CONFIG_BLK                                                0
#define GCREG_DE_SRC_CONFIG_Count                                              4
#define GCREG_DE_SRC_CONFIG_FieldMask                                 0xFFF0B1CF
#define GCREG_DE_SRC_CONFIG_ReadMask                                  0xFFF0B1CF
#define GCREG_DE_SRC_CONFIG_WriteMask                                 0xFFF0B1CF
#define GCREG_DE_SRC_CONFIG_ResetValue                                0x00000000

#define GCREG_DE_SRC_CONFIG_ENDIAN_CONTROL                                 31:30
#define GCREG_DE_SRC_CONFIG_ENDIAN_CONTROL_End                                31
#define GCREG_DE_SRC_CONFIG_ENDIAN_CONTROL_Start                              30
#define GCREG_DE_SRC_CONFIG_ENDIAN_CONTROL_Type                              U02
#define   GCREG_DE_SRC_CONFIG_ENDIAN_CONTROL_NO_SWAP                         0x0
#define   GCREG_DE_SRC_CONFIG_ENDIAN_CONTROL_SWAP_WORD                       0x1
#define   GCREG_DE_SRC_CONFIG_ENDIAN_CONTROL_SWAP_DWORD                      0x2
#define   GCREG_DE_SRC_CONFIG_ENDIAN_CONTROL_SWAP_DDWORD                     0x3

#define GCREG_DE_SRC_CONFIG_ENDIAN_CONTROL_EX                              29:29
#define GCREG_DE_SRC_CONFIG_ENDIAN_CONTROL_EX_End                             29
#define GCREG_DE_SRC_CONFIG_ENDIAN_CONTROL_EX_Start                           29
#define GCREG_DE_SRC_CONFIG_ENDIAN_CONTROL_EX_Type                           U01
#define   GCREG_DE_SRC_CONFIG_ENDIAN_CONTROL_EX_NO_SWAP                      0x0
#define   GCREG_DE_SRC_CONFIG_ENDIAN_CONTROL_EX_SWAP_BYTE                    0x1

#define GCREG_DE_SRC_CONFIG_SOURCE_FORMAT                                  28:24
#define GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_End                                 28
#define GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_Start                               24
#define GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_Type                               U05
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_X4R4G4B4                        0x00
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_A4R4G4B4                        0x01
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_X1R5G5B5                        0x02
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_A1R5G5B5                        0x03
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_R5G6B5                          0x04
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_X8R8G8B8                        0x05
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_A8R8G8B8                        0x06
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_YUY2                            0x07
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_UYVY                            0x08
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_INDEX8                          0x09
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_MONOCHROME                      0x0A
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_AYUV                            0x0B
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_RGB888_PLANAR                   0x0C
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_RGB888_PACKED                   0x0D
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_EXTENSION                       0x0E
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_YV12                            0x0F
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_A8                              0x10
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_NV12                            0x11
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_NV16                            0x12
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_RG16                            0x13
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_R8                              0x14
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_NV12_10BIT                      0x15
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_A2R10G10B10                     0x16
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_NV16_10BIT                      0x17
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_INDEX1                          0x18
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_INDEX2                          0x19
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_INDEX4                          0x1A
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_P010                            0x1B
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_NV12_10BIT_L1                   0x1C
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_NV16_10BIT_L1                   0x1D
#define   GCREG_DE_SRC_CONFIG_SOURCE_FORMAT_I010                            0x1E

#define GCREG_DE_SRC_CONFIG_DE_GAMMA                                       23:23
#define GCREG_DE_SRC_CONFIG_DE_GAMMA_End                                      23
#define GCREG_DE_SRC_CONFIG_DE_GAMMA_Start                                    23
#define GCREG_DE_SRC_CONFIG_DE_GAMMA_Type                                    U01
#define   GCREG_DE_SRC_CONFIG_DE_GAMMA_DISABLED                              0x0
#define   GCREG_DE_SRC_CONFIG_DE_GAMMA_ENABLED                               0x1

#define GCREG_DE_SRC_CONFIG_P010_BITS                                      22:22
#define GCREG_DE_SRC_CONFIG_P010_BITS_End                                     22
#define GCREG_DE_SRC_CONFIG_P010_BITS_Start                                   22
#define GCREG_DE_SRC_CONFIG_P010_BITS_Type                                   U01
#define   GCREG_DE_SRC_CONFIG_P010_BITS_P010_MSB                             0x0
#define   GCREG_DE_SRC_CONFIG_P010_BITS_P010_LSB                             0x1

#define GCREG_DE_SRC_CONFIG_SWIZZLE                                        21:20
#define GCREG_DE_SRC_CONFIG_SWIZZLE_End                                       21
#define GCREG_DE_SRC_CONFIG_SWIZZLE_Start                                     20
#define GCREG_DE_SRC_CONFIG_SWIZZLE_Type                                     U02
#define   GCREG_DE_SRC_CONFIG_SWIZZLE_ARGB                                   0x0
#define   GCREG_DE_SRC_CONFIG_SWIZZLE_RGBA                                   0x1
#define   GCREG_DE_SRC_CONFIG_SWIZZLE_ABGR                                   0x2
#define   GCREG_DE_SRC_CONFIG_SWIZZLE_BGRA                                   0x3

#define GCREG_DE_SRC_CONFIG_MONO_TRANSPARENCY                              15:15
#define GCREG_DE_SRC_CONFIG_MONO_TRANSPARENCY_End                             15
#define GCREG_DE_SRC_CONFIG_MONO_TRANSPARENCY_Start                           15
#define GCREG_DE_SRC_CONFIG_MONO_TRANSPARENCY_Type                           U01
#define   GCREG_DE_SRC_CONFIG_MONO_TRANSPARENCY_BACKGROUND                   0x0
#define   GCREG_DE_SRC_CONFIG_MONO_TRANSPARENCY_FOREGROUND                   0x1

#define GCREG_DE_SRC_CONFIG_PACK                                           13:12
#define GCREG_DE_SRC_CONFIG_PACK_End                                          13
#define GCREG_DE_SRC_CONFIG_PACK_Start                                        12
#define GCREG_DE_SRC_CONFIG_PACK_Type                                        U02
#define   GCREG_DE_SRC_CONFIG_PACK_PACKED8                                   0x0
#define   GCREG_DE_SRC_CONFIG_PACK_PACKED16                                  0x1
#define   GCREG_DE_SRC_CONFIG_PACK_PACKED32                                  0x2
#define   GCREG_DE_SRC_CONFIG_PACK_UNPACKED                                  0x3

#define GCREG_DE_SRC_CONFIG_LOCATION                                         8:8
#define GCREG_DE_SRC_CONFIG_LOCATION_End                                       8
#define GCREG_DE_SRC_CONFIG_LOCATION_Start                                     8
#define GCREG_DE_SRC_CONFIG_LOCATION_Type                                    U01
#define   GCREG_DE_SRC_CONFIG_LOCATION_MEMORY                                0x0
#define   GCREG_DE_SRC_CONFIG_LOCATION_STREAM                                0x1

#define GCREG_DE_SRC_CONFIG_TILED                                            7:7
#define GCREG_DE_SRC_CONFIG_TILED_End                                          7
#define GCREG_DE_SRC_CONFIG_TILED_Start                                        7
#define GCREG_DE_SRC_CONFIG_TILED_Type                                       U01
#define   GCREG_DE_SRC_CONFIG_TILED_DISABLED                                 0x0
#define   GCREG_DE_SRC_CONFIG_TILED_ENABLED                                  0x1

#define GCREG_DE_SRC_CONFIG_SRC_RELATIVE                                     6:6
#define GCREG_DE_SRC_CONFIG_SRC_RELATIVE_End                                   6
#define GCREG_DE_SRC_CONFIG_SRC_RELATIVE_Start                                 6
#define GCREG_DE_SRC_CONFIG_SRC_RELATIVE_Type                                U01
#define   GCREG_DE_SRC_CONFIG_SRC_RELATIVE_ABSOLUTE                          0x0
#define   GCREG_DE_SRC_CONFIG_SRC_RELATIVE_RELATIVE                          0x1

#define GCREG_DE_SRC_CONFIG_EXTRA_SOURCE_FORMAT                              3:0
#define GCREG_DE_SRC_CONFIG_EXTRA_SOURCE_FORMAT_End                            3
#define GCREG_DE_SRC_CONFIG_EXTRA_SOURCE_FORMAT_Start                          0
#define GCREG_DE_SRC_CONFIG_EXTRA_SOURCE_FORMAT_Type                         U04
#define   GCREG_DE_SRC_CONFIG_EXTRA_SOURCE_FORMAT_RGB_F16F16F16_PLANAR       0x0
#define   GCREG_DE_SRC_CONFIG_EXTRA_SOURCE_FORMAT_RGB_F16F16F16_PACKED       0x1
#define   GCREG_DE_SRC_CONFIG_EXTRA_SOURCE_FORMAT_RGB_F32F32F32_PLANAR       0x2
#define   GCREG_DE_SRC_CONFIG_EXTRA_SOURCE_FORMAT_RGB_F32F32F32_PACKED       0x3
#define   GCREG_DE_SRC_CONFIG_EXTRA_SOURCE_FORMAT_GRAY_I8                    0x4
#define   GCREG_DE_SRC_CONFIG_EXTRA_SOURCE_FORMAT_GRAY_F16                   0x5
#define   GCREG_DE_SRC_CONFIG_EXTRA_SOURCE_FORMAT_GRAY_F32                   0x6
#define   GCREG_DE_SRC_CONFIG_EXTRA_SOURCE_FORMAT_INT8_PLANAR                0x7
#define   GCREG_DE_SRC_CONFIG_EXTRA_SOURCE_FORMAT_INT8_PACKED                0x8
#define   GCREG_DE_SRC_CONFIG_EXTRA_SOURCE_FORMAT_INT16_PLANAR               0x9
#define   GCREG_DE_SRC_CONFIG_EXTRA_SOURCE_FORMAT_INT16_PACKED               0xA
#define   GCREG_DE_SRC_CONFIG_EXTRA_SOURCE_FORMAT_GRAY_U8                    0xB

#define gcregDESrcOriginRegAddrs                                          0x4A10
#define GCREG_DE_SRC_ORIGIN_Address                                      0x12840
#define GCREG_DE_SRC_ORIGIN_MSB                                               15
#define GCREG_DE_SRC_ORIGIN_LSB                                                2
#define GCREG_DE_SRC_ORIGIN_BLK                                                0
#define GCREG_DE_SRC_ORIGIN_Count                                              4
#define GCREG_DE_SRC_ORIGIN_FieldMask                                 0xFFFFFFFF
#define GCREG_DE_SRC_ORIGIN_ReadMask                                  0xFFFFFFFF
#define GCREG_DE_SRC_ORIGIN_WriteMask                                 0xFFFFFFFF
#define GCREG_DE_SRC_ORIGIN_ResetValue                                0x00000000

#define GCREG_DE_SRC_ORIGIN_Y                                              31:16
#define GCREG_DE_SRC_ORIGIN_Y_End                                             31
#define GCREG_DE_SRC_ORIGIN_Y_Start                                           16
#define GCREG_DE_SRC_ORIGIN_Y_Type                                           U16

#define GCREG_DE_SRC_ORIGIN_X                                               15:0
#define GCREG_DE_SRC_ORIGIN_X_End                                             15
#define GCREG_DE_SRC_ORIGIN_X_Start                                            0
#define GCREG_DE_SRC_ORIGIN_X_Type                                           U16

#define gcregDESrcSizeRegAddrs                                            0x4A14
#define GCREG_DE_SRC_SIZE_Address                                        0x12850
#define GCREG_DE_SRC_SIZE_MSB                                                 15
#define GCREG_DE_SRC_SIZE_LSB                                                  2
#define GCREG_DE_SRC_SIZE_BLK                                                  0
#define GCREG_DE_SRC_SIZE_Count                                                4
#define GCREG_DE_SRC_SIZE_FieldMask                                   0xFFFFFFFF
#define GCREG_DE_SRC_SIZE_ReadMask                                    0xFFFFFFFF
#define GCREG_DE_SRC_SIZE_WriteMask                                   0xFFFFFFFF
#define GCREG_DE_SRC_SIZE_ResetValue                                  0x00000000

#define GCREG_DE_SRC_SIZE_Y                                                31:16
#define GCREG_DE_SRC_SIZE_Y_End                                               31
#define GCREG_DE_SRC_SIZE_Y_Start                                             16
#define GCREG_DE_SRC_SIZE_Y_Type                                             U16

#define GCREG_DE_SRC_SIZE_X                                                 15:0
#define GCREG_DE_SRC_SIZE_X_End                                               15
#define GCREG_DE_SRC_SIZE_X_Start                                              0
#define GCREG_DE_SRC_SIZE_X_Type                                             U16

#define gcregDESrcColorBgRegAddrs                                         0x4A18
#define GCREG_DE_SRC_COLOR_BG_Address                                    0x12860
#define GCREG_DE_SRC_COLOR_BG_MSB                                             15
#define GCREG_DE_SRC_COLOR_BG_LSB                                              2
#define GCREG_DE_SRC_COLOR_BG_BLK                                              0
#define GCREG_DE_SRC_COLOR_BG_Count                                            4
#define GCREG_DE_SRC_COLOR_BG_FieldMask                               0xFFFFFFFF
#define GCREG_DE_SRC_COLOR_BG_ReadMask                                0xFFFFFFFF
#define GCREG_DE_SRC_COLOR_BG_WriteMask                               0xFFFFFFFF
#define GCREG_DE_SRC_COLOR_BG_ResetValue                              0x00000000

#define GCREG_DE_SRC_COLOR_BG_ALPHA                                        31:24
#define GCREG_DE_SRC_COLOR_BG_ALPHA_End                                       31
#define GCREG_DE_SRC_COLOR_BG_ALPHA_Start                                     24
#define GCREG_DE_SRC_COLOR_BG_ALPHA_Type                                     U08

#define GCREG_DE_SRC_COLOR_BG_RED                                          23:16
#define GCREG_DE_SRC_COLOR_BG_RED_End                                         23
#define GCREG_DE_SRC_COLOR_BG_RED_Start                                       16
#define GCREG_DE_SRC_COLOR_BG_RED_Type                                       U08

#define GCREG_DE_SRC_COLOR_BG_GREEN                                         15:8
#define GCREG_DE_SRC_COLOR_BG_GREEN_End                                       15
#define GCREG_DE_SRC_COLOR_BG_GREEN_Start                                      8
#define GCREG_DE_SRC_COLOR_BG_GREEN_Type                                     U08

#define GCREG_DE_SRC_COLOR_BG_BLUE                                           7:0
#define GCREG_DE_SRC_COLOR_BG_BLUE_End                                         7
#define GCREG_DE_SRC_COLOR_BG_BLUE_Start                                       0
#define GCREG_DE_SRC_COLOR_BG_BLUE_Type                                      U08

#define gcregDERopRegAddrs                                                0x4A1C
#define GCREG_DE_ROP_Address                                             0x12870
#define GCREG_DE_ROP_MSB                                                      15
#define GCREG_DE_ROP_LSB                                                       2
#define GCREG_DE_ROP_BLK                                                       0
#define GCREG_DE_ROP_Count                                                     4
#define GCREG_DE_ROP_FieldMask                                        0x0030FFFF
#define GCREG_DE_ROP_ReadMask                                         0x0030FFFF
#define GCREG_DE_ROP_WriteMask                                        0x0030FFFF
#define GCREG_DE_ROP_ResetValue                                       0x00000000

#define GCREG_DE_ROP_TYPE                                                  21:20
#define GCREG_DE_ROP_TYPE_End                                                 21
#define GCREG_DE_ROP_TYPE_Start                                               20
#define GCREG_DE_ROP_TYPE_Type                                               U02
#define   GCREG_DE_ROP_TYPE_ROP2_PATTERN                                     0x0
#define   GCREG_DE_ROP_TYPE_ROP2_SOURCE                                      0x1
#define   GCREG_DE_ROP_TYPE_ROP3                                             0x2
#define   GCREG_DE_ROP_TYPE_ROP4                                             0x3

#define GCREG_DE_ROP_ROP_BG                                                 15:8
#define GCREG_DE_ROP_ROP_BG_End                                               15
#define GCREG_DE_ROP_ROP_BG_Start                                              8
#define GCREG_DE_ROP_ROP_BG_Type                                             U08

#define GCREG_DE_ROP_ROP_FG                                                  7:0
#define GCREG_DE_ROP_ROP_FG_End                                                7
#define GCREG_DE_ROP_ROP_FG_Start                                              0
#define GCREG_DE_ROP_ROP_FG_Type                                             U08

#define gcregDEAlphaControlRegAddrs                                       0x4A20
#define GCREG_DE_ALPHA_CONTROL_Address                                   0x12880
#define GCREG_DE_ALPHA_CONTROL_MSB                                            15
#define GCREG_DE_ALPHA_CONTROL_LSB                                             2
#define GCREG_DE_ALPHA_CONTROL_BLK                                             0
#define GCREG_DE_ALPHA_CONTROL_Count                                           4
#define GCREG_DE_ALPHA_CONTROL_FieldMask                              0x00000001
#define GCREG_DE_ALPHA_CONTROL_ReadMask                               0x00000001
#define GCREG_DE_ALPHA_CONTROL_WriteMask                              0x00000001
#define GCREG_DE_ALPHA_CONTROL_ResetValue                             0x00000000

#define GCREG_DE_ALPHA_CONTROL_ENABLE                                        0:0
#define GCREG_DE_ALPHA_CONTROL_ENABLE_End                                      0
#define GCREG_DE_ALPHA_CONTROL_ENABLE_Start                                    0
#define GCREG_DE_ALPHA_CONTROL_ENABLE_Type                                   U01
#define   GCREG_DE_ALPHA_CONTROL_ENABLE_OFF                                  0x0
#define   GCREG_DE_ALPHA_CONTROL_ENABLE_ON                                   0x1

#define gcregDEAlphaModesRegAddrs                                         0x4A24
#define GCREG_DE_ALPHA_MODES_Address                                     0x12890
#define GCREG_DE_ALPHA_MODES_MSB                                              15
#define GCREG_DE_ALPHA_MODES_LSB                                               2
#define GCREG_DE_ALPHA_MODES_BLK                                               0
#define GCREG_DE_ALPHA_MODES_Count                                             4
#define GCREG_DE_ALPHA_MODES_FieldMask                                0xFF003311
#define GCREG_DE_ALPHA_MODES_ReadMask                                 0xFF003311
#define GCREG_DE_ALPHA_MODES_WriteMask                                0xFF003311
#define GCREG_DE_ALPHA_MODES_ResetValue                               0x00000000

#define GCREG_DE_ALPHA_MODES_SRC_ALPHA_MODE                                  0:0
#define GCREG_DE_ALPHA_MODES_SRC_ALPHA_MODE_End                                0
#define GCREG_DE_ALPHA_MODES_SRC_ALPHA_MODE_Start                              0
#define GCREG_DE_ALPHA_MODES_SRC_ALPHA_MODE_Type                             U01
#define   GCREG_DE_ALPHA_MODES_SRC_ALPHA_MODE_NORMAL                         0x0
#define   GCREG_DE_ALPHA_MODES_SRC_ALPHA_MODE_INVERSED                       0x1

#define GCREG_DE_ALPHA_MODES_DST_ALPHA_MODE                                  4:4
#define GCREG_DE_ALPHA_MODES_DST_ALPHA_MODE_End                                4
#define GCREG_DE_ALPHA_MODES_DST_ALPHA_MODE_Start                              4
#define GCREG_DE_ALPHA_MODES_DST_ALPHA_MODE_Type                             U01
#define   GCREG_DE_ALPHA_MODES_DST_ALPHA_MODE_NORMAL                         0x0
#define   GCREG_DE_ALPHA_MODES_DST_ALPHA_MODE_INVERSED                       0x1

#define GCREG_DE_ALPHA_MODES_GLOBAL_SRC_ALPHA_MODE                           9:8
#define GCREG_DE_ALPHA_MODES_GLOBAL_SRC_ALPHA_MODE_End                         9
#define GCREG_DE_ALPHA_MODES_GLOBAL_SRC_ALPHA_MODE_Start                       8
#define GCREG_DE_ALPHA_MODES_GLOBAL_SRC_ALPHA_MODE_Type                      U02
#define   GCREG_DE_ALPHA_MODES_GLOBAL_SRC_ALPHA_MODE_NORMAL                  0x0
#define   GCREG_DE_ALPHA_MODES_GLOBAL_SRC_ALPHA_MODE_GLOBAL                  0x1
#define   GCREG_DE_ALPHA_MODES_GLOBAL_SRC_ALPHA_MODE_SCALED                  0x2

#define GCREG_DE_ALPHA_MODES_GLOBAL_DST_ALPHA_MODE                         13:12
#define GCREG_DE_ALPHA_MODES_GLOBAL_DST_ALPHA_MODE_End                        13
#define GCREG_DE_ALPHA_MODES_GLOBAL_DST_ALPHA_MODE_Start                      12
#define GCREG_DE_ALPHA_MODES_GLOBAL_DST_ALPHA_MODE_Type                      U02
#define   GCREG_DE_ALPHA_MODES_GLOBAL_DST_ALPHA_MODE_NORMAL                  0x0
#define   GCREG_DE_ALPHA_MODES_GLOBAL_DST_ALPHA_MODE_GLOBAL                  0x1
#define   GCREG_DE_ALPHA_MODES_GLOBAL_DST_ALPHA_MODE_SCALED                  0x2

#define GCREG_DE_ALPHA_MODES_SRC_BLENDING_MODE                             26:24
#define GCREG_DE_ALPHA_MODES_SRC_BLENDING_MODE_End                            26
#define GCREG_DE_ALPHA_MODES_SRC_BLENDING_MODE_Start                          24
#define GCREG_DE_ALPHA_MODES_SRC_BLENDING_MODE_Type                          U03
#define   GCREG_DE_ALPHA_MODES_SRC_BLENDING_MODE_ZERO                        0x0
#define   GCREG_DE_ALPHA_MODES_SRC_BLENDING_MODE_ONE                         0x1
#define   GCREG_DE_ALPHA_MODES_SRC_BLENDING_MODE_NORMAL                      0x2
#define   GCREG_DE_ALPHA_MODES_SRC_BLENDING_MODE_INVERSED                    0x3
#define   GCREG_DE_ALPHA_MODES_SRC_BLENDING_MODE_COLOR                       0x4
#define   GCREG_DE_ALPHA_MODES_SRC_BLENDING_MODE_COLOR_INVERSED              0x5
#define   GCREG_DE_ALPHA_MODES_SRC_BLENDING_MODE_SATURATED_ALPHA             0x6
#define   GCREG_DE_ALPHA_MODES_SRC_BLENDING_MODE_SATURATED_DEST_ALPHA        0x7

#define GCREG_DE_ALPHA_MODES_SRC_ALPHA_FACTOR                              27:27
#define GCREG_DE_ALPHA_MODES_SRC_ALPHA_FACTOR_End                             27
#define GCREG_DE_ALPHA_MODES_SRC_ALPHA_FACTOR_Start                           27
#define GCREG_DE_ALPHA_MODES_SRC_ALPHA_FACTOR_Type                           U01
#define   GCREG_DE_ALPHA_MODES_SRC_ALPHA_FACTOR_DISABLED                     0x0
#define   GCREG_DE_ALPHA_MODES_SRC_ALPHA_FACTOR_ENABLED                      0x1

#define GCREG_DE_ALPHA_MODES_DST_BLENDING_MODE                             30:28
#define GCREG_DE_ALPHA_MODES_DST_BLENDING_MODE_End                            30
#define GCREG_DE_ALPHA_MODES_DST_BLENDING_MODE_Start                          28
#define GCREG_DE_ALPHA_MODES_DST_BLENDING_MODE_Type                          U03
#define   GCREG_DE_ALPHA_MODES_DST_BLENDING_MODE_ZERO                        0x0
#define   GCREG_DE_ALPHA_MODES_DST_BLENDING_MODE_ONE                         0x1
#define   GCREG_DE_ALPHA_MODES_DST_BLENDING_MODE_NORMAL                      0x2
#define   GCREG_DE_ALPHA_MODES_DST_BLENDING_MODE_INVERSED                    0x3
#define   GCREG_DE_ALPHA_MODES_DST_BLENDING_MODE_COLOR                       0x4
#define   GCREG_DE_ALPHA_MODES_DST_BLENDING_MODE_COLOR_INVERSED              0x5
#define   GCREG_DE_ALPHA_MODES_DST_BLENDING_MODE_SATURATED_ALPHA             0x6
#define   GCREG_DE_ALPHA_MODES_DST_BLENDING_MODE_SATURATED_DEST_ALPHA        0x7

#define GCREG_DE_ALPHA_MODES_DST_ALPHA_FACTOR                              31:31
#define GCREG_DE_ALPHA_MODES_DST_ALPHA_FACTOR_End                             31
#define GCREG_DE_ALPHA_MODES_DST_ALPHA_FACTOR_Start                           31
#define GCREG_DE_ALPHA_MODES_DST_ALPHA_FACTOR_Type                           U01
#define   GCREG_DE_ALPHA_MODES_DST_ALPHA_FACTOR_DISABLED                     0x0
#define   GCREG_DE_ALPHA_MODES_DST_ALPHA_FACTOR_ENABLED                      0x1

#define gcregDEAddressURegAddrs                                           0x4A28
#define GCREG_DE_ADDRESS_U_Address                                       0x128A0
#define GCREG_DE_ADDRESS_U_MSB                                                15
#define GCREG_DE_ADDRESS_U_LSB                                                 2
#define GCREG_DE_ADDRESS_U_BLK                                                 0
#define GCREG_DE_ADDRESS_U_Count                                               4
#define GCREG_DE_ADDRESS_U_FieldMask                                  0xFFFFFFFF
#define GCREG_DE_ADDRESS_U_ReadMask                                   0xFFFFFFFC
#define GCREG_DE_ADDRESS_U_WriteMask                                  0xFFFFFFFC
#define GCREG_DE_ADDRESS_U_ResetValue                                 0x00000000

#define GCREG_DE_ADDRESS_U_TYPE                                            31:31
#define GCREG_DE_ADDRESS_U_TYPE_End                                           31
#define GCREG_DE_ADDRESS_U_TYPE_Start                                         31
#define GCREG_DE_ADDRESS_U_TYPE_Type                                         U01
#define   GCREG_DE_ADDRESS_U_TYPE_SYSTEM                                     0x0
#define   GCREG_DE_ADDRESS_U_TYPE_VIRTUAL_SYSTEM                             0x1

#define GCREG_DE_ADDRESS_U_ADDRESS                                          30:0
#define GCREG_DE_ADDRESS_U_ADDRESS_End                                        30
#define GCREG_DE_ADDRESS_U_ADDRESS_Start                                       0
#define GCREG_DE_ADDRESS_U_ADDRESS_Type                                      U31

#define gcregDEStrideURegAddrs                                            0x4A2C
#define GCREG_DE_STRIDE_U_Address                                        0x128B0
#define GCREG_DE_STRIDE_U_MSB                                                 15
#define GCREG_DE_STRIDE_U_LSB                                                  2
#define GCREG_DE_STRIDE_U_BLK                                                  0
#define GCREG_DE_STRIDE_U_Count                                                4
#define GCREG_DE_STRIDE_U_FieldMask                                   0x0003FFFF
#define GCREG_DE_STRIDE_U_ReadMask                                    0x0003FFFC
#define GCREG_DE_STRIDE_U_WriteMask                                   0x0003FFFC
#define GCREG_DE_STRIDE_U_ResetValue                                  0x00000000

#define GCREG_DE_STRIDE_U_STRIDE                                            17:0
#define GCREG_DE_STRIDE_U_STRIDE_End                                          17
#define GCREG_DE_STRIDE_U_STRIDE_Start                                         0
#define GCREG_DE_STRIDE_U_STRIDE_Type                                        U18

#define gcregDEAddressVRegAddrs                                           0x4A30
#define GCREG_DE_ADDRESS_V_Address                                       0x128C0
#define GCREG_DE_ADDRESS_V_MSB                                                15
#define GCREG_DE_ADDRESS_V_LSB                                                 2
#define GCREG_DE_ADDRESS_V_BLK                                                 0
#define GCREG_DE_ADDRESS_V_Count                                               4
#define GCREG_DE_ADDRESS_V_FieldMask                                  0xFFFFFFFF
#define GCREG_DE_ADDRESS_V_ReadMask                                   0xFFFFFFFC
#define GCREG_DE_ADDRESS_V_WriteMask                                  0xFFFFFFFC
#define GCREG_DE_ADDRESS_V_ResetValue                                 0x00000000

#define GCREG_DE_ADDRESS_V_TYPE                                            31:31
#define GCREG_DE_ADDRESS_V_TYPE_End                                           31
#define GCREG_DE_ADDRESS_V_TYPE_Start                                         31
#define GCREG_DE_ADDRESS_V_TYPE_Type                                         U01
#define   GCREG_DE_ADDRESS_V_TYPE_SYSTEM                                     0x0
#define   GCREG_DE_ADDRESS_V_TYPE_VIRTUAL_SYSTEM                             0x1

#define GCREG_DE_ADDRESS_V_ADDRESS                                          30:0
#define GCREG_DE_ADDRESS_V_ADDRESS_End                                        30
#define GCREG_DE_ADDRESS_V_ADDRESS_Start                                       0
#define GCREG_DE_ADDRESS_V_ADDRESS_Type                                      U31

#define gcregDEStrideVRegAddrs                                            0x4A34
#define GCREG_DE_STRIDE_V_Address                                        0x128D0
#define GCREG_DE_STRIDE_V_MSB                                                 15
#define GCREG_DE_STRIDE_V_LSB                                                  2
#define GCREG_DE_STRIDE_V_BLK                                                  0
#define GCREG_DE_STRIDE_V_Count                                                4
#define GCREG_DE_STRIDE_V_FieldMask                                   0x0003FFFF
#define GCREG_DE_STRIDE_V_ReadMask                                    0x0003FFFC
#define GCREG_DE_STRIDE_V_WriteMask                                   0x0003FFFC
#define GCREG_DE_STRIDE_V_ResetValue                                  0x00000000

#define GCREG_DE_STRIDE_V_STRIDE                                            17:0
#define GCREG_DE_STRIDE_V_STRIDE_End                                          17
#define GCREG_DE_STRIDE_V_STRIDE_Start                                         0
#define GCREG_DE_STRIDE_V_STRIDE_Type                                        U18

#define gcregDESrcRotationHeightRegAddrs                                  0x4A38
#define GCREG_DE_SRC_ROTATION_HEIGHT_Address                             0x128E0
#define GCREG_DE_SRC_ROTATION_HEIGHT_MSB                                      15
#define GCREG_DE_SRC_ROTATION_HEIGHT_LSB                                       2
#define GCREG_DE_SRC_ROTATION_HEIGHT_BLK                                       0
#define GCREG_DE_SRC_ROTATION_HEIGHT_Count                                     4
#define GCREG_DE_SRC_ROTATION_HEIGHT_FieldMask                        0x0000FFFF
#define GCREG_DE_SRC_ROTATION_HEIGHT_ReadMask                         0x0000FFFF
#define GCREG_DE_SRC_ROTATION_HEIGHT_WriteMask                        0x0000FFFF
#define GCREG_DE_SRC_ROTATION_HEIGHT_ResetValue                       0x00000000

#define GCREG_DE_SRC_ROTATION_HEIGHT_HEIGHT                                 15:0
#define GCREG_DE_SRC_ROTATION_HEIGHT_HEIGHT_End                               15
#define GCREG_DE_SRC_ROTATION_HEIGHT_HEIGHT_Start                              0
#define GCREG_DE_SRC_ROTATION_HEIGHT_HEIGHT_Type                             U16

#define gcregDERotAngleRegAddrs                                           0x4A3C
#define GCREG_DE_ROT_ANGLE_Address                                       0x128F0
#define GCREG_DE_ROT_ANGLE_MSB                                                15
#define GCREG_DE_ROT_ANGLE_LSB                                                 2
#define GCREG_DE_ROT_ANGLE_BLK                                                 0
#define GCREG_DE_ROT_ANGLE_Count                                               4
#define GCREG_DE_ROT_ANGLE_FieldMask                                  0x03FBB33F
#define GCREG_DE_ROT_ANGLE_ReadMask                                   0x03FBB33F
#define GCREG_DE_ROT_ANGLE_WriteMask                                  0x03FBB33F
#define GCREG_DE_ROT_ANGLE_ResetValue                                 0x00000000

#define GCREG_DE_ROT_ANGLE_SRC                                               2:0
#define GCREG_DE_ROT_ANGLE_SRC_End                                             2
#define GCREG_DE_ROT_ANGLE_SRC_Start                                           0
#define GCREG_DE_ROT_ANGLE_SRC_Type                                          U03
#define   GCREG_DE_ROT_ANGLE_SRC_ROT0                                        0x0
#define   GCREG_DE_ROT_ANGLE_SRC_FLIP_X                                      0x1
#define   GCREG_DE_ROT_ANGLE_SRC_FLIP_Y                                      0x2
#define   GCREG_DE_ROT_ANGLE_SRC_ROT90                                       0x4
#define   GCREG_DE_ROT_ANGLE_SRC_ROT180                                      0x5
#define   GCREG_DE_ROT_ANGLE_SRC_ROT270                                      0x6

#define GCREG_DE_ROT_ANGLE_DST                                               5:3
#define GCREG_DE_ROT_ANGLE_DST_End                                             5
#define GCREG_DE_ROT_ANGLE_DST_Start                                           3
#define GCREG_DE_ROT_ANGLE_DST_Type                                          U03
#define   GCREG_DE_ROT_ANGLE_DST_ROT0                                        0x0
#define   GCREG_DE_ROT_ANGLE_DST_FLIP_X                                      0x1
#define   GCREG_DE_ROT_ANGLE_DST_FLIP_Y                                      0x2
#define   GCREG_DE_ROT_ANGLE_DST_ROT90                                       0x4
#define   GCREG_DE_ROT_ANGLE_DST_ROT180                                      0x5
#define   GCREG_DE_ROT_ANGLE_DST_ROT270                                      0x6

#define GCREG_DE_ROT_ANGLE_MASK_SRC                                          8:8
#define GCREG_DE_ROT_ANGLE_MASK_SRC_End                                        8
#define GCREG_DE_ROT_ANGLE_MASK_SRC_Start                                      8
#define GCREG_DE_ROT_ANGLE_MASK_SRC_Type                                     U01
#define   GCREG_DE_ROT_ANGLE_MASK_SRC_ENABLED                                0x0
#define   GCREG_DE_ROT_ANGLE_MASK_SRC_MASKED                                 0x1

#define GCREG_DE_ROT_ANGLE_MASK_DST                                          9:9
#define GCREG_DE_ROT_ANGLE_MASK_DST_End                                        9
#define GCREG_DE_ROT_ANGLE_MASK_DST_Start                                      9
#define GCREG_DE_ROT_ANGLE_MASK_DST_Type                                     U01
#define   GCREG_DE_ROT_ANGLE_MASK_DST_ENABLED                                0x0
#define   GCREG_DE_ROT_ANGLE_MASK_DST_MASKED                                 0x1

#define GCREG_DE_ROT_ANGLE_SRC_MIRROR                                      13:12
#define GCREG_DE_ROT_ANGLE_SRC_MIRROR_End                                     13
#define GCREG_DE_ROT_ANGLE_SRC_MIRROR_Start                                   12
#define GCREG_DE_ROT_ANGLE_SRC_MIRROR_Type                                   U02
#define   GCREG_DE_ROT_ANGLE_SRC_MIRROR_NONE                                 0x0
#define   GCREG_DE_ROT_ANGLE_SRC_MIRROR_MIRROR_X                             0x1
#define   GCREG_DE_ROT_ANGLE_SRC_MIRROR_MIRROR_Y                             0x2
#define   GCREG_DE_ROT_ANGLE_SRC_MIRROR_MIRROR_XY                            0x3

#define GCREG_DE_ROT_ANGLE_MASK_SRC_MIRROR                                 15:15
#define GCREG_DE_ROT_ANGLE_MASK_SRC_MIRROR_End                                15
#define GCREG_DE_ROT_ANGLE_MASK_SRC_MIRROR_Start                              15
#define GCREG_DE_ROT_ANGLE_MASK_SRC_MIRROR_Type                              U01
#define   GCREG_DE_ROT_ANGLE_MASK_SRC_MIRROR_ENABLED                         0x0
#define   GCREG_DE_ROT_ANGLE_MASK_SRC_MIRROR_MASKED                          0x1

#define GCREG_DE_ROT_ANGLE_DST_MIRROR                                      17:16
#define GCREG_DE_ROT_ANGLE_DST_MIRROR_End                                     17
#define GCREG_DE_ROT_ANGLE_DST_MIRROR_Start                                   16
#define GCREG_DE_ROT_ANGLE_DST_MIRROR_Type                                   U02
#define   GCREG_DE_ROT_ANGLE_DST_MIRROR_NONE                                 0x0
#define   GCREG_DE_ROT_ANGLE_DST_MIRROR_MIRROR_X                             0x1
#define   GCREG_DE_ROT_ANGLE_DST_MIRROR_MIRROR_Y                             0x2
#define   GCREG_DE_ROT_ANGLE_DST_MIRROR_MIRROR_XY                            0x3

#define GCREG_DE_ROT_ANGLE_MASK_DST_MIRROR                                 19:19
#define GCREG_DE_ROT_ANGLE_MASK_DST_MIRROR_End                                19
#define GCREG_DE_ROT_ANGLE_MASK_DST_MIRROR_Start                              19
#define GCREG_DE_ROT_ANGLE_MASK_DST_MIRROR_Type                              U01
#define   GCREG_DE_ROT_ANGLE_MASK_DST_MIRROR_ENABLED                         0x0
#define   GCREG_DE_ROT_ANGLE_MASK_DST_MIRROR_MASKED                          0x1

#define GCREG_DE_ROT_ANGLE_SRC_FLIP                                        21:20
#define GCREG_DE_ROT_ANGLE_SRC_FLIP_End                                       21
#define GCREG_DE_ROT_ANGLE_SRC_FLIP_Start                                     20
#define GCREG_DE_ROT_ANGLE_SRC_FLIP_Type                                     U02
#define   GCREG_DE_ROT_ANGLE_SRC_FLIP_NONE                                   0x0
#define   GCREG_DE_ROT_ANGLE_SRC_FLIP_FLIP_X                                 0x1
#define   GCREG_DE_ROT_ANGLE_SRC_FLIP_FLIP_Y                                 0x2
#define   GCREG_DE_ROT_ANGLE_SRC_FLIP_FLIP_XY                                0x3

#define GCREG_DE_ROT_ANGLE_MASK_SRC_FLIP                                   22:22
#define GCREG_DE_ROT_ANGLE_MASK_SRC_FLIP_End                                  22
#define GCREG_DE_ROT_ANGLE_MASK_SRC_FLIP_Start                                22
#define GCREG_DE_ROT_ANGLE_MASK_SRC_FLIP_Type                                U01
#define   GCREG_DE_ROT_ANGLE_MASK_SRC_FLIP_ENABLED                           0x0
#define   GCREG_DE_ROT_ANGLE_MASK_SRC_FLIP_MASKED                            0x1

#define GCREG_DE_ROT_ANGLE_DST_FLIP                                        24:23
#define GCREG_DE_ROT_ANGLE_DST_FLIP_End                                       24
#define GCREG_DE_ROT_ANGLE_DST_FLIP_Start                                     23
#define GCREG_DE_ROT_ANGLE_DST_FLIP_Type                                     U02
#define   GCREG_DE_ROT_ANGLE_DST_FLIP_NONE                                   0x0
#define   GCREG_DE_ROT_ANGLE_DST_FLIP_FLIP_X                                 0x1
#define   GCREG_DE_ROT_ANGLE_DST_FLIP_FLIP_Y                                 0x2
#define   GCREG_DE_ROT_ANGLE_DST_FLIP_FLIP_XY                                0x3

#define GCREG_DE_ROT_ANGLE_MASK_DST_FLIP                                   25:25
#define GCREG_DE_ROT_ANGLE_MASK_DST_FLIP_End                                  25
#define GCREG_DE_ROT_ANGLE_MASK_DST_FLIP_Start                                25
#define GCREG_DE_ROT_ANGLE_MASK_DST_FLIP_Type                                U01
#define   GCREG_DE_ROT_ANGLE_MASK_DST_FLIP_ENABLED                           0x0
#define   GCREG_DE_ROT_ANGLE_MASK_DST_FLIP_MASKED                            0x1

#define gcregDEGlobalSrcColorRegAddrs                                     0x4A40
#define GCREG_DE_GLOBAL_SRC_COLOR_Address                                0x12900
#define GCREG_DE_GLOBAL_SRC_COLOR_MSB                                         15
#define GCREG_DE_GLOBAL_SRC_COLOR_LSB                                          2
#define GCREG_DE_GLOBAL_SRC_COLOR_BLK                                          0
#define GCREG_DE_GLOBAL_SRC_COLOR_Count                                        4
#define GCREG_DE_GLOBAL_SRC_COLOR_FieldMask                           0xFFFFFFFF
#define GCREG_DE_GLOBAL_SRC_COLOR_ReadMask                            0xFFFFFFFF
#define GCREG_DE_GLOBAL_SRC_COLOR_WriteMask                           0xFFFFFFFF
#define GCREG_DE_GLOBAL_SRC_COLOR_ResetValue                          0x00000000

#define GCREG_DE_GLOBAL_SRC_COLOR_ALPHA                                    31:24
#define GCREG_DE_GLOBAL_SRC_COLOR_ALPHA_End                                   31
#define GCREG_DE_GLOBAL_SRC_COLOR_ALPHA_Start                                 24
#define GCREG_DE_GLOBAL_SRC_COLOR_ALPHA_Type                                 U08

#define GCREG_DE_GLOBAL_SRC_COLOR_RED                                      23:16
#define GCREG_DE_GLOBAL_SRC_COLOR_RED_End                                     23
#define GCREG_DE_GLOBAL_SRC_COLOR_RED_Start                                   16
#define GCREG_DE_GLOBAL_SRC_COLOR_RED_Type                                   U08

#define GCREG_DE_GLOBAL_SRC_COLOR_GREEN                                     15:8
#define GCREG_DE_GLOBAL_SRC_COLOR_GREEN_End                                   15
#define GCREG_DE_GLOBAL_SRC_COLOR_GREEN_Start                                  8
#define GCREG_DE_GLOBAL_SRC_COLOR_GREEN_Type                                 U08

#define GCREG_DE_GLOBAL_SRC_COLOR_BLUE                                       7:0
#define GCREG_DE_GLOBAL_SRC_COLOR_BLUE_End                                     7
#define GCREG_DE_GLOBAL_SRC_COLOR_BLUE_Start                                   0
#define GCREG_DE_GLOBAL_SRC_COLOR_BLUE_Type                                  U08

#define gcregDEGlobalDestColorRegAddrs                                    0x4A44
#define GCREG_DE_GLOBAL_DEST_COLOR_Address                               0x12910
#define GCREG_DE_GLOBAL_DEST_COLOR_MSB                                        15
#define GCREG_DE_GLOBAL_DEST_COLOR_LSB                                         2
#define GCREG_DE_GLOBAL_DEST_COLOR_BLK                                         0
#define GCREG_DE_GLOBAL_DEST_COLOR_Count                                       4
#define GCREG_DE_GLOBAL_DEST_COLOR_FieldMask                          0xFFFFFFFF
#define GCREG_DE_GLOBAL_DEST_COLOR_ReadMask                           0xFFFFFFFF
#define GCREG_DE_GLOBAL_DEST_COLOR_WriteMask                          0xFFFFFFFF
#define GCREG_DE_GLOBAL_DEST_COLOR_ResetValue                         0x00000000

#define GCREG_DE_GLOBAL_DEST_COLOR_ALPHA                                   31:24
#define GCREG_DE_GLOBAL_DEST_COLOR_ALPHA_End                                  31
#define GCREG_DE_GLOBAL_DEST_COLOR_ALPHA_Start                                24
#define GCREG_DE_GLOBAL_DEST_COLOR_ALPHA_Type                                U08

#define GCREG_DE_GLOBAL_DEST_COLOR_RED                                     23:16
#define GCREG_DE_GLOBAL_DEST_COLOR_RED_End                                    23
#define GCREG_DE_GLOBAL_DEST_COLOR_RED_Start                                  16
#define GCREG_DE_GLOBAL_DEST_COLOR_RED_Type                                  U08

#define GCREG_DE_GLOBAL_DEST_COLOR_GREEN                                    15:8
#define GCREG_DE_GLOBAL_DEST_COLOR_GREEN_End                                  15
#define GCREG_DE_GLOBAL_DEST_COLOR_GREEN_Start                                 8
#define GCREG_DE_GLOBAL_DEST_COLOR_GREEN_Type                                U08

#define GCREG_DE_GLOBAL_DEST_COLOR_BLUE                                      7:0
#define GCREG_DE_GLOBAL_DEST_COLOR_BLUE_End                                    7
#define GCREG_DE_GLOBAL_DEST_COLOR_BLUE_Start                                  0
#define GCREG_DE_GLOBAL_DEST_COLOR_BLUE_Type                                 U08

#define gcregDEColorMultiplyModesRegAddrs                                 0x4A48
#define GCREG_DE_COLOR_MULTIPLY_MODES_Address                            0x12920
#define GCREG_DE_COLOR_MULTIPLY_MODES_MSB                                     15
#define GCREG_DE_COLOR_MULTIPLY_MODES_LSB                                      2
#define GCREG_DE_COLOR_MULTIPLY_MODES_BLK                                      0
#define GCREG_DE_COLOR_MULTIPLY_MODES_Count                                    4
#define GCREG_DE_COLOR_MULTIPLY_MODES_FieldMask                       0x00300311
#define GCREG_DE_COLOR_MULTIPLY_MODES_ReadMask                        0x00300311
#define GCREG_DE_COLOR_MULTIPLY_MODES_WriteMask                       0x00300311
#define GCREG_DE_COLOR_MULTIPLY_MODES_ResetValue                      0x00000000

#define GCREG_DE_COLOR_MULTIPLY_MODES_SRC_PREMULTIPLY                        0:0
#define GCREG_DE_COLOR_MULTIPLY_MODES_SRC_PREMULTIPLY_End                      0
#define GCREG_DE_COLOR_MULTIPLY_MODES_SRC_PREMULTIPLY_Start                    0
#define GCREG_DE_COLOR_MULTIPLY_MODES_SRC_PREMULTIPLY_Type                   U01
#define   GCREG_DE_COLOR_MULTIPLY_MODES_SRC_PREMULTIPLY_DISABLE              0x0
#define   GCREG_DE_COLOR_MULTIPLY_MODES_SRC_PREMULTIPLY_ENABLE               0x1

#define GCREG_DE_COLOR_MULTIPLY_MODES_DST_PREMULTIPLY                        4:4
#define GCREG_DE_COLOR_MULTIPLY_MODES_DST_PREMULTIPLY_End                      4
#define GCREG_DE_COLOR_MULTIPLY_MODES_DST_PREMULTIPLY_Start                    4
#define GCREG_DE_COLOR_MULTIPLY_MODES_DST_PREMULTIPLY_Type                   U01
#define   GCREG_DE_COLOR_MULTIPLY_MODES_DST_PREMULTIPLY_DISABLE              0x0
#define   GCREG_DE_COLOR_MULTIPLY_MODES_DST_PREMULTIPLY_ENABLE               0x1

#define GCREG_DE_COLOR_MULTIPLY_MODES_SRC_GLOBAL_PREMULTIPLY                 9:8
#define GCREG_DE_COLOR_MULTIPLY_MODES_SRC_GLOBAL_PREMULTIPLY_End               9
#define GCREG_DE_COLOR_MULTIPLY_MODES_SRC_GLOBAL_PREMULTIPLY_Start             8
#define GCREG_DE_COLOR_MULTIPLY_MODES_SRC_GLOBAL_PREMULTIPLY_Type            U02
#define   GCREG_DE_COLOR_MULTIPLY_MODES_SRC_GLOBAL_PREMULTIPLY_DISABLE       0x0
#define   GCREG_DE_COLOR_MULTIPLY_MODES_SRC_GLOBAL_PREMULTIPLY_ALPHA         0x1
#define   GCREG_DE_COLOR_MULTIPLY_MODES_SRC_GLOBAL_PREMULTIPLY_COLOR         0x2

#define GCREG_DE_COLOR_MULTIPLY_MODES_DST_DEMULTIPLY                       20:20
#define GCREG_DE_COLOR_MULTIPLY_MODES_DST_DEMULTIPLY_End                      20
#define GCREG_DE_COLOR_MULTIPLY_MODES_DST_DEMULTIPLY_Start                    20
#define GCREG_DE_COLOR_MULTIPLY_MODES_DST_DEMULTIPLY_Type                    U01
#define   GCREG_DE_COLOR_MULTIPLY_MODES_DST_DEMULTIPLY_DISABLE               0x0
#define   GCREG_DE_COLOR_MULTIPLY_MODES_DST_DEMULTIPLY_ENABLE                0x1

#define GCREG_DE_COLOR_MULTIPLY_MODES_DST_DEMULTIPLY_ZERO                  21:21
#define GCREG_DE_COLOR_MULTIPLY_MODES_DST_DEMULTIPLY_ZERO_End                 21
#define GCREG_DE_COLOR_MULTIPLY_MODES_DST_DEMULTIPLY_ZERO_Start               21
#define GCREG_DE_COLOR_MULTIPLY_MODES_DST_DEMULTIPLY_ZERO_Type               U01
#define   GCREG_DE_COLOR_MULTIPLY_MODES_DST_DEMULTIPLY_ZERO_DISABLE          0x0
#define   GCREG_DE_COLOR_MULTIPLY_MODES_DST_DEMULTIPLY_ZERO_ENABLE           0x1

#define gcregPETransparencyRegAddrs                                       0x4A4C
#define GCREG_PE_TRANSPARENCY_Address                                    0x12930
#define GCREG_PE_TRANSPARENCY_MSB                                             15
#define GCREG_PE_TRANSPARENCY_LSB                                              2
#define GCREG_PE_TRANSPARENCY_BLK                                              0
#define GCREG_PE_TRANSPARENCY_Count                                            4
#define GCREG_PE_TRANSPARENCY_FieldMask                               0xB3331333
#define GCREG_PE_TRANSPARENCY_ReadMask                                0xB3331333
#define GCREG_PE_TRANSPARENCY_WriteMask                               0xB3331333
#define GCREG_PE_TRANSPARENCY_ResetValue                              0x00000000

#define GCREG_PE_TRANSPARENCY_SOURCE                                         1:0
#define GCREG_PE_TRANSPARENCY_SOURCE_End                                       1
#define GCREG_PE_TRANSPARENCY_SOURCE_Start                                     0
#define GCREG_PE_TRANSPARENCY_SOURCE_Type                                    U02
#define   GCREG_PE_TRANSPARENCY_SOURCE_OPAQUE                                0x0
#define   GCREG_PE_TRANSPARENCY_SOURCE_MASK                                  0x1
#define   GCREG_PE_TRANSPARENCY_SOURCE_KEY                                   0x2

#define GCREG_PE_TRANSPARENCY_PATTERN                                        5:4
#define GCREG_PE_TRANSPARENCY_PATTERN_End                                      5
#define GCREG_PE_TRANSPARENCY_PATTERN_Start                                    4
#define GCREG_PE_TRANSPARENCY_PATTERN_Type                                   U02
#define   GCREG_PE_TRANSPARENCY_PATTERN_OPAQUE                               0x0
#define   GCREG_PE_TRANSPARENCY_PATTERN_MASK                                 0x1
#define   GCREG_PE_TRANSPARENCY_PATTERN_KEY                                  0x2

#define GCREG_PE_TRANSPARENCY_DESTINATION                                    9:8
#define GCREG_PE_TRANSPARENCY_DESTINATION_End                                  9
#define GCREG_PE_TRANSPARENCY_DESTINATION_Start                                8
#define GCREG_PE_TRANSPARENCY_DESTINATION_Type                               U02
#define   GCREG_PE_TRANSPARENCY_DESTINATION_OPAQUE                           0x0
#define   GCREG_PE_TRANSPARENCY_DESTINATION_MASK                             0x1
#define   GCREG_PE_TRANSPARENCY_DESTINATION_KEY                              0x2

#define GCREG_PE_TRANSPARENCY_MASK_TRANSPARENCY                            12:12
#define GCREG_PE_TRANSPARENCY_MASK_TRANSPARENCY_End                           12
#define GCREG_PE_TRANSPARENCY_MASK_TRANSPARENCY_Start                         12
#define GCREG_PE_TRANSPARENCY_MASK_TRANSPARENCY_Type                         U01
#define   GCREG_PE_TRANSPARENCY_MASK_TRANSPARENCY_ENABLED                    0x0
#define   GCREG_PE_TRANSPARENCY_MASK_TRANSPARENCY_MASKED                     0x1

#define GCREG_PE_TRANSPARENCY_USE_SRC_OVERRIDE                             17:16
#define GCREG_PE_TRANSPARENCY_USE_SRC_OVERRIDE_End                            17
#define GCREG_PE_TRANSPARENCY_USE_SRC_OVERRIDE_Start                          16
#define GCREG_PE_TRANSPARENCY_USE_SRC_OVERRIDE_Type                          U02
#define   GCREG_PE_TRANSPARENCY_USE_SRC_OVERRIDE_DEFAULT                     0x0
#define   GCREG_PE_TRANSPARENCY_USE_SRC_OVERRIDE_USE_ENABLE                  0x1
#define   GCREG_PE_TRANSPARENCY_USE_SRC_OVERRIDE_USE_DISABLE                 0x2

#define GCREG_PE_TRANSPARENCY_USE_PAT_OVERRIDE                             21:20
#define GCREG_PE_TRANSPARENCY_USE_PAT_OVERRIDE_End                            21
#define GCREG_PE_TRANSPARENCY_USE_PAT_OVERRIDE_Start                          20
#define GCREG_PE_TRANSPARENCY_USE_PAT_OVERRIDE_Type                          U02
#define   GCREG_PE_TRANSPARENCY_USE_PAT_OVERRIDE_DEFAULT                     0x0
#define   GCREG_PE_TRANSPARENCY_USE_PAT_OVERRIDE_USE_ENABLE                  0x1
#define   GCREG_PE_TRANSPARENCY_USE_PAT_OVERRIDE_USE_DISABLE                 0x2

#define GCREG_PE_TRANSPARENCY_USE_DST_OVERRIDE                             25:24
#define GCREG_PE_TRANSPARENCY_USE_DST_OVERRIDE_End                            25
#define GCREG_PE_TRANSPARENCY_USE_DST_OVERRIDE_Start                          24
#define GCREG_PE_TRANSPARENCY_USE_DST_OVERRIDE_Type                          U02
#define   GCREG_PE_TRANSPARENCY_USE_DST_OVERRIDE_DEFAULT                     0x0
#define   GCREG_PE_TRANSPARENCY_USE_DST_OVERRIDE_USE_ENABLE                  0x1
#define   GCREG_PE_TRANSPARENCY_USE_DST_OVERRIDE_USE_DISABLE                 0x2

#define GCREG_PE_TRANSPARENCY_MASK_RESOURCE_OVERRIDE                       28:28
#define GCREG_PE_TRANSPARENCY_MASK_RESOURCE_OVERRIDE_End                      28
#define GCREG_PE_TRANSPARENCY_MASK_RESOURCE_OVERRIDE_Start                    28
#define GCREG_PE_TRANSPARENCY_MASK_RESOURCE_OVERRIDE_Type                    U01
#define   GCREG_PE_TRANSPARENCY_MASK_RESOURCE_OVERRIDE_ENABLED               0x0
#define   GCREG_PE_TRANSPARENCY_MASK_RESOURCE_OVERRIDE_MASKED                0x1

#define GCREG_PE_TRANSPARENCY_DFB_COLOR_KEY                                29:29
#define GCREG_PE_TRANSPARENCY_DFB_COLOR_KEY_End                               29
#define GCREG_PE_TRANSPARENCY_DFB_COLOR_KEY_Start                             29
#define GCREG_PE_TRANSPARENCY_DFB_COLOR_KEY_Type                             U01
#define   GCREG_PE_TRANSPARENCY_DFB_COLOR_KEY_DISABLED                       0x0
#define   GCREG_PE_TRANSPARENCY_DFB_COLOR_KEY_ENABLED                        0x1

#define GCREG_PE_TRANSPARENCY_MASK_DFB_COLOR_KEY                           31:31
#define GCREG_PE_TRANSPARENCY_MASK_DFB_COLOR_KEY_End                          31
#define GCREG_PE_TRANSPARENCY_MASK_DFB_COLOR_KEY_Start                        31
#define GCREG_PE_TRANSPARENCY_MASK_DFB_COLOR_KEY_Type                        U01
#define   GCREG_PE_TRANSPARENCY_MASK_DFB_COLOR_KEY_ENABLED                   0x0
#define   GCREG_PE_TRANSPARENCY_MASK_DFB_COLOR_KEY_MASKED                    0x1

#define gcregPEControlRegAddrs                                            0x4A50
#define GCREG_PE_CONTROL_Address                                         0x12940
#define GCREG_PE_CONTROL_MSB                                                  15
#define GCREG_PE_CONTROL_LSB                                                   2
#define GCREG_PE_CONTROL_BLK                                                   0
#define GCREG_PE_CONTROL_Count                                                 4
#define GCREG_PE_CONTROL_FieldMask                                    0x0009999B
#define GCREG_PE_CONTROL_ReadMask                                     0x0009999B
#define GCREG_PE_CONTROL_WriteMask                                    0x0009999B
#define GCREG_PE_CONTROL_ResetValue                                   0x00000000

#define GCREG_PE_CONTROL_YUV                                                 1:0
#define GCREG_PE_CONTROL_YUV_End                                               1
#define GCREG_PE_CONTROL_YUV_Start                                             0
#define GCREG_PE_CONTROL_YUV_Type                                            U02
#define   GCREG_PE_CONTROL_YUV_601                                           0x0
#define   GCREG_PE_CONTROL_YUV_709                                           0x1
#define   GCREG_PE_CONTROL_YUV_UserDefined                                   0x2
#define   GCREG_PE_CONTROL_YUV_2020                                          0x3

#define GCREG_PE_CONTROL_MASK_YUV                                            3:3
#define GCREG_PE_CONTROL_MASK_YUV_End                                          3
#define GCREG_PE_CONTROL_MASK_YUV_Start                                        3
#define GCREG_PE_CONTROL_MASK_YUV_Type                                       U01
#define   GCREG_PE_CONTROL_MASK_YUV_ENABLED                                  0x0
#define   GCREG_PE_CONTROL_MASK_YUV_MASKED                                   0x1

#define GCREG_PE_CONTROL_UV_SWIZZLE                                          4:4
#define GCREG_PE_CONTROL_UV_SWIZZLE_End                                        4
#define GCREG_PE_CONTROL_UV_SWIZZLE_Start                                      4
#define GCREG_PE_CONTROL_UV_SWIZZLE_Type                                     U01
#define   GCREG_PE_CONTROL_UV_SWIZZLE_UV                                     0x0
#define   GCREG_PE_CONTROL_UV_SWIZZLE_VU                                     0x1

#define GCREG_PE_CONTROL_MASK_UV_SWIZZLE                                     7:7
#define GCREG_PE_CONTROL_MASK_UV_SWIZZLE_End                                   7
#define GCREG_PE_CONTROL_MASK_UV_SWIZZLE_Start                                 7
#define GCREG_PE_CONTROL_MASK_UV_SWIZZLE_Type                                U01
#define   GCREG_PE_CONTROL_MASK_UV_SWIZZLE_ENABLED                           0x0
#define   GCREG_PE_CONTROL_MASK_UV_SWIZZLE_MASKED                            0x1

#define GCREG_PE_CONTROL_YUVRGB                                              8:8
#define GCREG_PE_CONTROL_YUVRGB_End                                            8
#define GCREG_PE_CONTROL_YUVRGB_Start                                          8
#define GCREG_PE_CONTROL_YUVRGB_Type                                         U01
#define   GCREG_PE_CONTROL_YUVRGB_DISABLED                                   0x0
#define   GCREG_PE_CONTROL_YUVRGB_ENABLED                                    0x1

#define GCREG_PE_CONTROL_MASK_YUVRGB                                       11:11
#define GCREG_PE_CONTROL_MASK_YUVRGB_End                                      11
#define GCREG_PE_CONTROL_MASK_YUVRGB_Start                                    11
#define GCREG_PE_CONTROL_MASK_YUVRGB_Type                                    U01
#define   GCREG_PE_CONTROL_MASK_YUVRGB_ENABLED                               0x0
#define   GCREG_PE_CONTROL_MASK_YUVRGB_MASKED                                0x1

#define GCREG_PE_CONTROL_YUV_CLAMP                                         12:12
#define GCREG_PE_CONTROL_YUV_CLAMP_End                                        12
#define GCREG_PE_CONTROL_YUV_CLAMP_Start                                      12
#define GCREG_PE_CONTROL_YUV_CLAMP_Type                                      U01
#define   GCREG_PE_CONTROL_YUV_CLAMP_DISABLED                                0x0
#define   GCREG_PE_CONTROL_YUV_CLAMP_ENABLED                                 0x1

#define GCREG_PE_CONTROL_MASK_YUV_CLAMP                                    15:15
#define GCREG_PE_CONTROL_MASK_YUV_CLAMP_End                                   15
#define GCREG_PE_CONTROL_MASK_YUV_CLAMP_Start                                 15
#define GCREG_PE_CONTROL_MASK_YUV_CLAMP_Type                                 U01
#define   GCREG_PE_CONTROL_MASK_YUV_CLAMP_ENABLED                            0x0
#define   GCREG_PE_CONTROL_MASK_YUV_CLAMP_MASKED                             0x1

#define GCREG_PE_CONTROL_GRAY_YUVRGB                                       16:16
#define GCREG_PE_CONTROL_GRAY_YUVRGB_End                                      16
#define GCREG_PE_CONTROL_GRAY_YUVRGB_Start                                    16
#define GCREG_PE_CONTROL_GRAY_YUVRGB_Type                                    U01
#define   GCREG_PE_CONTROL_GRAY_YUVRGB_DISABLED                              0x0
#define   GCREG_PE_CONTROL_GRAY_YUVRGB_ENABLED                               0x1

#define GCREG_PE_CONTROL_MASK_GRAY_YUVRGB                                  19:19
#define GCREG_PE_CONTROL_MASK_GRAY_YUVRGB_End                                 19
#define GCREG_PE_CONTROL_MASK_GRAY_YUVRGB_Start                               19
#define GCREG_PE_CONTROL_MASK_GRAY_YUVRGB_Type                               U01
#define   GCREG_PE_CONTROL_MASK_GRAY_YUVRGB_ENABLED                          0x0
#define   GCREG_PE_CONTROL_MASK_GRAY_YUVRGB_MASKED                           0x1

#define gcregDESrcColorKeyHighRegAddrs                                    0x4A54
#define GCREG_DE_SRC_COLOR_KEY_HIGH_Address                              0x12950
#define GCREG_DE_SRC_COLOR_KEY_HIGH_MSB                                       15
#define GCREG_DE_SRC_COLOR_KEY_HIGH_LSB                                        2
#define GCREG_DE_SRC_COLOR_KEY_HIGH_BLK                                        0
#define GCREG_DE_SRC_COLOR_KEY_HIGH_Count                                      4
#define GCREG_DE_SRC_COLOR_KEY_HIGH_FieldMask                         0xFFFFFFFF
#define GCREG_DE_SRC_COLOR_KEY_HIGH_ReadMask                          0xFFFFFFFF
#define GCREG_DE_SRC_COLOR_KEY_HIGH_WriteMask                         0xFFFFFFFF
#define GCREG_DE_SRC_COLOR_KEY_HIGH_ResetValue                        0x00000000

#define GCREG_DE_SRC_COLOR_KEY_HIGH_ALPHA                                  31:24
#define GCREG_DE_SRC_COLOR_KEY_HIGH_ALPHA_End                                 31
#define GCREG_DE_SRC_COLOR_KEY_HIGH_ALPHA_Start                               24
#define GCREG_DE_SRC_COLOR_KEY_HIGH_ALPHA_Type                               U08

#define GCREG_DE_SRC_COLOR_KEY_HIGH_RED                                    23:16
#define GCREG_DE_SRC_COLOR_KEY_HIGH_RED_End                                   23
#define GCREG_DE_SRC_COLOR_KEY_HIGH_RED_Start                                 16
#define GCREG_DE_SRC_COLOR_KEY_HIGH_RED_Type                                 U08

#define GCREG_DE_SRC_COLOR_KEY_HIGH_GREEN                                   15:8
#define GCREG_DE_SRC_COLOR_KEY_HIGH_GREEN_End                                 15
#define GCREG_DE_SRC_COLOR_KEY_HIGH_GREEN_Start                                8
#define GCREG_DE_SRC_COLOR_KEY_HIGH_GREEN_Type                               U08

#define GCREG_DE_SRC_COLOR_KEY_HIGH_BLUE                                     7:0
#define GCREG_DE_SRC_COLOR_KEY_HIGH_BLUE_End                                   7
#define GCREG_DE_SRC_COLOR_KEY_HIGH_BLUE_Start                                 0
#define GCREG_DE_SRC_COLOR_KEY_HIGH_BLUE_Type                                U08

#define gcregDESrcExConfigRegAddrs                                        0x4A58
#define GCREG_DE_SRC_EX_CONFIG_Address                                   0x12960
#define GCREG_DE_SRC_EX_CONFIG_MSB                                            15
#define GCREG_DE_SRC_EX_CONFIG_LSB                                             2
#define GCREG_DE_SRC_EX_CONFIG_BLK                                             0
#define GCREG_DE_SRC_EX_CONFIG_Count                                           4
#define GCREG_DE_SRC_EX_CONFIG_FieldMask                              0x00001109
#define GCREG_DE_SRC_EX_CONFIG_ReadMask                               0x00001109
#define GCREG_DE_SRC_EX_CONFIG_WriteMask                              0x00001109
#define GCREG_DE_SRC_EX_CONFIG_ResetValue                             0x00000000

#define GCREG_DE_SRC_EX_CONFIG_MULTI_TILED                                   0:0
#define GCREG_DE_SRC_EX_CONFIG_MULTI_TILED_End                                 0
#define GCREG_DE_SRC_EX_CONFIG_MULTI_TILED_Start                               0
#define GCREG_DE_SRC_EX_CONFIG_MULTI_TILED_Type                              U01
#define   GCREG_DE_SRC_EX_CONFIG_MULTI_TILED_DISABLED                        0x0
#define   GCREG_DE_SRC_EX_CONFIG_MULTI_TILED_ENABLED                         0x1

#define GCREG_DE_SRC_EX_CONFIG_SUPER_TILED                                   3:3
#define GCREG_DE_SRC_EX_CONFIG_SUPER_TILED_End                                 3
#define GCREG_DE_SRC_EX_CONFIG_SUPER_TILED_Start                               3
#define GCREG_DE_SRC_EX_CONFIG_SUPER_TILED_Type                              U01
#define   GCREG_DE_SRC_EX_CONFIG_SUPER_TILED_DISABLED                        0x0
#define   GCREG_DE_SRC_EX_CONFIG_SUPER_TILED_ENABLED                         0x1

#define GCREG_DE_SRC_EX_CONFIG_MINOR_TILED                                   8:8
#define GCREG_DE_SRC_EX_CONFIG_MINOR_TILED_End                                 8
#define GCREG_DE_SRC_EX_CONFIG_MINOR_TILED_Start                               8
#define GCREG_DE_SRC_EX_CONFIG_MINOR_TILED_Type                              U01
#define   GCREG_DE_SRC_EX_CONFIG_MINOR_TILED_DISABLED                        0x0
#define   GCREG_DE_SRC_EX_CONFIG_MINOR_TILED_ENABLED                         0x1

#define GCREG_DE_SRC_EX_CONFIG_CACHE_MODE                                  12:12
#define GCREG_DE_SRC_EX_CONFIG_CACHE_MODE_End                                 12
#define GCREG_DE_SRC_EX_CONFIG_CACHE_MODE_Start                               12
#define GCREG_DE_SRC_EX_CONFIG_CACHE_MODE_Type                               U01
#define   GCREG_DE_SRC_EX_CONFIG_CACHE_MODE_DISABLED                         0x0
#define   GCREG_DE_SRC_EX_CONFIG_CACHE_MODE_ENABLED                          0x1

#define gcregDESrcExAddressRegAddrs                                       0x4A5C
#define GCREG_DE_SRC_EX_ADDRESS_Address                                  0x12970
#define GCREG_DE_SRC_EX_ADDRESS_MSB                                           15
#define GCREG_DE_SRC_EX_ADDRESS_LSB                                            2
#define GCREG_DE_SRC_EX_ADDRESS_BLK                                            0
#define GCREG_DE_SRC_EX_ADDRESS_Count                                          4
#define GCREG_DE_SRC_EX_ADDRESS_FieldMask                             0xFFFFFFFF
#define GCREG_DE_SRC_EX_ADDRESS_ReadMask                              0xFFFFFFFC
#define GCREG_DE_SRC_EX_ADDRESS_WriteMask                             0xFFFFFFFC
#define GCREG_DE_SRC_EX_ADDRESS_ResetValue                            0x00000000

#define GCREG_DE_SRC_EX_ADDRESS_TYPE                                       31:31
#define GCREG_DE_SRC_EX_ADDRESS_TYPE_End                                      31
#define GCREG_DE_SRC_EX_ADDRESS_TYPE_Start                                    31
#define GCREG_DE_SRC_EX_ADDRESS_TYPE_Type                                    U01
#define   GCREG_DE_SRC_EX_ADDRESS_TYPE_SYSTEM                                0x0
#define   GCREG_DE_SRC_EX_ADDRESS_TYPE_VIRTUAL_SYSTEM                        0x1

#define GCREG_DE_SRC_EX_ADDRESS_ADDRESS                                     30:0
#define GCREG_DE_SRC_EX_ADDRESS_ADDRESS_End                                   30
#define GCREG_DE_SRC_EX_ADDRESS_ADDRESS_Start                                  0
#define GCREG_DE_SRC_EX_ADDRESS_ADDRESS_Type                                 U31

#define gcregDESrcPlane0AlignConfigRegAddrs                               0x4A60
#define GCREG_DE_SRC_PLANE0_ALIGN_CONFIG_Address                         0x12980
#define GCREG_DE_SRC_PLANE0_ALIGN_CONFIG_MSB                                  15
#define GCREG_DE_SRC_PLANE0_ALIGN_CONFIG_LSB                                   3
#define GCREG_DE_SRC_PLANE0_ALIGN_CONFIG_BLK                                   0
#define GCREG_DE_SRC_PLANE0_ALIGN_CONFIG_Count                                 8
#define GCREG_DE_SRC_PLANE0_ALIGN_CONFIG_FieldMask                    0x0001FFFF
#define GCREG_DE_SRC_PLANE0_ALIGN_CONFIG_ReadMask                     0x0001FFFF
#define GCREG_DE_SRC_PLANE0_ALIGN_CONFIG_WriteMask                    0x0001FFFF
#define GCREG_DE_SRC_PLANE0_ALIGN_CONFIG_ResetValue                   0x00000000

#define GCREG_DE_SRC_PLANE0_ALIGN_CONFIG_STRIDE_WIDTH                       16:0
#define GCREG_DE_SRC_PLANE0_ALIGN_CONFIG_STRIDE_WIDTH_End                     16
#define GCREG_DE_SRC_PLANE0_ALIGN_CONFIG_STRIDE_WIDTH_Start                    0
#define GCREG_DE_SRC_PLANE0_ALIGN_CONFIG_STRIDE_WIDTH_Type                   U17

#define gcregDESrcPlane1AlignConfigRegAddrs                               0x4A68
#define GCREG_DE_SRC_PLANE1_ALIGN_CONFIG_Address                         0x129A0
#define GCREG_DE_SRC_PLANE1_ALIGN_CONFIG_MSB                                  15
#define GCREG_DE_SRC_PLANE1_ALIGN_CONFIG_LSB                                   3
#define GCREG_DE_SRC_PLANE1_ALIGN_CONFIG_BLK                                   0
#define GCREG_DE_SRC_PLANE1_ALIGN_CONFIG_Count                                 8
#define GCREG_DE_SRC_PLANE1_ALIGN_CONFIG_FieldMask                    0x0001FFFF
#define GCREG_DE_SRC_PLANE1_ALIGN_CONFIG_ReadMask                     0x0001FFFF
#define GCREG_DE_SRC_PLANE1_ALIGN_CONFIG_WriteMask                    0x0001FFFF
#define GCREG_DE_SRC_PLANE1_ALIGN_CONFIG_ResetValue                   0x00000000

#define GCREG_DE_SRC_PLANE1_ALIGN_CONFIG_STRIDE_WIDTH                       16:0
#define GCREG_DE_SRC_PLANE1_ALIGN_CONFIG_STRIDE_WIDTH_End                     16
#define GCREG_DE_SRC_PLANE1_ALIGN_CONFIG_STRIDE_WIDTH_Start                    0
#define GCREG_DE_SRC_PLANE1_ALIGN_CONFIG_STRIDE_WIDTH_Type                   U17

#define gcregDESrcPlane2AlignConfigRegAddrs                               0x4A70
#define GCREG_DE_SRC_PLANE2_ALIGN_CONFIG_Address                         0x129C0
#define GCREG_DE_SRC_PLANE2_ALIGN_CONFIG_MSB                                  15
#define GCREG_DE_SRC_PLANE2_ALIGN_CONFIG_LSB                                   3
#define GCREG_DE_SRC_PLANE2_ALIGN_CONFIG_BLK                                   0
#define GCREG_DE_SRC_PLANE2_ALIGN_CONFIG_Count                                 8
#define GCREG_DE_SRC_PLANE2_ALIGN_CONFIG_FieldMask                    0x0001FFFF
#define GCREG_DE_SRC_PLANE2_ALIGN_CONFIG_ReadMask                     0x0001FFFF
#define GCREG_DE_SRC_PLANE2_ALIGN_CONFIG_WriteMask                    0x0001FFFF
#define GCREG_DE_SRC_PLANE2_ALIGN_CONFIG_ResetValue                   0x00000000

#define GCREG_DE_SRC_PLANE2_ALIGN_CONFIG_STRIDE_WIDTH                       16:0
#define GCREG_DE_SRC_PLANE2_ALIGN_CONFIG_STRIDE_WIDTH_End                     16
#define GCREG_DE_SRC_PLANE2_ALIGN_CONFIG_STRIDE_WIDTH_Start                    0
#define GCREG_DE_SRC_PLANE2_ALIGN_CONFIG_STRIDE_WIDTH_Type                   U17

#define gcregDESrcAddressExRegAddrs                                       0x4A80
#define GCREG_DE_SRC_ADDRESS_EX_Address                                  0x12A00
#define GCREG_DE_SRC_ADDRESS_EX_MSB                                           15
#define GCREG_DE_SRC_ADDRESS_EX_LSB                                            3
#define GCREG_DE_SRC_ADDRESS_EX_BLK                                            0
#define GCREG_DE_SRC_ADDRESS_EX_Count                                          8
#define GCREG_DE_SRC_ADDRESS_EX_FieldMask                             0xFFFFFFFF
#define GCREG_DE_SRC_ADDRESS_EX_ReadMask                              0xFFFFFFFC
#define GCREG_DE_SRC_ADDRESS_EX_WriteMask                             0xFFFFFFFC
#define GCREG_DE_SRC_ADDRESS_EX_ResetValue                            0x00000000

#define GCREG_DE_SRC_ADDRESS_EX_TYPE                                       31:31
#define GCREG_DE_SRC_ADDRESS_EX_TYPE_End                                      31
#define GCREG_DE_SRC_ADDRESS_EX_TYPE_Start                                    31
#define GCREG_DE_SRC_ADDRESS_EX_TYPE_Type                                    U01
#define   GCREG_DE_SRC_ADDRESS_EX_TYPE_SYSTEM                                0x0
#define   GCREG_DE_SRC_ADDRESS_EX_TYPE_VIRTUAL_SYSTEM                        0x1

#define GCREG_DE_SRC_ADDRESS_EX_ADDRESS                                     30:0
#define GCREG_DE_SRC_ADDRESS_EX_ADDRESS_End                                   30
#define GCREG_DE_SRC_ADDRESS_EX_ADDRESS_Start                                  0
#define GCREG_DE_SRC_ADDRESS_EX_ADDRESS_Type                                 U31

#define gcregDESrcStrideExRegAddrs                                        0x4A88
#define GCREG_DE_SRC_STRIDE_EX_Address                                   0x12A20
#define GCREG_DE_SRC_STRIDE_EX_MSB                                            15
#define GCREG_DE_SRC_STRIDE_EX_LSB                                             3
#define GCREG_DE_SRC_STRIDE_EX_BLK                                             0
#define GCREG_DE_SRC_STRIDE_EX_Count                                           8
#define GCREG_DE_SRC_STRIDE_EX_FieldMask                              0x0003FFFF
#define GCREG_DE_SRC_STRIDE_EX_ReadMask                               0x0003FFFC
#define GCREG_DE_SRC_STRIDE_EX_WriteMask                              0x0003FFFC
#define GCREG_DE_SRC_STRIDE_EX_ResetValue                             0x00000000

#define GCREG_DE_SRC_STRIDE_EX_STRIDE                                       17:0
#define GCREG_DE_SRC_STRIDE_EX_STRIDE_End                                     17
#define GCREG_DE_SRC_STRIDE_EX_STRIDE_Start                                    0
#define GCREG_DE_SRC_STRIDE_EX_STRIDE_Type                                   U18

#define gcregDESrcRotationConfigExRegAddrs                                0x4A90
#define GCREG_DE_SRC_ROTATION_CONFIG_EX_Address                          0x12A40
#define GCREG_DE_SRC_ROTATION_CONFIG_EX_MSB                                   15
#define GCREG_DE_SRC_ROTATION_CONFIG_EX_LSB                                    3
#define GCREG_DE_SRC_ROTATION_CONFIG_EX_BLK                                    0
#define GCREG_DE_SRC_ROTATION_CONFIG_EX_Count                                  8
#define GCREG_DE_SRC_ROTATION_CONFIG_EX_FieldMask                     0x0001FFFF
#define GCREG_DE_SRC_ROTATION_CONFIG_EX_ReadMask                      0x0001FFFF
#define GCREG_DE_SRC_ROTATION_CONFIG_EX_WriteMask                     0x0001FFFF
#define GCREG_DE_SRC_ROTATION_CONFIG_EX_ResetValue                    0x00000000

#define GCREG_DE_SRC_ROTATION_CONFIG_EX_WIDTH                               15:0
#define GCREG_DE_SRC_ROTATION_CONFIG_EX_WIDTH_End                             15
#define GCREG_DE_SRC_ROTATION_CONFIG_EX_WIDTH_Start                            0
#define GCREG_DE_SRC_ROTATION_CONFIG_EX_WIDTH_Type                           U16

#define GCREG_DE_SRC_ROTATION_CONFIG_EX_ROTATION                           16:16
#define GCREG_DE_SRC_ROTATION_CONFIG_EX_ROTATION_End                          16
#define GCREG_DE_SRC_ROTATION_CONFIG_EX_ROTATION_Start                        16
#define GCREG_DE_SRC_ROTATION_CONFIG_EX_ROTATION_Type                        U01
#define   GCREG_DE_SRC_ROTATION_CONFIG_EX_ROTATION_NORMAL                    0x0
#define   GCREG_DE_SRC_ROTATION_CONFIG_EX_ROTATION_ROTATED                   0x1

#define gcregDESrcConfigExRegAddrs                                        0x4A98
#define GCREG_DE_SRC_CONFIG_EX_Address                                   0x12A60
#define GCREG_DE_SRC_CONFIG_EX_MSB                                            15
#define GCREG_DE_SRC_CONFIG_EX_LSB                                             3
#define GCREG_DE_SRC_CONFIG_EX_BLK                                             0
#define GCREG_DE_SRC_CONFIG_EX_Count                                           8
#define GCREG_DE_SRC_CONFIG_EX_FieldMask                              0xFFFEB7CF
#define GCREG_DE_SRC_CONFIG_EX_ReadMask                               0xFFFEB7CF
#define GCREG_DE_SRC_CONFIG_EX_WriteMask                              0xFFFEB7CF
#define GCREG_DE_SRC_CONFIG_EX_ResetValue                             0x00000000

#define GCREG_DE_SRC_CONFIG_EX_ENDIAN_CONTROL                              31:30
#define GCREG_DE_SRC_CONFIG_EX_ENDIAN_CONTROL_End                             31
#define GCREG_DE_SRC_CONFIG_EX_ENDIAN_CONTROL_Start                           30
#define GCREG_DE_SRC_CONFIG_EX_ENDIAN_CONTROL_Type                           U02
#define   GCREG_DE_SRC_CONFIG_EX_ENDIAN_CONTROL_NO_SWAP                      0x0
#define   GCREG_DE_SRC_CONFIG_EX_ENDIAN_CONTROL_SWAP_WORD                    0x1
#define   GCREG_DE_SRC_CONFIG_EX_ENDIAN_CONTROL_SWAP_DWORD                   0x2
#define   GCREG_DE_SRC_CONFIG_EX_ENDIAN_CONTROL_SWAP_DDWORD                  0x3

#define GCREG_DE_SRC_CONFIG_EX_ENDIAN_CONTROL_EX                           29:29
#define GCREG_DE_SRC_CONFIG_EX_ENDIAN_CONTROL_EX_End                          29
#define GCREG_DE_SRC_CONFIG_EX_ENDIAN_CONTROL_EX_Start                        29
#define GCREG_DE_SRC_CONFIG_EX_ENDIAN_CONTROL_EX_Type                        U01
#define   GCREG_DE_SRC_CONFIG_EX_ENDIAN_CONTROL_EX_NO_SWAP                   0x0
#define   GCREG_DE_SRC_CONFIG_EX_ENDIAN_CONTROL_EX_SWAP_BYTE                 0x1

#define GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT                               28:24
#define GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_End                              28
#define GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_Start                            24
#define GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_Type                            U05
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_X4R4G4B4                     0x00
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_A4R4G4B4                     0x01
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_X1R5G5B5                     0x02
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_A1R5G5B5                     0x03
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_R5G6B5                       0x04
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_X8R8G8B8                     0x05
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_A8R8G8B8                     0x06
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_YUY2                         0x07
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_UYVY                         0x08
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_INDEX8                       0x09
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_MONOCHROME                   0x0A
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_AYUV                         0x0B
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_RGB888_PLANAR                0x0C
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_RGB888_PACKED                0x0D
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_EXTENSION                    0x0E
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_YV12                         0x0F
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_A8                           0x10
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_NV12                         0x11
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_NV16                         0x12
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_RG16                         0x13
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_R8                           0x14
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_NV12_10BIT                   0x15
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_A2R10G10B10                  0x16
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_NV16_10BIT                   0x17
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_INDEX1                       0x18
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_INDEX2                       0x19
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_INDEX4                       0x1A
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_P010                         0x1B
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_NV12_10BIT_L1                0x1C
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_NV16_10BIT_L1                0x1D
#define   GCREG_DE_SRC_CONFIG_EX_SOURCE_FORMAT_I010                         0x1E

#define GCREG_DE_SRC_CONFIG_EX_DE_GAMMA                                    23:23
#define GCREG_DE_SRC_CONFIG_EX_DE_GAMMA_End                                   23
#define GCREG_DE_SRC_CONFIG_EX_DE_GAMMA_Start                                 23
#define GCREG_DE_SRC_CONFIG_EX_DE_GAMMA_Type                                 U01
#define   GCREG_DE_SRC_CONFIG_EX_DE_GAMMA_DISABLED                           0x0
#define   GCREG_DE_SRC_CONFIG_EX_DE_GAMMA_ENABLED                            0x1

#define GCREG_DE_SRC_CONFIG_EX_P010_BITS                                   22:22
#define GCREG_DE_SRC_CONFIG_EX_P010_BITS_End                                  22
#define GCREG_DE_SRC_CONFIG_EX_P010_BITS_Start                                22
#define GCREG_DE_SRC_CONFIG_EX_P010_BITS_Type                                U01
#define   GCREG_DE_SRC_CONFIG_EX_P010_BITS_P010_MSB                          0x0
#define   GCREG_DE_SRC_CONFIG_EX_P010_BITS_P010_LSB                          0x1

#define GCREG_DE_SRC_CONFIG_EX_SWIZZLE                                     21:20
#define GCREG_DE_SRC_CONFIG_EX_SWIZZLE_End                                    21
#define GCREG_DE_SRC_CONFIG_EX_SWIZZLE_Start                                  20
#define GCREG_DE_SRC_CONFIG_EX_SWIZZLE_Type                                  U02
#define   GCREG_DE_SRC_CONFIG_EX_SWIZZLE_ARGB                                0x0
#define   GCREG_DE_SRC_CONFIG_EX_SWIZZLE_RGBA                                0x1
#define   GCREG_DE_SRC_CONFIG_EX_SWIZZLE_ABGR                                0x2
#define   GCREG_DE_SRC_CONFIG_EX_SWIZZLE_BGRA                                0x3

#define GCREG_DE_SRC_CONFIG_EX_DEC400_TILE_MODE                            19:17
#define GCREG_DE_SRC_CONFIG_EX_DEC400_TILE_MODE_End                           19
#define GCREG_DE_SRC_CONFIG_EX_DEC400_TILE_MODE_Start                         17
#define GCREG_DE_SRC_CONFIG_EX_DEC400_TILE_MODE_Type                         U03
#define   GCREG_DE_SRC_CONFIG_EX_DEC400_TILE_MODE_TILE8X8_XMAJOR             0x0
#define   GCREG_DE_SRC_CONFIG_EX_DEC400_TILE_MODE_TILE8X4                    0x1
#define   GCREG_DE_SRC_CONFIG_EX_DEC400_TILE_MODE_TILE4X8                    0x2
#define   GCREG_DE_SRC_CONFIG_EX_DEC400_TILE_MODE_TILE64X4                   0x3
#define   GCREG_DE_SRC_CONFIG_EX_DEC400_TILE_MODE_TILE32X4                   0x4
#define   GCREG_DE_SRC_CONFIG_EX_DEC400_TILE_MODE_TILE4X4                    0x5

#define GCREG_DE_SRC_CONFIG_EX_MONO_TRANSPARENCY                           15:15
#define GCREG_DE_SRC_CONFIG_EX_MONO_TRANSPARENCY_End                          15
#define GCREG_DE_SRC_CONFIG_EX_MONO_TRANSPARENCY_Start                        15
#define GCREG_DE_SRC_CONFIG_EX_MONO_TRANSPARENCY_Type                        U01
#define   GCREG_DE_SRC_CONFIG_EX_MONO_TRANSPARENCY_BACKGROUND                0x0
#define   GCREG_DE_SRC_CONFIG_EX_MONO_TRANSPARENCY_FOREGROUND                0x1

#define GCREG_DE_SRC_CONFIG_EX_PACK                                        13:12
#define GCREG_DE_SRC_CONFIG_EX_PACK_End                                       13
#define GCREG_DE_SRC_CONFIG_EX_PACK_Start                                     12
#define GCREG_DE_SRC_CONFIG_EX_PACK_Type                                     U02
#define   GCREG_DE_SRC_CONFIG_EX_PACK_PACKED8                                0x0
#define   GCREG_DE_SRC_CONFIG_EX_PACK_PACKED16                               0x1
#define   GCREG_DE_SRC_CONFIG_EX_PACK_PACKED32                               0x2
#define   GCREG_DE_SRC_CONFIG_EX_PACK_UNPACKED                               0x3

#define GCREG_DE_SRC_CONFIG_EX_TILE_MODE                                    10:9
#define GCREG_DE_SRC_CONFIG_EX_TILE_MODE_End                                  10
#define GCREG_DE_SRC_CONFIG_EX_TILE_MODE_Start                                 9
#define GCREG_DE_SRC_CONFIG_EX_TILE_MODE_Type                                U02
#define   GCREG_DE_SRC_CONFIG_EX_TILE_MODE_TILED4X4                          0x0
#define   GCREG_DE_SRC_CONFIG_EX_TILE_MODE_SUPER_TILED_XMAJOR                0x1
#define   GCREG_DE_SRC_CONFIG_EX_TILE_MODE_SUPER_TILED_YMAJOR                0x2
#define   GCREG_DE_SRC_CONFIG_EX_TILE_MODE_DEC400                            0x3

#define GCREG_DE_SRC_CONFIG_EX_LOCATION                                      8:8
#define GCREG_DE_SRC_CONFIG_EX_LOCATION_End                                    8
#define GCREG_DE_SRC_CONFIG_EX_LOCATION_Start                                  8
#define GCREG_DE_SRC_CONFIG_EX_LOCATION_Type                                 U01
#define   GCREG_DE_SRC_CONFIG_EX_LOCATION_MEMORY                             0x0
#define   GCREG_DE_SRC_CONFIG_EX_LOCATION_STREAM                             0x1

#define GCREG_DE_SRC_CONFIG_EX_TILED                                         7:7
#define GCREG_DE_SRC_CONFIG_EX_TILED_End                                       7
#define GCREG_DE_SRC_CONFIG_EX_TILED_Start                                     7
#define GCREG_DE_SRC_CONFIG_EX_TILED_Type                                    U01
#define   GCREG_DE_SRC_CONFIG_EX_TILED_DISABLED                              0x0
#define   GCREG_DE_SRC_CONFIG_EX_TILED_ENABLED                               0x1

#define GCREG_DE_SRC_CONFIG_EX_SRC_RELATIVE                                  6:6
#define GCREG_DE_SRC_CONFIG_EX_SRC_RELATIVE_End                                6
#define GCREG_DE_SRC_CONFIG_EX_SRC_RELATIVE_Start                              6
#define GCREG_DE_SRC_CONFIG_EX_SRC_RELATIVE_Type                             U01
#define   GCREG_DE_SRC_CONFIG_EX_SRC_RELATIVE_ABSOLUTE                       0x0
#define   GCREG_DE_SRC_CONFIG_EX_SRC_RELATIVE_RELATIVE                       0x1

#define GCREG_DE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT                           3:0
#define GCREG_DE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_End                         3
#define GCREG_DE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_Start                       0
#define GCREG_DE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_Type                      U04
#define   GCREG_DE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_RGB_F16F16F16_PLANAR    0x0
#define   GCREG_DE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_RGB_F16F16F16_PACKED    0x1
#define   GCREG_DE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_RGB_F32F32F32_PLANAR    0x2
#define   GCREG_DE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_RGB_F32F32F32_PACKED    0x3
#define   GCREG_DE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_GRAY_I8                 0x4
#define   GCREG_DE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_GRAY_F16                0x5
#define   GCREG_DE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_GRAY_F32                0x6
#define   GCREG_DE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_INT8_PLANAR             0x7
#define   GCREG_DE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_INT8_PACKED             0x8
#define   GCREG_DE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_INT16_PLANAR            0x9
#define   GCREG_DE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_INT16_PACKED            0xA
#define   GCREG_DE_SRC_CONFIG_EX_EXTRA_SOURCE_FORMAT_GRAY_U8                 0xB

#define gcregDESrcOriginExRegAddrs                                        0x4AA0
#define GCREG_DE_SRC_ORIGIN_EX_Address                                   0x12A80
#define GCREG_DE_SRC_ORIGIN_EX_MSB                                            15
#define GCREG_DE_SRC_ORIGIN_EX_LSB                                             3
#define GCREG_DE_SRC_ORIGIN_EX_BLK                                             0
#define GCREG_DE_SRC_ORIGIN_EX_Count                                           8
#define GCREG_DE_SRC_ORIGIN_EX_FieldMask                              0xFFFFFFFF
#define GCREG_DE_SRC_ORIGIN_EX_ReadMask                               0xFFFFFFFF
#define GCREG_DE_SRC_ORIGIN_EX_WriteMask                              0xFFFFFFFF
#define GCREG_DE_SRC_ORIGIN_EX_ResetValue                             0x00000000

#define GCREG_DE_SRC_ORIGIN_EX_Y                                           31:16
#define GCREG_DE_SRC_ORIGIN_EX_Y_End                                          31
#define GCREG_DE_SRC_ORIGIN_EX_Y_Start                                        16
#define GCREG_DE_SRC_ORIGIN_EX_Y_Type                                        U16

#define GCREG_DE_SRC_ORIGIN_EX_X                                            15:0
#define GCREG_DE_SRC_ORIGIN_EX_X_End                                          15
#define GCREG_DE_SRC_ORIGIN_EX_X_Start                                         0
#define GCREG_DE_SRC_ORIGIN_EX_X_Type                                        U16

#define gcregDESrcSizeExRegAddrs                                          0x4AA8
#define GCREG_DE_SRC_SIZE_EX_Address                                     0x12AA0
#define GCREG_DE_SRC_SIZE_EX_MSB                                              15
#define GCREG_DE_SRC_SIZE_EX_LSB                                               3
#define GCREG_DE_SRC_SIZE_EX_BLK                                               0
#define GCREG_DE_SRC_SIZE_EX_Count                                             8
#define GCREG_DE_SRC_SIZE_EX_FieldMask                                0xFFFFFFFF
#define GCREG_DE_SRC_SIZE_EX_ReadMask                                 0xFFFFFFFF
#define GCREG_DE_SRC_SIZE_EX_WriteMask                                0xFFFFFFFF
#define GCREG_DE_SRC_SIZE_EX_ResetValue                               0x00000000

#define GCREG_DE_SRC_SIZE_EX_Y                                             31:16
#define GCREG_DE_SRC_SIZE_EX_Y_End                                            31
#define GCREG_DE_SRC_SIZE_EX_Y_Start                                          16
#define GCREG_DE_SRC_SIZE_EX_Y_Type                                          U16

#define GCREG_DE_SRC_SIZE_EX_X                                              15:0
#define GCREG_DE_SRC_SIZE_EX_X_End                                            15
#define GCREG_DE_SRC_SIZE_EX_X_Start                                           0
#define GCREG_DE_SRC_SIZE_EX_X_Type                                          U16

#define gcregDESrcColorBgExRegAddrs                                       0x4AB0
#define GCREG_DE_SRC_COLOR_BG_EX_Address                                 0x12AC0
#define GCREG_DE_SRC_COLOR_BG_EX_MSB                                          15
#define GCREG_DE_SRC_COLOR_BG_EX_LSB                                           3
#define GCREG_DE_SRC_COLOR_BG_EX_BLK                                           0
#define GCREG_DE_SRC_COLOR_BG_EX_Count                                         8
#define GCREG_DE_SRC_COLOR_BG_EX_FieldMask                            0xFFFFFFFF
#define GCREG_DE_SRC_COLOR_BG_EX_ReadMask                             0xFFFFFFFF
#define GCREG_DE_SRC_COLOR_BG_EX_WriteMask                            0xFFFFFFFF
#define GCREG_DE_SRC_COLOR_BG_EX_ResetValue                           0x00000000

#define GCREG_DE_SRC_COLOR_BG_EX_ALPHA                                     31:24
#define GCREG_DE_SRC_COLOR_BG_EX_ALPHA_End                                    31
#define GCREG_DE_SRC_COLOR_BG_EX_ALPHA_Start                                  24
#define GCREG_DE_SRC_COLOR_BG_EX_ALPHA_Type                                  U08

#define GCREG_DE_SRC_COLOR_BG_EX_RED                                       23:16
#define GCREG_DE_SRC_COLOR_BG_EX_RED_End                                      23
#define GCREG_DE_SRC_COLOR_BG_EX_RED_Start                                    16
#define GCREG_DE_SRC_COLOR_BG_EX_RED_Type                                    U08

#define GCREG_DE_SRC_COLOR_BG_EX_GREEN                                      15:8
#define GCREG_DE_SRC_COLOR_BG_EX_GREEN_End                                    15
#define GCREG_DE_SRC_COLOR_BG_EX_GREEN_Start                                   8
#define GCREG_DE_SRC_COLOR_BG_EX_GREEN_Type                                  U08

#define GCREG_DE_SRC_COLOR_BG_EX_BLUE                                        7:0
#define GCREG_DE_SRC_COLOR_BG_EX_BLUE_End                                      7
#define GCREG_DE_SRC_COLOR_BG_EX_BLUE_Start                                    0
#define GCREG_DE_SRC_COLOR_BG_EX_BLUE_Type                                   U08

#define gcregDERopExRegAddrs                                              0x4AB8
#define GCREG_DE_ROP_EX_Address                                          0x12AE0
#define GCREG_DE_ROP_EX_MSB                                                   15
#define GCREG_DE_ROP_EX_LSB                                                    3
#define GCREG_DE_ROP_EX_BLK                                                    0
#define GCREG_DE_ROP_EX_Count                                                  8
#define GCREG_DE_ROP_EX_FieldMask                                     0x0030FFFF
#define GCREG_DE_ROP_EX_ReadMask                                      0x0030FFFF
#define GCREG_DE_ROP_EX_WriteMask                                     0x0030FFFF
#define GCREG_DE_ROP_EX_ResetValue                                    0x00000000

#define GCREG_DE_ROP_EX_TYPE                                               21:20
#define GCREG_DE_ROP_EX_TYPE_End                                              21
#define GCREG_DE_ROP_EX_TYPE_Start                                            20
#define GCREG_DE_ROP_EX_TYPE_Type                                            U02
#define   GCREG_DE_ROP_EX_TYPE_ROP2_PATTERN                                  0x0
#define   GCREG_DE_ROP_EX_TYPE_ROP2_SOURCE                                   0x1
#define   GCREG_DE_ROP_EX_TYPE_ROP3                                          0x2
#define   GCREG_DE_ROP_EX_TYPE_ROP4                                          0x3

#define GCREG_DE_ROP_EX_ROP_BG                                              15:8
#define GCREG_DE_ROP_EX_ROP_BG_End                                            15
#define GCREG_DE_ROP_EX_ROP_BG_Start                                           8
#define GCREG_DE_ROP_EX_ROP_BG_Type                                          U08

#define GCREG_DE_ROP_EX_ROP_FG                                               7:0
#define GCREG_DE_ROP_EX_ROP_FG_End                                             7
#define GCREG_DE_ROP_EX_ROP_FG_Start                                           0
#define GCREG_DE_ROP_EX_ROP_FG_Type                                          U08

#define gcregDEAlphaControlExRegAddrs                                     0x4AC0
#define GCREG_DE_ALPHA_CONTROL_EX_Address                                0x12B00
#define GCREG_DE_ALPHA_CONTROL_EX_MSB                                         15
#define GCREG_DE_ALPHA_CONTROL_EX_LSB                                          3
#define GCREG_DE_ALPHA_CONTROL_EX_BLK                                          0
#define GCREG_DE_ALPHA_CONTROL_EX_Count                                        8
#define GCREG_DE_ALPHA_CONTROL_EX_FieldMask                           0x00000001
#define GCREG_DE_ALPHA_CONTROL_EX_ReadMask                            0x00000001
#define GCREG_DE_ALPHA_CONTROL_EX_WriteMask                           0x00000001
#define GCREG_DE_ALPHA_CONTROL_EX_ResetValue                          0x00000000

#define GCREG_DE_ALPHA_CONTROL_EX_ENABLE                                     0:0
#define GCREG_DE_ALPHA_CONTROL_EX_ENABLE_End                                   0
#define GCREG_DE_ALPHA_CONTROL_EX_ENABLE_Start                                 0
#define GCREG_DE_ALPHA_CONTROL_EX_ENABLE_Type                                U01
#define   GCREG_DE_ALPHA_CONTROL_EX_ENABLE_OFF                               0x0
#define   GCREG_DE_ALPHA_CONTROL_EX_ENABLE_ON                                0x1

#define gcregDEAlphaModesExRegAddrs                                       0x4AC8
#define GCREG_DE_ALPHA_MODES_EX_Address                                  0x12B20
#define GCREG_DE_ALPHA_MODES_EX_MSB                                           15
#define GCREG_DE_ALPHA_MODES_EX_LSB                                            3
#define GCREG_DE_ALPHA_MODES_EX_BLK                                            0
#define GCREG_DE_ALPHA_MODES_EX_Count                                          8
#define GCREG_DE_ALPHA_MODES_EX_FieldMask                             0xFF003311
#define GCREG_DE_ALPHA_MODES_EX_ReadMask                              0xFF003311
#define GCREG_DE_ALPHA_MODES_EX_WriteMask                             0xFF003311
#define GCREG_DE_ALPHA_MODES_EX_ResetValue                            0x00000000

#define GCREG_DE_ALPHA_MODES_EX_SRC_ALPHA_MODE                               0:0
#define GCREG_DE_ALPHA_MODES_EX_SRC_ALPHA_MODE_End                             0
#define GCREG_DE_ALPHA_MODES_EX_SRC_ALPHA_MODE_Start                           0
#define GCREG_DE_ALPHA_MODES_EX_SRC_ALPHA_MODE_Type                          U01
#define   GCREG_DE_ALPHA_MODES_EX_SRC_ALPHA_MODE_NORMAL                      0x0
#define   GCREG_DE_ALPHA_MODES_EX_SRC_ALPHA_MODE_INVERSED                    0x1

#define GCREG_DE_ALPHA_MODES_EX_DST_ALPHA_MODE                               4:4
#define GCREG_DE_ALPHA_MODES_EX_DST_ALPHA_MODE_End                             4
#define GCREG_DE_ALPHA_MODES_EX_DST_ALPHA_MODE_Start                           4
#define GCREG_DE_ALPHA_MODES_EX_DST_ALPHA_MODE_Type                          U01
#define   GCREG_DE_ALPHA_MODES_EX_DST_ALPHA_MODE_NORMAL                      0x0
#define   GCREG_DE_ALPHA_MODES_EX_DST_ALPHA_MODE_INVERSED                    0x1

#define GCREG_DE_ALPHA_MODES_EX_GLOBAL_SRC_ALPHA_MODE                        9:8
#define GCREG_DE_ALPHA_MODES_EX_GLOBAL_SRC_ALPHA_MODE_End                      9
#define GCREG_DE_ALPHA_MODES_EX_GLOBAL_SRC_ALPHA_MODE_Start                    8
#define GCREG_DE_ALPHA_MODES_EX_GLOBAL_SRC_ALPHA_MODE_Type                   U02
#define   GCREG_DE_ALPHA_MODES_EX_GLOBAL_SRC_ALPHA_MODE_NORMAL               0x0
#define   GCREG_DE_ALPHA_MODES_EX_GLOBAL_SRC_ALPHA_MODE_GLOBAL               0x1
#define   GCREG_DE_ALPHA_MODES_EX_GLOBAL_SRC_ALPHA_MODE_SCALED               0x2

#define GCREG_DE_ALPHA_MODES_EX_GLOBAL_DST_ALPHA_MODE                      13:12
#define GCREG_DE_ALPHA_MODES_EX_GLOBAL_DST_ALPHA_MODE_End                     13
#define GCREG_DE_ALPHA_MODES_EX_GLOBAL_DST_ALPHA_MODE_Start                   12
#define GCREG_DE_ALPHA_MODES_EX_GLOBAL_DST_ALPHA_MODE_Type                   U02
#define   GCREG_DE_ALPHA_MODES_EX_GLOBAL_DST_ALPHA_MODE_NORMAL               0x0
#define   GCREG_DE_ALPHA_MODES_EX_GLOBAL_DST_ALPHA_MODE_GLOBAL               0x1
#define   GCREG_DE_ALPHA_MODES_EX_GLOBAL_DST_ALPHA_MODE_SCALED               0x2

#define GCREG_DE_ALPHA_MODES_EX_SRC_BLENDING_MODE                          26:24
#define GCREG_DE_ALPHA_MODES_EX_SRC_BLENDING_MODE_End                         26
#define GCREG_DE_ALPHA_MODES_EX_SRC_BLENDING_MODE_Start                       24
#define GCREG_DE_ALPHA_MODES_EX_SRC_BLENDING_MODE_Type                       U03
#define   GCREG_DE_ALPHA_MODES_EX_SRC_BLENDING_MODE_ZERO                     0x0
#define   GCREG_DE_ALPHA_MODES_EX_SRC_BLENDING_MODE_ONE                      0x1
#define   GCREG_DE_ALPHA_MODES_EX_SRC_BLENDING_MODE_NORMAL                   0x2
#define   GCREG_DE_ALPHA_MODES_EX_SRC_BLENDING_MODE_INVERSED                 0x3
#define   GCREG_DE_ALPHA_MODES_EX_SRC_BLENDING_MODE_COLOR                    0x4
#define   GCREG_DE_ALPHA_MODES_EX_SRC_BLENDING_MODE_COLOR_INVERSED           0x5
#define   GCREG_DE_ALPHA_MODES_EX_SRC_BLENDING_MODE_SATURATED_ALPHA          0x6
#define   GCREG_DE_ALPHA_MODES_EX_SRC_BLENDING_MODE_SATURATED_DEST_ALPHA     0x7

#define GCREG_DE_ALPHA_MODES_EX_SRC_ALPHA_FACTOR                           27:27
#define GCREG_DE_ALPHA_MODES_EX_SRC_ALPHA_FACTOR_End                          27
#define GCREG_DE_ALPHA_MODES_EX_SRC_ALPHA_FACTOR_Start                        27
#define GCREG_DE_ALPHA_MODES_EX_SRC_ALPHA_FACTOR_Type                        U01
#define   GCREG_DE_ALPHA_MODES_EX_SRC_ALPHA_FACTOR_DISABLED                  0x0
#define   GCREG_DE_ALPHA_MODES_EX_SRC_ALPHA_FACTOR_ENABLED                   0x1

#define GCREG_DE_ALPHA_MODES_EX_DST_BLENDING_MODE                          30:28
#define GCREG_DE_ALPHA_MODES_EX_DST_BLENDING_MODE_End                         30
#define GCREG_DE_ALPHA_MODES_EX_DST_BLENDING_MODE_Start                       28
#define GCREG_DE_ALPHA_MODES_EX_DST_BLENDING_MODE_Type                       U03
#define   GCREG_DE_ALPHA_MODES_EX_DST_BLENDING_MODE_ZERO                     0x0
#define   GCREG_DE_ALPHA_MODES_EX_DST_BLENDING_MODE_ONE                      0x1
#define   GCREG_DE_ALPHA_MODES_EX_DST_BLENDING_MODE_NORMAL                   0x2
#define   GCREG_DE_ALPHA_MODES_EX_DST_BLENDING_MODE_INVERSED                 0x3
#define   GCREG_DE_ALPHA_MODES_EX_DST_BLENDING_MODE_COLOR                    0x4
#define   GCREG_DE_ALPHA_MODES_EX_DST_BLENDING_MODE_COLOR_INVERSED           0x5
#define   GCREG_DE_ALPHA_MODES_EX_DST_BLENDING_MODE_SATURATED_ALPHA          0x6
#define   GCREG_DE_ALPHA_MODES_EX_DST_BLENDING_MODE_SATURATED_DEST_ALPHA     0x7

#define GCREG_DE_ALPHA_MODES_EX_DST_ALPHA_FACTOR                           31:31
#define GCREG_DE_ALPHA_MODES_EX_DST_ALPHA_FACTOR_End                          31
#define GCREG_DE_ALPHA_MODES_EX_DST_ALPHA_FACTOR_Start                        31
#define GCREG_DE_ALPHA_MODES_EX_DST_ALPHA_FACTOR_Type                        U01
#define   GCREG_DE_ALPHA_MODES_EX_DST_ALPHA_FACTOR_DISABLED                  0x0
#define   GCREG_DE_ALPHA_MODES_EX_DST_ALPHA_FACTOR_ENABLED                   0x1

#define gcregDEAddressUExRegAddrs                                         0x4AD0
#define GCREG_DE_ADDRESS_UEX_Address                                     0x12B40
#define GCREG_DE_ADDRESS_UEX_MSB                                              15
#define GCREG_DE_ADDRESS_UEX_LSB                                               3
#define GCREG_DE_ADDRESS_UEX_BLK                                               0
#define GCREG_DE_ADDRESS_UEX_Count                                             8
#define GCREG_DE_ADDRESS_UEX_FieldMask                                0xFFFFFFFF
#define GCREG_DE_ADDRESS_UEX_ReadMask                                 0xFFFFFFFC
#define GCREG_DE_ADDRESS_UEX_WriteMask                                0xFFFFFFFC
#define GCREG_DE_ADDRESS_UEX_ResetValue                               0x00000000

#define GCREG_DE_ADDRESS_UEX_TYPE                                          31:31
#define GCREG_DE_ADDRESS_UEX_TYPE_End                                         31
#define GCREG_DE_ADDRESS_UEX_TYPE_Start                                       31
#define GCREG_DE_ADDRESS_UEX_TYPE_Type                                       U01
#define   GCREG_DE_ADDRESS_UEX_TYPE_SYSTEM                                   0x0
#define   GCREG_DE_ADDRESS_UEX_TYPE_VIRTUAL_SYSTEM                           0x1

#define GCREG_DE_ADDRESS_UEX_ADDRESS                                        30:0
#define GCREG_DE_ADDRESS_UEX_ADDRESS_End                                      30
#define GCREG_DE_ADDRESS_UEX_ADDRESS_Start                                     0
#define GCREG_DE_ADDRESS_UEX_ADDRESS_Type                                    U31

#define gcregDEStrideUExRegAddrs                                          0x4AD8
#define GCREG_DE_STRIDE_UEX_Address                                      0x12B60
#define GCREG_DE_STRIDE_UEX_MSB                                               15
#define GCREG_DE_STRIDE_UEX_LSB                                                3
#define GCREG_DE_STRIDE_UEX_BLK                                                0
#define GCREG_DE_STRIDE_UEX_Count                                              8
#define GCREG_DE_STRIDE_UEX_FieldMask                                 0x0003FFFF
#define GCREG_DE_STRIDE_UEX_ReadMask                                  0x0003FFFC
#define GCREG_DE_STRIDE_UEX_WriteMask                                 0x0003FFFC
#define GCREG_DE_STRIDE_UEX_ResetValue                                0x00000000

#define GCREG_DE_STRIDE_UEX_STRIDE                                          17:0
#define GCREG_DE_STRIDE_UEX_STRIDE_End                                        17
#define GCREG_DE_STRIDE_UEX_STRIDE_Start                                       0
#define GCREG_DE_STRIDE_UEX_STRIDE_Type                                      U18

#define gcregDEAddressVExRegAddrs                                         0x4AE0
#define GCREG_DE_ADDRESS_VEX_Address                                     0x12B80
#define GCREG_DE_ADDRESS_VEX_MSB                                              15
#define GCREG_DE_ADDRESS_VEX_LSB                                               3
#define GCREG_DE_ADDRESS_VEX_BLK                                               0
#define GCREG_DE_ADDRESS_VEX_Count                                             8
#define GCREG_DE_ADDRESS_VEX_FieldMask                                0xFFFFFFFF
#define GCREG_DE_ADDRESS_VEX_ReadMask                                 0xFFFFFFFC
#define GCREG_DE_ADDRESS_VEX_WriteMask                                0xFFFFFFFC
#define GCREG_DE_ADDRESS_VEX_ResetValue                               0x00000000

#define GCREG_DE_ADDRESS_VEX_TYPE                                          31:31
#define GCREG_DE_ADDRESS_VEX_TYPE_End                                         31
#define GCREG_DE_ADDRESS_VEX_TYPE_Start                                       31
#define GCREG_DE_ADDRESS_VEX_TYPE_Type                                       U01
#define   GCREG_DE_ADDRESS_VEX_TYPE_SYSTEM                                   0x0
#define   GCREG_DE_ADDRESS_VEX_TYPE_VIRTUAL_SYSTEM                           0x1

#define GCREG_DE_ADDRESS_VEX_ADDRESS                                        30:0
#define GCREG_DE_ADDRESS_VEX_ADDRESS_End                                      30
#define GCREG_DE_ADDRESS_VEX_ADDRESS_Start                                     0
#define GCREG_DE_ADDRESS_VEX_ADDRESS_Type                                    U31

#define gcregDEStrideVExRegAddrs                                          0x4AE8
#define GCREG_DE_STRIDE_VEX_Address                                      0x12BA0
#define GCREG_DE_STRIDE_VEX_MSB                                               15
#define GCREG_DE_STRIDE_VEX_LSB                                                3
#define GCREG_DE_STRIDE_VEX_BLK                                                0
#define GCREG_DE_STRIDE_VEX_Count                                              8
#define GCREG_DE_STRIDE_VEX_FieldMask                                 0x0003FFFF
#define GCREG_DE_STRIDE_VEX_ReadMask                                  0x0003FFFC
#define GCREG_DE_STRIDE_VEX_WriteMask                                 0x0003FFFC
#define GCREG_DE_STRIDE_VEX_ResetValue                                0x00000000

#define GCREG_DE_STRIDE_VEX_STRIDE                                          17:0
#define GCREG_DE_STRIDE_VEX_STRIDE_End                                        17
#define GCREG_DE_STRIDE_VEX_STRIDE_Start                                       0
#define GCREG_DE_STRIDE_VEX_STRIDE_Type                                      U18

#define gcregDESrcRotationHeightExRegAddrs                                0x4AF0
#define GCREG_DE_SRC_ROTATION_HEIGHT_EX_Address                          0x12BC0
#define GCREG_DE_SRC_ROTATION_HEIGHT_EX_MSB                                   15
#define GCREG_DE_SRC_ROTATION_HEIGHT_EX_LSB                                    3
#define GCREG_DE_SRC_ROTATION_HEIGHT_EX_BLK                                    0
#define GCREG_DE_SRC_ROTATION_HEIGHT_EX_Count                                  8
#define GCREG_DE_SRC_ROTATION_HEIGHT_EX_FieldMask                     0x0000FFFF
#define GCREG_DE_SRC_ROTATION_HEIGHT_EX_ReadMask                      0x0000FFFF
#define GCREG_DE_SRC_ROTATION_HEIGHT_EX_WriteMask                     0x0000FFFF
#define GCREG_DE_SRC_ROTATION_HEIGHT_EX_ResetValue                    0x00000000

#define GCREG_DE_SRC_ROTATION_HEIGHT_EX_HEIGHT                              15:0
#define GCREG_DE_SRC_ROTATION_HEIGHT_EX_HEIGHT_End                            15
#define GCREG_DE_SRC_ROTATION_HEIGHT_EX_HEIGHT_Start                           0
#define GCREG_DE_SRC_ROTATION_HEIGHT_EX_HEIGHT_Type                          U16

#define gcregDERotAngleExRegAddrs                                         0x4AF8
#define GCREG_DE_ROT_ANGLE_EX_Address                                    0x12BE0
#define GCREG_DE_ROT_ANGLE_EX_MSB                                             15
#define GCREG_DE_ROT_ANGLE_EX_LSB                                              3
#define GCREG_DE_ROT_ANGLE_EX_BLK                                              0
#define GCREG_DE_ROT_ANGLE_EX_Count                                            8
#define GCREG_DE_ROT_ANGLE_EX_FieldMask                               0x03FBB33F
#define GCREG_DE_ROT_ANGLE_EX_ReadMask                                0x03FBB33F
#define GCREG_DE_ROT_ANGLE_EX_WriteMask                               0x03FBB33F
#define GCREG_DE_ROT_ANGLE_EX_ResetValue                              0x00000000

#define GCREG_DE_ROT_ANGLE_EX_SRC                                            2:0
#define GCREG_DE_ROT_ANGLE_EX_SRC_End                                          2
#define GCREG_DE_ROT_ANGLE_EX_SRC_Start                                        0
#define GCREG_DE_ROT_ANGLE_EX_SRC_Type                                       U03
#define   GCREG_DE_ROT_ANGLE_EX_SRC_ROT0                                     0x0
#define   GCREG_DE_ROT_ANGLE_EX_SRC_FLIP_X                                   0x1
#define   GCREG_DE_ROT_ANGLE_EX_SRC_FLIP_Y                                   0x2
#define   GCREG_DE_ROT_ANGLE_EX_SRC_ROT90                                    0x4
#define   GCREG_DE_ROT_ANGLE_EX_SRC_ROT180                                   0x5
#define   GCREG_DE_ROT_ANGLE_EX_SRC_ROT270                                   0x6

#define GCREG_DE_ROT_ANGLE_EX_DST                                            5:3
#define GCREG_DE_ROT_ANGLE_EX_DST_End                                          5
#define GCREG_DE_ROT_ANGLE_EX_DST_Start                                        3
#define GCREG_DE_ROT_ANGLE_EX_DST_Type                                       U03
#define   GCREG_DE_ROT_ANGLE_EX_DST_ROT0                                     0x0
#define   GCREG_DE_ROT_ANGLE_EX_DST_FLIP_X                                   0x1
#define   GCREG_DE_ROT_ANGLE_EX_DST_FLIP_Y                                   0x2
#define   GCREG_DE_ROT_ANGLE_EX_DST_ROT90                                    0x4
#define   GCREG_DE_ROT_ANGLE_EX_DST_ROT180                                   0x5
#define   GCREG_DE_ROT_ANGLE_EX_DST_ROT270                                   0x6

#define GCREG_DE_ROT_ANGLE_EX_MASK_SRC                                       8:8
#define GCREG_DE_ROT_ANGLE_EX_MASK_SRC_End                                     8
#define GCREG_DE_ROT_ANGLE_EX_MASK_SRC_Start                                   8
#define GCREG_DE_ROT_ANGLE_EX_MASK_SRC_Type                                  U01
#define   GCREG_DE_ROT_ANGLE_EX_MASK_SRC_ENABLED                             0x0
#define   GCREG_DE_ROT_ANGLE_EX_MASK_SRC_MASKED                              0x1

#define GCREG_DE_ROT_ANGLE_EX_MASK_DST                                       9:9
#define GCREG_DE_ROT_ANGLE_EX_MASK_DST_End                                     9
#define GCREG_DE_ROT_ANGLE_EX_MASK_DST_Start                                   9
#define GCREG_DE_ROT_ANGLE_EX_MASK_DST_Type                                  U01
#define   GCREG_DE_ROT_ANGLE_EX_MASK_DST_ENABLED                             0x0
#define   GCREG_DE_ROT_ANGLE_EX_MASK_DST_MASKED                              0x1

#define GCREG_DE_ROT_ANGLE_EX_SRC_MIRROR                                   13:12
#define GCREG_DE_ROT_ANGLE_EX_SRC_MIRROR_End                                  13
#define GCREG_DE_ROT_ANGLE_EX_SRC_MIRROR_Start                                12
#define GCREG_DE_ROT_ANGLE_EX_SRC_MIRROR_Type                                U02
#define   GCREG_DE_ROT_ANGLE_EX_SRC_MIRROR_NONE                              0x0
#define   GCREG_DE_ROT_ANGLE_EX_SRC_MIRROR_MIRROR_X                          0x1
#define   GCREG_DE_ROT_ANGLE_EX_SRC_MIRROR_MIRROR_Y                          0x2
#define   GCREG_DE_ROT_ANGLE_EX_SRC_MIRROR_MIRROR_XY                         0x3

#define GCREG_DE_ROT_ANGLE_EX_MASK_SRC_MIRROR                              15:15
#define GCREG_DE_ROT_ANGLE_EX_MASK_SRC_MIRROR_End                             15
#define GCREG_DE_ROT_ANGLE_EX_MASK_SRC_MIRROR_Start                           15
#define GCREG_DE_ROT_ANGLE_EX_MASK_SRC_MIRROR_Type                           U01
#define   GCREG_DE_ROT_ANGLE_EX_MASK_SRC_MIRROR_ENABLED                      0x0
#define   GCREG_DE_ROT_ANGLE_EX_MASK_SRC_MIRROR_MASKED                       0x1

#define GCREG_DE_ROT_ANGLE_EX_DST_MIRROR                                   17:16
#define GCREG_DE_ROT_ANGLE_EX_DST_MIRROR_End                                  17
#define GCREG_DE_ROT_ANGLE_EX_DST_MIRROR_Start                                16
#define GCREG_DE_ROT_ANGLE_EX_DST_MIRROR_Type                                U02
#define   GCREG_DE_ROT_ANGLE_EX_DST_MIRROR_NONE                              0x0
#define   GCREG_DE_ROT_ANGLE_EX_DST_MIRROR_MIRROR_X                          0x1
#define   GCREG_DE_ROT_ANGLE_EX_DST_MIRROR_MIRROR_Y                          0x2
#define   GCREG_DE_ROT_ANGLE_EX_DST_MIRROR_MIRROR_XY                         0x3

#define GCREG_DE_ROT_ANGLE_EX_MASK_DST_MIRROR                              19:19
#define GCREG_DE_ROT_ANGLE_EX_MASK_DST_MIRROR_End                             19
#define GCREG_DE_ROT_ANGLE_EX_MASK_DST_MIRROR_Start                           19
#define GCREG_DE_ROT_ANGLE_EX_MASK_DST_MIRROR_Type                           U01
#define   GCREG_DE_ROT_ANGLE_EX_MASK_DST_MIRROR_ENABLED                      0x0
#define   GCREG_DE_ROT_ANGLE_EX_MASK_DST_MIRROR_MASKED                       0x1

#define GCREG_DE_ROT_ANGLE_EX_SRC_FLIP                                     21:20
#define GCREG_DE_ROT_ANGLE_EX_SRC_FLIP_End                                    21
#define GCREG_DE_ROT_ANGLE_EX_SRC_FLIP_Start                                  20
#define GCREG_DE_ROT_ANGLE_EX_SRC_FLIP_Type                                  U02
#define   GCREG_DE_ROT_ANGLE_EX_SRC_FLIP_NONE                                0x0
#define   GCREG_DE_ROT_ANGLE_EX_SRC_FLIP_FLIP_X                              0x1
#define   GCREG_DE_ROT_ANGLE_EX_SRC_FLIP_FLIP_Y                              0x2
#define   GCREG_DE_ROT_ANGLE_EX_SRC_FLIP_FLIP_XY                             0x3

#define GCREG_DE_ROT_ANGLE_EX_MASK_SRC_FLIP                                22:22
#define GCREG_DE_ROT_ANGLE_EX_MASK_SRC_FLIP_End                               22
#define GCREG_DE_ROT_ANGLE_EX_MASK_SRC_FLIP_Start                             22
#define GCREG_DE_ROT_ANGLE_EX_MASK_SRC_FLIP_Type                             U01
#define   GCREG_DE_ROT_ANGLE_EX_MASK_SRC_FLIP_ENABLED                        0x0
#define   GCREG_DE_ROT_ANGLE_EX_MASK_SRC_FLIP_MASKED                         0x1

#define GCREG_DE_ROT_ANGLE_EX_DST_FLIP                                     24:23
#define GCREG_DE_ROT_ANGLE_EX_DST_FLIP_End                                    24
#define GCREG_DE_ROT_ANGLE_EX_DST_FLIP_Start                                  23
#define GCREG_DE_ROT_ANGLE_EX_DST_FLIP_Type                                  U02
#define   GCREG_DE_ROT_ANGLE_EX_DST_FLIP_NONE                                0x0
#define   GCREG_DE_ROT_ANGLE_EX_DST_FLIP_FLIP_X                              0x1
#define   GCREG_DE_ROT_ANGLE_EX_DST_FLIP_FLIP_Y                              0x2
#define   GCREG_DE_ROT_ANGLE_EX_DST_FLIP_FLIP_XY                             0x3

#define GCREG_DE_ROT_ANGLE_EX_MASK_DST_FLIP                                25:25
#define GCREG_DE_ROT_ANGLE_EX_MASK_DST_FLIP_End                               25
#define GCREG_DE_ROT_ANGLE_EX_MASK_DST_FLIP_Start                             25
#define GCREG_DE_ROT_ANGLE_EX_MASK_DST_FLIP_Type                             U01
#define   GCREG_DE_ROT_ANGLE_EX_MASK_DST_FLIP_ENABLED                        0x0
#define   GCREG_DE_ROT_ANGLE_EX_MASK_DST_FLIP_MASKED                         0x1

#define gcregDEGlobalSrcColorExRegAddrs                                   0x4B00
#define GCREG_DE_GLOBAL_SRC_COLOR_EX_Address                             0x12C00
#define GCREG_DE_GLOBAL_SRC_COLOR_EX_MSB                                      15
#define GCREG_DE_GLOBAL_SRC_COLOR_EX_LSB                                       3
#define GCREG_DE_GLOBAL_SRC_COLOR_EX_BLK                                       0
#define GCREG_DE_GLOBAL_SRC_COLOR_EX_Count                                     8
#define GCREG_DE_GLOBAL_SRC_COLOR_EX_FieldMask                        0xFFFFFFFF
#define GCREG_DE_GLOBAL_SRC_COLOR_EX_ReadMask                         0xFFFFFFFF
#define GCREG_DE_GLOBAL_SRC_COLOR_EX_WriteMask                        0xFFFFFFFF
#define GCREG_DE_GLOBAL_SRC_COLOR_EX_ResetValue                       0x00000000

#define GCREG_DE_GLOBAL_SRC_COLOR_EX_ALPHA                                 31:24
#define GCREG_DE_GLOBAL_SRC_COLOR_EX_ALPHA_End                                31
#define GCREG_DE_GLOBAL_SRC_COLOR_EX_ALPHA_Start                              24
#define GCREG_DE_GLOBAL_SRC_COLOR_EX_ALPHA_Type                              U08

#define GCREG_DE_GLOBAL_SRC_COLOR_EX_RED                                   23:16
#define GCREG_DE_GLOBAL_SRC_COLOR_EX_RED_End                                  23
#define GCREG_DE_GLOBAL_SRC_COLOR_EX_RED_Start                                16
#define GCREG_DE_GLOBAL_SRC_COLOR_EX_RED_Type                                U08

#define GCREG_DE_GLOBAL_SRC_COLOR_EX_GREEN                                  15:8
#define GCREG_DE_GLOBAL_SRC_COLOR_EX_GREEN_End                                15
#define GCREG_DE_GLOBAL_SRC_COLOR_EX_GREEN_Start                               8
#define GCREG_DE_GLOBAL_SRC_COLOR_EX_GREEN_Type                              U08

#define GCREG_DE_GLOBAL_SRC_COLOR_EX_BLUE                                    7:0
#define GCREG_DE_GLOBAL_SRC_COLOR_EX_BLUE_End                                  7
#define GCREG_DE_GLOBAL_SRC_COLOR_EX_BLUE_Start                                0
#define GCREG_DE_GLOBAL_SRC_COLOR_EX_BLUE_Type                               U08

#define gcregDEGlobalDestColorExRegAddrs                                  0x4B08
#define GCREG_DE_GLOBAL_DEST_COLOR_EX_Address                            0x12C20
#define GCREG_DE_GLOBAL_DEST_COLOR_EX_MSB                                     15
#define GCREG_DE_GLOBAL_DEST_COLOR_EX_LSB                                      3
#define GCREG_DE_GLOBAL_DEST_COLOR_EX_BLK                                      0
#define GCREG_DE_GLOBAL_DEST_COLOR_EX_Count                                    8
#define GCREG_DE_GLOBAL_DEST_COLOR_EX_FieldMask                       0xFFFFFFFF
#define GCREG_DE_GLOBAL_DEST_COLOR_EX_ReadMask                        0xFFFFFFFF
#define GCREG_DE_GLOBAL_DEST_COLOR_EX_WriteMask                       0xFFFFFFFF
#define GCREG_DE_GLOBAL_DEST_COLOR_EX_ResetValue                      0x00000000

#define GCREG_DE_GLOBAL_DEST_COLOR_EX_ALPHA                                31:24
#define GCREG_DE_GLOBAL_DEST_COLOR_EX_ALPHA_End                               31
#define GCREG_DE_GLOBAL_DEST_COLOR_EX_ALPHA_Start                             24
#define GCREG_DE_GLOBAL_DEST_COLOR_EX_ALPHA_Type                             U08

#define GCREG_DE_GLOBAL_DEST_COLOR_EX_RED                                  23:16
#define GCREG_DE_GLOBAL_DEST_COLOR_EX_RED_End                                 23
#define GCREG_DE_GLOBAL_DEST_COLOR_EX_RED_Start                               16
#define GCREG_DE_GLOBAL_DEST_COLOR_EX_RED_Type                               U08

#define GCREG_DE_GLOBAL_DEST_COLOR_EX_GREEN                                 15:8
#define GCREG_DE_GLOBAL_DEST_COLOR_EX_GREEN_End                               15
#define GCREG_DE_GLOBAL_DEST_COLOR_EX_GREEN_Start                              8
#define GCREG_DE_GLOBAL_DEST_COLOR_EX_GREEN_Type                             U08

#define GCREG_DE_GLOBAL_DEST_COLOR_EX_BLUE                                   7:0
#define GCREG_DE_GLOBAL_DEST_COLOR_EX_BLUE_End                                 7
#define GCREG_DE_GLOBAL_DEST_COLOR_EX_BLUE_Start                               0
#define GCREG_DE_GLOBAL_DEST_COLOR_EX_BLUE_Type                              U08

#define gcregDEColorMultiplyModesExRegAddrs                               0x4B10
#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_Address                         0x12C40
#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_MSB                                  15
#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_LSB                                   3
#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_BLK                                   0
#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_Count                                 8
#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_FieldMask                    0x00300311
#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_ReadMask                     0x00300311
#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_WriteMask                    0x00300311
#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_ResetValue                   0x00000000

#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_SRC_PREMULTIPLY                     0:0
#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_SRC_PREMULTIPLY_End                   0
#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_SRC_PREMULTIPLY_Start                 0
#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_SRC_PREMULTIPLY_Type                U01
#define   GCREG_DE_COLOR_MULTIPLY_MODES_EX_SRC_PREMULTIPLY_DISABLE           0x0
#define   GCREG_DE_COLOR_MULTIPLY_MODES_EX_SRC_PREMULTIPLY_ENABLE            0x1

#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_DST_PREMULTIPLY                     4:4
#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_DST_PREMULTIPLY_End                   4
#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_DST_PREMULTIPLY_Start                 4
#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_DST_PREMULTIPLY_Type                U01
#define   GCREG_DE_COLOR_MULTIPLY_MODES_EX_DST_PREMULTIPLY_DISABLE           0x0
#define   GCREG_DE_COLOR_MULTIPLY_MODES_EX_DST_PREMULTIPLY_ENABLE            0x1

#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_SRC_GLOBAL_PREMULTIPLY              9:8
#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_SRC_GLOBAL_PREMULTIPLY_End            9
#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_SRC_GLOBAL_PREMULTIPLY_Start          8
#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_SRC_GLOBAL_PREMULTIPLY_Type         U02
#define   GCREG_DE_COLOR_MULTIPLY_MODES_EX_SRC_GLOBAL_PREMULTIPLY_DISABLE    0x0
#define   GCREG_DE_COLOR_MULTIPLY_MODES_EX_SRC_GLOBAL_PREMULTIPLY_ALPHA      0x1
#define   GCREG_DE_COLOR_MULTIPLY_MODES_EX_SRC_GLOBAL_PREMULTIPLY_COLOR      0x2

#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_DST_DEMULTIPLY                    20:20
#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_DST_DEMULTIPLY_End                   20
#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_DST_DEMULTIPLY_Start                 20
#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_DST_DEMULTIPLY_Type                 U01
#define   GCREG_DE_COLOR_MULTIPLY_MODES_EX_DST_DEMULTIPLY_DISABLE            0x0
#define   GCREG_DE_COLOR_MULTIPLY_MODES_EX_DST_DEMULTIPLY_ENABLE             0x1

#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_DST_DEMULTIPLY_ZERO               21:21
#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_DST_DEMULTIPLY_ZERO_End              21
#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_DST_DEMULTIPLY_ZERO_Start            21
#define GCREG_DE_COLOR_MULTIPLY_MODES_EX_DST_DEMULTIPLY_ZERO_Type            U01
#define   GCREG_DE_COLOR_MULTIPLY_MODES_EX_DST_DEMULTIPLY_ZERO_DISABLE       0x0
#define   GCREG_DE_COLOR_MULTIPLY_MODES_EX_DST_DEMULTIPLY_ZERO_ENABLE        0x1

#define gcregPETransparencyExRegAddrs                                     0x4B18
#define GCREG_PE_TRANSPARENCY_EX_Address                                 0x12C60
#define GCREG_PE_TRANSPARENCY_EX_MSB                                          15
#define GCREG_PE_TRANSPARENCY_EX_LSB                                           3
#define GCREG_PE_TRANSPARENCY_EX_BLK                                           0
#define GCREG_PE_TRANSPARENCY_EX_Count                                         8
#define GCREG_PE_TRANSPARENCY_EX_FieldMask                            0xB3331333
#define GCREG_PE_TRANSPARENCY_EX_ReadMask                             0xB3331333
#define GCREG_PE_TRANSPARENCY_EX_WriteMask                            0xB3331333
#define GCREG_PE_TRANSPARENCY_EX_ResetValue                           0x00000000

#define GCREG_PE_TRANSPARENCY_EX_SOURCE                                      1:0
#define GCREG_PE_TRANSPARENCY_EX_SOURCE_End                                    1
#define GCREG_PE_TRANSPARENCY_EX_SOURCE_Start                                  0
#define GCREG_PE_TRANSPARENCY_EX_SOURCE_Type                                 U02
#define   GCREG_PE_TRANSPARENCY_EX_SOURCE_OPAQUE                             0x0
#define   GCREG_PE_TRANSPARENCY_EX_SOURCE_MASK                               0x1
#define   GCREG_PE_TRANSPARENCY_EX_SOURCE_KEY                                0x2

#define GCREG_PE_TRANSPARENCY_EX_PATTERN                                     5:4
#define GCREG_PE_TRANSPARENCY_EX_PATTERN_End                                   5
#define GCREG_PE_TRANSPARENCY_EX_PATTERN_Start                                 4
#define GCREG_PE_TRANSPARENCY_EX_PATTERN_Type                                U02
#define   GCREG_PE_TRANSPARENCY_EX_PATTERN_OPAQUE                            0x0
#define   GCREG_PE_TRANSPARENCY_EX_PATTERN_MASK                              0x1
#define   GCREG_PE_TRANSPARENCY_EX_PATTERN_KEY                               0x2

#define GCREG_PE_TRANSPARENCY_EX_DESTINATION                                 9:8
#define GCREG_PE_TRANSPARENCY_EX_DESTINATION_End                               9
#define GCREG_PE_TRANSPARENCY_EX_DESTINATION_Start                             8
#define GCREG_PE_TRANSPARENCY_EX_DESTINATION_Type                            U02
#define   GCREG_PE_TRANSPARENCY_EX_DESTINATION_OPAQUE                        0x0
#define   GCREG_PE_TRANSPARENCY_EX_DESTINATION_MASK                          0x1
#define   GCREG_PE_TRANSPARENCY_EX_DESTINATION_KEY                           0x2

#define GCREG_PE_TRANSPARENCY_EX_MASK_TRANSPARENCY                         12:12
#define GCREG_PE_TRANSPARENCY_EX_MASK_TRANSPARENCY_End                        12
#define GCREG_PE_TRANSPARENCY_EX_MASK_TRANSPARENCY_Start                      12
#define GCREG_PE_TRANSPARENCY_EX_MASK_TRANSPARENCY_Type                      U01
#define   GCREG_PE_TRANSPARENCY_EX_MASK_TRANSPARENCY_ENABLED                 0x0
#define   GCREG_PE_TRANSPARENCY_EX_MASK_TRANSPARENCY_MASKED                  0x1

#define GCREG_PE_TRANSPARENCY_EX_USE_SRC_OVERRIDE                          17:16
#define GCREG_PE_TRANSPARENCY_EX_USE_SRC_OVERRIDE_End                         17
#define GCREG_PE_TRANSPARENCY_EX_USE_SRC_OVERRIDE_Start                       16
#define GCREG_PE_TRANSPARENCY_EX_USE_SRC_OVERRIDE_Type                       U02
#define   GCREG_PE_TRANSPARENCY_EX_USE_SRC_OVERRIDE_DEFAULT                  0x0
#define   GCREG_PE_TRANSPARENCY_EX_USE_SRC_OVERRIDE_USE_ENABLE               0x1
#define   GCREG_PE_TRANSPARENCY_EX_USE_SRC_OVERRIDE_USE_DISABLE              0x2

#define GCREG_PE_TRANSPARENCY_EX_USE_PAT_OVERRIDE                          21:20
#define GCREG_PE_TRANSPARENCY_EX_USE_PAT_OVERRIDE_End                         21
#define GCREG_PE_TRANSPARENCY_EX_USE_PAT_OVERRIDE_Start                       20
#define GCREG_PE_TRANSPARENCY_EX_USE_PAT_OVERRIDE_Type                       U02
#define   GCREG_PE_TRANSPARENCY_EX_USE_PAT_OVERRIDE_DEFAULT                  0x0
#define   GCREG_PE_TRANSPARENCY_EX_USE_PAT_OVERRIDE_USE_ENABLE               0x1
#define   GCREG_PE_TRANSPARENCY_EX_USE_PAT_OVERRIDE_USE_DISABLE              0x2

#define GCREG_PE_TRANSPARENCY_EX_USE_DST_OVERRIDE                          25:24
#define GCREG_PE_TRANSPARENCY_EX_USE_DST_OVERRIDE_End                         25
#define GCREG_PE_TRANSPARENCY_EX_USE_DST_OVERRIDE_Start                       24
#define GCREG_PE_TRANSPARENCY_EX_USE_DST_OVERRIDE_Type                       U02
#define   GCREG_PE_TRANSPARENCY_EX_USE_DST_OVERRIDE_DEFAULT                  0x0
#define   GCREG_PE_TRANSPARENCY_EX_USE_DST_OVERRIDE_USE_ENABLE               0x1
#define   GCREG_PE_TRANSPARENCY_EX_USE_DST_OVERRIDE_USE_DISABLE              0x2

#define GCREG_PE_TRANSPARENCY_EX_MASK_RESOURCE_OVERRIDE                    28:28
#define GCREG_PE_TRANSPARENCY_EX_MASK_RESOURCE_OVERRIDE_End                   28
#define GCREG_PE_TRANSPARENCY_EX_MASK_RESOURCE_OVERRIDE_Start                 28
#define GCREG_PE_TRANSPARENCY_EX_MASK_RESOURCE_OVERRIDE_Type                 U01
#define   GCREG_PE_TRANSPARENCY_EX_MASK_RESOURCE_OVERRIDE_ENABLED            0x0
#define   GCREG_PE_TRANSPARENCY_EX_MASK_RESOURCE_OVERRIDE_MASKED             0x1

#define GCREG_PE_TRANSPARENCY_EX_DFB_COLOR_KEY                             29:29
#define GCREG_PE_TRANSPARENCY_EX_DFB_COLOR_KEY_End                            29
#define GCREG_PE_TRANSPARENCY_EX_DFB_COLOR_KEY_Start                          29
#define GCREG_PE_TRANSPARENCY_EX_DFB_COLOR_KEY_Type                          U01
#define   GCREG_PE_TRANSPARENCY_EX_DFB_COLOR_KEY_DISABLED                    0x0
#define   GCREG_PE_TRANSPARENCY_EX_DFB_COLOR_KEY_ENABLED                     0x1

#define GCREG_PE_TRANSPARENCY_EX_MASK_DFB_COLOR_KEY                        31:31
#define GCREG_PE_TRANSPARENCY_EX_MASK_DFB_COLOR_KEY_End                       31
#define GCREG_PE_TRANSPARENCY_EX_MASK_DFB_COLOR_KEY_Start                     31
#define GCREG_PE_TRANSPARENCY_EX_MASK_DFB_COLOR_KEY_Type                     U01
#define   GCREG_PE_TRANSPARENCY_EX_MASK_DFB_COLOR_KEY_ENABLED                0x0
#define   GCREG_PE_TRANSPARENCY_EX_MASK_DFB_COLOR_KEY_MASKED                 0x1

#define gcregPEControlExRegAddrs                                          0x4B20
#define GCREG_PE_CONTROL_EX_Address                                      0x12C80
#define GCREG_PE_CONTROL_EX_MSB                                               15
#define GCREG_PE_CONTROL_EX_LSB                                                3
#define GCREG_PE_CONTROL_EX_BLK                                                0
#define GCREG_PE_CONTROL_EX_Count                                              8
#define GCREG_PE_CONTROL_EX_FieldMask                                 0x0009999B
#define GCREG_PE_CONTROL_EX_ReadMask                                  0x0009999B
#define GCREG_PE_CONTROL_EX_WriteMask                                 0x0009999B
#define GCREG_PE_CONTROL_EX_ResetValue                                0x00000000

#define GCREG_PE_CONTROL_EX_YUV                                              1:0
#define GCREG_PE_CONTROL_EX_YUV_End                                            1
#define GCREG_PE_CONTROL_EX_YUV_Start                                          0
#define GCREG_PE_CONTROL_EX_YUV_Type                                         U02
#define   GCREG_PE_CONTROL_EX_YUV_601                                        0x0
#define   GCREG_PE_CONTROL_EX_YUV_709                                        0x1
#define   GCREG_PE_CONTROL_EX_YUV_UserDefined                                0x2
#define   GCREG_PE_CONTROL_EX_YUV_2020                                       0x3

#define GCREG_PE_CONTROL_EX_MASK_YUV                                         3:3
#define GCREG_PE_CONTROL_EX_MASK_YUV_End                                       3
#define GCREG_PE_CONTROL_EX_MASK_YUV_Start                                     3
#define GCREG_PE_CONTROL_EX_MASK_YUV_Type                                    U01
#define   GCREG_PE_CONTROL_EX_MASK_YUV_ENABLED                               0x0
#define   GCREG_PE_CONTROL_EX_MASK_YUV_MASKED                                0x1

#define GCREG_PE_CONTROL_EX_UV_SWIZZLE                                       4:4
#define GCREG_PE_CONTROL_EX_UV_SWIZZLE_End                                     4
#define GCREG_PE_CONTROL_EX_UV_SWIZZLE_Start                                   4
#define GCREG_PE_CONTROL_EX_UV_SWIZZLE_Type                                  U01
#define   GCREG_PE_CONTROL_EX_UV_SWIZZLE_UV                                  0x0
#define   GCREG_PE_CONTROL_EX_UV_SWIZZLE_VU                                  0x1

#define GCREG_PE_CONTROL_EX_MASK_UV_SWIZZLE                                  7:7
#define GCREG_PE_CONTROL_EX_MASK_UV_SWIZZLE_End                                7
#define GCREG_PE_CONTROL_EX_MASK_UV_SWIZZLE_Start                              7
#define GCREG_PE_CONTROL_EX_MASK_UV_SWIZZLE_Type                             U01
#define   GCREG_PE_CONTROL_EX_MASK_UV_SWIZZLE_ENABLED                        0x0
#define   GCREG_PE_CONTROL_EX_MASK_UV_SWIZZLE_MASKED                         0x1

#define GCREG_PE_CONTROL_EX_YUVRGB                                           8:8
#define GCREG_PE_CONTROL_EX_YUVRGB_End                                         8
#define GCREG_PE_CONTROL_EX_YUVRGB_Start                                       8
#define GCREG_PE_CONTROL_EX_YUVRGB_Type                                      U01
#define   GCREG_PE_CONTROL_EX_YUVRGB_DISABLED                                0x0
#define   GCREG_PE_CONTROL_EX_YUVRGB_ENABLED                                 0x1

#define GCREG_PE_CONTROL_EX_MASK_YUVRGB                                    11:11
#define GCREG_PE_CONTROL_EX_MASK_YUVRGB_End                                   11
#define GCREG_PE_CONTROL_EX_MASK_YUVRGB_Start                                 11
#define GCREG_PE_CONTROL_EX_MASK_YUVRGB_Type                                 U01
#define   GCREG_PE_CONTROL_EX_MASK_YUVRGB_ENABLED                            0x0
#define   GCREG_PE_CONTROL_EX_MASK_YUVRGB_MASKED                             0x1

#define GCREG_PE_CONTROL_EX_YUV_CLAMP                                      12:12
#define GCREG_PE_CONTROL_EX_YUV_CLAMP_End                                     12
#define GCREG_PE_CONTROL_EX_YUV_CLAMP_Start                                   12
#define GCREG_PE_CONTROL_EX_YUV_CLAMP_Type                                   U01
#define   GCREG_PE_CONTROL_EX_YUV_CLAMP_DISABLED                             0x0
#define   GCREG_PE_CONTROL_EX_YUV_CLAMP_ENABLED                              0x1

#define GCREG_PE_CONTROL_EX_MASK_YUV_CLAMP                                 15:15
#define GCREG_PE_CONTROL_EX_MASK_YUV_CLAMP_End                                15
#define GCREG_PE_CONTROL_EX_MASK_YUV_CLAMP_Start                              15
#define GCREG_PE_CONTROL_EX_MASK_YUV_CLAMP_Type                              U01
#define   GCREG_PE_CONTROL_EX_MASK_YUV_CLAMP_ENABLED                         0x0
#define   GCREG_PE_CONTROL_EX_MASK_YUV_CLAMP_MASKED                          0x1

#define GCREG_PE_CONTROL_EX_GRAY_YUVRGB                                    16:16
#define GCREG_PE_CONTROL_EX_GRAY_YUVRGB_End                                   16
#define GCREG_PE_CONTROL_EX_GRAY_YUVRGB_Start                                 16
#define GCREG_PE_CONTROL_EX_GRAY_YUVRGB_Type                                 U01
#define   GCREG_PE_CONTROL_EX_GRAY_YUVRGB_DISABLED                           0x0
#define   GCREG_PE_CONTROL_EX_GRAY_YUVRGB_ENABLED                            0x1

#define GCREG_PE_CONTROL_EX_MASK_GRAY_YUVRGB                               19:19
#define GCREG_PE_CONTROL_EX_MASK_GRAY_YUVRGB_End                              19
#define GCREG_PE_CONTROL_EX_MASK_GRAY_YUVRGB_Start                            19
#define GCREG_PE_CONTROL_EX_MASK_GRAY_YUVRGB_Type                            U01
#define   GCREG_PE_CONTROL_EX_MASK_GRAY_YUVRGB_ENABLED                       0x0
#define   GCREG_PE_CONTROL_EX_MASK_GRAY_YUVRGB_MASKED                        0x1

#define gcregDESrcColorKeyHighExRegAddrs                                  0x4B28
#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_Address                           0x12CA0
#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_MSB                                    15
#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_LSB                                     3
#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_BLK                                     0
#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_Count                                   8
#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_FieldMask                      0xFFFFFFFF
#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_ReadMask                       0xFFFFFFFF
#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_WriteMask                      0xFFFFFFFF
#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_ResetValue                     0x00000000

#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_ALPHA                               31:24
#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_ALPHA_End                              31
#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_ALPHA_Start                            24
#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_ALPHA_Type                            U08

#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_RED                                 23:16
#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_RED_End                                23
#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_RED_Start                              16
#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_RED_Type                              U08

#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_GREEN                                15:8
#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_GREEN_End                              15
#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_GREEN_Start                             8
#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_GREEN_Type                            U08

#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_BLUE                                  7:0
#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_BLUE_End                                7
#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_BLUE_Start                              0
#define GCREG_DE_SRC_COLOR_KEY_HIGH_EX_BLUE_Type                             U08

#define gcregDESrcExConfigExRegAddrs                                      0x4B30
#define GCREG_DE_SRC_EX_CONFIG_EX_Address                                0x12CC0
#define GCREG_DE_SRC_EX_CONFIG_EX_MSB                                         15
#define GCREG_DE_SRC_EX_CONFIG_EX_LSB                                          3
#define GCREG_DE_SRC_EX_CONFIG_EX_BLK                                          0
#define GCREG_DE_SRC_EX_CONFIG_EX_Count                                        8
#define GCREG_DE_SRC_EX_CONFIG_EX_FieldMask                           0x0001F109
#define GCREG_DE_SRC_EX_CONFIG_EX_ReadMask                            0x0001F109
#define GCREG_DE_SRC_EX_CONFIG_EX_WriteMask                           0x0001F109
#define GCREG_DE_SRC_EX_CONFIG_EX_ResetValue                          0x00000000

#define GCREG_DE_SRC_EX_CONFIG_EX_MULTI_TILED                                0:0
#define GCREG_DE_SRC_EX_CONFIG_EX_MULTI_TILED_End                              0
#define GCREG_DE_SRC_EX_CONFIG_EX_MULTI_TILED_Start                            0
#define GCREG_DE_SRC_EX_CONFIG_EX_MULTI_TILED_Type                           U01
#define   GCREG_DE_SRC_EX_CONFIG_EX_MULTI_TILED_DISABLED                     0x0
#define   GCREG_DE_SRC_EX_CONFIG_EX_MULTI_TILED_ENABLED                      0x1

#define GCREG_DE_SRC_EX_CONFIG_EX_SUPER_TILED                                3:3
#define GCREG_DE_SRC_EX_CONFIG_EX_SUPER_TILED_End                              3
#define GCREG_DE_SRC_EX_CONFIG_EX_SUPER_TILED_Start                            3
#define GCREG_DE_SRC_EX_CONFIG_EX_SUPER_TILED_Type                           U01
#define   GCREG_DE_SRC_EX_CONFIG_EX_SUPER_TILED_DISABLED                     0x0
#define   GCREG_DE_SRC_EX_CONFIG_EX_SUPER_TILED_ENABLED                      0x1

#define GCREG_DE_SRC_EX_CONFIG_EX_MINOR_TILED                                8:8
#define GCREG_DE_SRC_EX_CONFIG_EX_MINOR_TILED_End                              8
#define GCREG_DE_SRC_EX_CONFIG_EX_MINOR_TILED_Start                            8
#define GCREG_DE_SRC_EX_CONFIG_EX_MINOR_TILED_Type                           U01
#define   GCREG_DE_SRC_EX_CONFIG_EX_MINOR_TILED_DISABLED                     0x0
#define   GCREG_DE_SRC_EX_CONFIG_EX_MINOR_TILED_ENABLED                      0x1

#define GCREG_DE_SRC_EX_CONFIG_EX_CACHE_MODE                               12:12
#define GCREG_DE_SRC_EX_CONFIG_EX_CACHE_MODE_End                              12
#define GCREG_DE_SRC_EX_CONFIG_EX_CACHE_MODE_Start                            12
#define GCREG_DE_SRC_EX_CONFIG_EX_CACHE_MODE_Type                            U01
#define   GCREG_DE_SRC_EX_CONFIG_EX_CACHE_MODE_DISABLED                      0x0
#define   GCREG_DE_SRC_EX_CONFIG_EX_CACHE_MODE_ENABLED                       0x1

#define GCREG_DE_SRC_EX_CONFIG_EX_DC_COMPRESSION                           13:13
#define GCREG_DE_SRC_EX_CONFIG_EX_DC_COMPRESSION_End                          13
#define GCREG_DE_SRC_EX_CONFIG_EX_DC_COMPRESSION_Start                        13
#define GCREG_DE_SRC_EX_CONFIG_EX_DC_COMPRESSION_Type                        U01
#define   GCREG_DE_SRC_EX_CONFIG_EX_DC_COMPRESSION_DISABLED                  0x0
#define   GCREG_DE_SRC_EX_CONFIG_EX_DC_COMPRESSION_ENABLED                   0x1

#define GCREG_DE_SRC_EX_CONFIG_EX_MASK_DC_COMPRESSION                      14:14
#define GCREG_DE_SRC_EX_CONFIG_EX_MASK_DC_COMPRESSION_End                     14
#define GCREG_DE_SRC_EX_CONFIG_EX_MASK_DC_COMPRESSION_Start                   14
#define GCREG_DE_SRC_EX_CONFIG_EX_MASK_DC_COMPRESSION_Type                   U01
#define   GCREG_DE_SRC_EX_CONFIG_EX_MASK_DC_COMPRESSION_ENABLED              0x0
#define   GCREG_DE_SRC_EX_CONFIG_EX_MASK_DC_COMPRESSION_MASKED               0x1

#define GCREG_DE_SRC_EX_CONFIG_EX_DC_COMPRESSION_ENABLE_ALPHA              15:15
#define GCREG_DE_SRC_EX_CONFIG_EX_DC_COMPRESSION_ENABLE_ALPHA_End             15
#define GCREG_DE_SRC_EX_CONFIG_EX_DC_COMPRESSION_ENABLE_ALPHA_Start           15
#define GCREG_DE_SRC_EX_CONFIG_EX_DC_COMPRESSION_ENABLE_ALPHA_Type           U01
#define   GCREG_DE_SRC_EX_CONFIG_EX_DC_COMPRESSION_ENABLE_ALPHA_DISABLED     0x0
#define   GCREG_DE_SRC_EX_CONFIG_EX_DC_COMPRESSION_ENABLE_ALPHA_ENABLED      0x1

#define GCREG_DE_SRC_EX_CONFIG_EX_MASK_DC_COMPRESSION_ENABLE_ALPHA         16:16
#define GCREG_DE_SRC_EX_CONFIG_EX_MASK_DC_COMPRESSION_ENABLE_ALPHA_End        16
#define GCREG_DE_SRC_EX_CONFIG_EX_MASK_DC_COMPRESSION_ENABLE_ALPHA_Start      16
#define GCREG_DE_SRC_EX_CONFIG_EX_MASK_DC_COMPRESSION_ENABLE_ALPHA_Type      U01
#define   GCREG_DE_SRC_EX_CONFIG_EX_MASK_DC_COMPRESSION_ENABLE_ALPHA_ENABLED 0x0
#define   GCREG_DE_SRC_EX_CONFIG_EX_MASK_DC_COMPRESSION_ENABLE_ALPHA_MASKED  0x1

#define gcregDESrcExAddressExRegAddrs                                     0x4B38
#define GCREG_DE_SRC_EX_ADDRESS_EX_Address                               0x12CE0
#define GCREG_DE_SRC_EX_ADDRESS_EX_MSB                                        15
#define GCREG_DE_SRC_EX_ADDRESS_EX_LSB                                         3
#define GCREG_DE_SRC_EX_ADDRESS_EX_BLK                                         0
#define GCREG_DE_SRC_EX_ADDRESS_EX_Count                                       8
#define GCREG_DE_SRC_EX_ADDRESS_EX_FieldMask                          0xFFFFFFFF
#define GCREG_DE_SRC_EX_ADDRESS_EX_ReadMask                           0xFFFFFFFC
#define GCREG_DE_SRC_EX_ADDRESS_EX_WriteMask                          0xFFFFFFFC
#define GCREG_DE_SRC_EX_ADDRESS_EX_ResetValue                         0x00000000

#define GCREG_DE_SRC_EX_ADDRESS_EX_TYPE                                    31:31
#define GCREG_DE_SRC_EX_ADDRESS_EX_TYPE_End                                   31
#define GCREG_DE_SRC_EX_ADDRESS_EX_TYPE_Start                                 31
#define GCREG_DE_SRC_EX_ADDRESS_EX_TYPE_Type                                 U01
#define   GCREG_DE_SRC_EX_ADDRESS_EX_TYPE_SYSTEM                             0x0
#define   GCREG_DE_SRC_EX_ADDRESS_EX_TYPE_VIRTUAL_SYSTEM                     0x1

#define GCREG_DE_SRC_EX_ADDRESS_EX_ADDRESS                                  30:0
#define GCREG_DE_SRC_EX_ADDRESS_EX_ADDRESS_End                                30
#define GCREG_DE_SRC_EX_ADDRESS_EX_ADDRESS_Start                               0
#define GCREG_DE_SRC_EX_ADDRESS_EX_ADDRESS_Type                              U31

#define gcregDEClipTopLeftRegAddrs                                        0x4B40
#define GCREG_DE_CLIP_TOP_LEFT_Address                                   0x12D00
#define GCREG_DE_CLIP_TOP_LEFT_MSB                                            15
#define GCREG_DE_CLIP_TOP_LEFT_LSB                                             3
#define GCREG_DE_CLIP_TOP_LEFT_BLK                                             0
#define GCREG_DE_CLIP_TOP_LEFT_Count                                           8
#define GCREG_DE_CLIP_TOP_LEFT_FieldMask                              0x7FFF7FFF
#define GCREG_DE_CLIP_TOP_LEFT_ReadMask                               0x7FFF7FFF
#define GCREG_DE_CLIP_TOP_LEFT_WriteMask                              0x7FFF7FFF
#define GCREG_DE_CLIP_TOP_LEFT_ResetValue                             0x00000000

#define GCREG_DE_CLIP_TOP_LEFT_Y                                           30:16
#define GCREG_DE_CLIP_TOP_LEFT_Y_End                                          30
#define GCREG_DE_CLIP_TOP_LEFT_Y_Start                                        16
#define GCREG_DE_CLIP_TOP_LEFT_Y_Type                                        U15

#define GCREG_DE_CLIP_TOP_LEFT_X                                            14:0
#define GCREG_DE_CLIP_TOP_LEFT_X_End                                          14
#define GCREG_DE_CLIP_TOP_LEFT_X_Start                                         0
#define GCREG_DE_CLIP_TOP_LEFT_X_Type                                        U15

#define gcregDEClipBottomRightRegAddrs                                    0x4B48
#define GCREG_DE_CLIP_BOTTOM_RIGHT_Address                               0x12D20
#define GCREG_DE_CLIP_BOTTOM_RIGHT_MSB                                        15
#define GCREG_DE_CLIP_BOTTOM_RIGHT_LSB                                         3
#define GCREG_DE_CLIP_BOTTOM_RIGHT_BLK                                         0
#define GCREG_DE_CLIP_BOTTOM_RIGHT_Count                                       8
#define GCREG_DE_CLIP_BOTTOM_RIGHT_FieldMask                          0x7FFF7FFF
#define GCREG_DE_CLIP_BOTTOM_RIGHT_ReadMask                           0x7FFF7FFF
#define GCREG_DE_CLIP_BOTTOM_RIGHT_WriteMask                          0x7FFF7FFF
#define GCREG_DE_CLIP_BOTTOM_RIGHT_ResetValue                         0x00000000

#define GCREG_DE_CLIP_BOTTOM_RIGHT_Y                                       30:16
#define GCREG_DE_CLIP_BOTTOM_RIGHT_Y_End                                      30
#define GCREG_DE_CLIP_BOTTOM_RIGHT_Y_Start                                    16
#define GCREG_DE_CLIP_BOTTOM_RIGHT_Y_Type                                    U15

#define GCREG_DE_CLIP_BOTTOM_RIGHT_X                                        14:0
#define GCREG_DE_CLIP_BOTTOM_RIGHT_X_End                                      14
#define GCREG_DE_CLIP_BOTTOM_RIGHT_X_Start                                     0
#define GCREG_DE_CLIP_BOTTOM_RIGHT_X_Type                                    U15

#define gcregDEDestRectTopLeftRegAddrs                                    0x4B50
#define GCREG_DE_DEST_RECT_TOP_LEFT_Address                              0x12D40
#define GCREG_DE_DEST_RECT_TOP_LEFT_MSB                                       15
#define GCREG_DE_DEST_RECT_TOP_LEFT_LSB                                        3
#define GCREG_DE_DEST_RECT_TOP_LEFT_BLK                                        0
#define GCREG_DE_DEST_RECT_TOP_LEFT_Count                                      8
#define GCREG_DE_DEST_RECT_TOP_LEFT_FieldMask                         0x7FFF7FFF
#define GCREG_DE_DEST_RECT_TOP_LEFT_ReadMask                          0x7FFF7FFF
#define GCREG_DE_DEST_RECT_TOP_LEFT_WriteMask                         0x7FFF7FFF
#define GCREG_DE_DEST_RECT_TOP_LEFT_ResetValue                        0x00000000

#define GCREG_DE_DEST_RECT_TOP_LEFT_Y                                      30:16
#define GCREG_DE_DEST_RECT_TOP_LEFT_Y_End                                     30
#define GCREG_DE_DEST_RECT_TOP_LEFT_Y_Start                                   16
#define GCREG_DE_DEST_RECT_TOP_LEFT_Y_Type                                   U15

#define GCREG_DE_DEST_RECT_TOP_LEFT_X                                       14:0
#define GCREG_DE_DEST_RECT_TOP_LEFT_X_End                                     14
#define GCREG_DE_DEST_RECT_TOP_LEFT_X_Start                                    0
#define GCREG_DE_DEST_RECT_TOP_LEFT_X_Type                                   U15

#define gcregDEDestRectBottomRightRegAddrs                                0x4B58
#define GCREG_DE_DEST_RECT_BOTTOM_RIGHT_Address                          0x12D60
#define GCREG_DE_DEST_RECT_BOTTOM_RIGHT_MSB                                   15
#define GCREG_DE_DEST_RECT_BOTTOM_RIGHT_LSB                                    3
#define GCREG_DE_DEST_RECT_BOTTOM_RIGHT_BLK                                    0
#define GCREG_DE_DEST_RECT_BOTTOM_RIGHT_Count                                  8
#define GCREG_DE_DEST_RECT_BOTTOM_RIGHT_FieldMask                     0x7FFF7FFF
#define GCREG_DE_DEST_RECT_BOTTOM_RIGHT_ReadMask                      0x7FFF7FFF
#define GCREG_DE_DEST_RECT_BOTTOM_RIGHT_WriteMask                     0x7FFF7FFF
#define GCREG_DE_DEST_RECT_BOTTOM_RIGHT_ResetValue                    0x00000000

#define GCREG_DE_DEST_RECT_BOTTOM_RIGHT_Y                                  30:16
#define GCREG_DE_DEST_RECT_BOTTOM_RIGHT_Y_End                                 30
#define GCREG_DE_DEST_RECT_BOTTOM_RIGHT_Y_Start                               16
#define GCREG_DE_DEST_RECT_BOTTOM_RIGHT_Y_Type                               U15

#define GCREG_DE_DEST_RECT_BOTTOM_RIGHT_X                                   14:0
#define GCREG_DE_DEST_RECT_BOTTOM_RIGHT_X_End                                 14
#define GCREG_DE_DEST_RECT_BOTTOM_RIGHT_X_Start                                0
#define GCREG_DE_DEST_RECT_BOTTOM_RIGHT_X_Type                               U15

#define gcregDEStretchFactorLowRegAddrs                                   0x4B60
#define GCREG_DE_STRETCH_FACTOR_LOW_Address                              0x12D80
#define GCREG_DE_STRETCH_FACTOR_LOW_MSB                                       15
#define GCREG_DE_STRETCH_FACTOR_LOW_LSB                                        3
#define GCREG_DE_STRETCH_FACTOR_LOW_BLK                                        0
#define GCREG_DE_STRETCH_FACTOR_LOW_Count                                      8
#define GCREG_DE_STRETCH_FACTOR_LOW_FieldMask                         0x7FFFFFFF
#define GCREG_DE_STRETCH_FACTOR_LOW_ReadMask                          0x7FFFFFFF
#define GCREG_DE_STRETCH_FACTOR_LOW_WriteMask                         0x7FFFFFFF
#define GCREG_DE_STRETCH_FACTOR_LOW_ResetValue                        0x00000000

#define GCREG_DE_STRETCH_FACTOR_LOW_X                                       30:0
#define GCREG_DE_STRETCH_FACTOR_LOW_X_End                                     30
#define GCREG_DE_STRETCH_FACTOR_LOW_X_Start                                    0
#define GCREG_DE_STRETCH_FACTOR_LOW_X_Type                                   U31

#define gcregDEStretchFactorHighRegAddrs                                  0x4B68
#define GCREG_DE_STRETCH_FACTOR_HIGH_Address                             0x12DA0
#define GCREG_DE_STRETCH_FACTOR_HIGH_MSB                                      15
#define GCREG_DE_STRETCH_FACTOR_HIGH_LSB                                       3
#define GCREG_DE_STRETCH_FACTOR_HIGH_BLK                                       0
#define GCREG_DE_STRETCH_FACTOR_HIGH_Count                                     8
#define GCREG_DE_STRETCH_FACTOR_HIGH_FieldMask                        0x7FFFFFFF
#define GCREG_DE_STRETCH_FACTOR_HIGH_ReadMask                         0x7FFFFFFF
#define GCREG_DE_STRETCH_FACTOR_HIGH_WriteMask                        0x7FFFFFFF
#define GCREG_DE_STRETCH_FACTOR_HIGH_ResetValue                       0x00000000

#define GCREG_DE_STRETCH_FACTOR_HIGH_Y                                      30:0
#define GCREG_DE_STRETCH_FACTOR_HIGH_Y_End                                    30
#define GCREG_DE_STRETCH_FACTOR_HIGH_Y_Start                                   0
#define GCREG_DE_STRETCH_FACTOR_HIGH_Y_Type                                  U31

#define gcregDESrcVInitialErrorRegAddrs                                   0x4B70
#define GCREG_DE_SRC_VINITIAL_ERROR_Address                              0x12DC0
#define GCREG_DE_SRC_VINITIAL_ERROR_MSB                                       15
#define GCREG_DE_SRC_VINITIAL_ERROR_LSB                                        3
#define GCREG_DE_SRC_VINITIAL_ERROR_BLK                                        0
#define GCREG_DE_SRC_VINITIAL_ERROR_Count                                      8
#define GCREG_DE_SRC_VINITIAL_ERROR_FieldMask                         0xFFFFFFFF
#define GCREG_DE_SRC_VINITIAL_ERROR_ReadMask                          0xFFFFFFFF
#define GCREG_DE_SRC_VINITIAL_ERROR_WriteMask                         0xFFFFFFFF
#define GCREG_DE_SRC_VINITIAL_ERROR_ResetValue                        0x00000000

#define GCREG_DE_SRC_VINITIAL_ERROR_INITIAL_ERROR                           31:0
#define GCREG_DE_SRC_VINITIAL_ERROR_INITIAL_ERROR_End                         31
#define GCREG_DE_SRC_VINITIAL_ERROR_INITIAL_ERROR_Start                        0
#define GCREG_DE_SRC_VINITIAL_ERROR_INITIAL_ERROR_Type                       U32

#define gcregDESrcVSouthDeltaRegAddrs                                     0x4B78
#define GCREG_DE_SRC_VSOUTH_DELTA_Address                                0x12DE0
#define GCREG_DE_SRC_VSOUTH_DELTA_MSB                                         15
#define GCREG_DE_SRC_VSOUTH_DELTA_LSB                                          3
#define GCREG_DE_SRC_VSOUTH_DELTA_BLK                                          0
#define GCREG_DE_SRC_VSOUTH_DELTA_Count                                        8
#define GCREG_DE_SRC_VSOUTH_DELTA_FieldMask                           0xFFFFFFFF
#define GCREG_DE_SRC_VSOUTH_DELTA_ReadMask                            0xFFFFFFFF
#define GCREG_DE_SRC_VSOUTH_DELTA_WriteMask                           0xFFFFFFFF
#define GCREG_DE_SRC_VSOUTH_DELTA_ResetValue                          0x00000000

#define GCREG_DE_SRC_VSOUTH_DELTA_SOUTH_DELTA                               31:0
#define GCREG_DE_SRC_VSOUTH_DELTA_SOUTH_DELTA_End                             31
#define GCREG_DE_SRC_VSOUTH_DELTA_SOUTH_DELTA_Start                            0
#define GCREG_DE_SRC_VSOUTH_DELTA_SOUTH_DELTA_Type                           U32

#define gcregDESrcVNorthDeltaRegAddrs                                     0x4B80
#define GCREG_DE_SRC_VNORTH_DELTA_Address                                0x12E00
#define GCREG_DE_SRC_VNORTH_DELTA_MSB                                         15
#define GCREG_DE_SRC_VNORTH_DELTA_LSB                                          3
#define GCREG_DE_SRC_VNORTH_DELTA_BLK                                          0
#define GCREG_DE_SRC_VNORTH_DELTA_Count                                        8
#define GCREG_DE_SRC_VNORTH_DELTA_FieldMask                           0xFFFFFFFF
#define GCREG_DE_SRC_VNORTH_DELTA_ReadMask                            0xFFFFFFFF
#define GCREG_DE_SRC_VNORTH_DELTA_WriteMask                           0xFFFFFFFF
#define GCREG_DE_SRC_VNORTH_DELTA_ResetValue                          0x00000000

#define GCREG_DE_SRC_VNORTH_DELTA_NORTH_DELTA                               31:0
#define GCREG_DE_SRC_VNORTH_DELTA_NORTH_DELTA_End                             31
#define GCREG_DE_SRC_VNORTH_DELTA_NORTH_DELTA_Start                            0
#define GCREG_DE_SRC_VNORTH_DELTA_NORTH_DELTA_Type                           U32

#define gcregDESrcHInitialErrorRegAddrs                                   0x4B88
#define GCREG_DE_SRC_HINITIAL_ERROR_Address                              0x12E20
#define GCREG_DE_SRC_HINITIAL_ERROR_MSB                                       15
#define GCREG_DE_SRC_HINITIAL_ERROR_LSB                                        3
#define GCREG_DE_SRC_HINITIAL_ERROR_BLK                                        0
#define GCREG_DE_SRC_HINITIAL_ERROR_Count                                      8
#define GCREG_DE_SRC_HINITIAL_ERROR_FieldMask                         0xFFFFFFFF
#define GCREG_DE_SRC_HINITIAL_ERROR_ReadMask                          0xFFFFFFFF
#define GCREG_DE_SRC_HINITIAL_ERROR_WriteMask                         0xFFFFFFFF
#define GCREG_DE_SRC_HINITIAL_ERROR_ResetValue                        0x00000000

#define GCREG_DE_SRC_HINITIAL_ERROR_INITIAL_ERROR                           31:0
#define GCREG_DE_SRC_HINITIAL_ERROR_INITIAL_ERROR_End                         31
#define GCREG_DE_SRC_HINITIAL_ERROR_INITIAL_ERROR_Start                        0
#define GCREG_DE_SRC_HINITIAL_ERROR_INITIAL_ERROR_Type                       U32

#define gcregDESrcHSouthDeltaRegAddrs                                     0x4B90
#define GCREG_DE_SRC_HSOUTH_DELTA_Address                                0x12E40
#define GCREG_DE_SRC_HSOUTH_DELTA_MSB                                         15
#define GCREG_DE_SRC_HSOUTH_DELTA_LSB                                          3
#define GCREG_DE_SRC_HSOUTH_DELTA_BLK                                          0
#define GCREG_DE_SRC_HSOUTH_DELTA_Count                                        8
#define GCREG_DE_SRC_HSOUTH_DELTA_FieldMask                           0xFFFFFFFF
#define GCREG_DE_SRC_HSOUTH_DELTA_ReadMask                            0xFFFFFFFF
#define GCREG_DE_SRC_HSOUTH_DELTA_WriteMask                           0xFFFFFFFF
#define GCREG_DE_SRC_HSOUTH_DELTA_ResetValue                          0x00000000

#define GCREG_DE_SRC_HSOUTH_DELTA_SOUTH_DELTA                               31:0
#define GCREG_DE_SRC_HSOUTH_DELTA_SOUTH_DELTA_End                             31
#define GCREG_DE_SRC_HSOUTH_DELTA_SOUTH_DELTA_Start                            0
#define GCREG_DE_SRC_HSOUTH_DELTA_SOUTH_DELTA_Type                           U32

#define gcregDESrcHNorthDeltaRegAddrs                                     0x4B98
#define GCREG_DE_SRC_HNORTH_DELTA_Address                                0x12E60
#define GCREG_DE_SRC_HNORTH_DELTA_MSB                                         15
#define GCREG_DE_SRC_HNORTH_DELTA_LSB                                          3
#define GCREG_DE_SRC_HNORTH_DELTA_BLK                                          0
#define GCREG_DE_SRC_HNORTH_DELTA_Count                                        8
#define GCREG_DE_SRC_HNORTH_DELTA_FieldMask                           0xFFFFFFFF
#define GCREG_DE_SRC_HNORTH_DELTA_ReadMask                            0xFFFFFFFF
#define GCREG_DE_SRC_HNORTH_DELTA_WriteMask                           0xFFFFFFFF
#define GCREG_DE_SRC_HNORTH_DELTA_ResetValue                          0x00000000

#define GCREG_DE_SRC_HNORTH_DELTA_NORTH_DELTA                               31:0
#define GCREG_DE_SRC_HNORTH_DELTA_NORTH_DELTA_End                             31
#define GCREG_DE_SRC_HNORTH_DELTA_NORTH_DELTA_Start                            0
#define GCREG_DE_SRC_HNORTH_DELTA_NORTH_DELTA_Type                           U32

#define gcregMultiSrcConfigRegAddrs                                       0x4BA0
#define GCREG_MULTI_SRC_CONFIG_Address                                   0x12E80
#define GCREG_MULTI_SRC_CONFIG_MSB                                            15
#define GCREG_MULTI_SRC_CONFIG_LSB                                             3
#define GCREG_MULTI_SRC_CONFIG_BLK                                             0
#define GCREG_MULTI_SRC_CONFIG_Count                                           8
#define GCREG_MULTI_SRC_CONFIG_FieldMask                              0x000000FF
#define GCREG_MULTI_SRC_CONFIG_ReadMask                               0x000000FF
#define GCREG_MULTI_SRC_CONFIG_WriteMask                              0x000000FF
#define GCREG_MULTI_SRC_CONFIG_ResetValue                             0x00000000

#define GCREG_MULTI_SRC_CONFIG_SRC_CMD                                       2:0
#define GCREG_MULTI_SRC_CONFIG_SRC_CMD_End                                     2
#define GCREG_MULTI_SRC_CONFIG_SRC_CMD_Start                                   0
#define GCREG_MULTI_SRC_CONFIG_SRC_CMD_Type                                  U03
#define   GCREG_MULTI_SRC_CONFIG_SRC_CMD_BIT_BLT                             0x0
#define   GCREG_MULTI_SRC_CONFIG_SRC_CMD_STRETCH_BLT                         0x1

#define GCREG_MULTI_SRC_CONFIG_MASK_SRC_CMD                                  3:3
#define GCREG_MULTI_SRC_CONFIG_MASK_SRC_CMD_End                                3
#define GCREG_MULTI_SRC_CONFIG_MASK_SRC_CMD_Start                              3
#define GCREG_MULTI_SRC_CONFIG_MASK_SRC_CMD_Type                             U01
#define   GCREG_MULTI_SRC_CONFIG_MASK_SRC_CMD_ENABLED                        0x0
#define   GCREG_MULTI_SRC_CONFIG_MASK_SRC_CMD_MASKED                         0x1

#define GCREG_MULTI_SRC_CONFIG_GDI_STRE                                      4:4
#define GCREG_MULTI_SRC_CONFIG_GDI_STRE_End                                    4
#define GCREG_MULTI_SRC_CONFIG_GDI_STRE_Start                                  4
#define GCREG_MULTI_SRC_CONFIG_GDI_STRE_Type                                 U01
#define   GCREG_MULTI_SRC_CONFIG_GDI_STRE_DISABLED                           0x0
#define   GCREG_MULTI_SRC_CONFIG_GDI_STRE_ENABLED                            0x1

#define GCREG_MULTI_SRC_CONFIG_MASK_GDI_STRE                                 5:5
#define GCREG_MULTI_SRC_CONFIG_MASK_GDI_STRE_End                               5
#define GCREG_MULTI_SRC_CONFIG_MASK_GDI_STRE_Start                             5
#define GCREG_MULTI_SRC_CONFIG_MASK_GDI_STRE_Type                            U01
#define   GCREG_MULTI_SRC_CONFIG_MASK_GDI_STRE_ENABLED                       0x0
#define   GCREG_MULTI_SRC_CONFIG_MASK_GDI_STRE_MASKED                        0x1

#define GCREG_MULTI_SRC_CONFIG_INITIAL_ERROR                                 6:6
#define GCREG_MULTI_SRC_CONFIG_INITIAL_ERROR_End                               6
#define GCREG_MULTI_SRC_CONFIG_INITIAL_ERROR_Start                             6
#define GCREG_MULTI_SRC_CONFIG_INITIAL_ERROR_Type                            U01
#define   GCREG_MULTI_SRC_CONFIG_INITIAL_ERROR_DISABLED                      0x0
#define   GCREG_MULTI_SRC_CONFIG_INITIAL_ERROR_ENABLED                       0x1

#define GCREG_MULTI_SRC_CONFIG_MASK_INITIAL_ERROR                            7:7
#define GCREG_MULTI_SRC_CONFIG_MASK_INITIAL_ERROR_End                          7
#define GCREG_MULTI_SRC_CONFIG_MASK_INITIAL_ERROR_Start                        7
#define GCREG_MULTI_SRC_CONFIG_MASK_INITIAL_ERROR_Type                       U01
#define   GCREG_MULTI_SRC_CONFIG_MASK_INITIAL_ERROR_ENABLED                  0x0
#define   GCREG_MULTI_SRC_CONFIG_MASK_INITIAL_ERROR_MASKED                   0x1

#define gcregPatternFgColorRegAddrs                                       0x4BA8
#define GCREG_PATTERN_FG_COLOR_Address                                   0x12EA0
#define GCREG_PATTERN_FG_COLOR_MSB                                            15
#define GCREG_PATTERN_FG_COLOR_LSB                                             3
#define GCREG_PATTERN_FG_COLOR_BLK                                             0
#define GCREG_PATTERN_FG_COLOR_Count                                           8
#define GCREG_PATTERN_FG_COLOR_FieldMask                              0xFFFFFFFF
#define GCREG_PATTERN_FG_COLOR_ReadMask                               0xFFFFFFFF
#define GCREG_PATTERN_FG_COLOR_WriteMask                              0xFFFFFFFF
#define GCREG_PATTERN_FG_COLOR_ResetValue                             0x00000000

#define GCREG_PATTERN_FG_COLOR_ALPHA                                       31:24
#define GCREG_PATTERN_FG_COLOR_ALPHA_End                                      31
#define GCREG_PATTERN_FG_COLOR_ALPHA_Start                                    24
#define GCREG_PATTERN_FG_COLOR_ALPHA_Type                                    U08

#define GCREG_PATTERN_FG_COLOR_RED                                         23:16
#define GCREG_PATTERN_FG_COLOR_RED_End                                        23
#define GCREG_PATTERN_FG_COLOR_RED_Start                                      16
#define GCREG_PATTERN_FG_COLOR_RED_Type                                      U08

#define GCREG_PATTERN_FG_COLOR_GREEN                                        15:8
#define GCREG_PATTERN_FG_COLOR_GREEN_End                                      15
#define GCREG_PATTERN_FG_COLOR_GREEN_Start                                     8
#define GCREG_PATTERN_FG_COLOR_GREEN_Type                                    U08

#define GCREG_PATTERN_FG_COLOR_BLUE                                          7:0
#define GCREG_PATTERN_FG_COLOR_BLUE_End                                        7
#define GCREG_PATTERN_FG_COLOR_BLUE_Start                                      0
#define GCREG_PATTERN_FG_COLOR_BLUE_Type                                     U08

#define gcregPatternConfigRegAddrs                                        0x4BB0
#define GCREG_PATTERN_CONFIG_Address                                     0x12EC0
#define GCREG_PATTERN_CONFIG_MSB                                              15
#define GCREG_PATTERN_CONFIG_LSB                                               3
#define GCREG_PATTERN_CONFIG_BLK                                               0
#define GCREG_PATTERN_CONFIG_Count                                             8
#define GCREG_PATTERN_CONFIG_FieldMask                                0x00000010
#define GCREG_PATTERN_CONFIG_ReadMask                                 0x00000010
#define GCREG_PATTERN_CONFIG_WriteMask                                0x00000010
#define GCREG_PATTERN_CONFIG_ResetValue                               0x00000000

#define GCREG_PATTERN_CONFIG_TYPE                                            4:4
#define GCREG_PATTERN_CONFIG_TYPE_End                                          4
#define GCREG_PATTERN_CONFIG_TYPE_Start                                        4
#define GCREG_PATTERN_CONFIG_TYPE_Type                                       U01
#define   GCREG_PATTERN_CONFIG_TYPE_SOLID_COLOR                              0x0
#define   GCREG_PATTERN_CONFIG_TYPE_PATTERN                                  0x1

#define gcregDESrcTileStatusAddressRegAddrs                               0x4BB8
#define GCREG_DE_SRC_TILE_STATUS_ADDRESS_Address                         0x12EE0
#define GCREG_DE_SRC_TILE_STATUS_ADDRESS_MSB                                  15
#define GCREG_DE_SRC_TILE_STATUS_ADDRESS_LSB                                   3
#define GCREG_DE_SRC_TILE_STATUS_ADDRESS_BLK                                   0
#define GCREG_DE_SRC_TILE_STATUS_ADDRESS_Count                                 8
#define GCREG_DE_SRC_TILE_STATUS_ADDRESS_FieldMask                    0xFFFFFFFF
#define GCREG_DE_SRC_TILE_STATUS_ADDRESS_ReadMask                     0xFFFFFFFC
#define GCREG_DE_SRC_TILE_STATUS_ADDRESS_WriteMask                    0xFFFFFFFC
#define GCREG_DE_SRC_TILE_STATUS_ADDRESS_ResetValue                   0x00000000

#define GCREG_DE_SRC_TILE_STATUS_ADDRESS_TYPE                              31:31
#define GCREG_DE_SRC_TILE_STATUS_ADDRESS_TYPE_End                             31
#define GCREG_DE_SRC_TILE_STATUS_ADDRESS_TYPE_Start                           31
#define GCREG_DE_SRC_TILE_STATUS_ADDRESS_TYPE_Type                           U01
#define   GCREG_DE_SRC_TILE_STATUS_ADDRESS_TYPE_SYSTEM                       0x0
#define   GCREG_DE_SRC_TILE_STATUS_ADDRESS_TYPE_VIRTUAL_SYSTEM               0x1

#define GCREG_DE_SRC_TILE_STATUS_ADDRESS_ADDRESS                            30:0
#define GCREG_DE_SRC_TILE_STATUS_ADDRESS_ADDRESS_End                          30
#define GCREG_DE_SRC_TILE_STATUS_ADDRESS_ADDRESS_Start                         0
#define GCREG_DE_SRC_TILE_STATUS_ADDRESS_ADDRESS_Type                        U31

#define gcreg3rdPartyCompressionSrcConfigRegAddrs                         0x4BC0
#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_Address                    0x12F00
#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_MSB                             15
#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_LSB                              3
#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_BLK                              0
#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_Count                            8
#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FieldMask               0x000000FF
#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_ReadMask                0x000000FF
#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_WriteMask               0x000000FF
#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_ResetValue              0x00000000

#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_TPC                            0:0
#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_TPC_End                          0
#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_TPC_Start                        0
#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_TPC_Type                       U01
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_TPC_DISABLED                 0x0
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_TPC_ENABLED                  0x1

#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_MASK_TPC                       1:1
#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_MASK_TPC_End                     1
#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_MASK_TPC_Start                   1
#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_MASK_TPC_Type                  U01
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_MASK_TPC_ENABLED             0x0
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_MASK_TPC_MASKED              0x1

#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT                         6:2
#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_End                       6
#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_Start                     2
#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_Type                    U05
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_X4R4G4B4             0x00
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_A4R4G4B4             0x01
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_X1R5G5B5             0x02
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_A1R5G5B5             0x03
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_R5G6B5               0x04
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_X8R8G8B8             0x05
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_A8R8G8B8             0x06
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_YUY2                 0x07
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_UYVY                 0x08
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_INDEX8               0x09
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_MONOCHROME           0x0A
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_AYUV                 0x0B
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_RGB888_PLANAR        0x0C
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_RGB888_PACKED        0x0D
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_EXTENSION            0x0E
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_YV12                 0x0F
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_A8                   0x10
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_NV12                 0x11
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_NV16                 0x12
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_RG16                 0x13
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_R8                   0x14
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_NV12_10BIT           0x15
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_A2R10G10B10          0x16
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_NV16_10BIT           0x17
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_INDEX1               0x18
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_INDEX2               0x19
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_INDEX4               0x1A
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_P010                 0x1B
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_NV12_10BIT_L1        0x1C
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_NV16_10BIT_L1        0x1D
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_FORMAT_I010                 0x1E

#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_MASK_FORMAT                    7:7
#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_MASK_FORMAT_End                  7
#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_MASK_FORMAT_Start                7
#define GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_MASK_FORMAT_Type               U01
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_MASK_FORMAT_ENABLED          0x0
#define   GCREG3RD_PARTY_COMPRESSION_SRC_CONFIG_MASK_FORMAT_MASKED           0x1

#define gcreg3rdPartyCompressionSrcHeaderAddressRegAddrs                  0x4BC8
#define GCREG3RD_PARTY_COMPRESSION_SRC_HEADER_ADDRESS_Address            0x12F20
#define GCREG3RD_PARTY_COMPRESSION_SRC_HEADER_ADDRESS_MSB                     15
#define GCREG3RD_PARTY_COMPRESSION_SRC_HEADER_ADDRESS_LSB                      3
#define GCREG3RD_PARTY_COMPRESSION_SRC_HEADER_ADDRESS_BLK                      0
#define GCREG3RD_PARTY_COMPRESSION_SRC_HEADER_ADDRESS_Count                    8
#define GCREG3RD_PARTY_COMPRESSION_SRC_HEADER_ADDRESS_FieldMask       0xFFFFFFFF
#define GCREG3RD_PARTY_COMPRESSION_SRC_HEADER_ADDRESS_ReadMask        0xFFFFFFFC
#define GCREG3RD_PARTY_COMPRESSION_SRC_HEADER_ADDRESS_WriteMask       0xFFFFFFFC
#define GCREG3RD_PARTY_COMPRESSION_SRC_HEADER_ADDRESS_ResetValue      0x00000000

#define GCREG3RD_PARTY_COMPRESSION_SRC_HEADER_ADDRESS_TYPE                 31:31
#define GCREG3RD_PARTY_COMPRESSION_SRC_HEADER_ADDRESS_TYPE_End                31
#define GCREG3RD_PARTY_COMPRESSION_SRC_HEADER_ADDRESS_TYPE_Start              31
#define GCREG3RD_PARTY_COMPRESSION_SRC_HEADER_ADDRESS_TYPE_Type              U01
#define   GCREG3RD_PARTY_COMPRESSION_SRC_HEADER_ADDRESS_TYPE_SYSTEM          0x0
#define   GCREG3RD_PARTY_COMPRESSION_SRC_HEADER_ADDRESS_TYPE_VIRTUAL_SYSTEM  0x1

#define GCREG3RD_PARTY_COMPRESSION_SRC_HEADER_ADDRESS_ADDRESS               30:0
#define GCREG3RD_PARTY_COMPRESSION_SRC_HEADER_ADDRESS_ADDRESS_End             30
#define GCREG3RD_PARTY_COMPRESSION_SRC_HEADER_ADDRESS_ADDRESS_Start            0
#define GCREG3RD_PARTY_COMPRESSION_SRC_HEADER_ADDRESS_ADDRESS_Type           U31

#define gcreg3rdPartyCompressionSrcPayloadAddressRegAddrs                 0x4BD0
#define GCREG3RD_PARTY_COMPRESSION_SRC_PAYLOAD_ADDRESS_Address           0x12F40
#define GCREG3RD_PARTY_COMPRESSION_SRC_PAYLOAD_ADDRESS_MSB                    15
#define GCREG3RD_PARTY_COMPRESSION_SRC_PAYLOAD_ADDRESS_LSB                     3
#define GCREG3RD_PARTY_COMPRESSION_SRC_PAYLOAD_ADDRESS_BLK                     0
#define GCREG3RD_PARTY_COMPRESSION_SRC_PAYLOAD_ADDRESS_Count                   8
#define GCREG3RD_PARTY_COMPRESSION_SRC_PAYLOAD_ADDRESS_FieldMask      0xFFFFFFFF
#define GCREG3RD_PARTY_COMPRESSION_SRC_PAYLOAD_ADDRESS_ReadMask       0xFFFFFFFC
#define GCREG3RD_PARTY_COMPRESSION_SRC_PAYLOAD_ADDRESS_WriteMask      0xFFFFFFFC
#define GCREG3RD_PARTY_COMPRESSION_SRC_PAYLOAD_ADDRESS_ResetValue     0x00000000

#define GCREG3RD_PARTY_COMPRESSION_SRC_PAYLOAD_ADDRESS_TYPE                31:31
#define GCREG3RD_PARTY_COMPRESSION_SRC_PAYLOAD_ADDRESS_TYPE_End               31
#define GCREG3RD_PARTY_COMPRESSION_SRC_PAYLOAD_ADDRESS_TYPE_Start             31
#define GCREG3RD_PARTY_COMPRESSION_SRC_PAYLOAD_ADDRESS_TYPE_Type             U01
#define   GCREG3RD_PARTY_COMPRESSION_SRC_PAYLOAD_ADDRESS_TYPE_SYSTEM         0x0
#define   GCREG3RD_PARTY_COMPRESSION_SRC_PAYLOAD_ADDRESS_TYPE_VIRTUAL_SYSTEM 0x1

#define GCREG3RD_PARTY_COMPRESSION_SRC_PAYLOAD_ADDRESS_ADDRESS              30:0
#define GCREG3RD_PARTY_COMPRESSION_SRC_PAYLOAD_ADDRESS_ADDRESS_End            30
#define GCREG3RD_PARTY_COMPRESSION_SRC_PAYLOAD_ADDRESS_ADDRESS_Start           0
#define GCREG3RD_PARTY_COMPRESSION_SRC_PAYLOAD_ADDRESS_ADDRESS_Type          U31

#define gcregTPCNV12SrcConfigRegAddrs                                     0x4BD8
#define GCREG_TPCNV12_SRC_CONFIG_Address                                 0x12F60
#define GCREG_TPCNV12_SRC_CONFIG_MSB                                          15
#define GCREG_TPCNV12_SRC_CONFIG_LSB                                           3
#define GCREG_TPCNV12_SRC_CONFIG_BLK                                           0
#define GCREG_TPCNV12_SRC_CONFIG_Count                                         8
#define GCREG_TPCNV12_SRC_CONFIG_FieldMask                            0x0000003F
#define GCREG_TPCNV12_SRC_CONFIG_ReadMask                             0x0000003F
#define GCREG_TPCNV12_SRC_CONFIG_WriteMask                            0x0000003F
#define GCREG_TPCNV12_SRC_CONFIG_ResetValue                           0x00000000

#define GCREG_TPCNV12_SRC_CONFIG_TILE                                        0:0
#define GCREG_TPCNV12_SRC_CONFIG_TILE_End                                      0
#define GCREG_TPCNV12_SRC_CONFIG_TILE_Start                                    0
#define GCREG_TPCNV12_SRC_CONFIG_TILE_Type                                   U01
#define   GCREG_TPCNV12_SRC_CONFIG_TILE_DISABLED                             0x0
#define   GCREG_TPCNV12_SRC_CONFIG_TILE_ENABLED                              0x1

#define GCREG_TPCNV12_SRC_CONFIG_MASK_TILE                                   1:1
#define GCREG_TPCNV12_SRC_CONFIG_MASK_TILE_End                                 1
#define GCREG_TPCNV12_SRC_CONFIG_MASK_TILE_Start                               1
#define GCREG_TPCNV12_SRC_CONFIG_MASK_TILE_Type                              U01
#define   GCREG_TPCNV12_SRC_CONFIG_MASK_TILE_ENABLED                         0x0
#define   GCREG_TPCNV12_SRC_CONFIG_MASK_TILE_MASKED                          0x1

#define GCREG_TPCNV12_SRC_CONFIG_BIT10                                       2:2
#define GCREG_TPCNV12_SRC_CONFIG_BIT10_End                                     2
#define GCREG_TPCNV12_SRC_CONFIG_BIT10_Start                                   2
#define GCREG_TPCNV12_SRC_CONFIG_BIT10_Type                                  U01
#define   GCREG_TPCNV12_SRC_CONFIG_BIT10_DISABLED                            0x0
#define   GCREG_TPCNV12_SRC_CONFIG_BIT10_ENABLED                             0x1

#define GCREG_TPCNV12_SRC_CONFIG_MASK_BIT10                                  3:3
#define GCREG_TPCNV12_SRC_CONFIG_MASK_BIT10_End                                3
#define GCREG_TPCNV12_SRC_CONFIG_MASK_BIT10_Start                              3
#define GCREG_TPCNV12_SRC_CONFIG_MASK_BIT10_Type                             U01
#define   GCREG_TPCNV12_SRC_CONFIG_MASK_BIT10_ENABLED                        0x0
#define   GCREG_TPCNV12_SRC_CONFIG_MASK_BIT10_MASKED                         0x1

#define GCREG_TPCNV12_SRC_CONFIG_COMP                                        4:4
#define GCREG_TPCNV12_SRC_CONFIG_COMP_End                                      4
#define GCREG_TPCNV12_SRC_CONFIG_COMP_Start                                    4
#define GCREG_TPCNV12_SRC_CONFIG_COMP_Type                                   U01
#define   GCREG_TPCNV12_SRC_CONFIG_COMP_DISABLED                             0x0
#define   GCREG_TPCNV12_SRC_CONFIG_COMP_ENABLED                              0x1

#define GCREG_TPCNV12_SRC_CONFIG_MASK_COMP                                   5:5
#define GCREG_TPCNV12_SRC_CONFIG_MASK_COMP_End                                 5
#define GCREG_TPCNV12_SRC_CONFIG_MASK_COMP_Start                               5
#define GCREG_TPCNV12_SRC_CONFIG_MASK_COMP_Type                              U01
#define   GCREG_TPCNV12_SRC_CONFIG_MASK_COMP_ENABLED                         0x0
#define   GCREG_TPCNV12_SRC_CONFIG_MASK_COMP_MASKED                          0x1

#define gcregDESrcOriginFractionRegAddrs                                  0x4BE0
#define GCREG_DE_SRC_ORIGIN_FRACTION_Address                             0x12F80
#define GCREG_DE_SRC_ORIGIN_FRACTION_MSB                                      15
#define GCREG_DE_SRC_ORIGIN_FRACTION_LSB                                       3
#define GCREG_DE_SRC_ORIGIN_FRACTION_BLK                                       0
#define GCREG_DE_SRC_ORIGIN_FRACTION_Count                                     8
#define GCREG_DE_SRC_ORIGIN_FRACTION_FieldMask                        0xFFFFFFFF
#define GCREG_DE_SRC_ORIGIN_FRACTION_ReadMask                         0xFFFFFFFF
#define GCREG_DE_SRC_ORIGIN_FRACTION_WriteMask                        0xFFFFFFFF
#define GCREG_DE_SRC_ORIGIN_FRACTION_ResetValue                       0x00000000

#define GCREG_DE_SRC_ORIGIN_FRACTION_Y                                     31:16
#define GCREG_DE_SRC_ORIGIN_FRACTION_Y_End                                    31
#define GCREG_DE_SRC_ORIGIN_FRACTION_Y_Start                                  16
#define GCREG_DE_SRC_ORIGIN_FRACTION_Y_Type                                  U16

#define GCREG_DE_SRC_ORIGIN_FRACTION_X                                      15:0
#define GCREG_DE_SRC_ORIGIN_FRACTION_X_End                                    15
#define GCREG_DE_SRC_ORIGIN_FRACTION_X_Start                                   0
#define GCREG_DE_SRC_ORIGIN_FRACTION_X_Type                                  U16

#define gcregDESrcTileStatusClearRegAddrs                                 0x4BE8
#define GCREG_DE_SRC_TILE_STATUS_CLEAR_Address                           0x12FA0
#define GCREG_DE_SRC_TILE_STATUS_CLEAR_MSB                                    15
#define GCREG_DE_SRC_TILE_STATUS_CLEAR_LSB                                     3
#define GCREG_DE_SRC_TILE_STATUS_CLEAR_BLK                                     0
#define GCREG_DE_SRC_TILE_STATUS_CLEAR_Count                                   8
#define GCREG_DE_SRC_TILE_STATUS_CLEAR_FieldMask                      0xFFFFFFFF
#define GCREG_DE_SRC_TILE_STATUS_CLEAR_ReadMask                       0xFFFFFFFF
#define GCREG_DE_SRC_TILE_STATUS_CLEAR_WriteMask                      0xFFFFFFFF
#define GCREG_DE_SRC_TILE_STATUS_CLEAR_ResetValue                     0x00000000

#define GCREG_DE_SRC_TILE_STATUS_CLEAR_VALUE                                31:0
#define GCREG_DE_SRC_TILE_STATUS_CLEAR_VALUE_End                              31
#define GCREG_DE_SRC_TILE_STATUS_CLEAR_VALUE_Start                             0
#define GCREG_DE_SRC_TILE_STATUS_CLEAR_VALUE_Type                            U32

#define gcregDESrcTileStatusConfigRegAddrs                                0x4BF0
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_Address                          0x12FC0
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_MSB                                   15
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_LSB                                    3
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_BLK                                    0
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_Count                                  8
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_FieldMask                     0x001C01FF
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_ReadMask                      0x001C01FF
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_WriteMask                     0x001C01FF
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_ResetValue                    0x00000000

#define GCREG_DE_SRC_TILE_STATUS_CONFIG_CONTROL                              1:0
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_CONTROL_End                            1
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_CONTROL_Start                          0
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_CONTROL_Type                         U02

#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_CONTROL_DISABLE                    0x0

#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_CONTROL_ENABLE                     0x1

#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_CONTROL_COMPRESSED                 0x3

#define GCREG_DE_SRC_TILE_STATUS_CONFIG_FAST_CLEAR_EN_C                      2:2
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_FAST_CLEAR_EN_C_End                    2
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_FAST_CLEAR_EN_C_Start                  2
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_FAST_CLEAR_EN_C_Type                 U01
#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_FAST_CLEAR_EN_C_DISABLE            0x0
#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_FAST_CLEAR_EN_C_ENABLE             0x1

#define GCREG_DE_SRC_TILE_STATUS_CONFIG_COLOR64                              3:3
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_COLOR64_End                            3
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_COLOR64_Start                          3
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_COLOR64_Type                         U01
#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_COLOR64_COLOR16OR32                0x0
#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_COLOR64_COLOR64                    0x1

#define GCREG_DE_SRC_TILE_STATUS_CONFIG_FORMAT                               7:4
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_FORMAT_End                             7
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_FORMAT_Start                           4
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_FORMAT_Type                          U04
#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_FORMAT_ARGB4                       0x0
#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_FORMAT_A1RGB5                      0x1
#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_FORMAT_R5G6B5                      0x2
#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_FORMAT_ARGB8                       0x3
#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_FORMAT_RGB8                        0x4
#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_FORMAT_Z24S8                       0x5
#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_FORMAT_Z24                         0x6
#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_FORMAT_VAA16                       0x7
#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_FORMAT_Z16                         0x8
#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_FORMAT_S8                          0x9

#define GCREG_DE_SRC_TILE_STATUS_CONFIG_TILE_STATUS4_BIT                     8:8
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_TILE_STATUS4_BIT_End                   8
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_TILE_STATUS4_BIT_Start                 8
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_TILE_STATUS4_BIT_Type                U01
#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_TILE_STATUS4_BIT_DISABLED          0x0
#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_TILE_STATUS4_BIT_ENABLED           0x1

#define GCREG_DE_SRC_TILE_STATUS_CONFIG_MSAA_COMPRESSION                   18:18
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_MSAA_COMPRESSION_End                  18
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_MSAA_COMPRESSION_Start                18
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_MSAA_COMPRESSION_Type                U01
#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_MSAA_COMPRESSION_DISABLE           0x0
#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_MSAA_COMPRESSION_ENABLE            0x1

#define GCREG_DE_SRC_TILE_STATUS_CONFIG_TILE_MAJOR_TYPE                    19:19
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_TILE_MAJOR_TYPE_End                   19
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_TILE_MAJOR_TYPE_Start                 19
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_TILE_MAJOR_TYPE_Type                 U01
#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_TILE_MAJOR_TYPE_XMAJOR             0x0
#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_TILE_MAJOR_TYPE_YMAJOR             0x1

#define GCREG_DE_SRC_TILE_STATUS_CONFIG_COMPRESSION_SIZE                   20:20
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_COMPRESSION_SIZE_End                  20
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_COMPRESSION_SIZE_Start                20
#define GCREG_DE_SRC_TILE_STATUS_CONFIG_COMPRESSION_SIZE_Type                U01
#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_COMPRESSION_SIZE_Size128B          0x0
#define   GCREG_DE_SRC_TILE_STATUS_CONFIG_COMPRESSION_SIZE_Size256B          0x1

#define gcregDESrcTileStatusClear64RegAddrs                               0x4BF8
#define GCREG_DE_SRC_TILE_STATUS_CLEAR64_Address                         0x12FE0
#define GCREG_DE_SRC_TILE_STATUS_CLEAR64_MSB                                  15
#define GCREG_DE_SRC_TILE_STATUS_CLEAR64_LSB                                   3
#define GCREG_DE_SRC_TILE_STATUS_CLEAR64_BLK                                   0
#define GCREG_DE_SRC_TILE_STATUS_CLEAR64_Count                                 8
#define GCREG_DE_SRC_TILE_STATUS_CLEAR64_FieldMask                    0xFFFFFFFF
#define GCREG_DE_SRC_TILE_STATUS_CLEAR64_ReadMask                     0xFFFFFFFF
#define GCREG_DE_SRC_TILE_STATUS_CLEAR64_WriteMask                    0xFFFFFFFF
#define GCREG_DE_SRC_TILE_STATUS_CLEAR64_ResetValue                   0x00000000

#define GCREG_DE_SRC_TILE_STATUS_CLEAR64_VALUE                              31:0
#define GCREG_DE_SRC_TILE_STATUS_CLEAR64_VALUE_End                            31
#define GCREG_DE_SRC_TILE_STATUS_CLEAR64_VALUE_Start                           0
#define GCREG_DE_SRC_TILE_STATUS_CLEAR64_VALUE_Type                          U32

#define gcregDEDegammaLookupRegAddrs                                      0x4E00
#define GCREG_DE_DEGAMMA_LOOKUP_Address                                  0x13800
#define GCREG_DE_DEGAMMA_LOOKUP_MSB                                           15
#define GCREG_DE_DEGAMMA_LOOKUP_LSB                                            8
#define GCREG_DE_DEGAMMA_LOOKUP_BLK                                            8
#define GCREG_DE_DEGAMMA_LOOKUP_Count                                        256
#define GCREG_DE_DEGAMMA_LOOKUP_FieldMask                             0x00FFFFFF
#define GCREG_DE_DEGAMMA_LOOKUP_ReadMask                              0x00FFFFFF
#define GCREG_DE_DEGAMMA_LOOKUP_WriteMask                             0x00FFFFFF
#define GCREG_DE_DEGAMMA_LOOKUP_ResetValue                            0x00000000

#define GCREG_DE_DEGAMMA_LOOKUP_RED                                          7:0
#define GCREG_DE_DEGAMMA_LOOKUP_RED_End                                        7
#define GCREG_DE_DEGAMMA_LOOKUP_RED_Start                                      0
#define GCREG_DE_DEGAMMA_LOOKUP_RED_Type                                     U08

#define GCREG_DE_DEGAMMA_LOOKUP_GREEN                                       15:8
#define GCREG_DE_DEGAMMA_LOOKUP_GREEN_End                                     15
#define GCREG_DE_DEGAMMA_LOOKUP_GREEN_Start                                    8
#define GCREG_DE_DEGAMMA_LOOKUP_GREEN_Type                                   U08

#define GCREG_DE_DEGAMMA_LOOKUP_BLUE                                       23:16
#define GCREG_DE_DEGAMMA_LOOKUP_BLUE_End                                      23
#define GCREG_DE_DEGAMMA_LOOKUP_BLUE_Start                                    16
#define GCREG_DE_DEGAMMA_LOOKUP_BLUE_Type                                    U08

#define gcregDEGammaLookupRegAddrs                                        0x4F00
#define GCREG_DE_GAMMA_LOOKUP_Address                                    0x13C00
#define GCREG_DE_GAMMA_LOOKUP_MSB                                             15
#define GCREG_DE_GAMMA_LOOKUP_LSB                                              8
#define GCREG_DE_GAMMA_LOOKUP_BLK                                              8
#define GCREG_DE_GAMMA_LOOKUP_Count                                          256
#define GCREG_DE_GAMMA_LOOKUP_FieldMask                               0x3FFFFFFF
#define GCREG_DE_GAMMA_LOOKUP_ReadMask                                0x3FFFFFFF
#define GCREG_DE_GAMMA_LOOKUP_WriteMask                               0x3FFFFFFF
#define GCREG_DE_GAMMA_LOOKUP_ResetValue                              0x00000000

#define GCREG_DE_GAMMA_LOOKUP_RED                                            9:0
#define GCREG_DE_GAMMA_LOOKUP_RED_End                                          9
#define GCREG_DE_GAMMA_LOOKUP_RED_Start                                        0
#define GCREG_DE_GAMMA_LOOKUP_RED_Type                                       U10

#define GCREG_DE_GAMMA_LOOKUP_GREEN                                        19:10
#define GCREG_DE_GAMMA_LOOKUP_GREEN_End                                       19
#define GCREG_DE_GAMMA_LOOKUP_GREEN_Start                                     10
#define GCREG_DE_GAMMA_LOOKUP_GREEN_Type                                     U10

#define GCREG_DE_GAMMA_LOOKUP_BLUE                                         29:20
#define GCREG_DE_GAMMA_LOOKUP_BLUE_End                                        29
#define GCREG_DE_GAMMA_LOOKUP_BLUE_Start                                      20
#define GCREG_DE_GAMMA_LOOKUP_BLUE_Type                                      U10

#define gcregDEDegammaLookupExRegAddrs                                    0x7400
#define GCREG_DE_DEGAMMA_LOOKUP_EX_Address                               0x1D000
#define GCREG_DE_DEGAMMA_LOOKUP_EX_MSB                                        15
#define GCREG_DE_DEGAMMA_LOOKUP_EX_LSB                                        10
#define GCREG_DE_DEGAMMA_LOOKUP_EX_BLK                                        10
#define GCREG_DE_DEGAMMA_LOOKUP_EX_Count                                    1024
#define GCREG_DE_DEGAMMA_LOOKUP_EX_FieldMask                          0x3FFFFFFF
#define GCREG_DE_DEGAMMA_LOOKUP_EX_ReadMask                           0x3FFFFFFF
#define GCREG_DE_DEGAMMA_LOOKUP_EX_WriteMask                          0x3FFFFFFF
#define GCREG_DE_DEGAMMA_LOOKUP_EX_ResetValue                         0x00000000

#define GCREG_DE_DEGAMMA_LOOKUP_EX_RED                                       9:0
#define GCREG_DE_DEGAMMA_LOOKUP_EX_RED_End                                     9
#define GCREG_DE_DEGAMMA_LOOKUP_EX_RED_Start                                   0
#define GCREG_DE_DEGAMMA_LOOKUP_EX_RED_Type                                  U10

#define GCREG_DE_DEGAMMA_LOOKUP_EX_GREEN                                   19:10
#define GCREG_DE_DEGAMMA_LOOKUP_EX_GREEN_End                                  19
#define GCREG_DE_DEGAMMA_LOOKUP_EX_GREEN_Start                                10
#define GCREG_DE_DEGAMMA_LOOKUP_EX_GREEN_Type                                U10

#define GCREG_DE_DEGAMMA_LOOKUP_EX_BLUE                                    29:20
#define GCREG_DE_DEGAMMA_LOOKUP_EX_BLUE_End                                   29
#define GCREG_DE_DEGAMMA_LOOKUP_EX_BLUE_Start                                 20
#define GCREG_DE_DEGAMMA_LOOKUP_EX_BLUE_Type                                 U10

#define gcregDEAdjustSatGammaLUTRegAddrs                                  0x6600
#define GCREG_DE_ADJUST_SAT_GAMMA_LUT_Address                            0x19800
#define GCREG_DE_ADJUST_SAT_GAMMA_LUT_MSB                                     15
#define GCREG_DE_ADJUST_SAT_GAMMA_LUT_LSB                                      8
#define GCREG_DE_ADJUST_SAT_GAMMA_LUT_BLK                                      8
#define GCREG_DE_ADJUST_SAT_GAMMA_LUT_Count                                  256
#define GCREG_DE_ADJUST_SAT_GAMMA_LUT_FieldMask                       0xFFFFFFFF
#define GCREG_DE_ADJUST_SAT_GAMMA_LUT_ReadMask                        0xFFFFFFFF
#define GCREG_DE_ADJUST_SAT_GAMMA_LUT_WriteMask                       0xFFFFFFFF
#define GCREG_DE_ADJUST_SAT_GAMMA_LUT_ResetValue                      0x00000000

#define GCREG_DE_ADJUST_SAT_GAMMA_LUT_VALUE                                 31:0
#define GCREG_DE_ADJUST_SAT_GAMMA_LUT_VALUE_End                               31
#define GCREG_DE_ADJUST_SAT_GAMMA_LUT_VALUE_Start                              0
#define GCREG_DE_ADJUST_SAT_GAMMA_LUT_VALUE_Type                             U32

#define gcregDEAdjustValueGammaLUTRegAddrs                                0x6700
#define GCREG_DE_ADJUST_VALUE_GAMMA_LUT_Address                          0x19C00
#define GCREG_DE_ADJUST_VALUE_GAMMA_LUT_MSB                                   15
#define GCREG_DE_ADJUST_VALUE_GAMMA_LUT_LSB                                    8
#define GCREG_DE_ADJUST_VALUE_GAMMA_LUT_BLK                                    8
#define GCREG_DE_ADJUST_VALUE_GAMMA_LUT_Count                                256
#define GCREG_DE_ADJUST_VALUE_GAMMA_LUT_FieldMask                     0xFFFFFFFF
#define GCREG_DE_ADJUST_VALUE_GAMMA_LUT_ReadMask                      0xFFFFFFFF
#define GCREG_DE_ADJUST_VALUE_GAMMA_LUT_WriteMask                     0xFFFFFFFF
#define GCREG_DE_ADJUST_VALUE_GAMMA_LUT_ResetValue                    0x00000000

#define GCREG_DE_ADJUST_VALUE_GAMMA_LUT_VALUE                               31:0
#define GCREG_DE_ADJUST_VALUE_GAMMA_LUT_VALUE_End                             31
#define GCREG_DE_ADJUST_VALUE_GAMMA_LUT_VALUE_Start                            0
#define GCREG_DE_ADJUST_VALUE_GAMMA_LUT_VALUE_Type                           U32

#define gcregDEAdjustControlRegAddrs                                      0x0570
#define GCREG_DE_ADJUST_CONTROL_Address                                  0x015C0
#define GCREG_DE_ADJUST_CONTROL_MSB                                           15
#define GCREG_DE_ADJUST_CONTROL_LSB                                            0
#define GCREG_DE_ADJUST_CONTROL_BLK                                            0
#define GCREG_DE_ADJUST_CONTROL_Count                                          1
#define GCREG_DE_ADJUST_CONTROL_FieldMask                             0x00000733
#define GCREG_DE_ADJUST_CONTROL_ReadMask                              0x00000733
#define GCREG_DE_ADJUST_CONTROL_WriteMask                             0x00000733
#define GCREG_DE_ADJUST_CONTROL_ResetValue                            0x00000000

#define GCREG_DE_ADJUST_CONTROL_MODE                                         1:0
#define GCREG_DE_ADJUST_CONTROL_MODE_End                                       1
#define GCREG_DE_ADJUST_CONTROL_MODE_Start                                     0
#define GCREG_DE_ADJUST_CONTROL_MODE_Type                                    U02
#define   GCREG_DE_ADJUST_CONTROL_MODE_BYPASS                                0x0
#define   GCREG_DE_ADJUST_CONTROL_MODE_LINEAR_ENABLE                         0x1
#define   GCREG_DE_ADJUST_CONTROL_MODE_GAMMA_ENABLE                          0x2
#define   GCREG_DE_ADJUST_CONTROL_MODE_CONVERT_ONLY                          0x3

#define GCREG_DE_ADJUST_CONTROL_SRC_FORMAT                                   5:4
#define GCREG_DE_ADJUST_CONTROL_SRC_FORMAT_End                                 5
#define GCREG_DE_ADJUST_CONTROL_SRC_FORMAT_Start                               4
#define GCREG_DE_ADJUST_CONTROL_SRC_FORMAT_Type                              U02
#define   GCREG_DE_ADJUST_CONTROL_SRC_FORMAT_RGB8                            0x0
#define   GCREG_DE_ADJUST_CONTROL_SRC_FORMAT_HSV8                            0x1
#define   GCREG_DE_ADJUST_CONTROL_SRC_FORMAT_HLS8                            0x2

#define GCREG_DE_ADJUST_CONTROL_DST_FORMAT                                  10:8
#define GCREG_DE_ADJUST_CONTROL_DST_FORMAT_End                                10
#define GCREG_DE_ADJUST_CONTROL_DST_FORMAT_Start                               8
#define GCREG_DE_ADJUST_CONTROL_DST_FORMAT_Type                              U03
#define   GCREG_DE_ADJUST_CONTROL_DST_FORMAT_RGB8                            0x0
#define   GCREG_DE_ADJUST_CONTROL_DST_FORMAT_HSV8                            0x1
#define   GCREG_DE_ADJUST_CONTROL_DST_FORMAT_HLS8                            0x2
#define   GCREG_DE_ADJUST_CONTROL_DST_FORMAT_HSV32                           0x3
#define   GCREG_DE_ADJUST_CONTROL_DST_FORMAT_HLS32                           0x4

#define gcregDEAdjustFactor1RegAddrs                                      0x0571
#define GCREG_DE_ADJUST_FACTOR1_Address                                  0x015C4
#define GCREG_DE_ADJUST_FACTOR1_MSB                                           15
#define GCREG_DE_ADJUST_FACTOR1_LSB                                            0
#define GCREG_DE_ADJUST_FACTOR1_BLK                                            0
#define GCREG_DE_ADJUST_FACTOR1_Count                                          1
#define GCREG_DE_ADJUST_FACTOR1_FieldMask                             0xFFFFFFFF
#define GCREG_DE_ADJUST_FACTOR1_ReadMask                              0xFFFFFFFF
#define GCREG_DE_ADJUST_FACTOR1_WriteMask                             0xFFFFFFFF
#define GCREG_DE_ADJUST_FACTOR1_ResetValue                            0x00000000

#define GCREG_DE_ADJUST_FACTOR1_VALUE                                       31:0
#define GCREG_DE_ADJUST_FACTOR1_VALUE_End                                     31
#define GCREG_DE_ADJUST_FACTOR1_VALUE_Start                                    0
#define GCREG_DE_ADJUST_FACTOR1_VALUE_Type                                   U32

#define gcregDEAdjustFactor2RegAddrs                                      0x0572
#define GCREG_DE_ADJUST_FACTOR2_Address                                  0x015C8
#define GCREG_DE_ADJUST_FACTOR2_MSB                                           15
#define GCREG_DE_ADJUST_FACTOR2_LSB                                            0
#define GCREG_DE_ADJUST_FACTOR2_BLK                                            0
#define GCREG_DE_ADJUST_FACTOR2_Count                                          1
#define GCREG_DE_ADJUST_FACTOR2_FieldMask                             0xFFFFFFFF
#define GCREG_DE_ADJUST_FACTOR2_ReadMask                              0xFFFFFFFF
#define GCREG_DE_ADJUST_FACTOR2_WriteMask                             0xFFFFFFFF
#define GCREG_DE_ADJUST_FACTOR2_ResetValue                            0x00000000

#define GCREG_DE_ADJUST_FACTOR2_VALUE                                       31:0
#define GCREG_DE_ADJUST_FACTOR2_VALUE_End                                     31
#define GCREG_DE_ADJUST_FACTOR2_VALUE_Start                                    0
#define GCREG_DE_ADJUST_FACTOR2_VALUE_Type                                   U32

#define gcregDEHistControlRegAddrs                                        0x0573
#define GCREG_DE_HIST_CONTROL_Address                                    0x015CC
#define GCREG_DE_HIST_CONTROL_MSB                                             15
#define GCREG_DE_HIST_CONTROL_LSB                                              0
#define GCREG_DE_HIST_CONTROL_BLK                                              0
#define GCREG_DE_HIST_CONTROL_Count                                            1
#define GCREG_DE_HIST_CONTROL_FieldMask                               0x01FFFFFF
#define GCREG_DE_HIST_CONTROL_ReadMask                                0x01FFFFFF
#define GCREG_DE_HIST_CONTROL_WriteMask                               0x01FFFFFF
#define GCREG_DE_HIST_CONTROL_ResetValue                              0x00000000

#define GCREG_DE_HIST_CONTROL_ENABLE_RED_HIST_CALC                           0:0
#define GCREG_DE_HIST_CONTROL_ENABLE_RED_HIST_CALC_End                         0
#define GCREG_DE_HIST_CONTROL_ENABLE_RED_HIST_CALC_Start                       0
#define GCREG_DE_HIST_CONTROL_ENABLE_RED_HIST_CALC_Type                      U01
#define   GCREG_DE_HIST_CONTROL_ENABLE_RED_HIST_CALC_DISABLE                 0x0
#define   GCREG_DE_HIST_CONTROL_ENABLE_RED_HIST_CALC_ENABLE                  0x1

#define GCREG_DE_HIST_CONTROL_ENABLE_RED_GRADIENT_HIST_CALC                  1:1
#define GCREG_DE_HIST_CONTROL_ENABLE_RED_GRADIENT_HIST_CALC_End                1
#define GCREG_DE_HIST_CONTROL_ENABLE_RED_GRADIENT_HIST_CALC_Start              1
#define GCREG_DE_HIST_CONTROL_ENABLE_RED_GRADIENT_HIST_CALC_Type             U01
#define   GCREG_DE_HIST_CONTROL_ENABLE_RED_GRADIENT_HIST_CALC_DISABLE        0x0
#define   GCREG_DE_HIST_CONTROL_ENABLE_RED_GRADIENT_HIST_CALC_ENABLE         0x1

#define GCREG_DE_HIST_CONTROL_ENABLE_GREEN_HIST_CALC                         2:2
#define GCREG_DE_HIST_CONTROL_ENABLE_GREEN_HIST_CALC_End                       2
#define GCREG_DE_HIST_CONTROL_ENABLE_GREEN_HIST_CALC_Start                     2
#define GCREG_DE_HIST_CONTROL_ENABLE_GREEN_HIST_CALC_Type                    U01
#define   GCREG_DE_HIST_CONTROL_ENABLE_GREEN_HIST_CALC_DISABLE               0x0
#define   GCREG_DE_HIST_CONTROL_ENABLE_GREEN_HIST_CALC_ENABLE                0x1

#define GCREG_DE_HIST_CONTROL_ENABLE_GREEN_GRADIENT_HIST_CALC                3:3
#define GCREG_DE_HIST_CONTROL_ENABLE_GREEN_GRADIENT_HIST_CALC_End              3
#define GCREG_DE_HIST_CONTROL_ENABLE_GREEN_GRADIENT_HIST_CALC_Start            3
#define GCREG_DE_HIST_CONTROL_ENABLE_GREEN_GRADIENT_HIST_CALC_Type           U01
#define   GCREG_DE_HIST_CONTROL_ENABLE_GREEN_GRADIENT_HIST_CALC_DISABLE      0x0
#define   GCREG_DE_HIST_CONTROL_ENABLE_GREEN_GRADIENT_HIST_CALC_ENABLE       0x1

#define GCREG_DE_HIST_CONTROL_ENABLE_BLUE_HIST_CALC                          4:4
#define GCREG_DE_HIST_CONTROL_ENABLE_BLUE_HIST_CALC_End                        4
#define GCREG_DE_HIST_CONTROL_ENABLE_BLUE_HIST_CALC_Start                      4
#define GCREG_DE_HIST_CONTROL_ENABLE_BLUE_HIST_CALC_Type                     U01
#define   GCREG_DE_HIST_CONTROL_ENABLE_BLUE_HIST_CALC_DISABLE                0x0
#define   GCREG_DE_HIST_CONTROL_ENABLE_BLUE_HIST_CALC_ENABLE                 0x1

#define GCREG_DE_HIST_CONTROL_ENABLE_BLUE_GRADIENT_HIST_CALC                 5:5
#define GCREG_DE_HIST_CONTROL_ENABLE_BLUE_GRADIENT_HIST_CALC_End               5
#define GCREG_DE_HIST_CONTROL_ENABLE_BLUE_GRADIENT_HIST_CALC_Start             5
#define GCREG_DE_HIST_CONTROL_ENABLE_BLUE_GRADIENT_HIST_CALC_Type            U01
#define   GCREG_DE_HIST_CONTROL_ENABLE_BLUE_GRADIENT_HIST_CALC_DISABLE       0x0
#define   GCREG_DE_HIST_CONTROL_ENABLE_BLUE_GRADIENT_HIST_CALC_ENABLE        0x1

#define GCREG_DE_HIST_CONTROL_ENABLE_GRAY_HIST_CALC                          6:6
#define GCREG_DE_HIST_CONTROL_ENABLE_GRAY_HIST_CALC_End                        6
#define GCREG_DE_HIST_CONTROL_ENABLE_GRAY_HIST_CALC_Start                      6
#define GCREG_DE_HIST_CONTROL_ENABLE_GRAY_HIST_CALC_Type                     U01
#define   GCREG_DE_HIST_CONTROL_ENABLE_GRAY_HIST_CALC_DISABLE                0x0
#define   GCREG_DE_HIST_CONTROL_ENABLE_GRAY_HIST_CALC_ENABLE                 0x1

#define GCREG_DE_HIST_CONTROL_ENABLE_GRAY_GRADIENT_HIST_CALC                 7:7
#define GCREG_DE_HIST_CONTROL_ENABLE_GRAY_GRADIENT_HIST_CALC_End               7
#define GCREG_DE_HIST_CONTROL_ENABLE_GRAY_GRADIENT_HIST_CALC_Start             7
#define GCREG_DE_HIST_CONTROL_ENABLE_GRAY_GRADIENT_HIST_CALC_Type            U01
#define   GCREG_DE_HIST_CONTROL_ENABLE_GRAY_GRADIENT_HIST_CALC_DISABLE       0x0
#define   GCREG_DE_HIST_CONTROL_ENABLE_GRAY_GRADIENT_HIST_CALC_ENABLE        0x1

#define GCREG_DE_HIST_CONTROL_ENABLE_RED_HIST_EQUAL                          8:8
#define GCREG_DE_HIST_CONTROL_ENABLE_RED_HIST_EQUAL_End                        8
#define GCREG_DE_HIST_CONTROL_ENABLE_RED_HIST_EQUAL_Start                      8
#define GCREG_DE_HIST_CONTROL_ENABLE_RED_HIST_EQUAL_Type                     U01
#define   GCREG_DE_HIST_CONTROL_ENABLE_RED_HIST_EQUAL_DISABLE                0x0
#define   GCREG_DE_HIST_CONTROL_ENABLE_RED_HIST_EQUAL_ENABLE                 0x1

#define GCREG_DE_HIST_CONTROL_ENABLE_GREEN_HIST_EQUAL                        9:9
#define GCREG_DE_HIST_CONTROL_ENABLE_GREEN_HIST_EQUAL_End                      9
#define GCREG_DE_HIST_CONTROL_ENABLE_GREEN_HIST_EQUAL_Start                    9
#define GCREG_DE_HIST_CONTROL_ENABLE_GREEN_HIST_EQUAL_Type                   U01
#define   GCREG_DE_HIST_CONTROL_ENABLE_GREEN_HIST_EQUAL_DISABLE              0x0
#define   GCREG_DE_HIST_CONTROL_ENABLE_GREEN_HIST_EQUAL_ENABLE               0x1

#define GCREG_DE_HIST_CONTROL_ENABLE_BLUE_HIST_EQUAL                       10:10
#define GCREG_DE_HIST_CONTROL_ENABLE_BLUE_HIST_EQUAL_End                      10
#define GCREG_DE_HIST_CONTROL_ENABLE_BLUE_HIST_EQUAL_Start                    10
#define GCREG_DE_HIST_CONTROL_ENABLE_BLUE_HIST_EQUAL_Type                    U01
#define   GCREG_DE_HIST_CONTROL_ENABLE_BLUE_HIST_EQUAL_DISABLE               0x0
#define   GCREG_DE_HIST_CONTROL_ENABLE_BLUE_HIST_EQUAL_ENABLE                0x1

#define GCREG_DE_HIST_CONTROL_ENABLE_GRAY_HIST_EQUAL                       11:11
#define GCREG_DE_HIST_CONTROL_ENABLE_GRAY_HIST_EQUAL_End                      11
#define GCREG_DE_HIST_CONTROL_ENABLE_GRAY_HIST_EQUAL_Start                    11
#define GCREG_DE_HIST_CONTROL_ENABLE_GRAY_HIST_EQUAL_Type                    U01
#define   GCREG_DE_HIST_CONTROL_ENABLE_GRAY_HIST_EQUAL_DISABLE               0x0
#define   GCREG_DE_HIST_CONTROL_ENABLE_GRAY_HIST_EQUAL_ENABLE                0x1

#define GCREG_DE_HIST_CONTROL_RESET_HIST_COUNTER                           12:12
#define GCREG_DE_HIST_CONTROL_RESET_HIST_COUNTER_End                          12
#define GCREG_DE_HIST_CONTROL_RESET_HIST_COUNTER_Start                        12
#define GCREG_DE_HIST_CONTROL_RESET_HIST_COUNTER_Type                        U01
#define   GCREG_DE_HIST_CONTROL_RESET_HIST_COUNTER_DISABLE                   0x0
#define   GCREG_DE_HIST_CONTROL_RESET_HIST_COUNTER_ENABLE                    0x1

#define GCREG_DE_HIST_CONTROL_BIN_STEP                                     21:13
#define GCREG_DE_HIST_CONTROL_BIN_STEP_End                                    21
#define GCREG_DE_HIST_CONTROL_BIN_STEP_Start                                  13
#define GCREG_DE_HIST_CONTROL_BIN_STEP_Type                                  U09

#define GCREG_DE_HIST_CONTROL_HIST_CALC_MEMORY_MODE                        22:22
#define GCREG_DE_HIST_CONTROL_HIST_CALC_MEMORY_MODE_End                       22
#define GCREG_DE_HIST_CONTROL_HIST_CALC_MEMORY_MODE_Start                     22
#define GCREG_DE_HIST_CONTROL_HIST_CALC_MEMORY_MODE_Type                     U01
#define   GCREG_DE_HIST_CONTROL_HIST_CALC_MEMORY_MODE_PLANAR                 0x0
#define   GCREG_DE_HIST_CONTROL_HIST_CALC_MEMORY_MODE_PACKED                 0x1

#define GCREG_DE_HIST_CONTROL_HIST_EQUAL_OPT                               23:23
#define GCREG_DE_HIST_CONTROL_HIST_EQUAL_OPT_End                              23
#define GCREG_DE_HIST_CONTROL_HIST_EQUAL_OPT_Start                            23
#define GCREG_DE_HIST_CONTROL_HIST_EQUAL_OPT_Type                            U01
#define   GCREG_DE_HIST_CONTROL_HIST_EQUAL_OPT_FIRST                         0x0
#define   GCREG_DE_HIST_CONTROL_HIST_EQUAL_OPT_SECOND                        0x1

#define GCREG_DE_HIST_CONTROL_HIST_EQUAL_OUTPUT                            24:24
#define GCREG_DE_HIST_CONTROL_HIST_EQUAL_OUTPUT_End                           24
#define GCREG_DE_HIST_CONTROL_HIST_EQUAL_OUTPUT_Start                         24
#define GCREG_DE_HIST_CONTROL_HIST_EQUAL_OUTPUT_Type                         U01
#define   GCREG_DE_HIST_CONTROL_HIST_EQUAL_OUTPUT_DISABLE                    0x0
#define   GCREG_DE_HIST_CONTROL_HIST_EQUAL_OUTPUT_ENABLE                     0x1

#define gcregDEHistEqualMNRegAddrs                                        0x0574
#define GCREG_DE_HIST_EQUAL_MN_Address                                   0x015D0
#define GCREG_DE_HIST_EQUAL_MN_MSB                                            15
#define GCREG_DE_HIST_EQUAL_MN_LSB                                             0
#define GCREG_DE_HIST_EQUAL_MN_BLK                                             0
#define GCREG_DE_HIST_EQUAL_MN_Count                                           1
#define GCREG_DE_HIST_EQUAL_MN_FieldMask                              0xFFFFFFFF
#define GCREG_DE_HIST_EQUAL_MN_ReadMask                               0xFFFFFFFF
#define GCREG_DE_HIST_EQUAL_MN_WriteMask                              0xFFFFFFFF
#define GCREG_DE_HIST_EQUAL_MN_ResetValue                             0x00000000

#define GCREG_DE_HIST_EQUAL_MN_VALUE                                        31:0
#define GCREG_DE_HIST_EQUAL_MN_VALUE_End                                      31
#define GCREG_DE_HIST_EQUAL_MN_VALUE_Start                                     0
#define GCREG_DE_HIST_EQUAL_MN_VALUE_Type                                    U32

#define gcregDEHistSobelMatrixGx0RegAddrs                                 0x0575
#define GCREG_DE_HIST_SOBEL_MATRIX_GX0_Address                           0x015D4
#define GCREG_DE_HIST_SOBEL_MATRIX_GX0_MSB                                    15
#define GCREG_DE_HIST_SOBEL_MATRIX_GX0_LSB                                     0
#define GCREG_DE_HIST_SOBEL_MATRIX_GX0_BLK                                     0
#define GCREG_DE_HIST_SOBEL_MATRIX_GX0_Count                                   1
#define GCREG_DE_HIST_SOBEL_MATRIX_GX0_FieldMask                      0x00FFFFFF
#define GCREG_DE_HIST_SOBEL_MATRIX_GX0_ReadMask                       0x00FFFFFF
#define GCREG_DE_HIST_SOBEL_MATRIX_GX0_WriteMask                      0x00FFFFFF
#define GCREG_DE_HIST_SOBEL_MATRIX_GX0_ResetValue                     0x00000000

#define GCREG_DE_HIST_SOBEL_MATRIX_GX0_PARAM0                                7:0
#define GCREG_DE_HIST_SOBEL_MATRIX_GX0_PARAM0_End                              7
#define GCREG_DE_HIST_SOBEL_MATRIX_GX0_PARAM0_Start                            0
#define GCREG_DE_HIST_SOBEL_MATRIX_GX0_PARAM0_Type                           U08

#define GCREG_DE_HIST_SOBEL_MATRIX_GX0_PARAM1                               15:8
#define GCREG_DE_HIST_SOBEL_MATRIX_GX0_PARAM1_End                             15
#define GCREG_DE_HIST_SOBEL_MATRIX_GX0_PARAM1_Start                            8
#define GCREG_DE_HIST_SOBEL_MATRIX_GX0_PARAM1_Type                           U08

#define GCREG_DE_HIST_SOBEL_MATRIX_GX0_PARAM2                              23:16
#define GCREG_DE_HIST_SOBEL_MATRIX_GX0_PARAM2_End                             23
#define GCREG_DE_HIST_SOBEL_MATRIX_GX0_PARAM2_Start                           16
#define GCREG_DE_HIST_SOBEL_MATRIX_GX0_PARAM2_Type                           U08

#define gcregDEHistSobelMatrixGx1RegAddrs                                 0x0576
#define GCREG_DE_HIST_SOBEL_MATRIX_GX1_Address                           0x015D8
#define GCREG_DE_HIST_SOBEL_MATRIX_GX1_MSB                                    15
#define GCREG_DE_HIST_SOBEL_MATRIX_GX1_LSB                                     0
#define GCREG_DE_HIST_SOBEL_MATRIX_GX1_BLK                                     0
#define GCREG_DE_HIST_SOBEL_MATRIX_GX1_Count                                   1
#define GCREG_DE_HIST_SOBEL_MATRIX_GX1_FieldMask                      0x00FFFFFF
#define GCREG_DE_HIST_SOBEL_MATRIX_GX1_ReadMask                       0x00FFFFFF
#define GCREG_DE_HIST_SOBEL_MATRIX_GX1_WriteMask                      0x00FFFFFF
#define GCREG_DE_HIST_SOBEL_MATRIX_GX1_ResetValue                     0x00000000

#define GCREG_DE_HIST_SOBEL_MATRIX_GX1_PARAM3                                7:0
#define GCREG_DE_HIST_SOBEL_MATRIX_GX1_PARAM3_End                              7
#define GCREG_DE_HIST_SOBEL_MATRIX_GX1_PARAM3_Start                            0
#define GCREG_DE_HIST_SOBEL_MATRIX_GX1_PARAM3_Type                           U08

#define GCREG_DE_HIST_SOBEL_MATRIX_GX1_PARAM4                               15:8
#define GCREG_DE_HIST_SOBEL_MATRIX_GX1_PARAM4_End                             15
#define GCREG_DE_HIST_SOBEL_MATRIX_GX1_PARAM4_Start                            8
#define GCREG_DE_HIST_SOBEL_MATRIX_GX1_PARAM4_Type                           U08

#define GCREG_DE_HIST_SOBEL_MATRIX_GX1_PARAM5                              23:16
#define GCREG_DE_HIST_SOBEL_MATRIX_GX1_PARAM5_End                             23
#define GCREG_DE_HIST_SOBEL_MATRIX_GX1_PARAM5_Start                           16
#define GCREG_DE_HIST_SOBEL_MATRIX_GX1_PARAM5_Type                           U08

#define gcregDEHistSobelMatrixGx2RegAddrs                                 0x0577
#define GCREG_DE_HIST_SOBEL_MATRIX_GX2_Address                           0x015DC
#define GCREG_DE_HIST_SOBEL_MATRIX_GX2_MSB                                    15
#define GCREG_DE_HIST_SOBEL_MATRIX_GX2_LSB                                     0
#define GCREG_DE_HIST_SOBEL_MATRIX_GX2_BLK                                     0
#define GCREG_DE_HIST_SOBEL_MATRIX_GX2_Count                                   1
#define GCREG_DE_HIST_SOBEL_MATRIX_GX2_FieldMask                      0x00FFFFFF
#define GCREG_DE_HIST_SOBEL_MATRIX_GX2_ReadMask                       0x00FFFFFF
#define GCREG_DE_HIST_SOBEL_MATRIX_GX2_WriteMask                      0x00FFFFFF
#define GCREG_DE_HIST_SOBEL_MATRIX_GX2_ResetValue                     0x00000000

#define GCREG_DE_HIST_SOBEL_MATRIX_GX2_PARAM6                                7:0
#define GCREG_DE_HIST_SOBEL_MATRIX_GX2_PARAM6_End                              7
#define GCREG_DE_HIST_SOBEL_MATRIX_GX2_PARAM6_Start                            0
#define GCREG_DE_HIST_SOBEL_MATRIX_GX2_PARAM6_Type                           U08

#define GCREG_DE_HIST_SOBEL_MATRIX_GX2_PARAM7                               15:8
#define GCREG_DE_HIST_SOBEL_MATRIX_GX2_PARAM7_End                             15
#define GCREG_DE_HIST_SOBEL_MATRIX_GX2_PARAM7_Start                            8
#define GCREG_DE_HIST_SOBEL_MATRIX_GX2_PARAM7_Type                           U08

#define GCREG_DE_HIST_SOBEL_MATRIX_GX2_PARAM8                              23:16
#define GCREG_DE_HIST_SOBEL_MATRIX_GX2_PARAM8_End                             23
#define GCREG_DE_HIST_SOBEL_MATRIX_GX2_PARAM8_Start                           16
#define GCREG_DE_HIST_SOBEL_MATRIX_GX2_PARAM8_Type                           U08

#define gcregDEHistSobelMatrixGy0RegAddrs                                 0x0578
#define GCREG_DE_HIST_SOBEL_MATRIX_GY0_Address                           0x015E0
#define GCREG_DE_HIST_SOBEL_MATRIX_GY0_MSB                                    15
#define GCREG_DE_HIST_SOBEL_MATRIX_GY0_LSB                                     0
#define GCREG_DE_HIST_SOBEL_MATRIX_GY0_BLK                                     0
#define GCREG_DE_HIST_SOBEL_MATRIX_GY0_Count                                   1
#define GCREG_DE_HIST_SOBEL_MATRIX_GY0_FieldMask                      0x00FFFFFF
#define GCREG_DE_HIST_SOBEL_MATRIX_GY0_ReadMask                       0x00FFFFFF
#define GCREG_DE_HIST_SOBEL_MATRIX_GY0_WriteMask                      0x00FFFFFF
#define GCREG_DE_HIST_SOBEL_MATRIX_GY0_ResetValue                     0x00000000

#define GCREG_DE_HIST_SOBEL_MATRIX_GY0_PARAM0                                7:0
#define GCREG_DE_HIST_SOBEL_MATRIX_GY0_PARAM0_End                              7
#define GCREG_DE_HIST_SOBEL_MATRIX_GY0_PARAM0_Start                            0
#define GCREG_DE_HIST_SOBEL_MATRIX_GY0_PARAM0_Type                           U08

#define GCREG_DE_HIST_SOBEL_MATRIX_GY0_PARAM1                               15:8
#define GCREG_DE_HIST_SOBEL_MATRIX_GY0_PARAM1_End                             15
#define GCREG_DE_HIST_SOBEL_MATRIX_GY0_PARAM1_Start                            8
#define GCREG_DE_HIST_SOBEL_MATRIX_GY0_PARAM1_Type                           U08

#define GCREG_DE_HIST_SOBEL_MATRIX_GY0_PARAM2                              23:16
#define GCREG_DE_HIST_SOBEL_MATRIX_GY0_PARAM2_End                             23
#define GCREG_DE_HIST_SOBEL_MATRIX_GY0_PARAM2_Start                           16
#define GCREG_DE_HIST_SOBEL_MATRIX_GY0_PARAM2_Type                           U08

#define gcregDEHistSobelMatrixGy1RegAddrs                                 0x0579
#define GCREG_DE_HIST_SOBEL_MATRIX_GY1_Address                           0x015E4
#define GCREG_DE_HIST_SOBEL_MATRIX_GY1_MSB                                    15
#define GCREG_DE_HIST_SOBEL_MATRIX_GY1_LSB                                     0
#define GCREG_DE_HIST_SOBEL_MATRIX_GY1_BLK                                     0
#define GCREG_DE_HIST_SOBEL_MATRIX_GY1_Count                                   1
#define GCREG_DE_HIST_SOBEL_MATRIX_GY1_FieldMask                      0x00FFFFFF
#define GCREG_DE_HIST_SOBEL_MATRIX_GY1_ReadMask                       0x00FFFFFF
#define GCREG_DE_HIST_SOBEL_MATRIX_GY1_WriteMask                      0x00FFFFFF
#define GCREG_DE_HIST_SOBEL_MATRIX_GY1_ResetValue                     0x00000000

#define GCREG_DE_HIST_SOBEL_MATRIX_GY1_PARAM3                                7:0
#define GCREG_DE_HIST_SOBEL_MATRIX_GY1_PARAM3_End                              7
#define GCREG_DE_HIST_SOBEL_MATRIX_GY1_PARAM3_Start                            0
#define GCREG_DE_HIST_SOBEL_MATRIX_GY1_PARAM3_Type                           U08

#define GCREG_DE_HIST_SOBEL_MATRIX_GY1_PARAM4                               15:8
#define GCREG_DE_HIST_SOBEL_MATRIX_GY1_PARAM4_End                             15
#define GCREG_DE_HIST_SOBEL_MATRIX_GY1_PARAM4_Start                            8
#define GCREG_DE_HIST_SOBEL_MATRIX_GY1_PARAM4_Type                           U08

#define GCREG_DE_HIST_SOBEL_MATRIX_GY1_PARAM5                              23:16
#define GCREG_DE_HIST_SOBEL_MATRIX_GY1_PARAM5_End                             23
#define GCREG_DE_HIST_SOBEL_MATRIX_GY1_PARAM5_Start                           16
#define GCREG_DE_HIST_SOBEL_MATRIX_GY1_PARAM5_Type                           U08

#define gcregDEHistSobelMatrixGy2RegAddrs                                 0x057A
#define GCREG_DE_HIST_SOBEL_MATRIX_GY2_Address                           0x015E8
#define GCREG_DE_HIST_SOBEL_MATRIX_GY2_MSB                                    15
#define GCREG_DE_HIST_SOBEL_MATRIX_GY2_LSB                                     0
#define GCREG_DE_HIST_SOBEL_MATRIX_GY2_BLK                                     0
#define GCREG_DE_HIST_SOBEL_MATRIX_GY2_Count                                   1
#define GCREG_DE_HIST_SOBEL_MATRIX_GY2_FieldMask                      0x00FFFFFF
#define GCREG_DE_HIST_SOBEL_MATRIX_GY2_ReadMask                       0x00FFFFFF
#define GCREG_DE_HIST_SOBEL_MATRIX_GY2_WriteMask                      0x00FFFFFF
#define GCREG_DE_HIST_SOBEL_MATRIX_GY2_ResetValue                     0x00000000

#define GCREG_DE_HIST_SOBEL_MATRIX_GY2_PARAM6                                7:0
#define GCREG_DE_HIST_SOBEL_MATRIX_GY2_PARAM6_End                              7
#define GCREG_DE_HIST_SOBEL_MATRIX_GY2_PARAM6_Start                            0
#define GCREG_DE_HIST_SOBEL_MATRIX_GY2_PARAM6_Type                           U08

#define GCREG_DE_HIST_SOBEL_MATRIX_GY2_PARAM7                               15:8
#define GCREG_DE_HIST_SOBEL_MATRIX_GY2_PARAM7_End                             15
#define GCREG_DE_HIST_SOBEL_MATRIX_GY2_PARAM7_Start                            8
#define GCREG_DE_HIST_SOBEL_MATRIX_GY2_PARAM7_Type                           U08

#define GCREG_DE_HIST_SOBEL_MATRIX_GY2_PARAM8                              23:16
#define GCREG_DE_HIST_SOBEL_MATRIX_GY2_PARAM8_End                             23
#define GCREG_DE_HIST_SOBEL_MATRIX_GY2_PARAM8_Start                           16
#define GCREG_DE_HIST_SOBEL_MATRIX_GY2_PARAM8_Type                           U08

#define gcregHistBaseAddress0RegAddrs                                     0x057B
#define GCREG_HIST_BASE_ADDRESS0_Address                                 0x015EC
#define GCREG_HIST_BASE_ADDRESS0_MSB                                          15
#define GCREG_HIST_BASE_ADDRESS0_LSB                                           0
#define GCREG_HIST_BASE_ADDRESS0_BLK                                           0
#define GCREG_HIST_BASE_ADDRESS0_Count                                         1
#define GCREG_HIST_BASE_ADDRESS0_FieldMask                            0xFFFFFFFF
#define GCREG_HIST_BASE_ADDRESS0_ReadMask                             0xFFFFFFFC
#define GCREG_HIST_BASE_ADDRESS0_WriteMask                            0xFFFFFFFC
#define GCREG_HIST_BASE_ADDRESS0_ResetValue                           0x00000000

#define GCREG_HIST_BASE_ADDRESS0_TYPE                                      31:31
#define GCREG_HIST_BASE_ADDRESS0_TYPE_End                                     31
#define GCREG_HIST_BASE_ADDRESS0_TYPE_Start                                   31
#define GCREG_HIST_BASE_ADDRESS0_TYPE_Type                                   U01
#define   GCREG_HIST_BASE_ADDRESS0_TYPE_SYSTEM                               0x0
#define   GCREG_HIST_BASE_ADDRESS0_TYPE_VIRTUAL_SYSTEM                       0x1

#define GCREG_HIST_BASE_ADDRESS0_ADDRESS                                    30:0
#define GCREG_HIST_BASE_ADDRESS0_ADDRESS_End                                  30
#define GCREG_HIST_BASE_ADDRESS0_ADDRESS_Start                                 0
#define GCREG_HIST_BASE_ADDRESS0_ADDRESS_Type                                U31

#define gcregHistBaseAddress1RegAddrs                                     0x057C
#define GCREG_HIST_BASE_ADDRESS1_Address                                 0x015F0
#define GCREG_HIST_BASE_ADDRESS1_MSB                                          15
#define GCREG_HIST_BASE_ADDRESS1_LSB                                           0
#define GCREG_HIST_BASE_ADDRESS1_BLK                                           0
#define GCREG_HIST_BASE_ADDRESS1_Count                                         1
#define GCREG_HIST_BASE_ADDRESS1_FieldMask                            0xFFFFFFFF
#define GCREG_HIST_BASE_ADDRESS1_ReadMask                             0xFFFFFFFC
#define GCREG_HIST_BASE_ADDRESS1_WriteMask                            0xFFFFFFFC
#define GCREG_HIST_BASE_ADDRESS1_ResetValue                           0x00000000

#define GCREG_HIST_BASE_ADDRESS1_TYPE                                      31:31
#define GCREG_HIST_BASE_ADDRESS1_TYPE_End                                     31
#define GCREG_HIST_BASE_ADDRESS1_TYPE_Start                                   31
#define GCREG_HIST_BASE_ADDRESS1_TYPE_Type                                   U01
#define   GCREG_HIST_BASE_ADDRESS1_TYPE_SYSTEM                               0x0
#define   GCREG_HIST_BASE_ADDRESS1_TYPE_VIRTUAL_SYSTEM                       0x1

#define GCREG_HIST_BASE_ADDRESS1_ADDRESS                                    30:0
#define GCREG_HIST_BASE_ADDRESS1_ADDRESS_End                                  30
#define GCREG_HIST_BASE_ADDRESS1_ADDRESS_Start                                 0
#define GCREG_HIST_BASE_ADDRESS1_ADDRESS_Type                                U31

#define gcregHistBaseAddress2RegAddrs                                     0x057D
#define GCREG_HIST_BASE_ADDRESS2_Address                                 0x015F4
#define GCREG_HIST_BASE_ADDRESS2_MSB                                          15
#define GCREG_HIST_BASE_ADDRESS2_LSB                                           0
#define GCREG_HIST_BASE_ADDRESS2_BLK                                           0
#define GCREG_HIST_BASE_ADDRESS2_Count                                         1
#define GCREG_HIST_BASE_ADDRESS2_FieldMask                            0xFFFFFFFF
#define GCREG_HIST_BASE_ADDRESS2_ReadMask                             0xFFFFFFFC
#define GCREG_HIST_BASE_ADDRESS2_WriteMask                            0xFFFFFFFC
#define GCREG_HIST_BASE_ADDRESS2_ResetValue                           0x00000000

#define GCREG_HIST_BASE_ADDRESS2_TYPE                                      31:31
#define GCREG_HIST_BASE_ADDRESS2_TYPE_End                                     31
#define GCREG_HIST_BASE_ADDRESS2_TYPE_Start                                   31
#define GCREG_HIST_BASE_ADDRESS2_TYPE_Type                                   U01
#define   GCREG_HIST_BASE_ADDRESS2_TYPE_SYSTEM                               0x0
#define   GCREG_HIST_BASE_ADDRESS2_TYPE_VIRTUAL_SYSTEM                       0x1

#define GCREG_HIST_BASE_ADDRESS2_ADDRESS                                    30:0
#define GCREG_HIST_BASE_ADDRESS2_ADDRESS_End                                  30
#define GCREG_HIST_BASE_ADDRESS2_ADDRESS_Start                                 0
#define GCREG_HIST_BASE_ADDRESS2_ADDRESS_Type                                U31

#define DE2PE_COMMAND_PIXEL                                                  0:0
#define DE2PE_COMMAND_PIXEL_End                                                0
#define DE2PE_COMMAND_PIXEL_Start                                              0
#define DE2PE_COMMAND_PIXEL_Type                                             U01
#define   DE2PE_COMMAND_PIXEL_INVALID                                        0x0
#define   DE2PE_COMMAND_PIXEL_VALID                                          0x1
#define   DE2PE_COMMAND_PIXEL_OPAQUE                                         0x0
#define   DE2PE_COMMAND_PIXEL_TRANSPARENT                                    0x1

#define AQ_DRAWING_ENGINE_FORMAT_FORMAT_EX                                 19:16
#define AQ_DRAWING_ENGINE_FORMAT_FORMAT_EX_End                                19
#define AQ_DRAWING_ENGINE_FORMAT_FORMAT_EX_Start                              16
#define AQ_DRAWING_ENGINE_FORMAT_FORMAT_EX_Type                              U04
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_EX_RGB_F16F16F16_PLANAR            0x0
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_EX_RGB_F16F16F16_PACKED            0x1
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_EX_RGB_F32F32F32_PLANAR            0x2
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_EX_RGB_F32F32F32_PACKED            0x3
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_EX_GRAY_I8                         0x4
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_EX_GRAY_F16                        0x5
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_EX_GRAY_F32                        0x6
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_EX_INT8_PLANAR                     0x7
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_EX_INT8_PACKED                     0x8
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_EX_INT16_PLANAR                    0x9
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_EX_INT16_PACKED                    0xA
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_EX_GRAY_U8                         0xB

#define AQ_DRAWING_ENGINE_FORMAT_SUB_SAMPLE_MODE_YUV                       12:12
#define AQ_DRAWING_ENGINE_FORMAT_SUB_SAMPLE_MODE_YUV_End                      12
#define AQ_DRAWING_ENGINE_FORMAT_SUB_SAMPLE_MODE_YUV_Start                    12
#define AQ_DRAWING_ENGINE_FORMAT_SUB_SAMPLE_MODE_YUV_Type                    U01
#define   AQ_DRAWING_ENGINE_FORMAT_SUB_SAMPLE_MODE_YUV_MODE422               0x0
#define   AQ_DRAWING_ENGINE_FORMAT_SUB_SAMPLE_MODE_YUV_MODE420               0x1

#define AQ_DRAWING_ENGINE_FORMAT_SWIZZLE                                     9:8
#define AQ_DRAWING_ENGINE_FORMAT_SWIZZLE_End                                   9
#define AQ_DRAWING_ENGINE_FORMAT_SWIZZLE_Start                                 8
#define AQ_DRAWING_ENGINE_FORMAT_SWIZZLE_Type                                U02
#define   AQ_DRAWING_ENGINE_FORMAT_SWIZZLE_ARGB                              0x0
#define   AQ_DRAWING_ENGINE_FORMAT_SWIZZLE_RGBA                              0x1
#define   AQ_DRAWING_ENGINE_FORMAT_SWIZZLE_ABGR                              0x2
#define   AQ_DRAWING_ENGINE_FORMAT_SWIZZLE_BGRA                              0x3

#define AQ_DRAWING_ENGINE_FORMAT_FORMAT                                      4:0
#define AQ_DRAWING_ENGINE_FORMAT_FORMAT_End                                    4
#define AQ_DRAWING_ENGINE_FORMAT_FORMAT_Start                                  0
#define AQ_DRAWING_ENGINE_FORMAT_FORMAT_Type                                 U05
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_X4R4G4B4                          0x00
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_A4R4G4B4                          0x01
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_X1R5G5B5                          0x02
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_A1R5G5B5                          0x03
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_R5G6B5                            0x04
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_X8R8G8B8                          0x05
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_A8R8G8B8                          0x06
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_YUY2                              0x07
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_UYVY                              0x08
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_INDEX8                            0x09
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_MONOCHROME                        0x0A
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_AYUV                              0x0B
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_RGB888_PLANAR                     0x0C
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_RGB888_PACKED                     0x0D
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_EXTENSION                         0x0E
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_YV12                              0x0F
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_A8                                0x10
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_NV12                              0x11
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_NV16                              0x12
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_RG16                              0x13
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_R8                                0x14
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_NV12_10BIT                        0x15
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_A2R10G10B10                       0x16
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_NV16_10BIT                        0x17
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_INDEX1                            0x18
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_INDEX2                            0x19
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_INDEX4                            0x1A
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_P010                              0x1B
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_NV12_10BIT_L1                     0x1C
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_NV16_10BIT_L1                     0x1D
#define   AQ_DRAWING_ENGINE_FORMAT_FORMAT_I010                              0x1E

#define AQ_DRAWING_ENGINE_ALPHA_BLENDING_ALPHA_MODE                          0:0
#define AQ_DRAWING_ENGINE_ALPHA_BLENDING_ALPHA_MODE_End                        0
#define AQ_DRAWING_ENGINE_ALPHA_BLENDING_ALPHA_MODE_Start                      0
#define AQ_DRAWING_ENGINE_ALPHA_BLENDING_ALPHA_MODE_Type                     U01
#define   AQ_DRAWING_ENGINE_ALPHA_BLENDING_ALPHA_MODE_NORMAL                 0x0
#define   AQ_DRAWING_ENGINE_ALPHA_BLENDING_ALPHA_MODE_INVERSED               0x1

#define AQ_DRAWING_ENGINE_ALPHA_BLENDING_GLOBAL_ALPHA_MODE                   5:4
#define AQ_DRAWING_ENGINE_ALPHA_BLENDING_GLOBAL_ALPHA_MODE_End                 5
#define AQ_DRAWING_ENGINE_ALPHA_BLENDING_GLOBAL_ALPHA_MODE_Start               4
#define AQ_DRAWING_ENGINE_ALPHA_BLENDING_GLOBAL_ALPHA_MODE_Type              U02
#define   AQ_DRAWING_ENGINE_ALPHA_BLENDING_GLOBAL_ALPHA_MODE_NORMAL          0x0
#define   AQ_DRAWING_ENGINE_ALPHA_BLENDING_GLOBAL_ALPHA_MODE_GLOBAL          0x1
#define   AQ_DRAWING_ENGINE_ALPHA_BLENDING_GLOBAL_ALPHA_MODE_SCALED          0x2

#define AQ_DRAWING_ENGINE_ALPHA_BLENDING_COLOR_MODE                        13:12
#define AQ_DRAWING_ENGINE_ALPHA_BLENDING_COLOR_MODE_End                       13
#define AQ_DRAWING_ENGINE_ALPHA_BLENDING_COLOR_MODE_Start                     12
#define AQ_DRAWING_ENGINE_ALPHA_BLENDING_COLOR_MODE_Type                     U02
#define   AQ_DRAWING_ENGINE_ALPHA_BLENDING_COLOR_MODE_NORMAL                 0x0
#define   AQ_DRAWING_ENGINE_ALPHA_BLENDING_COLOR_MODE_MULTIPLY               0x1

#define AQ_DRAWING_ENGINE_ALPHA_BLENDING_BLENDING_MODE                     18:16
#define AQ_DRAWING_ENGINE_ALPHA_BLENDING_BLENDING_MODE_End                    18
#define AQ_DRAWING_ENGINE_ALPHA_BLENDING_BLENDING_MODE_Start                  16
#define AQ_DRAWING_ENGINE_ALPHA_BLENDING_BLENDING_MODE_Type                  U03
#define   AQ_DRAWING_ENGINE_ALPHA_BLENDING_BLENDING_MODE_ZERO                0x0
#define   AQ_DRAWING_ENGINE_ALPHA_BLENDING_BLENDING_MODE_ONE                 0x1
#define   AQ_DRAWING_ENGINE_ALPHA_BLENDING_BLENDING_MODE_NORMAL              0x2
#define   AQ_DRAWING_ENGINE_ALPHA_BLENDING_BLENDING_MODE_INVERSED            0x3
#define   AQ_DRAWING_ENGINE_ALPHA_BLENDING_BLENDING_MODE_COLOR               0x4
#define   AQ_DRAWING_ENGINE_ALPHA_BLENDING_BLENDING_MODE_COLOR_INVERSED      0x5
#define   AQ_DRAWING_ENGINE_ALPHA_BLENDING_BLENDING_MODE_SATURATED_ALPHA     0x6
#define   AQ_DRAWING_ENGINE_ALPHA_BLENDING_BLENDING_MODE_SATURATED_DEST_ALPHA 0x7

#define AQ_DRAWING_ENGINE_RESOURCE_USAGE_OVERRIDE                            1:0
#define AQ_DRAWING_ENGINE_RESOURCE_USAGE_OVERRIDE_End                          1
#define AQ_DRAWING_ENGINE_RESOURCE_USAGE_OVERRIDE_Start                        0
#define AQ_DRAWING_ENGINE_RESOURCE_USAGE_OVERRIDE_Type                       U02
#define   AQ_DRAWING_ENGINE_RESOURCE_USAGE_OVERRIDE_DEFAULT                  0x0
#define   AQ_DRAWING_ENGINE_RESOURCE_USAGE_OVERRIDE_USE_ENABLE               0x1
#define   AQ_DRAWING_ENGINE_RESOURCE_USAGE_OVERRIDE_USE_DISABLE              0x2

#define AQContextID0RegAddrs                                              0x6800
#define AQ_CONTEXT_ID0_Address                                           0x1A000
#define AQ_CONTEXT_ID0_MSB                                                    15
#define AQ_CONTEXT_ID0_LSB                                                     0
#define AQ_CONTEXT_ID0_BLK                                                     0
#define AQ_CONTEXT_ID0_Count                                                   1
#define AQ_CONTEXT_ID0_FieldMask                                      0xFFFFFFFF
#define AQ_CONTEXT_ID0_ReadMask                                       0xFFFFFFFF
#define AQ_CONTEXT_ID0_WriteMask                                      0xFFFFFFFF
#define AQ_CONTEXT_ID0_ResetValue                                     0x00000000

#define AQ_CONTEXT_ID0_PE_DST                                               15:0
#define AQ_CONTEXT_ID0_PE_DST_End                                             15
#define AQ_CONTEXT_ID0_PE_DST_Start                                            0
#define AQ_CONTEXT_ID0_PE_DST_Type                                           U16

#define AQ_CONTEXT_ID0_PE_SRC0                                             31:16
#define AQ_CONTEXT_ID0_PE_SRC0_End                                            31
#define AQ_CONTEXT_ID0_PE_SRC0_Start                                          16
#define AQ_CONTEXT_ID0_PE_SRC0_Type                                          U16

#define AQContextID1RegAddrs                                              0x6801
#define AQ_CONTEXT_ID1_Address                                           0x1A004
#define AQ_CONTEXT_ID1_MSB                                                    15
#define AQ_CONTEXT_ID1_LSB                                                     0
#define AQ_CONTEXT_ID1_BLK                                                     0
#define AQ_CONTEXT_ID1_Count                                                   1
#define AQ_CONTEXT_ID1_FieldMask                                      0xFFFFFFFF
#define AQ_CONTEXT_ID1_ReadMask                                       0xFFFFFFFF
#define AQ_CONTEXT_ID1_WriteMask                                      0xFFFFFFFF
#define AQ_CONTEXT_ID1_ResetValue                                     0x00000000

#define AQ_CONTEXT_ID1_PE_SRC1                                              15:0
#define AQ_CONTEXT_ID1_PE_SRC1_End                                            15
#define AQ_CONTEXT_ID1_PE_SRC1_Start                                           0
#define AQ_CONTEXT_ID1_PE_SRC1_Type                                          U16

#define AQ_CONTEXT_ID1_PE_SRC2                                             31:16
#define AQ_CONTEXT_ID1_PE_SRC2_End                                            31
#define AQ_CONTEXT_ID1_PE_SRC2_Start                                          16
#define AQ_CONTEXT_ID1_PE_SRC2_Type                                          U16

#define AQContextID2RegAddrs                                              0x6802
#define AQ_CONTEXT_ID2_Address                                           0x1A008
#define AQ_CONTEXT_ID2_MSB                                                    15
#define AQ_CONTEXT_ID2_LSB                                                     0
#define AQ_CONTEXT_ID2_BLK                                                     0
#define AQ_CONTEXT_ID2_Count                                                   1
#define AQ_CONTEXT_ID2_FieldMask                                      0xFFFFFFFF
#define AQ_CONTEXT_ID2_ReadMask                                       0xFFFFFFFF
#define AQ_CONTEXT_ID2_WriteMask                                      0xFFFFFFFF
#define AQ_CONTEXT_ID2_ResetValue                                     0x00000000

#define AQ_CONTEXT_ID2_PE_SRC3                                              15:0
#define AQ_CONTEXT_ID2_PE_SRC3_End                                            15
#define AQ_CONTEXT_ID2_PE_SRC3_Start                                           0
#define AQ_CONTEXT_ID2_PE_SRC3_Type                                          U16

#define AQ_CONTEXT_ID2_PE_SRC4                                             31:16
#define AQ_CONTEXT_ID2_PE_SRC4_End                                            31
#define AQ_CONTEXT_ID2_PE_SRC4_Start                                          16
#define AQ_CONTEXT_ID2_PE_SRC4_Type                                          U16

#define AQContextID3RegAddrs                                              0x6803
#define AQ_CONTEXT_ID3_Address                                           0x1A00C
#define AQ_CONTEXT_ID3_MSB                                                    15
#define AQ_CONTEXT_ID3_LSB                                                     0
#define AQ_CONTEXT_ID3_BLK                                                     0
#define AQ_CONTEXT_ID3_Count                                                   1
#define AQ_CONTEXT_ID3_FieldMask                                      0xFFFFFFFF
#define AQ_CONTEXT_ID3_ReadMask                                       0xFFFFFFFF
#define AQ_CONTEXT_ID3_WriteMask                                      0xFFFFFFFF
#define AQ_CONTEXT_ID3_ResetValue                                     0x00000000

#define AQ_CONTEXT_ID3_PE_SRC5                                              15:0
#define AQ_CONTEXT_ID3_PE_SRC5_End                                            15
#define AQ_CONTEXT_ID3_PE_SRC5_Start                                           0
#define AQ_CONTEXT_ID3_PE_SRC5_Type                                          U16

#define AQ_CONTEXT_ID3_PE_SRC6                                             31:16
#define AQ_CONTEXT_ID3_PE_SRC6_End                                            31
#define AQ_CONTEXT_ID3_PE_SRC6_Start                                          16
#define AQ_CONTEXT_ID3_PE_SRC6_Type                                          U16

#define AQContextID4RegAddrs                                              0x6804
#define AQ_CONTEXT_ID4_Address                                           0x1A010
#define AQ_CONTEXT_ID4_MSB                                                    15
#define AQ_CONTEXT_ID4_LSB                                                     0
#define AQ_CONTEXT_ID4_BLK                                                     0
#define AQ_CONTEXT_ID4_Count                                                   1
#define AQ_CONTEXT_ID4_FieldMask                                      0xFFFFFFFF
#define AQ_CONTEXT_ID4_ReadMask                                       0xFFFFFFFF
#define AQ_CONTEXT_ID4_WriteMask                                      0xFFFFFFFF
#define AQ_CONTEXT_ID4_ResetValue                                     0x00000000

#define AQ_CONTEXT_ID4_PE_SRC7                                              15:0
#define AQ_CONTEXT_ID4_PE_SRC7_End                                            15
#define AQ_CONTEXT_ID4_PE_SRC7_Start                                           0
#define AQ_CONTEXT_ID4_PE_SRC7_Type                                          U16

#define AQ_CONTEXT_ID4_TS                                                  31:16
#define AQ_CONTEXT_ID4_TS_End                                                 31
#define AQ_CONTEXT_ID4_TS_Start                                               16
#define AQ_CONTEXT_ID4_TS_Type                                               U16

#define AQContextID5RegAddrs                                              0x6805
#define AQ_CONTEXT_ID5_Address                                           0x1A014
#define AQ_CONTEXT_ID5_MSB                                                    15
#define AQ_CONTEXT_ID5_LSB                                                     0
#define AQ_CONTEXT_ID5_BLK                                                     0
#define AQ_CONTEXT_ID5_Count                                                   1
#define AQ_CONTEXT_ID5_FieldMask                                      0x0000FFFF
#define AQ_CONTEXT_ID5_ReadMask                                       0x0000FFFF
#define AQ_CONTEXT_ID5_WriteMask                                      0x0000FFFF
#define AQ_CONTEXT_ID5_ResetValue                                     0x00000000

#define AQ_CONTEXT_ID5_MMU                                                  15:0
#define AQ_CONTEXT_ID5_MMU_End                                                15
#define AQ_CONTEXT_ID5_MMU_Start                                               0
#define AQ_CONTEXT_ID5_MMU_Type                                              U16

#define AQTSAxiAddressRegAddrs                                            0x6806
#define AQTS_AXI_ADDRESS_Address                                         0x1A018
#define AQTS_AXI_ADDRESS_MSB                                                  15
#define AQTS_AXI_ADDRESS_LSB                                                   0
#define AQTS_AXI_ADDRESS_BLK                                                   0
#define AQTS_AXI_ADDRESS_Count                                                 1
#define AQTS_AXI_ADDRESS_FieldMask                                    0xFFFFFFFF
#define AQTS_AXI_ADDRESS_ReadMask                                     0xFFFFFFFF
#define AQTS_AXI_ADDRESS_WriteMask                                    0xFFFFFFFF
#define AQTS_AXI_ADDRESS_ResetValue                                   0x00000000

#define AQTS_AXI_ADDRESS_ADDRESS                                            31:0
#define AQTS_AXI_ADDRESS_ADDRESS_End                                          31
#define AQTS_AXI_ADDRESS_ADDRESS_Start                                         0
#define AQTS_AXI_ADDRESS_ADDRESS_Type                                        U32

#define AQFEAxiAddressRegAddrs                                            0x00AF
#define AQFE_AXI_ADDRESS_Address                                         0x002BC
#define AQFE_AXI_ADDRESS_MSB                                                  15
#define AQFE_AXI_ADDRESS_LSB                                                   0
#define AQFE_AXI_ADDRESS_BLK                                                   0
#define AQFE_AXI_ADDRESS_Count                                                 1
#define AQFE_AXI_ADDRESS_FieldMask                                    0xFFFFFFFF
#define AQFE_AXI_ADDRESS_ReadMask                                     0xFFFFFFFF
#define AQFE_AXI_ADDRESS_WriteMask                                    0xFFFFFFFF
#define AQFE_AXI_ADDRESS_ResetValue                                   0x00000000

#define AQFE_AXI_ADDRESS_ADDRESS                                            31:0
#define AQFE_AXI_ADDRESS_ADDRESS_End                                          31
#define AQFE_AXI_ADDRESS_ADDRESS_Start                                         0
#define AQFE_AXI_ADDRESS_ADDRESS_Type                                        U32

/* Register AQSelectNSAID **
** ~~~~~~~~~~~~~~~~~~~~~~ */

/* Control select security or no security. */

#define AQSelectNSAIDRegAddrs                                             0x0E73
#define AQ_SELECT_NSAID_Address                                          0x039CC
#define AQ_SELECT_NSAID_MSB                                                   15
#define AQ_SELECT_NSAID_LSB                                                    0
#define AQ_SELECT_NSAID_BLK                                                    0
#define AQ_SELECT_NSAID_Count                                                  1
#define AQ_SELECT_NSAID_FieldMask                                     0x00000003
#define AQ_SELECT_NSAID_ReadMask                                      0x00000003
#define AQ_SELECT_NSAID_WriteMask                                     0x00000003
#define AQ_SELECT_NSAID_ResetValue                                    0x00000000

/* Select_NSAID1 TS access. */
#define AQ_SELECT_NSAID_NSAID1                                               0:0
#define AQ_SELECT_NSAID_NSAID1_End                                             0
#define AQ_SELECT_NSAID_NSAID1_Start                                           0
#define AQ_SELECT_NSAID_NSAID1_Type                                          U01

/* Select_NSAID2 other access. */
#define AQ_SELECT_NSAID_NSAID2                                               1:1
#define AQ_SELECT_NSAID_NSAID2_End                                             1
#define AQ_SELECT_NSAID_NSAID2_Start                                           1
#define AQ_SELECT_NSAID_NSAID2_Type                                          U01

/* Register AQFESelectNSAID **
** ~~~~~~~~~~~~~~~~~~~~~~~~ */

/* Control select security or no security. */

#define AQFESelectNSAIDRegAddrs                                           0x0E74
#define AQFE_SELECT_NSAID_Address                                        0x039D0
#define AQFE_SELECT_NSAID_MSB                                                 15
#define AQFE_SELECT_NSAID_LSB                                                  0
#define AQFE_SELECT_NSAID_BLK                                                  0
#define AQFE_SELECT_NSAID_Count                                                1
#define AQFE_SELECT_NSAID_FieldMask                                   0x00000001
#define AQFE_SELECT_NSAID_ReadMask                                    0x00000001
#define AQFE_SELECT_NSAID_WriteMask                                   0x00000001
#define AQFE_SELECT_NSAID_ResetValue                                  0x00000000

/* Select_NSAID0 FE access. */
#define AQFE_SELECT_NSAID_NSAID0                                             0:0
#define AQFE_SELECT_NSAID_NSAID0_End                                           0
#define AQFE_SELECT_NSAID_NSAID0_Start                                         0
#define AQFE_SELECT_NSAID_NSAID0_Type                                        U01

/*******************************************************************************
**                                 ~~~~~~~~~~                                 **
**                                 Module MMU                                 **
**                                 ~~~~~~~~~~                                 **
*******************************************************************************/

#define gcregMMUSafeAddressRegAddrs                                       0x0060
#define gcregMMUConfigurationRegAddrs                                     0x0061

#define GCREG_MMU_CONFIGURATION_MODE                                         0:0
#define GCREG_MMU_CONFIGURATION_FLUSH                                        4:4
#define GCREG_MMU_CONFIGURATION_MASK_FLUSH                                   7:7
#define   GCREG_MMU_CONFIGURATION_MODE_MODE4_K                               0x0
#define   GCREG_MMU_CONFIGURATION_FLUSH_FLUSH                                0x1
#define   GCREG_MMU_CONFIGURATION_MASK_FLUSH_ENABLED                         0x0

#define GCREG_MMU_CONTROL_Address                                        0x0018C
#define GCREG_MMU_CONTROL_ENABLE                                             0:0
#define   GCREG_MMU_CONTROL_ENABLE_ENABLE                                    0x1

#define gcregMMUConfigurationExtRegAddrs                                  0x0068
#define GCREG_MMU_CONFIGURATION_EXT_SAFE_ADDRESS                             7:0
#define GCREG_MMU_CONFIGURATION_EXT_MASTER_TLB                             23:16

#define gcregMMUConfigRegAddrs                                            0x006B
#define GCREG_MMU_CONFIG_PAGE_TABLE_ID                                      15:0
#define GCREG_MMU_CONFIG_MASK_PAGE_TABLE_ID                                16:16
#define   GCREG_MMU_CONFIG_MASK_PAGE_TABLE_ID_ENABLED                        0x0

/*******************************************************************************
 **                            ~~~~~~~~~~~~~~~~~~~~                            **
 **                            Module HostInterface                            **
 **                            ~~~~~~~~~~~~~~~~~~~~                            **
 *******************************************************************************/

 /* Register AQHiClockControl **
 ** ~~~~~~~~~~~~~~~~~~~~~~~~~ */

/* Clock Control Register.  This register provides basic controls for  clock  **
** enable, frequency scaling value and core isolation.                        */

#define AQHiClockControlRegAddrs                                          0x0000
#define AQ_HI_CLOCK_CONTROL_Address                                      0x00000
#define AQ_HI_CLOCK_CONTROL_MSB                                               15
#define AQ_HI_CLOCK_CONTROL_LSB                                                0
#define AQ_HI_CLOCK_CONTROL_BLK                                                0
#define AQ_HI_CLOCK_CONTROL_Count                                              1
#define AQ_HI_CLOCK_CONTROL_FieldMask                                 0x1FFF7FFF
#define AQ_HI_CLOCK_CONTROL_ReadMask                                  0x1FFF7FFF
#define AQ_HI_CLOCK_CONTROL_WriteMask                                 0x1FF87FFF
#define AQ_HI_CLOCK_CONTROL_ResetValue                                0x00070900

/* Disable 3D clock. Software core clock disable signal  for 3d modules       **
** (clk_3d). When set to 1, this clock  is frozen.                            */
#define AQ_HI_CLOCK_CONTROL_CLK3D_DIS                                        0:0
#define AQ_HI_CLOCK_CONTROL_CLK3D_DIS_End                                      0
#define AQ_HI_CLOCK_CONTROL_CLK3D_DIS_Start                                    0
#define AQ_HI_CLOCK_CONTROL_CLK3D_DIS_Type                                   U01

/* Disable 2D clock.  Software clock disable signal. Both bits  CLK3D_DIS and **
** CLK2D_DIS should be controlled by software.  The AXI interface clock is    **
** the only block not stalled  at this point.                                 */
#define AQ_HI_CLOCK_CONTROL_CLK2D_DIS                                        1:1
#define AQ_HI_CLOCK_CONTROL_CLK2D_DIS_End                                      1
#define AQ_HI_CLOCK_CONTROL_CLK2D_DIS_Start                                    1
#define AQ_HI_CLOCK_CONTROL_CLK2D_DIS_Type                                   U01

/* Core clock frequency scale value.(reset=1) */
#define AQ_HI_CLOCK_CONTROL_FSCALE_VAL                                       8:2
#define AQ_HI_CLOCK_CONTROL_FSCALE_VAL_End                                     8
#define AQ_HI_CLOCK_CONTROL_FSCALE_VAL_Start                                   2
#define AQ_HI_CLOCK_CONTROL_FSCALE_VAL_Type                                  U07

/* Core clock frequency scale value enable.  When writing a 1 to this bit, it **
**  updates the frequency scale factor with the value  FSCALE_VAL[6:0]. The   **
** bit must be set back to 0 after  that. If this bit is set and FSCALE_VAL=0 **
** (an invalid  combination), the HREADYOUT output signal will get stuck  to  **
** 0.                                                                         */
#define AQ_HI_CLOCK_CONTROL_FSCALE_CMD_LOAD                                  9:9
#define AQ_HI_CLOCK_CONTROL_FSCALE_CMD_LOAD_End                                9
#define AQ_HI_CLOCK_CONTROL_FSCALE_CMD_LOAD_Start                              9
#define AQ_HI_CLOCK_CONTROL_FSCALE_CMD_LOAD_Type                             U01

/* Disables clock gating for rams. */
#define AQ_HI_CLOCK_CONTROL_DISABLE_RAM_CLOCK_GATING                       10:10
#define AQ_HI_CLOCK_CONTROL_DISABLE_RAM_CLOCK_GATING_End                      10
#define AQ_HI_CLOCK_CONTROL_DISABLE_RAM_CLOCK_GATING_Start                    10
#define AQ_HI_CLOCK_CONTROL_DISABLE_RAM_CLOCK_GATING_Type                    U01

/* Disable debug registers. If this bit is 1, debug  registers are clock      **
** gated.(reset=1) For cores from v620x, this field is obsolete.  Use         **
** gcregHiAHBControl[1]DEBUG_MODE instead.                                    */
#define AQ_HI_CLOCK_CONTROL_DISABLE_DEBUG_REGISTERS                        11:11
#define AQ_HI_CLOCK_CONTROL_DISABLE_DEBUG_REGISTERS_End                       11
#define AQ_HI_CLOCK_CONTROL_DISABLE_DEBUG_REGISTERS_Start                     11
#define AQ_HI_CLOCK_CONTROL_DISABLE_DEBUG_REGISTERS_Type                     U01

/* Soft resets the IP. For cores from v620x, this field is obsolete. Use      **
** gcregHiAHBControl[0]SOFT_RESET instead.                                    */
#define AQ_HI_CLOCK_CONTROL_SOFT_RESET                                     12:12
#define AQ_HI_CLOCK_CONTROL_SOFT_RESET_End                                    12
#define AQ_HI_CLOCK_CONTROL_SOFT_RESET_Start                                  12
#define AQ_HI_CLOCK_CONTROL_SOFT_RESET_Type                                  U01

/* Disables ram power optimization. */
#define AQ_HI_CLOCK_CONTROL_DISABLE_RAM_POWER_OPTIMIZATION                 13:13
#define AQ_HI_CLOCK_CONTROL_DISABLE_RAM_POWER_OPTIMIZATION_End                13
#define AQ_HI_CLOCK_CONTROL_DISABLE_RAM_POWER_OPTIMIZATION_Start              13
#define AQ_HI_CLOCK_CONTROL_DISABLE_RAM_POWER_OPTIMIZATION_Type              U01

/* Disables core speed up during AHB access. */
#define AQ_HI_CLOCK_CONTROL_DISABLE_CORE_SPEED_UP                          14:14
#define AQ_HI_CLOCK_CONTROL_DISABLE_CORE_SPEED_UP_End                         14
#define AQ_HI_CLOCK_CONTROL_DISABLE_CORE_SPEED_UP_Start                       14
#define AQ_HI_CLOCK_CONTROL_DISABLE_CORE_SPEED_UP_Type                       U01

/* 3D pipe is idle or not present. */
#define AQ_HI_CLOCK_CONTROL_IDLE3_D                                        16:16
#define AQ_HI_CLOCK_CONTROL_IDLE3_D_End                                       16
#define AQ_HI_CLOCK_CONTROL_IDLE3_D_Start                                     16
#define AQ_HI_CLOCK_CONTROL_IDLE3_D_Type                                     U01

/* 2D pipe is idle or not present. */
#define AQ_HI_CLOCK_CONTROL_IDLE2_D                                        17:17
#define AQ_HI_CLOCK_CONTROL_IDLE2_D_End                                       17
#define AQ_HI_CLOCK_CONTROL_IDLE2_D_Start                                     17
#define AQ_HI_CLOCK_CONTROL_IDLE2_D_Type                                     U01

/* VG pipe is idle or not present. */
#define AQ_HI_CLOCK_CONTROL_IDLE_VG                                        18:18
#define AQ_HI_CLOCK_CONTROL_IDLE_VG_End                                       18
#define AQ_HI_CLOCK_CONTROL_IDLE_VG_Start                                     18
#define AQ_HI_CLOCK_CONTROL_IDLE_VG_Type                                     U01

/* Isolate GPU bit. Used for power on/off, isolation  only for multi-core     **
** GPUs.                                                                      */
#define AQ_HI_CLOCK_CONTROL_ISOLATE_GPU                                    19:19
#define AQ_HI_CLOCK_CONTROL_ISOLATE_GPU_End                                   19
#define AQ_HI_CLOCK_CONTROL_ISOLATE_GPU_Start                                 19
#define AQ_HI_CLOCK_CONTROL_ISOLATE_GPU_Type                                 U01

/* Determines which HI/MC to use while reading registers.  */
#define AQ_HI_CLOCK_CONTROL_MULTI_PIPE_REG_SELECT                          23:20
#define AQ_HI_CLOCK_CONTROL_MULTI_PIPE_REG_SELECT_End                         23
#define AQ_HI_CLOCK_CONTROL_MULTI_PIPE_REG_SELECT_Start                       20
#define AQ_HI_CLOCK_CONTROL_MULTI_PIPE_REG_SELECT_Type                       U04

/* Force all the transactions to go to one AXI.  */
#define AQ_HI_CLOCK_CONTROL_MULTI_PIPE_USE_SINGLE_AXI                      27:24
#define AQ_HI_CLOCK_CONTROL_MULTI_PIPE_USE_SINGLE_AXI_End                     27
#define AQ_HI_CLOCK_CONTROL_MULTI_PIPE_USE_SINGLE_AXI_Start                   24
#define AQ_HI_CLOCK_CONTROL_MULTI_PIPE_USE_SINGLE_AXI_Type                   U04

/* Soft resets the L2. */
#define AQ_HI_CLOCK_CONTROL_SOFT_RESET_L2                                  28:28
#define AQ_HI_CLOCK_CONTROL_SOFT_RESET_L2_End                                 28
#define AQ_HI_CLOCK_CONTROL_SOFT_RESET_L2_Start                               28
#define AQ_HI_CLOCK_CONTROL_SOFT_RESET_L2_Type                               U01

#endif /* __nano2D_engine_h__ */
