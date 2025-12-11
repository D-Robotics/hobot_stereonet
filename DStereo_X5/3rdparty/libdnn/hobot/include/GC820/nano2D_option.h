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

#ifndef _nano2d_option_h__
#define _nano2d_option_h__

/* Enable/disable Nano2D driver MMU support for 2D GPU that has MMU feature.
*/
#ifndef NANO2D_MMU_ENABLE
    #define NANO2D_MMU_ENABLE                       1
#endif

/*
 * Set hardware DEC400 compression alignment mode
 *  0 : 16 bytes aligned.
 *  1 : 32 bytes aligned.
 *  2 : 64 bytes aligned.
*/
#ifndef NANO2D_COMPRESSION_DEC400_ALIGN_MODE
    #define NANO2D_COMPRESSION_DEC400_ALIGN_MODE    1
#endif

/* Set maximum number of GPU cores that Nano2D driver supports
*/
#ifndef NANO2D_CORE_MAX
    #define NANO2D_CORE_MAX      (NANO2D_DEVICE_MAX * NANO2D_DEV_CORE_COUNT)
#endif

/* Set maximum number of devices */
#ifndef NANO2D_DEVICE_MAX
    #define NANO2D_DEVICE_MAX                       8
#endif

/* Set maximum number of cores for one device */
#ifndef NANO2D_DEV_CORE_COUNT
    #define NANO2D_DEV_CORE_COUNT                   8
#endif

/*
 * Set NANO2D_ENABLE_WRITEBUFFER to 1 for Linux Write Combined memory access
 * Set NANO2D_ENABLE_WRITEBUFFER to 0 for Linux Uncacheable memory access
*/
#ifndef NANO2D_ENABLE_WRITEBUFFER
    #define NANO2D_ENABLE_WRITEBUFFER               1
#endif

/* Set Nano2D command buffer size in bytes.
*/
#ifndef NANO2D_CMDBUFFER_SIZE
    #define NANO2D_CMDBUFFER_SIZE                   (16 << 10)
#endif

/* Set command buffer count used for rendering
*/
#ifndef NANO2D_CMDBUFFERS_COUNT
    #define NANO2D_CMDBUFFERS_COUNT                 2
#endif

/* Set maximum command buffer count
*/
#ifndef NANO2D_CMDBUFFERS_COUNT_MAX
    #define NANO2D_CMDBUFFERS_COUNT_MAX             2
#endif

/* Set N2D_TRACE to 1 to print information about loading picture
*/
#ifndef N2D_TRACE
    #define N2D_TRACE                               0
#endif

/* Set NANO2D_DUMP to 1 to dump 2D commands and surface for debugging.
*/
#ifndef NANO2D_DUMP
    #define NANO2D_DUMP                             0
#endif

/* Set NANO2D_PERF_TIMER to 1 to print perf time stamp.
*/
#ifndef NANO2D_PERF_TIMER
    #define NANO2D_PERF_TIMER                       0
#endif

/* PCIe order-preserving mode, need ensure data is written into DDR by reading.
*/
#ifndef NANO2D_PCIE_ORDER
    #define NANO2D_PCIE_ORDER                       0
#endif

/*  Use WAIT-LINK mode only, even chip support N2D_FEATURE_FRAME_DONE_INTR.
*/
#ifndef NANO2D_WAIT_LINK_ONLY
    #define NANO2D_WAIT_LINK_ONLY                   1
#endif

/*
 * This define specified the number of milliseconds the system will wait
 * before it broadcasts the GPU is stuck.  In other words, it will define
 * the timeout of any operation that needs to wait for the GPU.
 *
 * If the value is 0, no timeout will be checked for.
 */
#ifndef N2D_GPU_TIMEOUT
#if (defined EMULATOR) || (defined LINUXEMULATOR)
# define N2D_GPU_TIMEOUT                            0
#else
# define N2D_GPU_TIMEOUT                        10000
#endif
#endif

#endif /* _nano2d_option_h__ */
