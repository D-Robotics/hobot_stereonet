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

#ifndef _nano2D_driver_shared_h_
#define _nano2D_driver_shared_h_

#include "nano2D_dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif

#define N2D_ALLOC_FLAG_NONE         0x00000000
#define N2D_ALLOC_FLAG_CONTIGUOUS   0x00000001
#define N2D_ALLOC_FLAG_NOCONTIGUOUS 0x00000002
#define N2D_ALLOC_FLAG_FROM_USER    0x00000020

#define N2D_ALLOC_FLAG_DMABUF       0x00000004
#define N2D_ALLOC_FLAG_USERMEMORY   0x00000008
#define N2D_ALLOC_FLAG_4G           0x00000010

/* bits[31-28] is used to represent allocate function */
#define N2D_ALLOC_FLAG_RESERVED_ALLOCATOR   0x10000000
#define N2D_ALLOC_FLAG_GFP_ALLOCATOR        0x20000000
#define N2D_ALLOC_FLAG_DMA_ALLOCATOR        0x40000000
#define N2D_ALLOC_FLAG_WRAP_USER            0x80000000

typedef struct n2d_user_event {
    n2d_uintptr_t next;
    n2d_ioctl_interface_t iface;
} n2d_user_event_t;

#ifdef __cplusplus
}
#endif

#endif
