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

#ifndef _nano2D_dispatch_h_
#define _nano2D_dispatch_h_

#include "nano2D_types.h"
#include "nano2D_enum.h"

#ifdef __cplusplus
extern "C" {
#endif

/* User signal command codes. */
typedef enum n2d_user_signal_command {
    N2D_USER_SIGNAL_CREATE,
    N2D_USER_SIGNAL_DESTROY,
    N2D_USER_SIGNAL_SIGNAL,
    N2D_USER_SIGNAL_WAIT,
} n2d_user_signal_command_t;

typedef enum n2d_kernel_command {
    /* 0  */ N2D_KERNEL_COMMAND_OPEN,
    /* 1  */ N2D_KERNEL_COMMAND_CLOSE,
    /* 2  */ N2D_KERNEL_COMMAND_ALLOCATE,
    /* 3  */ N2D_KERNEL_COMMAND_FREE,
    /* 4  */ N2D_KERNEL_COMMAND_MAP,
    /* 5  */ N2D_KERNEL_COMMAND_UNMAP,
    /* 6  */ N2D_KERNEL_COMMAND_COMMIT,
    /* 7  */ N2D_KERNEL_COMMAND_USER_SIGNAL,
    /* 8  */ N2D_KERNEL_COMMAND_SIGNAL,
    /* 9  */ N2D_KERNEL_COMMAND_EVENT_COMMIT,
    /* 10 */ N2D_KERNEL_COMMAND_GET_HW_INFO,
    /* 11 */ N2D_KERNEL_COMMAND_WRAP_USER_MEMORY,
    /* 12 */ N2D_KERNEL_COMMAND_EXPORT_VIDMEM,
    /* 13 */ N2D_KERNEL_COMMAND_CACHE,
} n2d_kernel_command_t;

typedef struct n2d_kernel_command_get_hw_info {
    /* [out] */ n2d_uint32_t    chipModel;
    /* [out] */ n2d_uint32_t    chipRevision;
    /* [out] */ n2d_uint32_t    productId;
    /* [out] */ n2d_uint32_t    cid;
    /* [out] */ n2d_uint32_t    chipFeatures;
    /* [out] */ n2d_uint32_t    chipMinorFeatures;
    /* [out] */ n2d_uint32_t    chipMinorFeatures1;
    /* [out] */ n2d_uint32_t    chipMinorFeatures2;
    /* [out] */ n2d_uint32_t    chipMinorFeatures3;
    /* [out] */ n2d_uint32_t    chipMinorFeatures4;
    /* [out] */ n2d_uint32_t    dev_core_num;
    /* [out] */ n2d_uint32_t    device_num;
} n2d_kernel_command_get_hw_info_t;

typedef struct n2d_kernel_command_allocate {
    /* [in] */  n2d_uint32_t         size;
    /* [in] */  n2d_uint32_t         flag;
    /* [in] */  n2d_vidmem_type_t    type;
    /* [in] */  n2d_vidmem_pool_t    pool;
    /* [out] */ n2d_uintptr_t        handle;
} n2d_kernel_command_allocate_t;

typedef struct n2d_kernel_command_free {
    /* [in] */  n2d_uintptr_t   handle;
} n2d_kernel_command_free_t;

typedef struct n2d_kernel_command_map {
    /* [in] */  n2d_uintptr_t   handle;
    /* [out] */ n2d_uint32_t    secure;
    /* [out] */ n2d_uintptr_t   logical;
    /* [out] */ n2d_uint64_t    address;
    /* [out] */ n2d_uint64_t    physical;
} n2d_kernel_command_map_t;

typedef struct n2d_kernel_command_unmap {
    /* [in] */  n2d_uintptr_t   handle;
} n2d_kernel_command_unmap_t;

typedef struct n2d_kernel_command_commit {
    /* [in] */  n2d_uint64_t    handle; /*node handle, used to flush cache.*/
    /* [in] */  n2d_uintptr_t   logical;
    /* [in] */  n2d_uint64_t    address;
    /* [in] */  n2d_uint32_t    offset;
    /* [in] */  n2d_uint32_t    size;
    /* [in] */  n2d_uintptr_t   queue;
    /*return value*/
    /* [out] */  n2d_uint32_t   ret;
} n2d_kernel_command_commit_t;

typedef struct n2d_kernel_command_user_signal {
    /* [in] */  n2d_user_signal_command_t   command;
    /* [in] */  n2d_uintptr_t               handle;
    /* [in] */  n2d_bool_t                  manual_reset;
    /* [in] */  n2d_uint32_t                wait;
    /* [in] */  n2d_bool_t                  state;
} n2d_kernel_command_user_signal_t;

/* gcvHAL_SIGNAL. */
typedef struct n2d_kernel_command_signal {
    /* [in] */ n2d_uintptr_t    handle;
    /* [in] */ n2d_uint32_t     process;
    /* [in] */ n2d_bool_t       from_kernel;
} n2d_kernel_command_signal_t;

typedef struct n2d_kernel_command_event_commit {
    /* [in] */  n2d_uintptr_t   queue;
    /* [in] */  n2d_bool_t      submit;
} n2d_kernel_command_event_commit_t;

/* nano2d wrap user memory */
typedef struct n2d_kernel_command_wrap_user_memory {
    /* [in] */  n2d_uint32_t    flag;

    /* dma_buf */
    /* [in] */  n2d_int32_t     fd_handle; /* dma_buf fd */
    /* [in] */  n2d_uintptr_t   dmabuf;

    /* user memory */
    /* [in] */  n2d_uintptr_t   logical;
    /* [in] */  n2d_uintptr_t   physical;
    /* [in] */  n2d_size_t      size;

    /* [out] */ n2d_uintptr_t   handle; /* wraped node handle */
} n2d_kernel_command_wrap_user_memory_t;

/* Export vidmem to dma_buf fd. */
typedef struct n2d_kernel_export_video_memory {
    /* Allocated video memory. */
    /* [in] */  n2d_uint32_t          handle;
    /* [in] */  n2d_uint32_t          flags; /* Export flags, mode flags for the file */
    /* [out] */ n2d_int32_t           fd;    /* Exported dma_buf fd */
} n2d_kernel_export_video_memory_t;

/* nano2d cache operation. */
typedef struct n2d_kernel_command_cache {
    /* [in] */ n2d_cache_op_t   operation;
    /* [in] */ n2d_uint64_t     bytes;
    /* [in] */ n2d_uint64_t     offset;
    /* [in] */ n2d_uint32_t     handle;
} n2d_kernel_command_cache_t;

typedef struct n2d_ioctl_interface {
    n2d_kernel_command_t        command;
    n2d_device_id_t             dev_id;
    n2d_core_id_t               core;
    n2d_error_t                 error;

    union _u {
        n2d_kernel_command_allocate_t       command_allocate;
        n2d_kernel_command_free_t           command_free;
        n2d_kernel_command_map_t            command_map;
        n2d_kernel_command_unmap_t          command_unmap;
        n2d_kernel_command_commit_t         command_commit;
        n2d_kernel_command_user_signal_t    command_user_signal;
        n2d_kernel_command_signal_t         command_signal;
        n2d_kernel_command_event_commit_t   command_event_commit;
        n2d_kernel_command_get_hw_info_t    command_get_hw_info;
        n2d_kernel_command_wrap_user_memory_t    command_wrap_user_memory;
        n2d_kernel_export_video_memory_t    command_export_vidmem;
        n2d_kernel_command_cache_t          command_cache;
    } u;
} n2d_ioctl_interface_t;

#ifdef  __cplusplus
}
#endif

#endif /* _nano2D_dispatch_h_ */
