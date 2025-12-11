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

#ifndef _nano2D_types_h__
#define _nano2D_types_h__

#ifdef __cplusplus
extern "C" {
#endif

#define IN
#define OUT

typedef int                 n2d_int_t;
typedef unsigned int        n2d_uint_t;

typedef int                 n2d_bool_t;
typedef unsigned char       n2d_uint8_t;
typedef char                n2d_int8_t;
typedef short               n2d_int16_t;
typedef unsigned short      n2d_uint16_t;
typedef int                 n2d_int32_t;
typedef unsigned int        n2d_uint32_t;
typedef unsigned long long  n2d_uint64_t;
typedef float               n2d_float_t;
typedef double              n2d_double_t;
typedef char                n2d_char;
typedef char*               n2d_string;
typedef void*               n2d_pointer;
typedef void                n2d_void;

typedef const char*         n2d_const_string;

typedef n2d_int32_t         n2d_color_t;

#if __KERNEL__
#include <asm/bitsperlong.h>
#if BITS_PER_LONG == 64
    typedef unsigned long long n2d_uintptr_t;
    typedef unsigned long long n2d_size_t;
#else
    typedef unsigned int n2d_uintptr_t;
    typedef unsigned int n2d_size_t;
#endif
#else
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
    typedef uintptr_t n2d_uintptr_t;
    typedef size_t n2d_size_t;
#endif


typedef enum _file_type {
    BMP,
    VIMG,
}
n2d_file_type;

#if !defined(__KERNEL__) && defined(__GNUC__) && (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__ >= 40300) \
   && !defined(__VXWORKS__)
#  define gcmBSWAP32(x)     __builtin_bswap32(x)
#else
#  define gcmBSWAP32(x) ((n2d_uint32_t)( \
    (((n2d_uint32_t)(x) & (n2d_uint32_t)0x000000FFUL) << 24) | \
    (((n2d_uint32_t)(x) & (n2d_uint32_t)0x0000FF00UL) << 8)  | \
    (((n2d_uint32_t)(x) & (n2d_uint32_t)0x00FF0000UL) >> 8)  | \
    (((n2d_uint32_t)(x) & (n2d_uint32_t)0xFF000000UL) >> 24)))
#endif

#define N2D_MAX(a, b) ((a) > (b) ? (a) : (b))
#define N2D_MIN(a, b) ((a) < (b) ? (a) : (b))

#define N2D_POW     pow
#define N2D_ABS     abs
#define N2D_ATOI    atoi

#define N2D_INFINITE ((n2d_uint32_t)(~0U))
#define N2D_INVALID_ADDRESS ((n2d_uintptr_t)(~0))

#define N2D_INVALID_HANDLE      0


#define N2D_INT16_MAX   (32767)
#define N2D_INT16_MIN   (-32768)
#define N2D_INT25_MAX   (16777215)
#define N2D_INT25_MIN   (-16777216)

#define N2D_MAXUINT32   0xffffffff

#ifdef __cplusplus
#define N2D_NULL 0
#else
#define N2D_NULL ((void *) 0)
#endif

#define N2D_TRUE  1
#define N2D_FALSE 0

#define N2D_ZERO  0

#define N2D_SIZEOF(a) \
( \
    (n2d_size_t) (sizeof(a)) \
)

#define N2D_COUNTOF(a) \
( \
    sizeof(a) / sizeof(a[0]) \
)

#define __gcmSTART(reg_field) \
    (0 ? reg_field)

#define __gcmEND(reg_field) \
    (1 ? reg_field)

#define __gcmGETSIZE(reg_field) \
    (__gcmEND(reg_field) - __gcmSTART(reg_field) + 1)

#define __gcmALIGN(data, reg_field) \
    (((n2d_uint32_t) (data)) << __gcmSTART(reg_field))

#define __gcmMASK(reg_field) \
    ((n2d_uint32_t) ((__gcmGETSIZE(reg_field) == 32) \
        ?  ~0 \
        : (~(~0 << __gcmGETSIZE(reg_field)))))

#define gcmVERIFYFIELDVALUE(data, reg, field, value) \
( \
    (((n2d_uint32_t) (data)) >> __gcmSTART(reg##_##field) & \
                             __gcmMASK(reg##_##field)) \
        == \
    (reg##_##field##_##value & __gcmMASK(reg##_##field)) \
)

#define gcmSETFIELD(data, reg, field, value) \
( \
    (((n2d_uint32_t) (data)) \
        & ~__gcmALIGN(__gcmMASK(reg##_##field), reg##_##field)) \
        |  __gcmALIGN((n2d_uint32_t) (value) \
            & __gcmMASK(reg##_##field), reg##_##field) \
)

#define gcmGETFIELD(data, reg, field) \
( \
    ((((n2d_uint32_t) (data)) >> __gcmSTART(reg##_##field)) \
        & __gcmMASK(reg##_##field)) \
)

#define gcmSETFIELDVALUE(data, reg, field, value) \
( \
    (((n2d_uint32_t) (data)) \
        & ~__gcmALIGN(__gcmMASK(reg##_##field), reg##_##field)) \
        |  __gcmALIGN(reg##_##field##_##value \
            & __gcmMASK(reg##_##field), reg##_##field) \
)

#define gcmSETMASKEDFIELDVALUE(reg, field, value) \
( \
    gcmSETFIELDVALUE(~0, reg,          field, value) & \
    gcmSETFIELDVALUE(~0, reg, MASK_ ## field, ENABLED) \
)

#define gcmSETMASKEDFIELD(reg, field, value) \
( \
    gcmSETFIELD     (~0, reg,          field, value) & \
    gcmSETFIELDVALUE(~0, reg, MASK_ ## field, ENABLED) \
)

#define gcmALIGN(n, align) \
( \
    ((n) + ((align) - 1)) & ~((align) - 1) \
)


#define gcmGET_PAGE_COUNT(size, page_size) \
( \
    ((size) + (page_size) - 1) / page_size \
)

#define gcmINT2PTR(i) \
( \
    (n2d_pointer)(n2d_uintptr_t)(i) \
)

#define gcmPTR2INT(p) \
( \
    (n2d_uint32_t)(n2d_uintptr_t)(p) \
)

#define gcmPTR_TO_UINT64(p) \
( \
    (n2d_uint64_t)(n2d_uintptr_t)(p)\
)

#define gcmUINT64_TO_PTR(u) \
( \
    (n2d_pointer)(n2d_uintptr_t)(u)\
)

#define gcmLONG2PTR(i) \
( \
    (n2d_pointer)(n2d_uint64_t)(i) \
)

#define gcmPTR2LONG(p) \
( \
    (n2d_uint64_t)(p) \
)

#define gcmFLOAT2INT(p) \
( \
    (n2d_int32_t)(p) \
)

#define gcmBIT(x) (1ULL << x)

#define N2D_IS_SUCCESS(error) (error == N2D_SUCCESS)
#define N2D_IS_ERROR(error)   (error != N2D_SUCCESS)

#define N2D_ON_ERROR(func) \
    do \
    { \
        error = func; \
        if (N2D_IS_ERROR(error)) \
        { \
            printf("[%s(%d)] error = 0x%x\n",__FUNCTION__,__LINE__,error);\
            goto on_error; \
        } \
    } \
    while (0)

#define ONERROR(func) \
    do \
    { \
        error = func; \
        if (N2D_IS_ERROR(error)) \
        { \
            n2d_kernel_os_print((char*)"ONERROR: %d %s(%d)\n", error, __FUNCTION__, __LINE__); \
            goto on_error; \
        } \
    } \
    while (0)

#define gcmkTRACE(...)

#define gcmkASSERT(exp)                                  \
    do {                                                 \
        if (!(exp)) {                                    \
            n2d_kernel_os_print("ASSERT at %s(%d)\n",    \
                                __FUNCTION__, __LINE__); \
        }                                                \
    } while (N2D_FALSE)

#define gcmVERIFY_ARGUMENT(arg) \
       do \
       { \
           if (!(arg)) \
           { \
               error = N2D_INVALID_ARGUMENT;\
               goto on_error; \
           } \
       } \
       while (N2D_FALSE)

#if defined(ANDROID) && defined(__BIONIC_FORTIFY)
#   if defined(__clang__)
#       if (__clang_major__ >= 10)
#           define gcmINLINE            __inline__ __attribute__ ((always_inline))
#       else
#           define gcmINLINE            __inline__ __attribute__ ((always_inline)) __attribute__ ((gnu_inline))
#       endif
#   else
#       define gcmINLINE            __inline__ __attribute__ ((always_inline)) __attribute__ ((gnu_inline)) __attribute__ ((artificial))
#   endif
#elif ((defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)) || defined(__APPLE__))
#   define gcmINLINE            inline      /* C99 keyword. */
#elif defined(__GNUC__)
#   define gcmINLINE            __inline__  /* GNU keyword. */
#elif defined(_MSC_VER) || defined(UNDER_CE)
#   define gcmINLINE            __inline    /* Internal keyword. */
#else
#   error "gcmINLINE: Platform could not be determined"
#endif

#ifndef N2D_ASSERT
#   define N2D_ASSERT(exp) \
        do \
        { \
            if (!(exp)) \
            { \
                printf( "ASSERT at %s(%d)", \
                              __FUNCTION__, __LINE__); \
                abort(); \
            } \
        } \
        while (N2D_FALSE)
#endif

#define N2D_SAFECASTVA(x, y)                                       \
    do {                                                           \
        n2d_uint32_t tmp = (n2d_uint32_t)(y);                      \
        if (N2D_SIZEOF(n2d_uint64_t) > N2D_SIZEOF(n2d_uint32_t)) { \
            gcmkASSERT(tmp <= N2D_MAXUINT32);                      \
        }                                                          \
        (x) = tmp;                                                 \
    } while (N2D_FALSE)

#ifdef __cplusplus
}
#endif

#endif
