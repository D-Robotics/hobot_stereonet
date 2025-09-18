#ifndef __CL_EXT_VIV_H
#define __CL_EXT_VIV_H

#include <CL/cl.h>
#include <CL/cl_platform.h>

#ifdef __cplusplus
extern "C" {
#endif

/* address or size is not aligned */
#define CL_NOT_ALIGNED          -1143


/********************************************************************
* cl_vivante_device_attribute_query
********************************************************************/
#define CL_MEM_USE_UNCACHED_HOST_MEMORY_VIV         (1 << 28)

/* for CL_MEM_USE_HOST_PHYSICAL_ADDR_VIV, application must make
   sure the physical address passed in is a 40 bit address
*/
#define CL_MEM_USE_HOST_PHYSICAL_ADDR_VIV           (1 << 29)


/* external sram */
#define CL_MEM_ALLOC_FROM_EXTERNAL_SRAM_VIV         (((cl_mem_flags)1) << 63)
/* return type: size_t[] */
#define CL_DEVICE_EXTERNAL_SRAM_SIZE_VIV            0x4280
/* return type: cl_uint */
#define CL_DEVICE_VIDMEM_CHANNEL_COUNT_VIV          0x4281


/* properties for clCreateBufferWithProperties and clCreateImageWithProperties */
/* property value type: cl_uint */
#define CL_DEVICE_ALLOC_FROM_EXTSRAM_INDEX_VIV      0x4282
/* property value type: cl_uint */
#define CL_DEVICE_ALLOC_FROM_VIDMEM_INDEX_VIV       0x4283

/* dump IR for compiler */
/* function name: clSetCompilerDumpOptionVIV */
#define DUMP_OPTION_OFF                             0x0
#define DUMP_OPTION_ALL                             0xFFFFFFFF
typedef CL_API_ENTRY cl_int (CL_API_CALL *clSetCompilerDumpOptionVIV_fn)(cl_uint option);

#ifdef __cplusplus
}
#endif

#endif
