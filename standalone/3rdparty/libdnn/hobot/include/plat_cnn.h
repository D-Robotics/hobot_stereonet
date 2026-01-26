/*************************************************************************
 *                     COPYRIGHT NOTICE
 *            Copyright 2016-2020 Horizon Robotics, Inc.
 *                   All rights reserved.
 *************************************************************************/

#ifndef __PLAT_CNN_H__
#define __PLAT_CNN_H__

#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>

#define BPU_PLAT_VERSION_MAJOR 1
#define BPU_PLAT_VERSION_MINOR 3
#define BPU_PLAT_VERSION_PATCH 6

/**
 * hb_bpu_version() - Gets the currently BPU library version
 * @major: major version pointer
 * @minor: minor pointer
 * @patch: patch pointer
 *
 * when success version value will be set to the param value
 *
 * Return:
 * * =0                        - success
 * * <0                        - error code
 */
int32_t hb_bpu_version(uint32_t *major, uint32_t *minor, uint32_t *patch);

/**
 * hb_bpu_build_commit() - Gets the currently BPU library the build commit
 * @commit: git commit pointer
 *
 * when success commit value will be set to the param value
 *
 * Return:
 * * =0                        - success
 * * <0                        - error code
 */
int32_t hb_bpu_build_commit(char *commit);

//////////////////////////CNN Core API//////////////////////////
#define DEFAULT_CORE_MASK      (0xFFFFFFFFu)
enum cnn_start_method {
	START_FROM_UNKNOWN,
	START_FROM_DDR,
	START_FROM_PYM,
	START_FROM_RESIZER,
	START_METHOD_NUM
};

/**
 * enum cnn_core_type - list bpu core types.
 * @CORE_TYPE_UNKNOWN: unkown pe type.
 * @CORE_TYPE_4PE: 4pe type.
 * @CORE_TYPE_1PE: 1pe type.
 * @CORE_TYPE_2PE: 2pe type.
 * @CORE_TYPE_ANY: Support all pe type.
 * @CORE_TYPE_INVALID: Invalid
 *
 * list the BPU support PE type
 */
enum cnn_core_type {
	CORE_TYPE_UNKNOWN,
	CORE_TYPE_4PE,
	CORE_TYPE_1PE,
	CORE_TYPE_2PE,
	CORE_TYPE_ANY,
	CORE_TYPE_INVALID,
};

typedef struct {
	int32_t left;
	int32_t top;
	int32_t right;
	int32_t bottom;

	/* origin image info */
	uint64_t y_ddr_base_addr;
	uint64_t uv_ddr_base_addr;
	int32_t img_w;
	int32_t img_h;
	int32_t img_w_stride;

	/* resizer y/uv data or not */
	int32_t y_en;
	int32_t uv_en;

	/* resize dst size */
	int32_t dst_w;
	int32_t dst_h;

	/* other set for resizer*/
	uint32_t padding_mode;
	uint64_t extra_params[4];
} roi_box_t;

/**
 * typedef fc_done_cb - callcack for fc process done.
 * @id: the fc id which process done
 * @err: error status
 *
 * If hb_bpu_set_fc* api use a valid fc_done param,
 * the callback will be called when the fc process
 * done.
 *
 * NOTE: the callback will affect the whole process,
 * so need return as soon as possible.
 */
typedef void (*fc_done_cb)(uint32_t id, int32_t err);

/**
 * (Legacy)cnn_core_num() - refer to hb_bpu_core_num
 */
int cnn_core_num(void);
/**
 * (Legacy)cnn_core_type() - refer to hb_bpu_core_type
 */
int cnn_core_type(unsigned int core_index);
/**
 * (Legacy)cnn_core_open() - refer to hb_bpu_core_open
 */
int cnn_core_open(unsigned int core_index);
/**
 * (Legacy)cnn_core_close() - refer to hb_bpu_core_close
 */
void cnn_core_close(unsigned int core_index);

/**
 * (Legacy)cnn_core_set_fc() - refer to hb_bpu_core_set_fc
 */
int cnn_core_set_fc(void *fc, int num,
		unsigned int core_index, fc_done_cb done_cb);
/**
 * (Legacy)cnn_core_set_fc_with_rsz() - refer to hb_bpu_core_set_fc_with_rsz
 */
int cnn_core_set_fc_with_rsz(void *fc, int num,
		unsigned int core_index, fc_done_cb done_cb,
		roi_box_t *boxes, int boxes_num);

/**
 * (Legacy)cnn_core_fc_avl_id() - refer to hb_bpu_core_avl_id
 */
int cnn_core_fc_avl_id(unsigned int core_index);
/**
 * (Legacy)cnn_core_fc_all_cap() - refer to hb_bpu_core_all_cap
 */
int cnn_core_fc_all_cap(unsigned int core_index);
/**
 * (Legacy)cnn_core_fc_avl_cap() - refer to hb_bpu_core_avl_cap
 */
int cnn_core_fc_avl_cap(unsigned int core_index);
/**
 * (Legacy)cnn_core_wait_fc_done() - refer to hb_bpu_core_wait_fc_done
 */
int cnn_core_wait_fc_done(unsigned int core_index, int timeout);

/**
 * (Legacy)cnn_core_check_fc_done() - refer to hb_bpu_core_check_fc_done
 */
int cnn_core_check_fc_done(unsigned int core_index,
		uint32_t id, int timeout);

/**
 * (Legacy)cnn_set_group_proportion() - refer to hb_bpu_set_group_proportion
 */
int cnn_set_group_proportion(int group_id, int proportion);
/**
 * (Legacy)cnn_core_set_fc_group() - refer to hb_bpu_core_set_fc_group
 */
int cnn_core_set_fc_group(void *fc, int num,
		unsigned int core_index, fc_done_cb done_cb,
		int group_id);
/**
 * (Legacy)cnn_core_set_fc_with_rsz_group() - refer to hb_bpu_core_set_fc_with_rsz_group
 */
int cnn_core_set_fc_with_rsz_group(void *fc, int num,
		unsigned int core_index, fc_done_cb done_cb,
		roi_box_t *boxes, int boxes_num, int group_id);

/**
 * hb_bpu_core_num() - Get BPU Core number
 *
 * Get the bpu core number which support
 *
 * Return: bpu core number
 */
int32_t hb_bpu_core_num(void);
/**
 * hb_bpu_core_type() - Get BPU Core support instruction type
 * @core_index: bpu core index
 *
 * The BPU Core type formate:
 * bit0-23: BPU Algorithm architecture
 * bit24-27: PE type(CORE_TYPE**)
 *
 * Return:
 * * >0                         - Core type
 * * <=0                        - error code
 */
int32_t hb_bpu_core_type(uint32_t core_index);
/**
 * hb_bpu_core_open() - Open BPU Core
 * @core_index: bpu core index
 *
 * Enable BPU Core hardware and init software
 * resource.
 *
 * Return:
 * * =0                        - success
 * * <0                        - error code
 */
int32_t hb_bpu_core_open(uint32_t core_index);
/**
 * hb_bpu_core_close() - Close BPU Core
 * @core_index: bpu core index
 *
 * Disable BPU Core hardware and deinit software
 * resource.
 */
void hb_bpu_core_close(uint32_t core_index);

/**
 * hb_bpu_core_set_fc() - Set BPU Task(functioncall)
 * 						  to process by BPU Core
 * @fc: functioncall pointer which from hbrt tool
 * @num: functioncall number
 * @core_mask: bpu core index bit mask
 * @done_cb: callback function which whill be call
 * 			 when task process done(could be null)
 *
 * Functioncall from hbrt(horizon runtimelib) which parse
 * from algorithm model;
 * core_mask's bit identify the bpu core
 * for example, 0x1 is core0, 0x3 is core0 and core1
 * DEFAULT_CORE_MASK is all bpu cores.
 *
 * Return:
 * * >=0                       - success
 * * <0                        - error code
 */
int32_t hb_bpu_core_set_fc(void *fc, uint32_t num,
		uint32_t core_mask, fc_done_cb done_cb);

/**
 * (Legacy)hb_bpu_core_set_fc_with_rsz() - Set BPU Task(functioncall)
 * 						  		   		   to process by BPU Core with
 * 						  		   		   roi box information
 * @fc: functioncall pointer which from hbrt tool
 * @num: functioncall number
 * @core_mask: bpu core index bit mask
 * @done_cb: callback function which whill be call
 * 			 when task process done(could be null)
 * @boxes: roi box pointers
 * @boxes_num: roi box number
 *
 * Functioncall from hbrt(horizon runtimelib) which parse
 * from algorithm model;
 * core_mask's bit identify the bpu core
 * for example, 0x1 is core0, 0x3 is core0 and core1
 * DEFAULT_CORE_MASK is all bpu cores.
 *
 * NOTE: only for old HBDK2 version ,which legacy for XJ2
 *
 * Return:
 * * >=0                       - success
 * * <0                        - error code
 */
int32_t hb_bpu_core_set_fc_with_rsz(void *fc,
		uint32_t num, uint32_t core_mask,
		fc_done_cb done_cb, roi_box_t *boxes,
		uint32_t boxes_num);

/**
 * hb_bpu_core_fc_avl_id() - Get recommendation fc id
 * @core_index: bpu core index
 *
 * fc id user to identify different task
 *
 * Return:
 * * >0                        - id value
 * * <=0                       - error code
 */
int32_t hb_bpu_core_fc_avl_id(uint32_t core_index);
/**
 * hb_bpu_core_fc_all_cap() - Get fc buffer size of the BPU Core
 * @core_index: bpu core index
 *
 * Return:
 * * >0                        - id value
 * * <=0                       - error code
 */
int32_t hb_bpu_core_fc_all_cap(uint32_t core_index);
/**
 * hb_bpu_core_fc_avl_cap() - Get freefc buffer size of the BPU Core
 * 							  in the moment.
 * @core_index: bpu core index
 *
 * Return:
 * * >0                        - id value
 * * <=0                       - error code
 */
int32_t hb_bpu_core_fc_avl_cap(uint32_t core_index);

/**
 * hb_bpu_core_wait_fc_done() - Wait for the fc task to be process
 * 								done on the specified coremask
 * @core_mask: bpu core index bit mask
 * @timeout: wait timeout(ms)
 *
 * core_mask same with the value which used in *set_fc*
 *
 * NOTE: the API can't be used when done_cb not NULL when
 * 		 *set_fc*
 *
 * Return:
 * * >0                       - fc id which process done
 * * <0                       - error code
 */
int32_t hb_bpu_core_wait_fc_done(uint32_t core_mask, int32_t timeout);

/**
 * hb_bpu_core_check_fc_done() - Wait/check for the specified fc
 * 								task to be process done on the
 * 								specified coremask
 * @core_mask: bpu core index bit mask
 * @id: the id of the fc task which want to wait/check
 * @timeout: wait timeout(ms)
 *
 * core_mask same with the value which used in *set_fc*
 *
 * NOTE: the API can't be used when done_cb not NULL when
 * 		 *set_fc*
 *
 * Return:
 * * >0                       - fc id which process done
 * * <0                       - error code
 */
int32_t hb_bpu_core_check_fc_done(uint32_t core_mask,
		uint32_t id, int32_t timeout);

/**
 * hb_bpu_set_group_proportion() - set group run proportion
 * @group_id: group id for fc task bind
 * @proportion: the max bpu running proportion for the fcs
 *				which bind with the group_id
 *
 * the proportion could be [0, 100], if 0, mean delete the group
 *
 * NOTE: Default group_id = 0, and proportion = 100
 *
 * Return:
 * * =0                       - success
 * * <0                       - error code
 */
int32_t hb_bpu_set_group_proportion(uint32_t group_id,
		uint32_t proportion);

/**
 * hb_bpu_core_set_fc_group() - Set BPU Task(functioncall)
 * 						  		to process by BPU Core bind
 * 						  		task group
 * @fc: functioncall pointer which from hbrt tool
 * @num: functioncall number
 * @core_mask: bpu core index bit mask
 * @done_cb: callback function which whill be call
 * 			 when task process done(could be null)
 * @group_id: the group id which configure before
 *
 * Functioncall from hbrt(horizon runtimelib) which parse
 * from algorithm model;
 * core_mask's bit identify the bpu core
 * group_id high 16bit use for prio level, group_id 0 is
 * default group not net configure by user
 *
 * Return:
 * * >=0                       - success
 * * <0                        - error code
 */
int32_t hb_bpu_core_set_fc_group(void *fc, uint32_t num,
		uint32_t core_mask, fc_done_cb done_cb,
		uint32_t group_id);

/**
 * (Legacy)hb_bpu_core_set_fc_with_rsz_group() - Set BPU Task(functioncall)
 * 						  		   		   		 to process by BPU Core with
 * 						  		   		   		 roi box information and bind
 * 						  		   		   		 task group.
 * @fc: functioncall pointer which from hbrt tool
 * @num: functioncall number
 * @core_mask: bpu core index bit mask
 * @done_cb: callback function which whill be call
 * 			 when task process done(could be null)
 * @boxes: roi box pointers
 * @boxes_num: roi box number
 * @group_id: the group id which configure before
 *
 * Functioncall from hbrt(horizon runtimelib) which parse
 * from algorithm model;
 * core_mask's bit identify the bpu core
 * for example, 0x1 is core0, 0x3 is core0 and core1
 * DEFAULT_CORE_MASK is all bpu cores.
 * group_id high 16bit use for prio level, group_id 0 is
 * default group not net configure by user
 *
 * NOTE: only for old HBDK2 version ,which legacy for XJ2
 *
 * Return:
 * * >=0                       - success
 * * <0                        - error code
 */
int32_t hb_bpu_core_set_fc_with_rsz_group(void *fc, uint32_t num,
		uint32_t core_mask, fc_done_cb done_cb,
		roi_box_t *boxes, uint32_t boxes_num,
		uint32_t group_id);

/**
 * hb_bpu_core_set_fc_prio() - Set BPU Task(functioncall)
 * 						  		to process by BPU Core bind
 * 						  		task priority
 * @fc: functioncall pointer which from hbrt tool
 * @num: functioncall number
 * @core_mask: bpu core index bit mask
 * @done_cb: callback function which whill be call
 * 			 when task process done(could be null)
 * @prio_level: the priority level
 *
 * Functioncall from hbrt(horizon runtimelib) which parse
 * from algorithm model;
 * core_mask's bit identify the bpu core
 * group_id high 16bit can use for prio level
 * priority strategy need to be coordinated with HBDK-specific usage
 *
 * Return:
 * * >=0                       - success
 * * <0                        - error code
 */
int32_t hb_bpu_core_set_fc_prio(void *fc, uint32_t num,
		uint32_t core_mask, fc_done_cb done_cb,
		uint32_t prio_level);

/**
 * (Legacy)hb_bpu_core_set_fc_with_rsz_prio() - Set BPU Task(functioncall)
 * 						  		   		   		 to process by BPU Core with
 * 						  		   		   		 roi box information and bind
 * 										  		 task priority
 * @fc: functioncall pointer which from hbrt tool
 * @num: functioncall number
 * @core_mask: bpu core index bit mask
 * @done_cb: callback function which whill be call
 * 			 when task process done(could be null)
 * @boxes: roi box pointers
 * @boxes_num: roi box number
 * @prio_level: the priority level
 *
 * Functioncall from hbrt(horizon runtimelib) which parse
 * from algorithm model;
 * core_mask's bit identify the bpu core
 * for example, 0x1 is core0, 0x3 is core0 and core1
 * DEFAULT_CORE_MASK is all bpu cores.
 * priority strategy need to be coordinated with HBDK-specific usage
 *
 * NOTE: only for old HBDK2 version ,which legacy for XJ2
 *
 * Return:
 * * >=0                       - success
 * * <0                        - error code
 */
int32_t hb_bpu_core_set_fc_with_rsz_prio(void *fc, uint32_t num,
		uint32_t core_mask, fc_done_cb done_cb,
		roi_box_t *boxes, uint32_t boxes_num,
		uint32_t prio_level);
////////////////////////////////////////////////////////////////

//////////////////////////Memory API////////////////////////////
#define bpu_addr_t uint64_t
#define BPU_NON_CACHEABLE (0)
#define BPU_CACHEABLE (1)
#define BPU_CORE0 (0x10000)
#define BPU_CORE1 (0x20000)
#define ARM_TO_CNN (0)
#define CNN_TO_ARM (1)
#define BPU_MEM_INVALIDATE (1)
#define BPU_MEM_CLEAN (2)

/**
 * (Legacy)bpu_mem_register() - refer to hb_bpu_mem_register
 */
bpu_addr_t bpu_mem_register(void *phy_addr, int size);
/**
 * (Legacy)bpu_mem_unregister() - refer to hb_bpu_mem_unregister
 */
void bpu_mem_unregister(bpu_addr_t addr);

/**
 * (Legacy)bpu_mem_alloc() - refer to hb_bpu_mem_alloc
 */
bpu_addr_t bpu_mem_alloc(int size, int flag);
/**
 * (Legacy)bpu_cpumem_alloc() - refer to hb_bpu_cpumem_alloc
 */
bpu_addr_t bpu_cpumem_alloc(int size, int flag);
/**
 * (Legacy)bpu_mem_alloc_with_label() - refer to hb_bpu_mem_alloc_with_label
 */
bpu_addr_t bpu_mem_alloc_with_label(int size, int flag, const char* label);
/**
 * (Legacy)bpu_cpumem_alloc_with_label() - refer to hb_bpu_cpumem_alloc_with_label
 */
bpu_addr_t bpu_cpumem_alloc_with_label(int size, int flag, const char* label);
/**
 * (Legacy)bpu_mem_free() - refer to hb_bpu_mem_free
 */
void bpu_mem_free(bpu_addr_t addr);
/**
 * (Legacy)bpu_cpumem_free() - refer to hb_bpu_cpumem_free
 */
void bpu_cpumem_free(bpu_addr_t addr);
/**
 * (Legacy)bpu_memcpy() - refer to hb_bpu_memcpy
 */
int bpu_memcpy(bpu_addr_t dst_addr, bpu_addr_t src_addr, unsigned size, int direction);
/**
 * (Legacy)bpu_mem_cache_flush() - refer to hb_bpu_mem_cache_flush
 */
void bpu_mem_cache_flush(bpu_addr_t addr, int size, int flag);
/**
 * (Legacy)bpu_mem_is_cacheable() - refer to hb_bpu_mem_is_cacheable
 */
int bpu_mem_is_cacheable(bpu_addr_t addr);
/**
 * (Legacy)bpu_mem_phyaddr() - refer to hb_bpu_mem_phyaddr
 */
uint64_t bpu_mem_phyaddr(bpu_addr_t addr);

/**
 * hb_bpu_mem_register() - register mem space for libbpu manage
 * @phy_addr: the memory space start address
 * @size: mem space size
 *
 * mem space must be continuous ddr space
 * phy_addr must Page(0x1000) align
 *
 * Return:
 * * !=0                        - valid bpu_addr_t
 * * =0                        - failed
 */
bpu_addr_t hb_bpu_mem_register(void *phy_addr, uint32_t size);
/**
 * hb_bpu_mem_unregister() - unregister mem space which
 * 						   reigsted in libbpu manage
 * @addr: the valid bpu_addr_t which get form
 * 		  reigster API.
 */
void hb_bpu_mem_unregister(bpu_addr_t addr);

/**
 * hb_bpu_mem_alloc() - alloc bpu mem space from libbpu manage
 * @size: mem space size
 * @flag: mem alloc flag which define before
 *
 * The memory which can be accessed by BPU HW, some platform can
 * also be accessed by CPU.
 * Default label is NULL
 *
 * Return:
 * * !=0                       - valid bpu_addr_t
 * * =0                        - failed
 */
bpu_addr_t hb_bpu_mem_alloc(uint32_t size, uint32_t flag);
/**
 * hb_bpu_cpumem_alloc() - alloc cpu mem space from libbpu manage
 * @size: mem space size
 * @flag: mem alloc flag which define before
 *
 * The memory which can be accessed by CPU, some platform can
 * also be accessed by BPU_HW.
 *
 * Return:
 * * !=0                       - valid bpu_addr_t
 * * =0                        - failed
 */
bpu_addr_t hb_bpu_cpumem_alloc(uint32_t size, uint32_t flag);
/**
 * hb_bpu_mem_alloc_with_label() - alloc bpu mem space from libbpu manage
 * @size: mem space size
 * @flag: mem alloc flag which define before
 * @label: the string label(could be NULL)
 *
 * The memory which can be accessed by BPU HW, some platform can
 * also be accessed by CPU.
 *
 * Return:
 * * !=0                       - valid bpu_addr_t
 * * =0                        - failed
 */
bpu_addr_t hb_bpu_mem_alloc_with_label(uint32_t size,
		uint32_t flag, const char* label);
/**
 * hb_bpu_cpumem_alloc_with_label() - alloc cpu mem space from libbpu manage
 * @size: mem space size
 * @flag: mem alloc flag which define before
 * @label: the string label(could be NULL)
 *
 * The memory which can be accessed by CPU, some platform can
 * also be accessed by BPU_HW.
 *
 * Return:
 * * !=0                       - valid bpu_addr_t
 * * =0                        - failed
 */
bpu_addr_t hb_bpu_cpumem_alloc_with_label(uint32_t size,
		uint32_t flag, const char* label);

/**
 * hb_bpu_mem_free() - free mem space which alloc by hb_bpu_mem_allc*
 * @addr: the valid bpu_addr_t which get form
 * 		  alloc API.
 */
void hb_bpu_mem_free(bpu_addr_t addr);
/**
 * hb_bpu_cpumem_free() - free mem space which alloc by hb_bpu_cpumem_allc*
 * @addr: the valid bpu_addr_t which get form
 * 		  alloc API.
 */
void hb_bpu_cpumem_free(bpu_addr_t addr);

/**
 * hb_bpu_memcpy() - copy data from diff type hb_addr_t
 * @dst_addr: valid destination bpu_addr
 * @src_addr: valid source bpu_addr
 * @size: copy data size
 * @direction: the mem space copy direction type, type
 * 			   could be (ARM_TO_CNN/CNN_TO_ARM)
 *
 * The realize use dma transfer
 *
 * Return:
 * * =0                       - success
 * * <0                       - error code
 */
int32_t hb_bpu_memcpy(bpu_addr_t dst_addr, bpu_addr_t src_addr,
		uint32_t size, uint32_t direction);

/**
 * hb_bpu_mem_cache_flush() - cache operation for bpu addr
 * @addr: bpu_addr start
 * @size: mem size
 * @flag: cache flush flag(BPU_MEM_INVALIDATE/BPU_MEM_CLEAN)
 *
 * The operation for bpu addr which alloc use BPU_CACHEABLE flag
 */
void hb_bpu_mem_cache_flush(bpu_addr_t addr, uint32_t size,
		uint32_t flag);

/**
 * hb_bpu_mem_is_cacheable() - get the bpu_addr cacheable state
 * @addr: bpu_addr
 *
 * Return:
 * * >0                       - cacheable
 * * =0                       - non-cacheable
 * * <0                       - invalid bpu_addr
 */
int32_t hb_bpu_mem_is_cacheable(bpu_addr_t addr);
/**
 * hb_bpu_mem_phyaddr() - get the bpu_addr ddr phyical address
 * @addr: bpu_addr
 *
 * Return:
 * * !=0                       - phyical address
 * * =0                        - invalid bpu_addr
 */
uint64_t hb_bpu_mem_phyaddr(bpu_addr_t addr);
////////////////////////////////////////////////////////////////

//////////////////////// Hardware oprations API////////////////
#define BPU_HIGHEST_FRQ  0
#define BPU_POWER_OFF    0
#define BPU_POWER_ON    1
#define BPU_CLK_OFF      0
#define BPU_CLK_ON      1

/**
 * (Legacy)bpu_set_power() - refer to hb_bpu_set_power
 */
int bpu_set_power(int core_index, unsigned int status);
/**
 * (Legacy)bpu_set_frq_level() - refer to hb_bpu_set_frq_level
 */
int bpu_set_frq_level(int core_index, int level);
/**
 * (Legacy)bpu_set_clk() - refer to hb_bpu_set_clk
 */
int bpu_set_clk(int core_index, unsigned int status);
/**
 * (Legacy)bpu_get_total_level() - refer to hb_bpu_get_total_level
 */
int bpu_get_total_level(unsigned int core_index);
/**
 * (Legacy)bpu_get_cur_level() - refer to hb_bpu_get_cur_level
 */
int bpu_get_cur_level(unsigned int core_index);

/**
 * hb_bpu_set_power() - power on/off bpu core when bpu running
 * @core_index: bpu core index
 * @status: BPU_POWER_OFF/BPU_POWER_ON
 *
 * Only can use after hb_bpu_core_open, and before
 * hb_bpu_core_close
 *
 * Return:
 * * =0                       - success
 * * <0                       - error code
 */
int32_t hb_bpu_set_power(uint32_t core_index, uint32_t status);
/**
 * hb_bpu_set_frq_level() - set bpu core work frequency level
 * @core_index: bpu core index
 * @level: frequency level
 *
 * BPU_HIGHEST_FRQ is the highest frequency level
 * (BPU_HIGHEST_FRQ - n) will lower the frequency
 *
 * Return:
 * * =0                       - success
 * * <0                       - error code
 */
int32_t hb_bpu_set_frq_level(uint32_t core_index, int32_t level);
/**
 * hb_bpu_set_clk() - set bpu core work clk status or rate
 * @core_index: bpu core index
 * @val: status or rate value
 *
 * BPU_CLK_OFF/BPU_CLK_ON status to clock off or clock on
 * Specific frequency val if not status.
 *
 * NOTE: Don't set clock rate only when you know the
 * exactly available rates
 *
 * Return:
 * * =0                       - success
 * * <0                       - error code
 */
int32_t hb_bpu_set_clk(uint32_t core_index, uint64_t val);
/**
 * hb_bpu_get_clk() - get bpu core clk rate
 * @core_index: bpu core index
 *
 * Return:
 * * >0                       - working clock rate
 * * =0                       - bpu core not power on
 */
uint64_t hb_bpu_get_clk(uint32_t core_index);
/**
 * hb_bpu_get_total_level() - get bpu core total support frequency level
 * @core_index: bpu core index
 *
 * Return:
 * * >0                       - total frequency level number
 * * <0                       - error code
 */
int32_t hb_bpu_get_total_level(uint32_t core_index);
/**
 * hb_bpu_get_cur_level() - get bpu core working frequency level
 * @core_index: bpu core index
 *
 * Return:
 * * >=0                      - working frequency level
 * * <0                       - error code
 */
int32_t hb_bpu_get_cur_level(uint32_t core_index);

/**
 * hb_bpu_core_estimate_loading() - estimate total bpu core buffered
 * tasks process time
 *
 * @core_index: bpu core index
 * @prio_level: the priority level
 *
 *
 * Return: <0: error; >= 0: estimate time(us)
 */
int64_t hb_bpu_core_estimate_loading(uint32_t core_index, uint32_t prio_level);

#ifdef __cplusplus
}
#endif
#endif
