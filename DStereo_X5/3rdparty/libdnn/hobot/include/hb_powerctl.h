/***
 *                     COPYRIGHT NOTICE
 *            Copyright (C) 2023 -2023 Horizon Robotics, Inc.
 *                   All rights reserved.
 ***/
#ifndef HB_POWERCTL_H_
#define HB_POWERCTL_H_
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define RESTART		0   /* CMD  RESTART */
#define POWEROFF	1   /* CMD  POWEROFF */


/**
 * @NO{S15E06C01I}
 * @ASIL{B}
 * @brief Interface to control system safe restart
 *
 * @param[in] cmd: reboot cmd
 * @param[in] arg: reboot args
 * @param[out] None
 *
 * @retval -1: invalid args
 * @retval others: reboot cmd ret
 *
 * @data_read None
 * @data_updated None
 *
 * @compatibility HW: J5/J6
 * @compatibility SW: v1.0.0
 *
 * @callgraph
 * @callergraph
 * @design
 */
unsigned int  hb_powerctl_reboot(unsigned int  cmd, char *arg);


/* For client */
#define PWR_CLIENT_MAGIC (0xdeeaf051)
#define PWR_CLIENT_VERSION (0x0)

typedef enum {
    PWR_ACT_SUSPEND,
    PWR_ACT_RESUME,
} pwr_act_t;

struct pwr_client_notify_msg {
    const uint32_t magic;
    const uint32_t version;
    pwr_act_t action;
};

struct pwr_client_status {
    int32_t status;
};

typedef int32_t (*hb_pwr_callback_t) (pwr_act_t act, void *arg);

struct hb_power_cb_handle;

struct hb_power_cb_handle *hb_powerctl_register_notify(
    hb_pwr_callback_t cb, void *arg);

void hb_powerctl_unregister_notify(struct hb_power_cb_handle *handle);

#ifdef __cplusplus
}
#endif

#endif
