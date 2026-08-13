// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef VP_HB_VP_JPEG_CODEC_H_
#define VP_HB_VP_JPEG_CODEC_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include "hb_vp.h"

#define HB_VP_INITIALIZE_JPEG_ENC_PARAM(param) \
  {                                            \
    (param)->qualityFactor = 50;               \
    (param)->extendedSequential = false;       \
    (param)->outBufCount = 5;                  \
    (param)->backend = HB_UCP_JPU_CORE_0;      \
  }

#define HB_VP_INITIALIZE_JPEG_DEC_PARAM(param) \
  {                                            \
    (param)->outBufCount = 5;                  \
    (param)->backend = HB_UCP_JPU_CORE_0;      \
  }

/**
 * the jpeg encoding parameters
 * extendedSequential: enable 12bit encoding method，values[0, 1].
                       Currently only support 8bit encoding.
 * imageFormat: the format of input image. 
                support HB_VP_IMAGE_FORMAT_NV12, HB_VP_IMAGE_FORMAT_YUV444
                HB_VP_IMAGE_FORMAT_YUV444_P and HB_VP_IMAGE_FORMAT_YUV420.
 * width: the width of input image, values[32, 8192].
 * height: the height of input image, values[32, 8192].
 * qualityFactor: quality factor, values[1, 100], recommended value is 50.
 * outBufCount: the count of output buffers, values[1, 1000], recommended value is 5.
 * backend: specifies the execution backend for JPEG encoding tasks.
*/
typedef struct {
  uint8_t extendedSequential;
  uint8_t imageFormat;
  int32_t width;
  int32_t height;
  uint32_t qualityFactor;
  uint32_t outBufCount;
  uint64_t backend;
} hbVPJPEGEncParam;

/**
 * JPEG decoding parameters
 * imageFormat: the format of output image.
                support HB_VP_IMAGE_FORMAT_NV12, HB_VP_IMAGE_FORMAT_YUV444
                HB_VP_IMAGE_FORMAT_YUV444_P and HB_VP_IMAGE_FORMAT_YUV420.
 * outBufCount: the count of output buffers. values[1, 31], recommended value is 5.
 * backend: specifies the execution backend for JPEG decoding tasks.
 */
typedef struct {
  uint8_t imageFormat;
  uint32_t outBufCount;
  uint64_t backend;
} hbVPJPEGDecParam;

typedef void *hbVPJPEGContext;

/**
 * @brief generate JPEG encoding context
 *
 * @param[out] context JPEG encoding context, dispatched on JPU
 * @param[in] param param for jpeg encode
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPCreateJPEGEncContext(hbVPJPEGContext *context,
                                 hbVPJPEGEncParam const *param);

/**
 * @brief release JPEG encoding context
 *
 * @param[in] context JPEG encoding context 
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPReleaseJPEGEncContext(hbVPJPEGContext context);

/**
 * @brief Encode YUV image to JPEG
 * 
 * @param[out] taskHandle handle of task
 * case1: given *taskHandle is nullptr, create new task handle
 * @param[in] srcImg the image to be encoded
 * @param[in] context JPEG encode context
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPJPEGEncode(hbUCPTaskHandle_t *taskHandle, hbVPImage const *srcImg,
                       hbVPJPEGContext context);

/**
 * @brief get output buffer from codec
 
 * @param[out] outBuf the buffer to store the encoded JPEG data.
 *                    The buffer is allocated internally by the codec, 
 *                    contains valid data upon the task is successfully completed, 
 *                    and is released during the task release phase.
 * @param[in] taskHandle handle of task, given handle created by hbVPJPEGEncode
 * @return 0 if success, return defined error code otherwise
*/
int32_t hbVPGetJPEGEncOutputBuffer(hbVPArray *outBuf,
                                   hbUCPTaskHandle_t taskHandle);
/**
 * @brief generate JPEG decoding context
 *
 * @param[out] context JPEG decoding  context, dispatched on JPU
 * @param[in] param param for jpeg decode
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPCreateJPEGDecContext(hbVPJPEGContext *context,
                                 hbVPJPEGDecParam const *param);

/**
 * @brief release JPEG decoding context
 *
 * @param[in] context JPEG decoding context 
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPReleaseJPEGDecContext(hbVPJPEGContext context);

/**
 * @brief Decode JPEG to YUV
 * 
 * @param[out] taskHandle handle of task
 * case1: given *taskHandle is nullptr, create new task handle
 * @param[in] srcBuf the JPEG data to be decoded
 * @param[in] context JPEG decoding context
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPJPEGDecode(hbUCPTaskHandle_t *taskHandle, hbVPArray const *srcBuf,
                       hbVPJPEGContext context);

/**
 * @brief get output buffer from codec
 
 * @param[out] outImg the buffer to store the decoded Image data.
 *                    The buffer is allocated internally by the codec, 
 *                    contains valid data upon the task is successfully completed, 
 *                    and is released during the task release phase.
 * @param[in] taskHandle handle of task, given handle created by hbVPJPEGDecode
 * @return 0 if success, return defined error code otherwise
*/
int32_t hbVPGetJPEGDecOutputBuffer(hbVPImage *outImg,
                                   hbUCPTaskHandle_t taskHandle);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // VP_HB_VP_JPEG_CODEC_H_
