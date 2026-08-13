// Copyright (c) [2021-2023] [Horizon Robotics][Horizon Bole].
//
// You can use this software according to the terms and conditions of
// the Apache v2.0.
// You may obtain a copy of Apache v2.0. at:
//
//     http: //www.apache.org/licenses/LICENSE-2.0
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See Apache v2.0 for more details.

#ifndef VP_HB_VP_VIDEO_CODEC_H_
#define VP_HB_VP_VIDEO_CODEC_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include "hb_vp.h"

/**
 * HB_VP_VIDEO_TYPE_H264: video type is H264
 * HB_VP_VIDEO_TYPE_H265: video type is H265
 */
typedef enum { HB_VP_VIDEO_TYPE_H264, HB_VP_VIDEO_TYPE_H265 } hbVPVideoType;

/**
 * the rate control mode
*/
typedef enum {
  HB_VP_VIDEO_RC_MODE_H264_CBR,
  HB_VP_VIDEO_RC_MODE_H264_VBR,
  HB_VP_VIDEO_RC_MODE_H264_AVBR,
  HB_VP_VIDEO_RC_MODE_H264_FIXQP,
  HB_VP_VIDEO_RC_MODE_H264_QPMAP,
  HB_VP_VIDEO_RC_MODE_H265_CBR,
  HB_VP_VIDEO_RC_MODE_H265_VBR,
  HB_VP_VIDEO_RC_MODE_H265_AVBR,
  HB_VP_VIDEO_RC_MODE_H265_FIXQP,
  HB_VP_VIDEO_RC_MODE_H265_QPMAP
} hbVPVideoRcMode;

/**
 * the parameter of H264 Constant Bit Rate(CBR).
 * intraPeriod: I frame interval, values[0, 2047] and the recommended value is 28.
 * bitRate: the target average bitrate of the encoded data in kbps, multiplied 
 *          by 1024 must be greater than or equal to the frame rate.
 *          values[1, 700000]kbps and the recommended value is 1000kps.
 * frameRate: the target frame rate of the encoded data in fps, 
 *            values[1, 240] and the recommended value is 30
 * initialRcQp: specifies the initial QP by user, 
 *              values[0, 51], The larger the value, the worse the image quality.
 *              If the value exceeds 51, it is determined by the VPU hardware.
*/
typedef struct {
  uint32_t intraPeriod;
  uint32_t bitRate;
  uint32_t frameRate;
  uint32_t initialRcQp;
} hbVPVideoH264Cbr;

/**
 * the parameter of H264 Variable Bit Rate(VBR).
 * intraPeriod: I frame interval, values[0, 2047] and the recommended value is 28.
 * intraQp: the quantization parameter of intra picture,
 *          values[0, 51] and the recommended value is 30
 * frameRate: the target frame rate of the encoded data in fps, 
 *            values[1, 240] and the recommended value is 30
*/
typedef struct {
  uint32_t intraPeriod;
  uint32_t intraQp;
  uint32_t frameRate;
} hbVPVideoH264Vbr;

/**
 * the parameter of H264 Fix Qp.
 * intraPeriod: I frame interval, values[0, 2047] and the recommended value is 28.
 * frameRate: the target frame rate of the encoded data in fps, 
 *            values[1, 240] and the recommended value is 30
 * qpI: a picture quantization parameter for I picture,
 *      values[0, 51] and the recommended value is 0
 * qpP: a picture quantization parameter for P picture,
 *      values[0, 51] and the recommended value is 0
 * qpB: a picture quantization parameter for B picture,
 *      values[0, 51] and the recommended value is 0
*/
typedef struct {
  uint32_t intraPeriod;
  uint32_t frameRate;
  uint32_t qpI;
  uint32_t qpP;
  uint32_t qpB;
} hbVPVideoH264FixQp;

/**
 * the parameter of H264 Qp Map.
 * intraPeriod: I frame interval, values[0, 2047] and the recommended value is 28.
 * frameRate: the target frame rate of the encoded data in fps, 
 *            values[1, 240] and the recommended value is 30
 * qpMapArrayCount: specify the qp map number, calculated as 
 *                  (ALIGN16(width) >> 4) * (ALIGN16(height) >> 4)
 * qpMapArray: specify the qp map, QP values[0, 51]
*/
typedef struct {
  uint32_t intraPeriod;
  uint32_t frameRate;
  uint32_t qpMapArrayCount;
  uint8_t *qpMapArray;
} hbVPVideoH264QpMap;

typedef hbVPVideoH264Cbr hbVPVideoH264Avbr;
typedef hbVPVideoH264Cbr hbVPVideoH265Cbr;
typedef hbVPVideoH264Vbr hbVPVideoH265Vbr;
typedef hbVPVideoH264Avbr hbVPVideoH265Avbr;
typedef hbVPVideoH264FixQp hbVPVideoH265FixQp;
typedef hbVPVideoH264QpMap hbVPVideoH265QpMap;

/**
 * the parameter of rate control.
 * mode: rate control mode
*/
typedef struct {
  hbVPVideoRcMode mode;
  union {
    hbVPVideoH264Cbr h264Cbr;
    hbVPVideoH264Vbr h264Vbr;
    hbVPVideoH264Avbr h264Avbr;
    hbVPVideoH264QpMap h264QpMap;
    hbVPVideoH264FixQp h264FixQp;

    hbVPVideoH265Cbr h265Cbr;
    hbVPVideoH265Vbr h265Vbr;
    hbVPVideoH265Avbr h265Avbr;
    hbVPVideoH265QpMap h265QpMap;
    hbVPVideoH265FixQp h265FixQp;
  };
} hbVPVideoRcParam;

/**
 * the parameters of GOP structure.
 * decodingRefreshType: the type of I picture to be inserted at every intraPeriod, the recommended value is 2.
 *                      0 : Non-IRAP(I picture, not a clean random access point)
 *                      1 : CRA(non-IDR clean random access point)
 *                      2 : IDR
 * gopPresetIdx: a GOP structure preset option, the recommended value is 2.
 *               1 : I-I-I-I,..I (all intra, gop_size=1)
 *               2 : I-P-P-P,… P (consecutive P, gop_size=1)
 *               3 : I-B-B-B,…B (consecutive B, gop_size=1)
 *               6 : I-P-P-P-P,… (consecutive P, gop_size=4)
 *               7 : I-B-B-B-B,… (consecutive B, gop_size=4)
 *               9 : I-P-P-P,… P (consecutive P, gop_size = 1, with single reference)
*/
typedef struct {
  int32_t decodingRefreshType;
  uint32_t gopPresetIdx;
} hbVPVideoGopParam;

/**
 * the video codec encoding parameters.
 * pixelFormat: the format of input video.
 *              support HB_VP_IMAGE_FORMAT_NV12 and HB_VP_IMAGE_FORMAT_YUV420. 
 * width: the width of input video. 
 *        H265: values[256, 8192], aligned with 32.
 *        H264: values[256, 8192], aligned with 32.
 * height: the height of input video.
 *         H265: values[128, 4096], aligned with 8.
 *         H264: values[128, 4096], aligned with 8.
 * outBufCount: the count of output buffers.
 *               values[1, 1000], recommended value is 5.
 * backend: specifies the execution backend for video encoding tasks.
 * videoType: the encoding types of video
 * rcParams: the rate control parameters.
 * gopParams: the gop parameters
*/
typedef struct {
  uint8_t pixelFormat;
  int32_t width;
  int32_t height;
  uint32_t outBufCount;
  uint64_t backend;
  hbVPVideoType videoType;
  hbVPVideoRcParam rcParam;
  hbVPVideoGopParam gopParam;
} hbVPVideoEncParam;

typedef void *hbVPVideoContext;

/**
 * @brief Generate video encoding default parameters
 *
 * @param[out] param Pointer to a hbVPVideoEncParam structure that will filled
 *                   with default encoding parameters. The structure be 
 *                   allocated by the caller and the videoType field must be set 
 *                   (e.g., HB_VP_VIDEO_TYPE_H264 or HB_VP_VIDEO_TYPE_H265)
 *                   before calling this function.         
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPGetDefaultVideoEncParam(hbVPVideoEncParam *param);

/**
 * @brief Generate video encoding context
 *
 * @param[out] context video encoding context, dispatched on VPU
 * @param[in] param param for video encode
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPCreateVideoEncContext(hbVPVideoContext *context,
                                  hbVPVideoEncParam const *param);

/**
 * @brief Release video encoding context
 *
 * @param[in] context video encoding context
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPReleaseVideoEncContext(hbVPVideoContext context);

/**
 * @brief Encode YUV to H264/H265 data
 * 
 * @param[out] taskHandle handle of task
 * case1: given *taskHandle is nullptr, create new task handle
 * @param[in] srcImg the image to be encoded
 * @param[in] context video encoding param context
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPVideoEncode(hbUCPTaskHandle_t *taskHandle, hbVPImage const *srcImg,
                        hbVPVideoContext context);

/**
 * @brief Get output buffer from codec
 *
 * @param[out] outBuf the buffer to store the encoded H264/H265 data.
 *                    The buffer is allocated internally by the codec, 
 *                    contains valid data upon the task is successfully completed, 
 *                    and is released during the task release phase.
 * @param[in] taskHandle handle of task, given handle created by hbVPVideoEncode
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPGetVideoEncOutputBuffer(hbVPArray *outBuf,
                                    hbUCPTaskHandle_t taskHandle);

/**
 * @brief The video codec decoding parameters.
 *  
 * @param pixelFormat the format of input video. 
 *                    support HB_VP_IMAGE_FORMAT_NV12 and HB_VP_IMAGE_FORMAT_YUV420. 
 * @param inBufSize Specify the size of bitstream buffer for codec inner using. 
 *                  It's size should be larger than the feeding size and should align with 1024,
 *                  values[1024, 8192 * 4096 * 3] and the recommended value is 10*1024*1024.
 *                   usually set its value to height * width * Size(pixelFormat)
 * @param outBufCount the count of output buffers. values[1, 31], recommended value is 5.
 * @param backend specifies the execution backend for video decoding tasks.
 * @param videoType the decoding types of video
*/
typedef struct {
  uint8_t pixelFormat;
  uint32_t inBufSize;
  uint32_t outBufCount;
  uint64_t backend;
  hbVPVideoType videoType;
} hbVPVideoDecParam;

/**
 * @brief Generate video decoding default parameters
 *
 * @param[out] param Pointer to a hbVPVideoDecParam structure that will filled
 *                   with default decoding parameters. The structure be 
 *                   allocated by the caller and the videoType field must be set 
 *                   (e.g., HB_VP_VIDEO_TYPE_H264 or HB_VP_VIDEO_TYPE_H265)
 *                   before calling this function.  
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPGetDefaultVideoDecParam(hbVPVideoDecParam *param);

/**
 * @brief Generate video decoding context
 *
 * @param[out] context video decoding context, dispatched on VPU
 * @param[in] param param for video decode
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPCreateVideoDecContext(hbVPVideoContext *context,
                                  hbVPVideoDecParam const *param);

/**
 * @brief Release video decoding context
 *
 * @param[in] context video decoding context
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPReleaseVideoDecContext(hbVPVideoContext context);

/**
 * @brief Decode H264/H265 data to YUV
 * 
 * @param[out] taskHandle handle of task
 * case1: given *taskHandle is nullptr, create new task handle
 * @param[in] srcBuf the H264/H265 data to be decoded
 * @param[in] context video decoding context
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPVideoDecode(hbUCPTaskHandle_t *taskHandle, hbVPArray const *srcBuf,
                        hbVPVideoContext const context);

/**
 * @brief Get output buffer from codec
 *
 * @param[out] outImg the buffer to store the decoded Image data.
 *                    The buffer is allocated internally by the codec, 
 *                    contains valid data upon the task is successfully completed, 
 *                    and is released during the task release phase.
 * @param[in] taskHandle handle of task, given handle created by hbVPVideoDecode
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPGetVideoDecOutputBuffer(hbVPImage *outImg,
                                    hbUCPTaskHandle_t taskHandle);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // VP_HB_VP_VIDEO_CODEC_H_
