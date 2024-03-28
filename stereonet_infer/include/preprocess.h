// Copyright (c) 2022，Horizon Robotics.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef BEV_INCLUDE_BEV_BEV_MODEL_PREPROCESS_H_
#define BEV_INCLUDE_BEV_BEV_MODEL_PREPROCESS_H_
#include <memory>
#include <string>
#include <vector>

#include "dnn_node/dnn_node.h"

#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#define ALIGNED_2E(w, alignment) \
  ((static_cast<uint32_t>(w) + (alignment - 1U)) & (~(alignment - 1U)))
#define ALIGN_4(w) ALIGNED_2E(w, 4U)
#define ALIGN_8(w) ALIGNED_2E(w, 8U)
#define ALIGN_16(w) ALIGNED_2E(w, 16U)

namespace hobot {
namespace stereonet {

using hobot::dnn_node::DNNInput;
using hobot::dnn_node::DNNTensor;
using hobot::dnn_node::Model;
using hobot::dnn_node::NV12PyramidInput;

struct FeedbackData {
  // size is 2 and seq is:
  // image_front_left
  // image_front_right
  std::vector<std::string> image_files {
    "./config/image_left.jpg",
    "./config/image_right.jpg"
  };

  // norm后的数据
  std::string bin_file {"numpy/img_adapter.bin"};
};

class Tools {
public:
  static int32_t BGRToNv12(cv::Mat &bgr_mat, cv::Mat &img_nv12) {
    auto height = bgr_mat.rows;
    auto width = bgr_mat.cols;

    if (height % 2 || width % 2) {
      std::cerr << "input img height and width must aligned by 2!";
      return -1;
    }
    cv::Mat yuv_mat;
    cv::cvtColor(bgr_mat, yuv_mat, cv::COLOR_BGR2YUV_I420);
    if (yuv_mat.data == nullptr) {
      std::cerr << "yuv_mat.data is null pointer" << std::endl;
      return -1;
    }

    auto *yuv = yuv_mat.ptr<uint8_t>();
    if (yuv == nullptr) {
      std::cerr << "yuv is null pointer" << std::endl;
      return -1;
    }
    img_nv12 = cv::Mat(height * 3 / 2, width, CV_8UC1);
    auto *ynv12 = img_nv12.ptr<uint8_t>();

    int32_t uv_height = height / 2;
    int32_t uv_width = width / 2;

    // copy y data
    int32_t y_size = height * width;
    memcpy(ynv12, yuv, y_size);

    // copy uv data
    uint8_t *nv12 = ynv12 + y_size;
    uint8_t *u_data = yuv + y_size;
    uint8_t *v_data = u_data + uv_height * uv_width;

    for (int32_t i = 0; i < uv_width * uv_height; i++) {
      *nv12++ = *u_data++;
      *nv12++ = *v_data++;
    }
    return 0;
  }

  static void nhwc2nchw(std::vector<uint8_t>& out_data, uint8_t *in_data, int height, int width, int channels) {
    for (int c = 0; c < channels; ++c)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                int dstIdx = c * height * width + h * width + w;
                int srcIdx = h * width * channels + w * channels + c;
                out_data.push_back(in_data[srcIdx]);
            }
        }
    }
  }

  static void ncwh2nhwc(std::vector<int8_t>& out_data, const int8_t* in_data, int channels, int height, int width) {
    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            for (int c = 0; c < channels; ++c)
            {
                int dstIdx = h * width * channels + w * channels + c;
                int srcIdx = c * height * width + h * width + w;
                out_data.push_back(in_data[srcIdx]);
            }
        }
    }
  };

  static void YUV420TOYUV444(const unsigned char *inbuf, unsigned char *outbuf, int w, int h) {
    const unsigned char *srcY = NULL, *srcU = NULL, *srcV = NULL;
    unsigned char *desY = NULL, *desU = NULL, *desV = NULL;
    srcY = inbuf;//Y
    srcU = srcY + w * h;//U
    srcV = srcU + w * h / 4;;//V

    desY = outbuf;
    desU = desY + w * h;
    desV = desU + w * h;
    memcpy(desY, srcY, w * h * sizeof(unsigned char));//Y分量直接拷贝即可
    //UV分量转换
    int i, j;
    for (i = 0; i < h; i += 2) {//行
      for (j = 0; j < w; j += 2) {//列
                    //U
        desU[i*w + j] = srcU[i / 2 * w / 2 + j / 2];
        desU[i*w + j + 1] = srcU[i / 2 * w / 2 + j / 2];
        desU[(i + 1)*w + j] = srcU[i / 2 * w / 2 + j / 2];
        desU[(i + 1)*w + j + 1] = srcU[i / 2 * w / 2 + j / 2];
        //V
        desV[i*w + j] = srcV[i / 2 * w / 2 + j / 2];
        desV[i*w + j + 1] = srcV[i / 2 * w / 2 + j / 2];
        desV[(i + 1)*w + j] = srcV[i / 2 * w / 2 + j / 2];
        desV[(i + 1)*w + j + 1] = srcV[i / 2 * w / 2 + j / 2];
      }
    }
  }

  static void YUV444TOYUV420(const unsigned char *inbuf, unsigned char *outbuf, int w, int h) {
    const unsigned char *srcY = NULL, *srcU = NULL, *srcV = NULL;
    unsigned char *desY = NULL, *desU = NULL, *desV = NULL;
    srcY = inbuf;//Y
    srcU = srcY + w * h;//U
    srcV = srcU + w * h;//V

    desY = outbuf;
    desU = desY + w * h;
    desV = desU + w * h / 4;

    int half_width = w / 2;
    int half_height = h / 2;
    memcpy(desY, srcY, w * h * sizeof(unsigned char));//Y分量直接拷贝即可
      //UV
    for (int i = 0; i < half_height; i++) {
      for (int j = 0; j < half_width; j++) {
        *desU = *srcU;
        *desV = *srcV;
        desU++;
        desV++;
        srcU += 2;
        srcV += 2;
      }
      srcU = srcU + w;
      srcV = srcV + w;
    }
  }
};

class PreProcess {
 public:
  explicit PreProcess(const std::string &config_file);
  int CvtImgData2Tensors(
      std::vector<std::shared_ptr<DNNTensor>> &input_tensors,
      Model *pmodel,
      const std::shared_ptr<FeedbackData>& sp_feedback_data
      );
  int CvtBinData2Tensors(
      std::vector<std::shared_ptr<DNNTensor>> &input_tensors,
      Model *pmodel,
      const std::shared_ptr<FeedbackData>& sp_feedback_data
      );
  int CvtBinData2Tensors(
      std::vector<std::shared_ptr<DNNTensor>> &input_tensors,
      Model *pmodel,
      const std::string& img_l,
      const std::string& img_r
      );
  int CvtNV12Data2Tensors(
      std::vector<std::shared_ptr<DNNTensor>> &input_tensors,
      Model *pmodel,
      const unsigned char* img_l,
      const unsigned char* img_r
      );
  int CvtNV12File2Tensors(
      std::vector<std::shared_ptr<DNNTensor>> &input_tensors,
      Model *pmodel,
      std::string img_l,
      std::string img_r
      );

 private:
  int model_in_w_ = 1280;
  int model_in_h_ = 720;
  
  void FreeTensors(
      const std::vector<std::shared_ptr<DNNTensor>> &input_tensors);

  void GetNV12Tensor(const std::string &image_file,
                    std::shared_ptr<DNNTensor> &dnn_tensor,
                    int scaled_img_height, int scaled_img_width);

  cv::Mat NormalizeImage(cv::Mat image, float mean, float std);

  // scale: 0.0078125
  // zero_point: 0.5
  // min: -128
  // max: 127
  inline int8_t Quantize(float32_t value,
                  float32_t const scale = 0.0078125,
                  float32_t const zero_point = 0.5,
                  float32_t const min = -128,
                  float32_t const max = 127);

  std::vector<std::shared_ptr<DNNTensor>> input_tensors_;
};  // class PreProcess

}  // namespace stereonet
}  // namespace hobot

#endif  // BEV_INCLUDE_BEV_BEV_MODEL_PREPROCESS_H_
