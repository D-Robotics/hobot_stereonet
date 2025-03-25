//
// Created by zhy on 7/22/24.
//

#ifndef STEREONET_MODEL_INCLUDE_IMAGE_CONVERSION_H_
#define STEREONET_MODEL_INCLUDE_IMAGE_CONVERSION_H_

#include <opencv2/opencv.hpp>

struct image_conversion {
  static void nv12_to_bgr24_neon(uint8_t* nv12, uint8_t* bgr24, int width, int height);
  static void bgr24_to_nv12_neon(uint8_t* bgr24, uint8_t* nv12, int width, int height);
  static void bgr_to_nv12(const cv::Mat &bgr, cv::Mat &nv12);
  static void nv12_to_bgr(const cv::Mat &nv12, cv::Mat &bgr);
};

#endif //STEREONET_MODEL_INCLUDE_IMAGE_CONVERSION_H_
