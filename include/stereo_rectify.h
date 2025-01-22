//
// Created by zhy on 8/21/24.
//

#ifndef STEREO_HOBOT_STEREONET_INCLUDE_STEREO_RECTIFY_H_
#define STEREO_HOBOT_STEREONET_INCLUDE_STEREO_RECTIFY_H_

#include <opencv2/opencv.hpp>

namespace stereonet {


struct StereoRectify {

  StereoRectify(const cv::FileNode &fs, int model_input_w, int model_input_h) {
    // Reading cam0 data
    std::vector<double> cam0_distortion_coeffs;
    std::vector<double> cam0_intrinsics;
    std::vector<int> cam0_resolution;

    fs["cam0"]["distortion_coeffs"] >> cam0_distortion_coeffs;
    fs["cam0"]["intrinsics"] >> cam0_intrinsics;
    fs["cam0"]["resolution"] >> cam0_resolution;
    fs["cam0"]["camera_model"] >> calib_model;

    // Reading cam1 data
    std::vector<std::vector<double>> cam1_T_cn_cnm1;
    std::vector<double> cam1_distortion_coeffs;
    std::vector<double> cam1_intrinsics;
    std::vector<int> cam1_resolution;

    fs["cam1"]["T_cn_cnm1"] >> cam1_T_cn_cnm1;
    fs["cam1"]["distortion_coeffs"] >> cam1_distortion_coeffs;
    fs["cam1"]["intrinsics"] >> cam1_intrinsics;
    fs["cam1"]["resolution"] >> cam1_resolution;

    Dl = cv::Mat(1, cam0_distortion_coeffs.size(), CV_64F, cam0_distortion_coeffs.data()).clone();
    Kl = cv::Mat::zeros(3, 3, CV_64F);
    Kl.at<double>(0, 0) = cam0_intrinsics[0];
    Kl.at<double>(0, 2) = cam0_intrinsics[2];
    Kl.at<double>(1, 1) = cam0_intrinsics[1];
    Kl.at<double>(1, 2) = cam0_intrinsics[3];
    Kl.at<double>(2, 2) = 1;

    R_rl = cv::Mat::zeros(3, 3, CV_64F);
    t_rl = cv::Mat::zeros(3, 1, CV_64F);

    R_rl.at<double>(0, 0) = cam1_T_cn_cnm1[0][0];
    R_rl.at<double>(0, 1) = cam1_T_cn_cnm1[0][1];
    R_rl.at<double>(0, 2) = cam1_T_cn_cnm1[0][2];
    R_rl.at<double>(1, 0) = cam1_T_cn_cnm1[1][0];
    R_rl.at<double>(1, 1) = cam1_T_cn_cnm1[1][1];
    R_rl.at<double>(1, 2) = cam1_T_cn_cnm1[1][2];
    R_rl.at<double>(2, 0) = cam1_T_cn_cnm1[2][0];
    R_rl.at<double>(2, 1) = cam1_T_cn_cnm1[2][1];
    R_rl.at<double>(2, 2) = cam1_T_cn_cnm1[2][2];

    t_rl.at<double>(0, 0) = cam1_T_cn_cnm1[0][3];
    t_rl.at<double>(1, 0) = cam1_T_cn_cnm1[1][3];
    t_rl.at<double>(2, 0) = cam1_T_cn_cnm1[2][3];

    Dr = cv::Mat(1, cam1_distortion_coeffs.size(), CV_64F, cam1_distortion_coeffs.data()).clone();

    Kr = cv::Mat::zeros(3, 3, CV_64F);
    Kr.at<double>(0, 0) = cam1_intrinsics[0];
    Kr.at<double>(0, 2) = cam1_intrinsics[2];
    Kr.at<double>(1, 1) = cam1_intrinsics[1];
    Kr.at<double>(1, 2) = cam1_intrinsics[3];
    Kr.at<double>(2, 2) = 1;

    float fov_scale = 0.8;
    if (calib_model == "pinhole")
    {
      cv::stereoRectify(Kl, Dl, Kr, Dr,
                        cv::Size(cam0_resolution[0], cam0_resolution[1]), R_rl, t_rl, Rl, Rr, Pl, Pr, Q,
                        cv::CALIB_ZERO_DISPARITY, 0, cv::Size(model_input_w, model_input_h));

      cv::initUndistortRectifyMap(Kl, Dl, Rl, Pl,
                                  cv::Size(model_input_w, model_input_h), CV_32FC1, undistmap1l, undistmap2l);
      cv::initUndistortRectifyMap(Kr, Dr, Rr, Pr,
                                  cv::Size(model_input_w, model_input_h), CV_32FC1, undistmap1r, undistmap2r);
    } else if (calib_model == "fish") {
      fs["cam1"]["fov_scale"] >> fov_scale;
      if (fov_scale == 0) fov_scale = 0.8;
      cv::fisheye::stereoRectify(Kl, Dl, Kr, Dr,
                                 cv::Size(cam0_resolution[0], cam0_resolution[1]), R_rl, t_rl, Rl, Rr, Pl, Pr, Q,
                                 cv::fisheye::CALIB_ZERO_DISPARITY, cv::Size(model_input_w, model_input_h), 0.0, fov_scale);
      cv::fisheye::initUndistortRectifyMap(Kl, Dl, Rl, Pl,
                                           cv::Size(model_input_w, model_input_h), CV_32FC1, undistmap1l, undistmap2l);
      cv::fisheye::initUndistortRectifyMap(Kr, Dr, Rr, Pr,
                                           cv::Size(model_input_w, model_input_h), CV_32FC1, undistmap1r, undistmap2r);
    }

    camera_fx = Q.at<double>(2, 3);
    camera_fy = Q.at<double>(2, 3);
    camera_cx = -Q.at<double>(0, 3);
    camera_cy = -Q.at<double>(1, 3);
    //  const cv::Mat t = Rr * t_rl;
    base_line = std::abs(1 / Q.at<double>(3, 2));

    std::cout << "calib model: " << calib_model << std::endl
              << "Kl:" << std::endl
              << Kl << std::endl
              << "Dl:" << std::endl
              << Dl << std::endl
              << "Kr: " << std::endl
              << Kr << std::endl
              << "Dr:" << std::endl
              << Dr << std::endl
              << "R, t: " << std::endl
              << R_rl << std::endl
              << t_rl << std::endl
              << "calib file width, height: " << cam0_resolution[0] << ", " << cam0_resolution[1] << std::endl
              << "model input width, height: " << model_input_w << ", " << model_input_h << std::endl
              << "fov_scale: " << fov_scale << std::endl
              << std::endl;
  }

  void Rectify(const cv::Mat &left_image,
               const cv::Mat &right_image,
               cv::Mat &rectified_left_image,
               cv::Mat &rectified_right_image) {
    cv::remap(left_image, rectified_left_image, undistmap1l, undistmap2l, cv::INTER_LINEAR);
    cv::remap(right_image, rectified_right_image, undistmap1r, undistmap2r, cv::INTER_LINEAR);
  }

  void GetIntrinsic(float &cx, float &cy, float &fx, float &fy, float &bl) {
    cx = camera_cx;
    cy = camera_cy;
    fx = camera_fx;
    fy = camera_fy;
    bl = base_line;
  }

  std::string calib_model;
  cv::Mat Rl, Rr, Pl, Pr, Q;
  cv::Mat Kl, Kr, Dl, Dr, R_rl, t_rl;
  cv::Mat undistmap1l, undistmap2l, undistmap1r, undistmap2r;
  float camera_cx, camera_cy, camera_fx, camera_fy, base_line;
};

}

#endif //STEREO_HOBOT_STEREONET_INCLUDE_STEREO_RECTIFY_H_
