#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <csignal>
#include <fstream>
#include <sstream>

// =============== stereonet ===============
#include "log_macros.h"
#include "camera_intrinsic.h"
#include "stereonet_process.h"
#include "img_convert_utils.h"
#include "file_utils.h"
#include "feature_epipolar_align.h"
// =============== stereonet ===============

namespace fs = std::filesystem;

bool readCameraIntrinsicFromFile(const std::string &file_path, stereonet::CameraIntrinsic &intrinsic) {
  std::ifstream infile(file_path);
  if (!infile.is_open()) {
    std::cerr << "Failed to open file: " << file_path << std::endl;
    return false;
  }

  std::vector<double> values;
  std::string line;

  while (std::getline(infile, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::istringstream ss(line);
    double v;
    while (ss >> v) {
      values.push_back(v);
    }
  }

  // ---------- format 1 ----------
  // fx fy cx cy baseline
  if (values.size() == 5) {
    intrinsic.fx = values[0];
    intrinsic.fy = values[1];
    intrinsic.cx = values[2];
    intrinsic.cy = values[3];
    intrinsic.baseline = values[4];

    return intrinsic.is_valid();
  }

  // ---------- format 2 ----------
  // 3x3 K + baseline
  // fx 0 cx
  // 0 fy cy
  // 0 0 1
  // baseline
  if (values.size() == 10) {

    intrinsic.fx = values[0];
    intrinsic.cx = values[2];
    intrinsic.fy = values[4];
    intrinsic.cy = values[5];
    intrinsic.baseline = values[9];

    return intrinsic.is_valid();
  }

  std::cerr << "Unsupported intrinsic format in: " << file_path << std::endl;
  return false;
}

void saveCameraIntrinsic(const std::string &dir, const stereonet::CameraIntrinsic &intr) {
  std::filesystem::create_directories(dir);

  // camera_intrinsic.txt
  std::string file1 = dir + "/camera_intrinsic.txt";
  std::ofstream f1(file1);
  f1 << "# fx fy cx cy baseline(m)" << std::endl;
  f1 << intr.fx << " " << intr.fy << " " << intr.cx << " " << intr.cy << " " << intr.baseline << std::endl;
  f1.close();

  // K.txt
  std::string file2 = dir + "/K.txt";
  std::ofstream f2(file2);
  f2 << intr.fx << " 0.0 " << intr.cx << " 0.0 " << intr.fy << " " << intr.cy << " 0.0 0.0 1.0" << std::endl;
  f2 << intr.baseline << std::endl;
  f2.close();
}

void print_help(const char *prog_name) {
  std::cout << R"(Usage:)" << prog_name << R"( [model_path] [local_img_dir] [uncertainty_th]

Arguments:
  model_path         Path to stereo model (.bin)
                     default: ./model/DStereoV2.4_int16.bin
  local_img_dir      Path to local image directory
                     default: ./img
  uncertainty_th     Uncertainty threshold
                     default: -0.10

Examples: )" << prog_name
            << R"( ./model/DStereoV2.4_int16.bin ./img -0.10)" << std::endl;
}

int main(int argc, char **argv) {
  // help
  if (argc > 1) {
    std::string arg1(argv[1]);
    if (arg1 == "-h" || arg1 == "--help") {
      print_help(argv[0]);
      return 0;
    }
  }

  // parse arguments
  std::string model_path = "./model/DStereoV2.4_int16.bin";
  std::string local_img_dir = "./img";
  float uncertainty_th = -0.10;
  if (argc > 1) model_path = argv[1];
  if (argc > 2) local_img_dir = argv[2];
  if (argc > 3) uncertainty_th = std::stof(argv[3]);

  if (!fs::exists(model_path)) {
    LOG_ERROR(nullptr, "=> model file not exist: " << model_path);
    return -1;
  }

  // init StereoNetProcess
  auto stereonet_process = std::make_shared<stereonet::StereonetProcess>();
  stereonet_process->init(model_path, "auto");
  int model_input_w = 0, model_input_h = 0;
  stereonet_process->get_model_input_size(model_input_w, model_input_h);

  // process
  std::string result_root = "./result";
  fs::create_directories(result_root);
  for (auto &dir_entry : fs::directory_iterator(local_img_dir)) {
    // check dir
    if (!dir_entry.is_directory()) continue;
    std::string sub_dir = dir_entry.path().string();
    std::string sub_name = dir_entry.path().filename().string();
    LOG_INFO(nullptr, "=> ==============================================");
    LOG_INFO(nullptr, "=> processing folder: " << sub_name);
    std::string result_dir = result_root + "/" + sub_name;
    fs::create_directories(result_dir);

    // read camera intrinsic
    stereonet::CameraIntrinsic camera_intrinsic;
    std::string intrinsic_file;
    std::string file1 = sub_dir + "/camera_intrinsic.txt";
    std::string file2 = sub_dir + "/K.txt";
    if (std::filesystem::exists(file1)) {
      intrinsic_file = file1;
      if (readCameraIntrinsicFromFile(intrinsic_file, camera_intrinsic)) {
        LOG_INFO(nullptr, "=> cam intrinsic [fx,fy,cx,cy,baseline]: "
                              << camera_intrinsic.fx << ", " << camera_intrinsic.fy << ", " << camera_intrinsic.cx
                              << ", " << camera_intrinsic.cy << ", " << camera_intrinsic.baseline);
        saveCameraIntrinsic(result_dir, camera_intrinsic);
      }
    } else if (std::filesystem::exists(file2)) {
      intrinsic_file = file2;
      if (readCameraIntrinsicFromFile(intrinsic_file, camera_intrinsic)) {
        LOG_INFO(nullptr, "=> cam intrinsic [fx,fy,cx,cy,baseline]: "
                              << camera_intrinsic.fx << ", " << camera_intrinsic.fy << ", " << camera_intrinsic.cx
                              << ", " << camera_intrinsic.cy << ", " << camera_intrinsic.baseline);
        saveCameraIntrinsic(result_dir, camera_intrinsic);
      }
    } else {
      LOG_WARN(nullptr, "=> no intrinsic file found in " << sub_dir);
    }

    std::vector<std::pair<std::string, std::string>> img_pairs = FileUtils::find_pairs(sub_dir);
    bool update_cam_intr = false;
    for (auto &img_pair : img_pairs) {
      LOG_INFO(nullptr, "=> processing image pair: " << img_pair.first << " " << img_pair.second);
      // read image
      std::string left_img_path = img_pair.first;
      std::string right_img_path = img_pair.second;
      std::string left_img_name = fs::path(left_img_path).filename().string();
      std::string right_img_name = fs::path(right_img_path).filename().string();
      cv::Mat left_img = cv::imread(left_img_path);
      cv::Mat right_img = cv::imread(right_img_path);
      if (left_img.empty() || right_img.empty()) {
        LOG_ERROR(nullptr, "=> image read failed");
        continue;
      }

      // resize
      cv::Mat left_img_resize, right_img_resize;
      if (left_img.cols != model_input_w || left_img.rows != model_input_h) {
        LOG_INFO(nullptr, "=> left img size [" << left_img.cols << ", " << left_img.rows << "], right img size ["
                                               << right_img.cols << ", " << right_img.rows << "], need reszie to ["
                                               << model_input_w << ", " << model_input_h << "]");
        cv::resize(left_img, left_img_resize, cv::Size(model_input_w, model_input_h));
        cv::resize(right_img, right_img_resize, cv::Size(model_input_w, model_input_h));
        if (!update_cam_intr && camera_intrinsic.is_valid()) {
          camera_intrinsic.fx = camera_intrinsic.fx * model_input_w / left_img.cols;
          camera_intrinsic.fy = camera_intrinsic.fy * model_input_h / left_img.rows;
          camera_intrinsic.cx = camera_intrinsic.cx * model_input_w / left_img.cols;
          camera_intrinsic.cy = camera_intrinsic.cy * model_input_h / left_img.rows;
          LOG_INFO(nullptr, "=> after resize, cam intrinsic [fx, fy, cx, cy, baseline]: ["
                                << camera_intrinsic.fx << ", " << camera_intrinsic.fy << ", " << camera_intrinsic.cx
                                << ", " << camera_intrinsic.cy << ", " << camera_intrinsic.baseline << "]");
          saveCameraIntrinsic(result_dir, camera_intrinsic);
          update_cam_intr = true;
        }
      } else {
        left_img_resize = left_img;
        right_img_resize = right_img;
      }

      // convert to nv12
      size_t model_input_nv12_size = model_input_w * model_input_h * 3 / 2;
      std::vector<uint8_t> left_img_nv12(model_input_nv12_size);
      std::vector<uint8_t> right_img_nv12(model_input_nv12_size);
      ImgConvertUtils::bgr_mat_to_nv12(left_img_resize, left_img_nv12.data());
      ImgConvertUtils::bgr_mat_to_nv12(right_img_resize, right_img_nv12.data());

      // infer
      cv::Mat disp, uncert;
      stereonet_process->forward_sync(left_img_nv12, right_img_nv12, uncertainty_th, disp, uncert);
      cv::Mat depth;
      if (camera_intrinsic.is_valid()) stereonet_process->disp_to_depth(disp, depth, camera_intrinsic);

      // epipolar check
      cv::Mat epipolar_visual;
      std::shared_ptr<stereonet::CameraIntrinsic> camera_intrinsic_ptr = std::make_shared<stereonet::CameraIntrinsic>();
      camera_intrinsic_ptr->fx = camera_intrinsic.fx;
      camera_intrinsic_ptr->fy = camera_intrinsic.fy;
      camera_intrinsic_ptr->cx = camera_intrinsic.cx;
      camera_intrinsic_ptr->cy = camera_intrinsic.cy;
      camera_intrinsic_ptr->baseline = camera_intrinsic.baseline;
      FeatureEpipolarAlign::check_epipolar_alignment(left_img_resize, right_img_resize, camera_intrinsic_ptr,
                                                     epipolar_visual);

      // save
      std::string prefix = left_img_name.substr(0, left_img_name.find_last_of("."));
      cv::imwrite(result_dir + "/" + left_img_name, left_img_resize);
      cv::imwrite(result_dir + "/" + right_img_name, right_img_resize);
      cv::imwrite(result_dir + "/disp_" + prefix + ".pfm", disp);
      if (!uncert.empty()) cv::imwrite(result_dir + "/uncert_" + prefix + ".pfm", uncert);
      cv::Mat visual_img_disp = stereonet_process->render_disp_or_depth(disp);
      cv::imwrite(result_dir + "/visual_disp_" + prefix + ".png", visual_img_disp);
      cv::imwrite(result_dir + "/epipolar_visual_" + prefix + ".png", epipolar_visual);
      if (camera_intrinsic.is_valid()) {
        cv::imwrite(result_dir + "/depth_" + prefix + ".png", depth);
        cv::Mat visual_img;
        stereonet_process->convert_visual_img(left_img_resize, disp, depth, camera_intrinsic, visual_img);
        cv::imwrite(result_dir + "/visual_" + prefix + ".png", visual_img);
        std::vector<stereonet::PointXYZRGB> pointcloud;
        stereonet_process->depth_to_pointcloud_rgb(depth, left_img_resize, camera_intrinsic, pointcloud);
        stereonet_process->dump_pcd_file_rgb(result_dir + "/pointcloud_" + prefix + ".pcd", pointcloud);
      }
    }
  }
  LOG_INFO(nullptr, "=> ==============================================");

  stereonet_process.reset();

  return 0;
}