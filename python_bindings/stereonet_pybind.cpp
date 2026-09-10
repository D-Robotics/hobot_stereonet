// Copyright (c) 2025,D-Robotics.
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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <string>
#include <stdexcept>
#include <memory>
#include <cstring>
#include <opencv2/opencv.hpp>

#include "camera_intrinsic.h"
#include "stereonet_process.h"
#include "feature_epipolar_align.h"

namespace py = pybind11;
using namespace stereonet;

static py::array mat_to_numpy_copy(const cv::Mat &mat) {
  if (mat.empty()) {
    throw std::runtime_error("cv::Mat is empty");
  }

  int typ = mat.type();

  if (typ == CV_32FC1) {
    py::array_t<float> arr({mat.rows, mat.cols});
    std::memcpy(arr.mutable_data(), mat.ptr<float>(), mat.total() * sizeof(float));
    return arr;
  } else if (typ == CV_32FC3) {
    py::array_t<float> arr({mat.rows, mat.cols, 3});
    std::memcpy(arr.mutable_data(), mat.ptr<float>(), mat.total() * 3 * sizeof(float));
    return arr;
  } else if (typ == CV_8UC1) {
    py::array_t<uint8_t> arr({mat.rows, mat.cols});
    std::memcpy(arr.mutable_data(), mat.ptr<uint8_t>(), mat.total() * sizeof(uint8_t));
    return arr;
  } else if (typ == CV_8UC3) {
    py::array_t<uint8_t> arr({mat.rows, mat.cols, 3});
    std::memcpy(arr.mutable_data(), mat.ptr<uint8_t>(), mat.total() * 3 * sizeof(uint8_t));
    return arr;
  } else if (typ == CV_16UC1) {
    py::array_t<uint16_t> arr({mat.rows, mat.cols});
    std::memcpy(arr.mutable_data(), mat.ptr<uint16_t>(), mat.total() * sizeof(uint16_t));
    return arr;
  }

  throw std::runtime_error("Unsupported cv::Mat type");
}

static std::vector<uint8_t> numpy_to_vector_u8(const py::array_t<uint8_t> &arr) {
  py::buffer_info info = arr.request();
  if (!info.ptr) {
    throw std::runtime_error("Input numpy array is null");
  }

  size_t total_bytes = 1;
  for (auto s : info.shape) {
    total_bytes *= static_cast<size_t>(s);
  }

  std::vector<uint8_t> out(total_bytes);
  std::memcpy(out.data(), info.ptr, total_bytes * sizeof(uint8_t));
  return out;
}

// Convert a numpy array (float32 HxW, uint16 HxW, uint8 HxW or uint8 HxWx3) to a
// cv::Mat. The dtype is preserved so that functions such as render_disp_or_depth,
// which branches on the input type, receive the correct one.
static cv::Mat numpy_to_mat(const py::array &arr) {
  py::buffer_info info = arr.request();
  if (!info.ptr) {
    throw std::runtime_error("numpy array has no data");
  }
  if (info.ndim < 2) {
    throw std::runtime_error("expected a 2D (or HxWx3) numpy array");
  }

  int h = static_cast<int>(info.shape[0]);
  int w = static_cast<int>(info.shape[1]);
  std::string fmt = info.format;

  if (fmt == py::format_descriptor<float>::format() && info.ndim == 2) {
    cv::Mat m(h, w, CV_32FC1);
    std::memcpy(m.data, info.ptr, m.total() * sizeof(float));
    return m;
  }
  if (fmt == py::format_descriptor<uint16_t>::format() && info.ndim == 2) {
    cv::Mat m(h, w, CV_16UC1);
    std::memcpy(m.data, info.ptr, m.total() * sizeof(uint16_t));
    return m;
  }
  if (fmt == py::format_descriptor<uint8_t>::format() && info.ndim == 2) {
    cv::Mat m(h, w, CV_8UC1);
    std::memcpy(m.data, info.ptr, m.total() * sizeof(uint8_t));
    return m;
  }
  if (fmt == py::format_descriptor<uint8_t>::format() && info.ndim == 3 && info.shape[2] == 3) {
    cv::Mat m(h, w, CV_8UC3);
    std::memcpy(m.data, info.ptr, m.total() * 3 * sizeof(uint8_t));
    return m;
  }

  throw std::runtime_error("unsupported numpy dtype/shape (need float32/uint16 HxW or uint8 HxW/HxWx3)");
}

PYBIND11_MODULE(dstereonet, m) {
  m.doc() = "pybind11 wrapper for StereonetProcess";

  py::class_<CameraIntrinsic>(m, "CameraIntrinsic")
      .def(py::init<>())
      .def_readwrite("cx", &CameraIntrinsic::cx)
      .def_readwrite("cy", &CameraIntrinsic::cy)
      .def_readwrite("fx", &CameraIntrinsic::fx)
      .def_readwrite("fy", &CameraIntrinsic::fy)
      .def_readwrite("baseline", &CameraIntrinsic::baseline)
      .def_readwrite("doffs", &CameraIntrinsic::doffs)
      .def_readwrite("rectify_model", &CameraIntrinsic::rectify_model)
      .def("is_valid", &CameraIntrinsic::is_valid);

  py::class_<PointXYZ>(m, "PointXYZ")
      .def(py::init<>())
      .def(py::init<float, float, float>(), py::arg("x"), py::arg("y"), py::arg("z"))
      .def_readwrite("x", &PointXYZ::X)
      .def_readwrite("y", &PointXYZ::Y)
      .def_readwrite("z", &PointXYZ::Z);

  py::class_<PointXYZRGB>(m, "PointXYZRGB")
      .def(py::init<>())
      .def(py::init<float, float, float, uint8_t, uint8_t, uint8_t>(), py::arg("x"), py::arg("y"), py::arg("z"),
           py::arg("r"), py::arg("g"), py::arg("b"))
      .def_readwrite("x", &PointXYZRGB::X)
      .def_readwrite("y", &PointXYZRGB::Y)
      .def_readwrite("z", &PointXYZRGB::Z)
      .def_readwrite("r", &PointXYZRGB::R)
      .def_readwrite("g", &PointXYZRGB::G)
      .def_readwrite("b", &PointXYZRGB::B);

  py::class_<StereonetProcess>(m, "StereonetProcess")
      .def(py::init<>())

      .def(
          "init",
          [](StereonetProcess &self, const std::string &model_path, const std::string &post_version,
             int max_memory_count) {
            int ret = self.init(model_path, post_version, max_memory_count);
            if (ret != 0) {
              throw std::runtime_error("StereonetProcess::init failed, ret=" + std::to_string(ret));
            }
          },
          py::arg("model_path"), py::arg("post_version") = "auto", py::arg("max_memory_count") = 5)

      .def("get_model_input_size",
           [](const StereonetProcess &self) {
             int w = 0, h = 0;
             self.get_model_input_size(w, h);
             return py::make_tuple(w, h);
           })

      .def(
          "forward_sync",
          [](StereonetProcess &self, py::array_t<uint8_t, py::array::c_style | py::array::forcecast> left_nv12,
             py::array_t<uint8_t, py::array::c_style | py::array::forcecast> right_nv12, double uncertainty_th) {
            std::vector<uint8_t> left_vec = numpy_to_vector_u8(left_nv12);
            std::vector<uint8_t> right_vec = numpy_to_vector_u8(right_nv12);

            cv::Mat disp, uncert;
            int ret = self.forward_sync(left_vec, right_vec, uncertainty_th, disp, uncert);
            if (ret != 0) {
              throw std::runtime_error("StereonetProcess::forward_sync failed, ret=" + std::to_string(ret));
            }

            if (disp.empty()) {
              throw std::runtime_error("disp is empty");
            }

            py::object disp_obj = mat_to_numpy_copy(disp);
            py::object uncert_obj = py::none();

            if (!uncert.empty()) {
              uncert_obj = mat_to_numpy_copy(uncert);
            }

            return py::make_tuple(disp_obj, uncert_obj);
          },
          py::arg("left_nv12"), py::arg("right_nv12"), py::arg("uncertainty_th") = -0.10)

      // Perspective disparity -> depth (uint16 mm), same as infer.cpp:
      //   depth_mm = fx * baseline * 1000 / (disp + doffs) for disp > 0, else 0.
      .def_static(
          "perspective_disparity_to_depth",
          [](const py::array &disp, const CameraIntrinsic &camera_intrinsic) {
            cv::Mat depth;
            StereonetProcess::perspective_disparity_to_depth(numpy_to_mat(disp), depth, camera_intrinsic);
            return mat_to_numpy_copy(depth);
          },
          py::arg("disp"), py::arg("camera_intrinsic"))

      // Dispatch disparity -> depth based on camera_intrinsic.rectify_model
      // (longlati for RECTIFY_LONGLATI, perspective otherwise).
      .def_static(
          "disparity_to_depth",
          [](const py::array &disp, const CameraIntrinsic &camera_intrinsic) {
            cv::Mat depth;
            StereonetProcess::disparity_to_depth(numpy_to_mat(disp), depth, camera_intrinsic);
            return mat_to_numpy_copy(depth);
          },
          py::arg("disp"), py::arg("camera_intrinsic"))

      // Render a disparity (float32) or depth (uint16) map to a pseudo-color image.
      .def_static(
          "render_disp_or_depth",
          [](const py::array &input, float min_disp, float max_disp, float min_depth, float max_depth,
             bool enable_speckle_filter, int speckle_size, double speckle_diff, int speckle_connectivity) {
            cv::Mat out = StereonetProcess::render_disp_or_depth(
                numpy_to_mat(input), min_disp, max_disp, min_depth, max_depth, enable_speckle_filter, speckle_size,
                speckle_diff, speckle_connectivity);
            return mat_to_numpy_copy(out);
          },
          py::arg("input"), py::arg("min_disp") = 0.0f, py::arg("max_disp") = 192.0f, py::arg("min_depth") = 0.0f,
          py::arg("max_depth") = 10000.0f, py::arg("enable_speckle_filter") = false, py::arg("speckle_size") = 100,
          py::arg("speckle_diff") = 2.0, py::arg("speckle_connectivity") = 8)

      // Overlay the left RGB image with the depth pseudo-color (visual output).
      .def_static(
          "convert_visual_img",
          [](const py::array &rgb, const py::array &disp, const py::array &depth,
             const CameraIntrinsic &camera_intrinsic, int depth_decimal_num) {
            cv::Mat visual_img;
            StereonetProcess::convert_visual_img(numpy_to_mat(rgb), numpy_to_mat(disp), numpy_to_mat(depth),
                                                 camera_intrinsic, visual_img, depth_decimal_num);
            return mat_to_numpy_copy(visual_img);
          },
          py::arg("rgb"), py::arg("disp"), py::arg("depth"), py::arg("camera_intrinsic"),
          py::arg("depth_decimal_num") = 2)

      // Depth + RGB -> colored point cloud (list of PointXYZRGB).
      .def_static(
          "depth_to_pointcloud_rgb",
          [](const py::array &depth, const py::array &rgb, const CameraIntrinsic &camera_intrinsic, float max_depth) {
            std::vector<PointXYZRGB> pointcloud;
            StereonetProcess::depth_to_pointcloud_rgb(numpy_to_mat(depth), numpy_to_mat(rgb), camera_intrinsic,
                                                      pointcloud, max_depth);
            return pointcloud;
          },
          py::arg("depth"), py::arg("rgb"), py::arg("camera_intrinsic"), py::arg("max_depth") = 10.0f)

      // Dump a colored point cloud to a PCD file.
      .def_static(
          "dump_pcd_file_rgb",
          [](const std::string &filename, const std::vector<PointXYZRGB> &pointcloud, const std::string &format) {
            StereonetProcess::dump_pcd_file_rgb(filename, pointcloud, format);
          },
          py::arg("filename"), py::arg("pointcloud"), py::arg("format") = "binary");

  // Feature-based epipolar alignment check: returns the visualization image, or
  // None when no valid matches were found (mirrors FeatureEpipolarAlign::check_epipolar_alignment).
  m.def(
      "check_epipolar_alignment",
      [](const py::array &left_img, const py::array &right_img, const CameraIntrinsic &camera_intrinsic) -> py::object {
        cv::Mat visualize;
        FeatureEpipolarAlign::check_epipolar_alignment(numpy_to_mat(left_img), numpy_to_mat(right_img), camera_intrinsic,
                                                       visualize);
        if (visualize.empty()) {
          return py::none();
        }
        return mat_to_numpy_copy(visualize);
      },
      py::arg("left_img"), py::arg("right_img"), py::arg("camera_intrinsic"));
}
