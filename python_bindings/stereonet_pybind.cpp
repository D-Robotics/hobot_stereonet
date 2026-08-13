// Copyright (c) 2025，D-Robotics.
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
#include <stdexcept>
#include <cstring>
#include <opencv2/opencv.hpp>

#include "stereonet_process.h"

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

PYBIND11_MODULE(dstereonet, m) {
  m.doc() = "pybind11 wrapper for StereonetProcess";

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
          py::arg("left_nv12"), py::arg("right_nv12"), py::arg("uncertainty_th") = -0.10);
}