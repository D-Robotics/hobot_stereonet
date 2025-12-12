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

#ifndef HOBOT_STEREONET_STANDALONE_DATA_LOADER_H_
#define HOBOT_STEREONET_STANDALONE_DATA_LOADER_H_

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"


struct StereoImageSet {
  bool is_valid() const {
    return !left_image_file.empty() && !right_image_file.empty() && !disparity_image_file.empty();
  }

  template<typename T>
  void get_camera_parameter(T &fx, T &fy, T &cx, T &cy, T &base_line) {
    fx = camera_params[0];
    fy = camera_params[1];
    cx = camera_params[2];
    cy = camera_params[3];
    base_line = camera_params[4];
  }

  std::string left_image_file;
  std::string right_image_file;
  std::string disparity_image_file;
  std::vector<double> camera_params;
};

class StereoDataLoader {
 public:
  static std::vector<StereoImageSet> load(const std::string& filename) {
    std::vector<StereoImageSet> stereo_sets;

    FILE* fp = fopen(filename.c_str(), "r");
    if (!fp) {
      std::cerr << "Error: Cannot open file " << filename << std::endl;
      return stereo_sets;
    }

    char readBuffer[65536];
    rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));

    rapidjson::Document doc;
    doc.ParseStream(is);
    fclose(fp);

    if (doc.HasParseError()) {
      std::cerr << "JSON parse error: " << doc.GetParseError() << std::endl;
      return stereo_sets;
    }

    if (!doc.IsArray()) {
      std::cerr << "Error: Root element must be an array" << std::endl;
      return stereo_sets;
    }

    // Check if array size is multiple of 3
    if (doc.Size() % 3 != 0) {
      std::cerr << "Warning: Array size (" << doc.Size() << ") is not multiple of 3" << std::endl;
    }

    for (rapidjson::SizeType i = 0; i + 2 < doc.Size(); i += 3) {
      StereoImageSet stereo_set = parse(doc[i], doc[i + 1], doc[i + 2], i);
      if (stereo_set.is_valid()) {
        stereo_sets.push_back(stereo_set);
      }
    }

    return stereo_sets;
  }

 private:
  static StereoImageSet parse(const rapidjson::Value& left_item,
                                const rapidjson::Value& right_item,
                                const rapidjson::Value& disparity_item,
                                int base_index) {
    StereoImageSet stereo_set;

    // Parse left image
    if (left_item.IsObject() && left_item.HasMember("filename") && left_item["filename"].IsString()) {
      stereo_set.left_image_file = left_item["filename"].GetString();

      // Extract camera parameters from left image
      if (left_item.HasMember("params") && left_item["params"].IsArray()) {
        const rapidjson::Value& params = left_item["params"];
        for (rapidjson::SizeType j = 0; j < params.Size(); j++) {
          if (params[j].IsNumber()) {
            stereo_set.camera_params.push_back(params[j].GetDouble());
          }
        }
      }
    }

    // Parse right image
    if (right_item.IsObject() && right_item.HasMember("filename") && right_item["filename"].IsString()) {
      stereo_set.right_image_file = right_item["filename"].GetString();
    }

    // Parse disparity image
    if (disparity_item.IsObject() && disparity_item.HasMember("filename") && disparity_item["filename"].IsString()) {
      stereo_set.disparity_image_file = disparity_item["filename"].GetString();
    }

    // Validate the set forms a complete stereo pair
    if (!stereo_set.is_valid()) {
      std::cerr << "Warning: Incomplete stereo set at indices " << base_index
                << "-" << base_index + 2 << ", skipping" << std::endl;
    }

    return stereo_set;
  }
};


#endif //HOBOT_STEREONET_STANDALONE_DATA_LOADER_H_
