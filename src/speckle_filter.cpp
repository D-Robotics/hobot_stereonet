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

#include "speckle_filter.h"

void SpeckleFilter::filter(cv::Mat &img, float newVal, int maxSpeckleSize, float maxDiff) {
  CV_Assert(img.type() == CV_32FC1);

  const int rows = img.rows;
  const int cols = img.cols;
  const int imgSize = rows * cols;

  // -1: 未访问; 0: 无效; >0: 区域 label
  std::vector<int> labels(imgSize, -1);

  int curLabel = 0;

  // 4 邻域偏移
  const int dx[4] = {-1, 1, 0, 0};
  const int dy[4] = {0, 0, -1, 1};

#pragma omp parallel for schedule(dynamic, 16)
  for (int y = 0; y < rows; y++) {
    for (int x = 0; x < cols; x++) {
      int idx = y * cols + x;
      float val = img.at<float>(y, x);

      if (val == newVal || labels[idx] >= 0) continue; // 已标记或无效

      // 新区域
      curLabel++;
      std::vector<int> region;
      region.reserve(1024);
      region.push_back(idx);
      labels[idx] = curLabel;

      // 区域生长 BFS
      for (size_t ri = 0; ri < region.size(); ri++) {
        int pidx = region[ri];
        int py = pidx / cols;
        int px = pidx % cols;
        float pval = img.at<float>(py, px);

        for (int k = 0; k < 4; k++) {
          int nx = px + dx[k];
          int ny = py + dy[k];
          if (nx < 0 || nx >= cols || ny < 0 || ny >= rows) continue;

          int nidx = ny * cols + nx;
          if (labels[nidx] >= 0) continue;

          float nval = img.at<float>(ny, nx);
          if (nval == newVal) continue;

          if (std::fabs(pval - nval) <= maxDiff) {
            labels[nidx] = curLabel;
            region.push_back(nidx);
          }
        }
      }

      // 如果区域太小 -> 清零
      if ((int)region.size() <= maxSpeckleSize) {
        for (int ridx : region) {
          int ry = ridx / cols;
          int rx = ridx % cols;
          img.at<float>(ry, rx) = newVal;
        }
      }
    }
  }
}
