//
// Created by zhy on 7/9/24.
//
#include <cstdio>
#include "dnn/hb_dnn.h"

int main(int argc, char **argv) {
  int32_t model_count = 0;
  hbDNNTensorProperties properties;
  hbPackedDNNHandle_t packed_dnn_handle;
  hbDNNHandle_t dnn_handle;
  const char **model_name_list;
  char model_file[128];
  printf(" StereonetProcess::stereonet_init: %s\n", model_file);
  int ret = hbDNNInitializeFromFiles(&packed_dnn_handle, (char const **)&model_file, 1);
  return 0;
}
