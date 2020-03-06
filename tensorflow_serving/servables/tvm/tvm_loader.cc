/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_serving/servables/tvm/tvm_loader.h"

#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>
#include <unordered_set>
#include <stdlib.h>
#include <string.h>

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace serving {
using tvm::runtime::NDArray;

Status TVMLoadModel(const std::string& export_dir,
                  TVMBundle* const bundle) {

  // tvm module for compiled functions
  tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(export_dir + "/model.so");

  // json graph
  std::ifstream json_in(export_dir + "/model.json", std::ios::in);
  std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
  json_in.close();

  // parameters in binary
  std::ifstream params_in(export_dir + "/model.params", std::ios::binary);
  std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
  params_in.close();

  // parameters need to be TVMByteArray type to indicate the binary data
  TVMByteArray params_arr;
  params_arr.data = params_data.c_str();
  params_arr.size = params_data.length();

  int device_type = kDLCPU;
  int device_id = 0;

  // get global function module for graph runtime
  const char* tvm_runtime_log = getenv("TVM_RUNTIME_LOG");
  if ( strcmp(tvm_runtime_log, "1") == 0
       || strcmp(tvm_runtime_log, "true") == 0
       || strcmp(tvm_runtime_log, "True") == 0 ){
    bundle->mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime_debug.create"))(json_data, mod_syslib, device_type, device_id);
  }
  else{
    bundle->mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, device_type, device_id);
  }

  // get the function from the module(load patameters)
  tvm::runtime::PackedFunc load_params = bundle->mod.GetFunction("load_params");
  load_params(params_arr);

  // Setup the meta data information.
  SignatureDef signature_def;

  int num_inputs = bundle->mod.GetFunction("get_num_inputs")();
  int num_outputs = bundle->mod.GetFunction("get_num_inputs")();

  NDArray ndarray;

  std::string name_hint;
  for(int i = 0; i < num_inputs ; ++i) {
    ndarray = bundle->mod.GetFunction("get_graph_input")(i);
    name_hint = GetNameHint(bundle, 0, i);
    LOG(INFO) << "Input:" << name_hint;
    (*signature_def.mutable_inputs())[name_hint] =
      MakeTensorInforFromNDArray(ndarray, name_hint);
  }
  for(int i = 0; i < num_outputs ; ++i) {
    ndarray = bundle->mod.GetFunction("get_output")(i);
    name_hint = GetNameHint(bundle, 1, i);
    LOG(INFO) << "Output:" << name_hint;
    (*signature_def.mutable_outputs())[name_hint] =
      MakeTensorInforFromNDArray(ndarray, name_hint);
  }
  (*bundle->meta_graph_def.mutable_signature_def())["serving_default"] = signature_def;

  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
