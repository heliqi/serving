/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/servables/tvm/predict_impl.h"

#include <string>
#include <utility>
#include<iostream>
#include<chrono>
#include <string.h>
#include <stdlib.h>

#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/servables/tensorflow/predict_util.h"
#include "tensorflow_serving/servables/tensorflow/util.h"
#include "tensorflow_serving/servables/tvm/ndarray_util.h"

namespace tensorflow {
namespace serving {
using tvm::runtime::NDArray;

Status TVMPredictor::Predict(ServerCore* core, const ModelSpec& model_spec,
               const PredictRequest& request, PredictResponse* response) {

  ServableHandle<TVMBundle> bundle;
  TF_RETURN_IF_ERROR(core->GetServableHandle(model_spec, &bundle));

  for (const auto& kv : request.inputs()) {
    const auto& name = kv.first;
    const TensorProto tensor = kv.second;

    NDArray ndarray_in = bundle->mod.GetFunction("get_graph_input")(name);
    TF_RETURN_IF_ERROR(CopyNDArrayFromTensorProto(ndarray_in, tensor));
  }
  auto t1 = std::chrono::system_clock::now();
  auto t2 = std::chrono::system_clock::now();

  const char* tvm_runtime_log = getenv("TVM_RUNTIME_LOG");
  if ( strcmp(tvm_runtime_log, "1") == 0
       || strcmp(tvm_runtime_log, "true") == 0
       || strcmp(tvm_runtime_log, "True") == 0 ){
    t1 = std::chrono::system_clock::now();
    string s = bundle->mod.GetFunction("run_individual")(10,1,1);
    t2 = std::chrono::system_clock::now();
    if (strcmp(getenv("TVM_LOG"), "2") == 0)
      LOG(INFO) << "TVM Debug:" << s ;
  }
  else{
    t1 = std::chrono::system_clock::now();
    bundle->mod.GetFunction("run")();
    t2 = std::chrono::system_clock::now();
  }

  int num_outputs = bundle->mod.GetFunction("get_num_outputs")();

  NDArray ndarray_out;
  std::string name_hint;
  TensorShapeProto tshape;
  TensorProto ptensor_out;
  DataType dtype;
  for(int i = 0; i < num_outputs ; ++i) {
    ndarray_out = bundle->mod.GetFunction("get_output")(i);
    name_hint = GetNameHint(bundle.operator->(), 1, i);
    dtype = DLTypeToDataType(ndarray_out);
    for(int i=0;i<ndarray_out.operator->()->ndim;++i) {
      tshape.add_dim()->set_size(ndarray_out.operator->()->shape[i]);
    }
    auto otensor = Tensor(dtype, tshape);
    CopyTensorFromNDArray(otensor, ndarray_out);
    otensor.AsProtoField(&ptensor_out);
    (*response->mutable_outputs())[name_hint] = ptensor_out;
  }

  const char* tvm_log_env = getenv("TVM_LOG");
  if ( strcmp(tvm_log_env, "1") == 0
       || strcmp(tvm_log_env, "true") == 0
       || strcmp(tvm_log_env, "True") == 0 ){

      LOG(INFO) << "TVM Predict:"
            << "  " <<std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
  }


  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
