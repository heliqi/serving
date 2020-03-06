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

#ifndef TENSORFLOW_SERVING_SERVABLES_TVM_TVM_LOADER_H_
#define TENSORFLOW_SERVING_SERVABLES_TVM_TVM_LOADER_H_

#include <string>
#include <unordered_set>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow_serving/servables/tvm/ndarray_util.h"

// TVM Headers
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/ndarray.h>

namespace tensorflow {
namespace serving {

struct TVMBundle {
  ~TVMBundle() {
  }

  tvm::runtime::Module mod;
  // We need to support this to be friendly with gRPC and other frontend API.
  MetaGraphDef meta_graph_def;

  TVMBundle() = default;
};

// Loads a SavedModel from the specified export directory. 
Status TVMLoadModel(const std::string& export_dir,
                  TVMBundle* const bundle);
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TVM_TVM_LOADER_H_
