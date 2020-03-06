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

#ifndef TENSORFLOW_SERVING_SERVABLES_TVM_NDARRAY_UTIL_H_
#define TENSORFLOW_SERVING_SERVABLES_TVM_NDARRAY_UTIL_H_

#include <string>
#include <unordered_set>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/servables/tvm/tvm_loader.h"
#include "tensorflow/core/framework/tensor.h"

// TVM Headers
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/ndarray.h>

namespace tensorflow {
namespace serving {
using tvm::runtime::NDArray;
class TVMBundle;

std::string GetNameHint(TVMBundle *bundle, int type, int index);
DataType DLTypeToDataType(NDArray &ndarray);
size_t GetNDArraySize(NDArray &ndarray);
Status CopyNDArrayFromTensorProto(NDArray &ndarray, const TensorProto &ptensor);
Status CopyTensorFromNDArray(Tensor &tensor, NDArray &ndarray);
TensorInfo MakeTensorInforFromNDArray(NDArray &ndarray, std::string name);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TVM_NDARRAY_UTIL_H_
