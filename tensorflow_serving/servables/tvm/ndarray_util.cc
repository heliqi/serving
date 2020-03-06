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

#include "tensorflow_serving/servables/tvm/ndarray_util.h"

#include <string>
#include <utility>

#include "tensorflow_serving/core/servable_handle.h"

namespace tensorflow {
namespace serving {
using tvm::runtime::NDArray;

std::string GetNameHint(TVMBundle *bundle, int type, int index) {
  tvm::runtime::TVMRetValue rv;
  rv = bundle->mod.GetFunction("get_name_hint")(type, index);
  if (rv.type_code() != kTVMDataType) {
    return *rv.ptr<std::string>();
  } else {
    return rv.operator std::string();
  }
};

DataType DLTypeToDataType(NDArray &ndarray) {
  // TODO : All error handling.
  switch(ndarray.operator->()->dtype.code) {
    case kDLInt:
      switch(ndarray.operator->()->dtype.bits) {
        case 8:
          return DT_INT8;
        case 16:
          return DT_INT16;
        case 32:
          return DT_INT32;
        case 64:
          return DT_INT64;
      }
    case kDLUInt:
      switch(ndarray.operator->()->dtype.bits) {
        case 8:
          return DT_UINT8;
        case 16:
          return DT_UINT16;
        case 32:
          return DT_UINT32;
        case 64:
          return DT_UINT64;
      }
    case kDLFloat:
      switch(ndarray.operator->()->dtype.bits) {
        case 32:
          return DT_FLOAT;
        case 64:
          return DT_DOUBLE;
      }
  }
}

size_t GetNDArraySize(NDArray &ndarray) {
  size_t size = 1;
  for (int i=0; i < ndarray.operator->()->ndim; ++i) {
    size *= ndarray.operator->()->shape[i];
  }
  return size * ndarray.operator->()->dtype.bits/8;
}

Status CopyNDArrayFromTensorProto(NDArray &ndarray, const TensorProto &ptensor) {
  TVMArrayHandle handle;
  if(TVMArrayFromDLPack(ndarray.ToDLPack(), &handle)) {
    return Status(tensorflow::error::INTERNAL, TVMGetLastError());
  }

  Tensor tensor;
  if (!tensor.FromProto(ptensor)) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "tensor parsing error");
  }

  if(TVMArrayCopyFromBytes (handle, (void*)tensor.tensor_data().data(),
			    GetNDArraySize(ndarray))) {
    return Status(tensorflow::error::INTERNAL, TVMGetLastError());
  }
  return Status::OK();
}

Status CopyTensorFromNDArray(Tensor &tensor, NDArray &ndarray) {
  TVMArrayHandle handle;
  if(TVMArrayFromDLPack(ndarray.ToDLPack(), &handle)) {
    return Status(tensorflow::error::INTERNAL, TVMGetLastError());
  }

  if(TVMArrayCopyToBytes (handle, (void*)tensor.tensor_data().data(),
			    GetNDArraySize(ndarray))) {
    return Status(tensorflow::error::INTERNAL, TVMGetLastError());
  }
  return Status::OK();
}

TensorInfo MakeTensorInforFromNDArray(NDArray &ndarray, std::string name) {
  TensorInfo tfinfo;
  DataType dtype;

  tfinfo.set_name(name);
  auto tshape = tfinfo.mutable_tensor_shape();
  dtype = DLTypeToDataType(ndarray);
  for(int i=0;i<ndarray.operator->()->ndim;++i) {
    tshape->add_dim()->set_size(ndarray.operator->()->shape[i]);
  }
  tfinfo.set_dtype(dtype);
  return tfinfo;
}

}  // namespace serving
}  // namespace tensorflow
