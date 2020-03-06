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

#include "tensorflow_serving/servables/tvm/tvm_factory.h"
#include "tensorflow_serving/servables/tvm/tvm_loader.h"

#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace serving {

Status TVMFactory::Create(
    const TVMConfig& config,
    std::unique_ptr<TVMFactory>* factory) {
  factory->reset(new TVMFactory(config));
  return Status::OK();
}

Status TVMFactory::EstimateResourceRequirement(
    const string& path, ResourceAllocation* estimate) const {
  // TODO
  Status status;
  return status;
  //return EstimateResourceFromPath(path, estimate);
}

Status TVMFactory::CreateTVM(
    const string& path, std::unique_ptr<TVMBundle>* bundle) {

  bundle->reset(new TVMBundle);
  TF_RETURN_IF_ERROR(TVMLoadModel(path, bundle->get()));

  return Status::OK();
}

TVMFactory::TVMFactory(
    const TVMConfig& config)
    : config_(config) {
    }
}  // namespace serving
}  // namespace tensorflow
