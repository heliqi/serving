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

#ifndef TENSORFLOW_SERVING_SERVABLES_TVM_TVM_FACTORY_H_
#define TENSORFLOW_SERVING_SERVABLES_TVM_TVM_FACTORY_H_

#include "tensorflow_serving/resources/resources.pb.h"
#include "tensorflow_serving/servables/tvm/tvm_config.pb.h"
#include "tensorflow_serving/servables/tvm/tvm_loader.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace serving {

// A factory that creates TVMBundle from export paths.
//
// TVMBundle holds the necessary objects of TVM runtime required for inference.
//
// The factory can also estimate the resource (e.g. RAM) requirements of a
// TVMBundle based on the export.
//
// This class is thread-safe.
class TVMFactory {
 public:
  static Status Create(const TVMConfig& config,
                       std::unique_ptr<TVMFactory>* factory);

  // Instantiates a bundle from a given export path.
  Status CreateTVM(const string& path,
                             std::unique_ptr<TVMBundle>* bundle);

  // Estimates the resources a session bundle will use once loaded, from its
  // export path.
  Status EstimateResourceRequirement(const string& path,
                                     ResourceAllocation* estimate) const;

 private:

  TVMFactory(const TVMConfig& config);
  const TVMConfig config_;

  TF_DISALLOW_COPY_AND_ASSIGN(TVMFactory);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TVM_TVM_FACTORY_H_
