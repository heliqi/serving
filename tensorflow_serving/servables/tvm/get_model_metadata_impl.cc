/* Copyright 2017 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/servables/tvm/get_model_metadata_impl.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/servables/tvm/tvm_loader.h"

namespace tensorflow {
namespace serving {

Status TVMModelGetSignatureDef(
       ServerCore* core, const ModelSpec& model_spec,
       const GetModelMetadataRequest& request,
       GetModelMetadataResponse* response) {
  ServableHandle<TVMBundle> bundle;
  TF_RETURN_IF_ERROR(core->GetServableHandle(model_spec, &bundle));
  SignatureDefMap signature_def_map;
  for (const auto& signature : bundle->meta_graph_def.signature_def()) {
    (*signature_def_map.mutable_signature_def())[signature.first] =
        signature.second;
  }
  auto response_model_spec = response->mutable_model_spec();
  // TODO: name ??
  response_model_spec->set_name("tvm");
  response_model_spec->mutable_version()->set_value(bundle.id().version);

  (*response->mutable_metadata())["signature_def"].PackFrom(
      signature_def_map);
  return tensorflow::Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
