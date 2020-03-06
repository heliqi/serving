# Description: Tensorflow Serving tvm servable.

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = ["-layering_check"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "g3doc/sitemap.md",
        ],
    ),
)

cc_library(
    name = "tvm_source_adapter",
    srcs = ["tvm_source_adapter.cc", "tvm_factory.cc"],
    hdrs = ["tvm_source_adapter.h", "tvm_factory.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":tvm_source_adapter_proto",
        ":tvm_config_proto",
        ":tvm_loader",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "//tensorflow_serving/apis:input_proto",
        "//tensorflow_serving/core:simple_loader",
        "//tensorflow_serving/core:source_adapter",
        "//tensorflow_serving/core:storage_path",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tvm_loader",
    srcs = ["tvm_loader.cc"],
    hdrs = ["tvm_loader.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@com_github_dmlc_tvm//:tvm_runtime",
        ":ndarray_util",
        "//tensorflow_serving/core:simple_loader",
    ],
    includes = ["@com_github_dmlc_tvm//3rdparty/dmlc-core/include",
	        "@com_github_dmlc_tvm//include",
	        "@com_github_dmlc_tvm//3rdparty/dlpack/include",
		"@com_github_dmlc_tvm//src/runtime"],
    alwayslink = 1,
)

#cc_library(
#    name = "tvm_runtime",
#    srcs = ["tvm_runtime_pack.cc"],
#    hdrs = [
#      "//third_party/tvm/src/runtime/c_runtime_api.cc",
#      "//third_party/tvm/src/runtime/runtime_base.h",
#      "//third_party/tvm/src/runtime/cpu_device_api.cc",
#      "//third_party/tvm/src/runtime/workspace_pool.h",
#      "//third_party/tvm/src/runtime/workspace_pool.cc",
#      "//third_party/tvm/src/runtime/module_util.cc",
#      "//third_party/tvm/src/runtime/module_util.h",
#      "//third_party/tvm/src/runtime/module.cc",
#      "//third_party/tvm/src/runtime/file_util.h",
#      "//third_party/tvm/src/runtime/meta_data.h",
#      "//third_party/tvm/src/runtime/registry.cc",
#      "//third_party/tvm/src/runtime/file_util.cc",
#      "//third_party/tvm/src/runtime/threading_backend.cc",
#      "//third_party/tvm/src/runtime/thread_pool.cc",
#      "//third_party/tvm/src/runtime/ndarray.cc",
#      "//third_party/tvm/src/runtime/dso_module.cc",
#      "//third_party/tvm/src/runtime/system_lib_module.cc",
#      "//third_party/tvm/src/runtime/graph/graph_runtime.cc",
#      "//third_party/tvm/src/runtime/graph/graph_runtime.h",
#    ],
#    visibility = [
#        "//visibility:public",
#    ],
#    includes = ["//third_party/tvm/3rdparty/dmlc-core/include",
#	        "//third_party/tvm/include",
#	        "//third_party/tvm/3rdparty/dlpack/include",
#		"//third_party/tvm/tvm/src/runtime"],
#    alwayslink = 1,
#)

cc_library(
    name = "predict_impl",
    srcs = ["predict_impl.cc"],
    hdrs = ["predict_impl.h",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":ndarray_util",
        "//tensorflow_serving/apis:predict_proto",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/model_servers:server_core",
    ],
)

cc_library(
    name = "ndarray_util",
    srcs = ["ndarray_util.cc"],
    hdrs = ["ndarray_util.h", "tvm_loader.h",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@com_github_dmlc_tvm//:tvm_runtime",
        "//tensorflow_serving/apis:predict_proto",
        "//tensorflow_serving/core:servable_handle",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
    includes = ["@com_github_dmlc_tvm//3rdparty/dmlc-core/include",
	        "@com_github_dmlc_tvm//include",
	        "@com_github_dmlc_tvm//3rdparty/dlpack/include",
		"@com_github_dmlc_tvm//src/runtime"],
)


cc_library(
    name = "get_model_metadata_impl",
    srcs = ["get_model_metadata_impl.cc"],
    hdrs = ["get_model_metadata_impl.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow_serving/apis:get_model_metadata_proto",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/model_servers:server_core",
    ],
)

cc_test(
    name = "tvm_source_adapter_test",
    size = "medium",
    srcs = ["tvm_source_adapter_test.cc"],
    deps = [
        ":tvm_source_adapter",
        ":tvm_source_adapter_proto",
        ":tvm_config_proto",
        "//tensorflow_serving/core:loader",
        "//tensorflow_serving/core:servable_data",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/util:any_ptr",
    ],
)

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

serving_proto_library(
    name = "tvm_source_adapter_proto",
    srcs = ["tvm_source_adapter.proto"],
    deps = [
        ":tvm_config_proto",
    ],
    visibility = [
        "//visibility:public",
    ],
    cc_api_version = 2,
)

serving_proto_library(
    name = "tvm_config_proto",
    srcs = ["tvm_config.proto"],
    visibility = [
        "//visibility:public",
    ],
    cc_api_version = 2,
)

