# TVM (tvm.ai) library.
# from https://github.com/dmlc/tvm.git

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache

exports_files(["LICENSE"])

cc_library(
    name = "tvm_runtime",
    srcs = [
        "src/runtime/c_runtime_api.cc",
        "src/runtime/cpu_device_api.cc",
        "src/runtime/workspace_pool.cc",
        "src/runtime/library_module.cc",
        #"src/runtime/module_util.cc",
        "src/runtime/module.cc",
        "src/runtime/registry.cc",
        "src/runtime/file_util.cc",
        "src/runtime/threading_backend.cc",
        "src/runtime/thread_pool.cc",
        "src/runtime/ndarray.cc",
        "src/runtime/object.cc", #新增
        "src/runtime/dso_library.cc",
        #"src/runtime/dso_module.cc",
        "src/runtime/system_library.cc",
        #"src/runtime/system_lib_module.cc",
        "src/runtime/graph/graph_runtime.cc",
        "src/runtime/graph/debug/graph_runtime_debug.cc",
    ],
    hdrs = glob([
      "3rdparty/dmlc-core/include/**/*.h",
      "include/**/*.h",
      "3rdparty/dlpack/include/**/*.h",
      "src/runtime/**/*.h",
      "src/runtime/graph/debug/*.h",
    ]),
    includes = ["3rdparty/dmlc-core/include",
	        "include",
	        "3rdparty/dlpack/include",
		"src/runtime",
    ],
    alwayslink = 1,
)
