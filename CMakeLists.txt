project(blink-mm-native-lib)
cmake_minimum_required(VERSION 3.0)

option(USE_TVM "OFF")

add_executable(
    energy_profiler
    "native/energy_profiler/energy_profiler.cc"
    "native/energy_profiler/main.cc"
)
target_include_directories(
    energy_profiler PUBLIC
    "native"
)

add_subdirectory("native/third_party/pybind11")
pybind11_add_module(blink_mm_native_lib native/main.cc)

if(NOT "${USE_TVM}" STREQUAL "OFF")
    add_subdirectory("${USE_TVM}" "tvm_bin")

    add_library(tvm_deploy INTERFACE)
    target_compile_definitions(
        tvm_deploy INTERFACE
        "DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>"
    )
    target_include_directories(
        tvm_deploy INTERFACE
        "${USE_TVM}/include"
        "${USE_TVM}/3rdparty/dmlc-core/include"
        "${USE_TVM}/3rdparty/dlpack/include"
    )
    target_link_libraries(
        tvm_deploy INTERFACE
        tvm_runtime
    )

    add_executable(
        profile_tvm_model
        "native/profile_tvm_model.cc"
    )
    target_link_libraries(
        profile_tvm_model
        PRIVATE tvm_deploy
    )
endif()
