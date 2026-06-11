# ============================================================
# FindOnnxRuntime.cmake
#
# 用法:
#   find_package(OnnxRuntime REQUIRED)
#
# 可设置变量:
#   ONNXRUNTIME_DIR — 指向 ONNX Runtime 安装根目录
#   (默认搜索 /usr/local, /usr, 以及常见的解压路径)
#
# 导出变量:
#   ONNXRUNTIME_INCLUDE_DIR
#   ONNXRUNTIME_LIBRARY
#   OnnxRuntime_FOUND
# ============================================================

# 允许用户通过 -DONNXRUNTIME_DIR=... 指定
if(NOT ONNXRUNTIME_DIR)
    # 尝试自动检测常见的安装/解压路径
    file(GLOB ONNX_CANDIDATES
        /usr/local/lib/libonnxruntime.so*
        /usr/lib/*/libonnxruntime.so*
        $ENV{HOME}/onnxruntime-linux-*/lib/libonnxruntime.so*
        $ENV{HOME}/.local/lib/libonnxruntime.so*
    )
    foreach(_cand ${ONNX_CANDIDATES})
        get_filename_component(_libdir "${_cand}" DIRECTORY)
        get_filename_component(_prefix  "${_libdir}" DIRECTORY)
        if(EXISTS "${_prefix}/include/onnxruntime/onnxruntime_cxx_api.h")
            set(ONNXRUNTIME_DIR "${_prefix}" CACHE PATH "ONNX Runtime install prefix")
            break()
        endif()
        # 某些发行版将头文件放在 include/onnxruntime_cxx_api.h
        if(EXISTS "${_prefix}/include/onnxruntime_cxx_api.h")
            set(ONNXRUNTIME_DIR "${_prefix}" CACHE PATH "ONNX Runtime install prefix")
            break()
        endif()
    endforeach()
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_DIR}/include
          ${ONNXRUNTIME_DIR}/include/onnxruntime
          /usr/local/include/onnxruntime
          /usr/include/onnxruntime)

find_library(ONNXRUNTIME_LIBRARY onnxruntime
    HINTS ${ONNXRUNTIME_DIR}/lib
          ${ONNXRUNTIME_DIR}
          /usr/local/lib
          /usr/lib
          /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OnnxRuntime
    REQUIRED_VARS ONNXRUNTIME_LIBRARY ONNXRUNTIME_INCLUDE_DIR)

if(OnnxRuntime_FOUND AND NOT TARGET OnnxRuntime::onnxruntime)
    add_library(OnnxRuntime::onnxruntime SHARED IMPORTED)
    set_target_properties(OnnxRuntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")
endif()
