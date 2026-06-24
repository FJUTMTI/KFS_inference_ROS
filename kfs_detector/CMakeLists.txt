cmake_minimum_required(VERSION 3.16)
project(kfs_core VERSION 1.0.0 LANGUAGES C CXX CUDA)

# ============================================================
# C++17 标准
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# 编译选项: DEBUG / RELEASE
# ============================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Debug 宏传给代码
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG_MODE)
endif()

# ============================================================
# CMake 模块路径
# ============================================================
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# ============================================================
# 检测构建模式: colcon (ROS2) 或 独立 CMake
# ============================================================
find_package(ament_cmake QUIET)
set(HAS_ROS2 ${ament_cmake_FOUND})

if(HAS_ROS2)
    message(STATUS "=== ROS2 / colcon 构建模式 ===")
else()
    message(STATUS "=== 独立 CMake 构建模式 (无 ROS2) ===")
endif()

# ============================================================
# 依赖查找
# ============================================================

# --- yaml-cpp (配置文件解析) ---
find_package(yaml-cpp REQUIRED)
message(STATUS "yaml-cpp found")

# --- OpenCV (显示 + 预处理) ---
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# --- ONNX Runtime (通过 FindOnnxRuntime.cmake 模块) ---
find_package(OnnxRuntime REQUIRED)
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime lib:     ${ONNXRUNTIME_LIBRARY}")

# --- librealsense2 ---
find_package(realsense2 QUIET)
if(NOT realsense2_FOUND)
    find_path(REALSENSE_INCLUDE_DIR librealsense2/rs.hpp
        HINTS /usr/local/include
              /usr/include)

    find_library(REALSENSE_LIBRARY realsense2
        HINTS /usr/local/lib
              /usr/lib)

    if(REALSENSE_INCLUDE_DIR AND REALSENSE_LIBRARY)
        set(realsense2_FOUND TRUE)
        set(realsense2_INCLUDE_DIR ${REALSENSE_INCLUDE_DIR})
        set(realsense2_LIBRARY   ${REALSENSE_LIBRARY})
    endif()
endif()

if(realsense2_FOUND)
    message(STATUS "RealSense2 include: ${realsense2_INCLUDE_DIR}")
    message(STATUS "RealSense2 lib:     ${realsense2_LIBRARY}")
else()
    message(WARNING
        "librealsense2 未找到！RealSense D415 功能不可用。\n"
        "USB 摄像头 (--camera usb) 仍可正常使用。\n"
        "如需 RealSense 支持: sudo apt install librealsense2-dev")
endif()

# ============================================================
# ROS2 依赖 (仅 colcon 模式)
# ============================================================
if(HAS_ROS2)
    find_package(rclcpp REQUIRED)
    find_package(std_msgs REQUIRED)
    find_package(std_srvs REQUIRED)
    find_package(sensor_msgs REQUIRED)
    find_package(cv_bridge REQUIRED)
    find_package(rosidl_default_generators REQUIRED)
endif()

# ============================================================
# ROS2 自定义接口 (msg / srv)
# ============================================================
if(HAS_ROS2)
    rosidl_generate_interfaces(${PROJECT_NAME}
        "msg/InferResult.msg"
        "msg/InferResults.msg"
        "srv/SetInferState.srv"
        DEPENDENCIES std_msgs
    )
endif()

# ============================================================
# KFS_CORE 共享库 (核心逻辑, 供 ROS2 节点 / CLI 链接)
# ============================================================
set(KFS_CORE_SOURCES
    src/yolo_detector.cpp
    src/usb_capture.cpp
    src/config.cpp
    src/camera_factory.cpp
    src/weaponhead_detector.cpp
)

set(KFS_CORE_HEADERS
    include/yolo_detector.h
    include/usb_capture.h
    include/kfs_core/icamera_capture.h
    include/kfs_core/config.h
    include/kfs_core/camera_factory.h
    include/weaponhead_detector.h
)

if(realsense2_FOUND)
    list(APPEND KFS_CORE_SOURCES src/rs_capture.cpp)
    list(APPEND KFS_CORE_HEADERS include/rs_capture.h)
endif()

add_library(kfs_core_lib SHARED ${KFS_CORE_SOURCES} ${KFS_CORE_HEADERS})

target_include_directories(kfs_core_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(kfs_core_lib PUBLIC
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
    yaml-cpp
    pthread
)

if(realsense2_FOUND)
    target_include_directories(kfs_core_lib PUBLIC ${realsense2_INCLUDE_DIR})
    target_link_libraries(kfs_core_lib PUBLIC ${realsense2_LIBRARY})
endif()

# 传递 HAS_REALSENSE 给库的使用者 (通过 target 而非全局)
target_compile_definitions(kfs_core_lib PUBLIC
    $<$<BOOL:${realsense2_FOUND}>:HAS_REALSENSE>
)

# ============================================================
# CLI 可执行文件 (薄 Demo, 仅用于快速测试)
# ============================================================
add_executable(kfs_detect src/main.cpp)
target_link_libraries(kfs_detect PRIVATE kfs_core_lib)

# ============================================================
# ROS2 推理节点 (仅 colcon 模式)
# ============================================================
if(HAS_ROS2)
    add_executable(kfs_infer_node src/kfs_infer_node.cpp)

    ament_target_dependencies(kfs_infer_node
        rclcpp
        std_msgs
        std_srvs
        sensor_msgs
        cv_bridge
    )

    # 链接生成的接口 (头文件 + 类型支持)
    rosidl_target_interfaces(kfs_infer_node
        ${PROJECT_NAME} "rosidl_typesupport_cpp")

    # 确保能找到 rosidl 生成的 kfs_core/msg/*.hpp (在纯 cmake 构建中)
    target_include_directories(kfs_infer_node PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp
    )

    target_link_libraries(kfs_infer_node kfs_core_lib)
endif()

# ============================================================
# 安装规则
# ============================================================
include(GNUInstallDirs)

# 库 + CLI 可执行文件
install(TARGETS kfs_core_lib kfs_detect
    EXPORT kfs_coreTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# 头文件
install(FILES ${KFS_CORE_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/kfs_core
)

# CMake 配置 (供其他项目 find_package)
install(EXPORT kfs_coreTargets
    FILE kfs_coreTargets.cmake
    NAMESPACE kfs::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kfs_core
)

install(FILES cmake/FindOnnxRuntime.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kfs_core
)

# ROS2 节点 + 接口 (仅 colcon 模式)
if(HAS_ROS2)
    install(TARGETS kfs_infer_node
        DESTINATION lib/${PROJECT_NAME}
    )

    # 接口安装由 rosidl_generate_interfaces 自动处理

    ament_package()
endif()

# ============================================================
# 安装规则
# ============================================================
install(TARGETS kfs_detect RUNTIME DESTINATION bin)

# ============================================================
# 帮助信息 (注意：当前源码在 kfs_detector/ 子目录，工作空间根有 build/)
# ============================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "  KFS 目标检测 (weaponhead + YOLO)")
message(STATUS "  3 类检测: R1 / T / F  +  weaponhead_detector (左右边界)")
message(STATUS "  配置: kfs_detector/config/kfs_config.yaml")
message(STATUS "========================================")
message(STATUS "  colcon: 从工作空间根运行 colcon build --base-paths kfs_detector ...")
message(STATUS "  standalone: cd kfs_detector; mkdir build; cd build; cmake .. ; make")
message(STATUS "  运行测试: build/kfs_core/kfs_detect --image kfs_detector/assets")
message(STATUS "========================================")
