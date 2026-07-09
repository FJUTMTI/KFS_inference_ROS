cmake_minimum_required(VERSION 3.16)
project(kfs_core VERSION 1.0.0 LANGUAGES C CXX)

# ============================================================
# 编译选项
# ============================================================
option(BUILD_WITH_REALSENSE "Build with RealSense D415/D435 support" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

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

# --- OpenCV ---
# 必须使用系统 OpenCV 4.5d (与 ROS2 cv_bridge 一致), 避免 ABI 冲突
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio calib3d
    PATHS /usr/lib/x86_64-linux-gnu/cmake/opencv4
          /lib/x86_64-linux-gnu/cmake/opencv4
    NO_DEFAULT_PATH)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# --- ONNX Runtime ---
find_package(OnnxRuntime REQUIRED)
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")

# --- librealsense2 (可选) ---
if(BUILD_WITH_REALSENSE)
    find_package(realsense2 QUIET)
    if(NOT realsense2_FOUND)
        find_path(REALSENSE_INCLUDE_DIR librealsense2/rs.hpp
            HINTS /usr/local/include /usr/include)
        find_library(REALSENSE_LIBRARY realsense2
            HINTS /usr/local/lib /usr/lib)
        if(REALSENSE_INCLUDE_DIR AND REALSENSE_LIBRARY)
            set(realsense2_FOUND TRUE)
            set(realsense2_INCLUDE_DIR ${REALSENSE_INCLUDE_DIR})
            set(realsense2_LIBRARY   ${REALSENSE_LIBRARY})
        endif()
    endif()
    if(realsense2_FOUND)
        message(STATUS "RealSense2: ENABLED (${realsense2_INCLUDE_DIR})")
    else()
        message(WARNING "librealsense2 未找到, RealSense 功能不可用")
    endif()
else()
    message(STATUS "RealSense2: DISABLED (USB 摄像头可用)")
    set(realsense2_FOUND FALSE)
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
# KFS_CORE 共享库 (核心逻辑)
# ============================================================
set(KFS_CORE_SOURCES
    src/yolo_detector.cpp
    src/lightbar_detector.cpp
    src/usb_capture.cpp
    src/config.cpp
    src/camera_factory.cpp
)

set(KFS_CORE_HEADERS
    include/yolo_detector.h
    include/lightbar_detector.h
    include/usb_capture.h
    include/kfs_core/icamera_capture.h
    include/kfs_core/config.h
    include/kfs_core/camera_factory.h
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

target_compile_definitions(kfs_core_lib PUBLIC
    $<$<BOOL:${realsense2_FOUND}>:HAS_REALSENSE>
)

# ============================================================
# CLI 可执行文件 (薄 Demo, 仅用于快速测试)
# ============================================================
add_executable(kfs_detect src/main.cpp)
target_link_libraries(kfs_detect PRIVATE kfs_core_lib)

# ============================================================
# ROS2 推理节点
# ============================================================
if(HAS_ROS2)
    add_executable(kfs_infer_node src/kfs_infer_node.cpp)

    ament_target_dependencies(kfs_infer_node
        rclcpp std_msgs std_srvs sensor_msgs cv_bridge ament_index_cpp
    )

    # 使用现代 API (替代弃用的 rosidl_target_interfaces)
    rosidl_get_typesupport_target(cpp_typesupport_target
        ${PROJECT_NAME} "rosidl_typesupport_cpp")
    target_link_libraries(kfs_infer_node ${cpp_typesupport_target})

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

# ROS2 节点 + 资源
if(HAS_ROS2)
    install(TARGETS kfs_infer_node
        DESTINATION lib/${PROJECT_NAME}
    )
    install(DIRECTORY config models
        DESTINATION share/${PROJECT_NAME}
    )
    # 接口安装由 rosidl_generate_interfaces 自动处理
    ament_package()
endif()

# ============================================================
# 帮助信息
# ============================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "  KFS 目标检测 (YOLO + lightbar_detector)")
message(STATUS "  类别: R1 / T / F + LIGHTBAR_{RED,GREEN,BLUE,YELLOW}")
message(STATUS "  RealSense: $<IF:$<BOOL:${realsense2_FOUND}>,ON,OFF>")
message(STATUS "========================================")
message(STATUS "  灯条测试: build/kfs_core/kfs_detect --config kfs_core/config/kfs_config.yaml --image kfs_core/assets/lightbar")
message(STATUS "========================================")
