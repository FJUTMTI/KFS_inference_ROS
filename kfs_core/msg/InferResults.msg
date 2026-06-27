# ============================================================
# KFS 目标检测结果 — 数组消息（推荐使用）
# ============================================================
# 每帧只发布一条消息，包含该帧全部检测结果 + 元数据。
# 替代原来“每检测一框发布一条 InferResult”的设计。
#
# 下游节点订阅 ~/result 即可一次性拿到一帧所有目标。

std_msgs/Header header

# 当前帧的所有检测框
# （数组元素复用 InferResult 结构，元素中的 header 字段可忽略）
InferResult[] detections

# 本次推理耗时（毫秒）
float32 inference_ms

# 原始图像尺寸（来自相机）
uint32 frame_width
uint32 frame_height
