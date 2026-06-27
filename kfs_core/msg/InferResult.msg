# 推理结果 — 单个检测框的核心信息
# 供下游功能包订阅 (如: 跟踪、规划、可视化等)
#
# 注意：此消息现在主要作为 InferResults.msg 中的数组元素使用。
# 直接订阅 ~/result 推荐使用 InferResults（一帧一个消息）。

std_msgs/Header header    # 时间戳 + frame_id (关联相机坐标系)

int32   class_id          # 类别 ID: 0=R1, 1=T, 2=F
string  class_name        # 类别名
float32 confidence        # 置信度 [0, 1]

float32 center_x          # 检测框中心 X (图像像素坐标)
float32 center_y          # 检测框中心 Y (图像像素坐标)

# 以下为可选扩展
float32 bbox_width        # 边界框宽度 (像素)
float32 bbox_height       # 边界框高度 (像素)
