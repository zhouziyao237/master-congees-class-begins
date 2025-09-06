# Master-Congees-Class-Begins

本项目主要实现基于YOLOv5模型的ONNX格式推理接口，支持图像和视频的目标检测任务。项目包含多个版本的API实现，适用于不同的应用场景。

## 项目结构

- `onnx_API/ONNX_API.py`: YOLOv5推理接口主文件（通用版本）
- `onnx_API/ONNX_API_v1.0/`: 第一版API实现目录
  - `ONNX_API.py`: YOLOv5推理接口文件
  - `ONNX_API_CS_photo_v1.0.py`: 图像检测接口实现
  - `ONNX_API_CS_video_v1.0.py`: 视频检测接口实现
- `onnx_API/ONNX_API_v2.0/`: 第二版API实现目录
  - `ONNX_API.py`: YOLOv5推理接口文件
  - `ONNX_API_CS_v2.0.py`: 检测接口实现
  - `result.txt`: 结果输出文件

## 主要功能

- 支持加载ONNX格式的YOLOv5模型
- 提供图像和视频的目标检测功能
- 包含预处理、推理和后处理全流程实现
- 支持边界框转换、非极大值抑制(NMS)等常用目标检测操作

## 使用方法

1. 准备ONNX格式的YOLOv5模型和类别标签文件
2. 根据需求选择合适的API版本
3. 调用相应的推理方法进行检测

## 类和方法说明

`YOLOV5`类提供了主要的推理功能，包含以下关键方法：
- `__init__`: 初始化模型路径、类别标签等参数
- `get_input_name`: 获取模型输入名称
- `get_output_name`: 获取模型输出名称
- `get_input_feed`: 构建模型输入
- `nms`: 非极大值抑制，用于去除重复检测框
- `xywh2xyxy`: 将边界框格式从[x,y,w,h]转换为[x1,y1,x2,y2]
- `filter_box`: 过滤掉无用的检测框
- `inference_photo`: 图像检测接口
- `inference_video`: 视频检测接口
- `draw`: 绘制检测结果

## 版本说明

项目包含多个API版本实现，不同版本可能在功能完整性和接口设计上有所差异，建议优先使用最新版本。

## 许可证

请查看项目根目录下的LICENSE文件获取许可证信息。