
### `__init__(self, onnxpath, name_file, CLASSES)`
- **作用**：类的构造函数，用于初始化YOLOv5对象。
- **参数**：
  - `onnxpath`：ONNX模型文件的路径。
  - `name_file`：用于保存检测结果的文件名。
  - `CLASSES`：一个列表，包含模型可以识别的所有类别名称。
- **使用方法**：
  ```python
  classes = ["class1", "class2", ...]  # 替换为实际类别
  model_path = "path/to/your/model.onnx"
  yolo = YOLOV5(model_path, "output.txt", classes)
  ```

### `get_input_name(self)`
- **作用**：获取ONNX模型的输入节点名称。
- **使用方法**：通常不需要直接调用，构造函数中已使用。

### `get_output_name(self)`
- **作用**：获取ONNX模型的输出节点名称。
- **使用方法**：通常不需要直接调用，构造函数中已使用。

### `get_input_feed(self, img_tensor)`
- **作用**：准备输入数据的字典，用于ONNX Runtime推理。
- **参数**：
  - `img_tensor`：要进行推理的图像张量。
- **使用方法**：通常不需要直接调用，`inference_photo`和`inference_video`函数中已使用。

### `nms(self, dets, thresh)`
- **作用**：执行非极大值抑制（Non-Maximum Suppression），用于去除重叠的检测框。
- **参数**：
  - `dets`：检测结果。
  - `thresh`：非极大值抑制的阈值。
- **使用方法**：
  ```python
  keep_indices = yolo.nms(detections, 0.4)  # 0.4为示例阈值
  ```

### `xywh2xyxy(self, x)`
- **作用**：将检测框的坐标从中心点和宽高表示转换为左上角和右下角的坐标表示。
- **使用方法**：通常不需要直接调用，`filter_box`函数中已使用。

### `filter_box(self, org_box, conf_thres, iou_thres)`
- **作用**：过滤检测结果，移除低置信度的框，并应用非极大值抑制。
- **参数**：
  - `org_box`：原始的检测框。
  - `conf_thres`：置信度阈值。
  - `iou_thres`：IOU阈值，用于非极大值抑制。
- **使用方法**：
  ```python
  filtered_boxes = yolo.filter_box(raw_detections, 0.25, 0.45)  # 0.25和0.45为示例阈值
  ```

### `inference_photo(self, img_path)`
- **作用**：对单张图片进行推理。
- **参数**：
  - `img_path`：图片的路径。
- **使用方法**：
  ```python
  predictions, processed_image, original_size = yolo.inference_photo("path/to/image.jpg")
  ```

### `inference_video(self, img)`
- **作用**：对视频帧进行推理。
- **参数**：
  - `img`：视频帧的图像数据。
- **使用方法**：
  ```python
  predictions, processed_image, original_size = yolo.inference_video(frame)
  ```

### `draw(self, image, box_data, size, RP=False, return_class=False, WRP=False, img_return=True)`
- **作用**：在图像上绘制检测框和类别标签。
- **参数**：
  - `image`：要绘制的图像。
  - `box_data`：检测框数据。
  - `size`：原始图像的尺寸。
  - `RP`：是否在控制台打印结果。
  - `return_class`：是否返回类别信息。
  - `WRP`：是否将结果写入文件。
  - `img_return`：是否返回处理后的图像。
- **使用方法**：
  ```python
  result_image = yolo.draw(processed_image, filtered_boxes, original_size, RP=True, return_class=True)
  ```

这些函数提供了使用YOLOv5模型进行目标检测的完整流程，包括加载模型、处理图像、执行推理、过滤结果和绘制检测框。
