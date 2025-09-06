# Master Congees Class Begins

This project primarily implements an ONNX format inference interface based on the YOLOv5 model, supporting object detection tasks in images and videos. The project contains multiple versions of API implementations, suitable for different application scenarios.

## Project Structure

- `onnx_API/ONNX_API.py`: Main YOLOv5 inference interface file (generic version)
- `onnx_API/ONNX_API_v1.0/`: First version API implementation directory
  - `ONNX_API.py`: YOLOv5 inference interface file
  - `ONNX_API_CS_photo_v1.0.py`: Image detection interface implementation
  - `ONNX_API_CS_video_v1.0.py`: Video detection interface implementation
- `onnx_API/ONNX_API_v2.0/`: Second version API implementation directory
  - `ONNX_API.py`: YOLOv5 inference interface file
  - `ONNX_API_CS_v2.0.py`: Detection interface implementation
  - `result.txt`: Result output file

## Main Features

- Supports loading YOLOv5 models in ONNX format
- Provides object detection capabilities for images and videos
- Includes full-process implementation of preprocessing, inference, and post-processing
- Supports common object detection operations such as bounding box conversion and Non-Maximum Suppression (NMS)

## Usage

1. Prepare the YOLOv5 model in ONNX format and the class label file
2. Choose the appropriate API version according to your needs
3. Call the corresponding inference method for detection

## Class and Method Description

The `YOLOV5` class provides the main inference functionality, including the following key methods:
- `__init__`: Initialize parameters such as model path and class labels
- `get_input_name`: Get the model input name
- `get_output_name`: Get the model output name
- `get_input_feed`: Build the model input
- `nms`: Non-Maximum Suppression, used to remove duplicate detection boxes
- `xywh2xyxy`: Convert the bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
- `filter_box`: Filter out useless detection boxes
- `inference_photo`: Image detection interface
- `inference_video`: Video detection interface
- `draw`: Draw detection results

## Version Notes

The project contains multiple API version implementations. Different versions may vary in terms of functional completeness and interface design. It is recommended to use the latest version first.

## License

Please check the LICENSE file in the project root directory for license information.