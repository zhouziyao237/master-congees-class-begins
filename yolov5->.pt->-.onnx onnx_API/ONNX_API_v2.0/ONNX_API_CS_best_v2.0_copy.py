from PID import PID
from ONNX_API import YOLOV5
import cv2
import serial
import json
import random
import threading


CLASSES=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'] #coco80类别

Servo_i = False
yaw_i = 0
pitch_i = 0


def send_json():
    global Servo_i, yaw_i, pitch_i
    while True:
        if Servo_i == False:
            try:
                data = data = {
                    "pitch:": pitch_i,
                    "yaw:": yaw_i
                }
                json_str = json.dumps(data)

                ser.write(json_str.encode())

                print("JSON data sent:", json_str.encode())
                # 等待接收确认消息
                response = ser.readline().decode().strip()
                print("Received confirmation message:", response)

            except Exception as e:
                print("Error sending JSON data:", str(e))

            Servo_i = True

def yolov5_run(video_ID):

    cap = cv2.VideoCapture(video_ID)

    frame_interval = 6  # 设置抽帧间隔，例如每10帧识别一次
    frame_count = 0  # 初始化帧计数器
    global Servo_i, yaw_i, pitch_i

    while True:
        ret, img = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 如果读取成功，显示图像
        if ret:
            frame_count += 1  # 增加帧计数器
            # 检查是否达到抽帧间隔
            if frame_count % frame_interval == 0:
                output,or_img,size=model.inference_video(img)
                outbox=model.filter_box(output,0.3,0.2)#此处为筛选识别信息，前者为可信度，后者为IOU thres
                box, score, cl, cs_img = model.draw(image = or_img, box_data = outbox, size = size, return_class = True, WRP = True)
                
                if cl == 0:
                    width, height = img.shape[:2]
                    
                    top, left, right, bottom = box

                    box_width = right - left
                    box_height = bottom - top


                    center_x = box_width / 2
                    center_y = box_height / 2

                    # (top, left), (right, bottom)
                    #   x     y     x + w   y + h

                    error_x = (right / 2) - (480 / 2)
                    error_y = (640 / 2) - (right / 2)  # 反转误差方向

                    print("center x: ", center_x," center y: ",center_y)
                    print("error x: ", error_x," error y: ",error_y)


            if frame_count % frame_interval == 0:
                print(box, score, cl)
                print("fps: ", fps)
                cv2.imshow('frame', cs_img)
                if Servo_i == True:
                    yaw_i = random.randint(-90, 90)
                    pitch_i = random.randint(-45, 45)
                    Servo_i = False
                

                                                                                                                                                                       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    onnx_path='yolov5s.onnx'
    model=YOLOV5(onnx_path, "result.txt", CLASSES)
    #清空输出结果文件
    
    ser = serial.Serial('COM6', 250000, timeout=1)

    thread2 = threading.Thread(target=yolov5_run, args=(0,))
    thread1 = threading.Thread(target=send_json)

    # 启动线程
    thread1.start()
    thread2.start()

    # 等待所有线程完成
    thread1.join()
    thread2.join()

    ser.close()

