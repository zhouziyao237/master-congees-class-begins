from ONNX_API import YOLOV5
import cv2



CLASSES=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'] #coco80类别

def yolov5_run(video_ID):

    cap = cv2.VideoCapture(video_ID)

    frame_interval = 5  # 设置抽帧间隔，例如每10帧识别一次
    frame_count = 0  # 初始化帧计数器
    global Servo_i, yaw_i, pitch_i

    servo_x = 0
    servo_y = 0


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


            if frame_count % frame_interval == 0:
                print(box, score, cl)
                print("fps: ", fps)
                cv2.imshow('frame', cs_img)

                                                                                                                                                                       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    onnx_path='./weights/yolov5s.onnx'
    model=YOLOV5(onnx_path, "result.txt", CLASSES)

    yolov5_run()


