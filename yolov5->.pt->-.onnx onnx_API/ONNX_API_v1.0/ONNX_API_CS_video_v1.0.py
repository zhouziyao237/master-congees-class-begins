
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

    while True:
        ret, img = cap.read()

        # 如果读取成功，显示图像
        if ret:
            output,or_img,size=model.inference_video(img)
            outbox=model.filter_box(output,0.3,0.2)#此处为筛选识别信息，前者为可信度，后者为IOU thres
            box, score, cl, img = model.draw(image = or_img, box_data = outbox, size = size, return_class = True, WRP = True)
            
            cv2.imshow('frame', img)
            print(box,score,cl)
                                                                                                                                                                       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    onnx_path='yolov5s.onnx'
    model=YOLOV5(onnx_path, "result.txt", CLASSES)
    #清空输出结果文件
    

    yolov5_run(0)
