import os
import cv2
import numpy as np
import onnxruntime
import time
import os

class YOLOV5():
    def __init__(self,onnxpath, name_file, CLASSES):
        self.onnx_session = onnxruntime.InferenceSession(onnxpath)
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()
        self.CLASSES = CLASSES

        self.name_file = name_file
        with open(name_file, "a") as f:
            f.write("")
        self.save_path = os.path.join("./", self.name_file)
        os.remove(self.save_path)
        


    def get_input_name(self):
        input_name=[]
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
    def get_output_name(self):
        output_name=[]
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_feed(self,img_tensor):
        input_feed={}
        for name in self.input_name:
            input_feed[name]=img_tensor
        return input_feed

    def nms(self, dets, thresh):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        scores = dets[:, 4]
        keep = []
        index = scores.argsort()[::-1] 

        while index.size > 0:
            i = index[0]
            keep.append(i)

            x11 = np.maximum(x1[i], x1[index[1:]]) 
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])

            w = np.maximum(0, x22 - x11 + 1)                              
            h = np.maximum(0, y22 - y11 + 1) 

            overlaps = w * h

            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
            idx = np.where(ious <= thresh)[0]
            index = index[idx + 1]
        return keep


    def xywh2xyxy(self, x):
        # [x, y, w, h] to [x1, y1, x2, y2]
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y


    def filter_box(self, org_box,conf_thres,iou_thres): #过滤掉无用的框

        org_box=np.squeeze(org_box)
        conf = org_box[..., 4] > conf_thres
        box = org_box[conf == True]

        cls_cinf = box[..., 5:]
        cls = []
        for i in range(len(cls_cinf)):
            cls.append(int(np.argmax(cls_cinf[i])))
        all_cls = list(set(cls))     

        output = []
        for i in range(len(all_cls)):
            curr_cls = all_cls[i]
            curr_cls_box = []
            curr_out_box = []
            for j in range(len(cls)):
                if cls[j] == curr_cls:
                    box[j][5] = curr_cls
                    curr_cls_box.append(box[j][:6])
            curr_cls_box = np.array(curr_cls_box)
            # curr_cls_box_old = np.copy(curr_cls_box)
            curr_cls_box = self.xywh2xyxy(curr_cls_box)
            curr_out_box = self.nms(curr_cls_box,iou_thres)
            for k in curr_out_box:
                output.append(curr_cls_box[k])
        output = np.array(output)
        return output

    def inference_photo(self,img_path):
        img=cv2.imread(img_path)
        size = img.shape #原始图像格式
        or_img=cv2.resize(img,(640,640))
        img=or_img[:,:,::-1].transpose(2,0,1)  #BGR2RGB和HWC2CHW
        img=img.astype(dtype=np.float32)
        img/=255.0
        img=np.expand_dims(img,axis=0)
        input_feed=self.get_input_feed(img)
        pred=self.onnx_session.run(None,input_feed)[0]
        return pred,or_img,size
    
    def inference_video(self,img):
        size = img.shape #原始图像格式
        or_img=cv2.resize(img,(640,640))
        img=or_img[:,:,::-1].transpose(2,0,1)  #BGR2RGB和HWC2CHW
        img=img.astype(dtype=np.float32)
        img/=255.0
        img=np.expand_dims(img,axis=0)
        input_feed=self.get_input_feed(img)
        pred=self.onnx_session.run(None,input_feed)[0]
        return pred,or_img,size


    def draw(self, image, box_data, size, RP = False, return_class = False, WRP = False, img_return = True):  

        if len(box_data) <= 0:  #有可能没有识别到物品
            img=cv2.resize(image,(size[1],size[0]))
            return None, None, None, image

        img=cv2.resize(image,(size[1],size[0]))
        h,w = size[0],size[1]
        boxes=box_data[...,:4].astype(np.int32) 
        scores=box_data[...,4]
        classes=box_data[...,5].astype(np.int32) 


        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = box
            
            if RP == True:
                print('class: {}, score: {}'.format(self.CLASSES[cl], score))
                print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

            if WRP == True:
                with open(str(self.save_path), 'a') as f:  #将识别信息记录到txt文件中
                    f.write('class: {}, score: {}'.format(self.CLASSES[cl], score))
                    f.write('({},{}) ({},{})\n'.format(top*w//640, left*h//640, right*w//640, bottom*h//640))


            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(self.CLASSES[cl], score),
                        (top+10, left+10 ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

            if return_class == True:
                if img_return == True:
                    return box, score, cl, image
                else:
                    return box, score, cl
            else:
                if img_return == True:
                    return image
                else:
                    return None
