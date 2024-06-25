# Selfie
Several components for processing face tasks such as face parsing, face detection, etc in selfies with ONNXRuntime

# Download checkpoint
[YoloSeg](https://drive.google.com/file/d/1tT6-jNY4TXD-oWIc2G4lTZC4Ts4lZLLy/view?usp=drive_link)

[YoloDetect-base](https://drive.google.com/file/d/1-8r31t1zUPU7pt6WrzTQ8ONGbpMFPLwK/view?usp=drive_link)

[Face parsing](https://huggingface.co/jonathandinu/face-parsing/tree/main/onnx)

## 1. Yolo Segmetation and Yolo Detection
```python
import os
import cv2
from src.yolo.detnet import YoloDet
from src.yolo.segnet import YoloSeg


def test_seg(image, model):
    image = cv2.imread(image)
        
    # save model path
    processed_img, mask_img, vis_img = model.predict(
        image,
        conf_thres=0.7,
        iou_thres=0.45, 
        classes=0 # [0,5,8] specific classes want to get. Default is None
    )
    
    cv2.imwrite('processed_img.jpg', processed_img)
    cv2.imwrite('mask_img.jpg', mask_img)
    cv2.imwrite('vis_img.jpg', vis_img)

def test_detect(inp_path, opt_path, model):
    image = cv2.imread(inp_path)
    
    bboxes, scores, _, _ = model.predict(image, det_thres=0.7, get_layer='face')
    
    if len(bboxes) > 0:
        for xyxy, score in zip(bboxes, scores):
            x1, y1, x2, y2 = xyxy.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            cv2.putText(
                image,
                f"{score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )
    
    cv2.imwrite(opt_path, image)
    

if __name__ == '__main__':
    # Detection
    model = YoloDet('weights/yolov7-headface-v1.onnx')
    test_detect('samples/aa1.jpg', 'output/aa1.jpg', model)
    
    ### Segmentation
    model = YoloSeg('weights/yolov7-seg-480-640.onnx')
    test_seg("samples/case.jpg", model)
```

Find COCO classes 
```python
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
         'hair drier', 'toothbrush']
```

## 2. Human parsing
Currently, we need ImageProcessor class from transformer for several steps. So. let install the transformer package

``` pip install -q transformer ```

And now, testing the performance with ONNX
```python
from src.parser.faceparser import ParserNet

model = ParserNet('weights/model_quantized.onnx')
img = cv2.imread('samples/testface2.jpg')[:,:,::-1]
labels = model.predict(img)

cv2.imwrite('result.jpg', labels)
```