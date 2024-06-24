# yolov7-segmentation

Implement Yolov7-segmentation with ONNX

# Download checkpoint
[YoloSeg](https://drive.google.com/file/d/1tT6-jNY4TXD-oWIc2G4lTZC4Ts4lZLLy/view?usp=drive_link)

[YoloDetect-tiny](https://drive.google.com/file/d/1Pj1im1OSAIdiK63_yF-jI278kdZTGe70/view?usp=drive_link)

[YoloDetect-base](https://drive.google.com/file/d/1-8r31t1zUPU7pt6WrzTQ8ONGbpMFPLwK/view?usp=drive_link)

# Inference
1. For Yolo segmentation
```python
import os
import cv2
from src.yolo import YoloSeg

# build model
model = YoloSeg(os.path.join('weights', 'yolov7-seg-480-640.onnx'))

# read images
img = cv2.imread(os.path.join("samples", "bus.jpg"))

# save model path
mask_img, vis_img = model.inference(
    img, 
    saved_path='output',
    conf_thres=0.7,
    iou_thres=0.45, 
    classes=0 # [0,5,8] specific classes want to get. Default is None
)
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

2. For Yolo Detection

```python
image = cv2.imread(os.path.join('samples', 'bus.jpg'))
    
# load model
model = DetectBase('weights/yolov7-tiny-v0.onnx')


# Face & Body: det_thresh = 0.6
# Head: det_thresh: 0.8

bboxes, scores, labels, kpts = model.inference(image, det_thres=0.6, get_layer='face') # change the get layer 'body' || 'face' || 'head'

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

cv2.imwrite('output/test.jpg', image)
```

