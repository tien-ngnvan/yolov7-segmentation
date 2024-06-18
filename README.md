# yolov7-segmentation

Implement Yolov7-segmentation with ONNX


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