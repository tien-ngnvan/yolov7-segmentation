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