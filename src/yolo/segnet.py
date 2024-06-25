import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import cv2
import numpy as np
from src import BaseInference
from src.yolo.utils import (
    non_max_suppression,
    process_mask,
    scale_boxes,
    gen_color, 
    vis_result,
    letterbox
)


CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
         'hair drier', 'toothbrush'
]


class YoloSeg(BaseInference):
    def __init__(self, model_path):
        super().__init__(model_path)
        
    def predict(self, image: np.array, conf_thres=0.7, iou_thres=0.45, classes=None):
        tensor_img, org_img = self.preprocess(image)
        
        result =  self.sess.run(self.output_names, dict(zip(self.input_names, [tensor_img])))
        
        pred_img = [result[0], result[4]]
        results = self.postprocess(pred_img, tensor_img, org_img, conf_thres, iou_thres, classes)[0]
        
        if isinstance(results[1], np.ndarray):
            color = gen_color(len(CLASSES))
            processed_img, mask_img, vis_img = vis_result(org_img, results, color, CLASSES)
        else:
            print("No segmentation result")
            return [], [], []
            
        return processed_img, mask_img, vis_img
    
    def preprocess(self, im:np.array, img_size=(480, 640)) -> list:
        """ Preprocessing function with reshape and normalize input

        Args:
            im (np.array, optional): input image
            img_size (tuple, optional): new shape to resize. Defaults to (480, 640).

        Returns:
            im_4c: The tensor image after normalize and resize 
            im: The image after resize 
        """
        image = im
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, _, _ = letterbox(image, new_shape=img_size)
        
        image_4c = image.transpose((2, 0, 1))
        image_4c = np.expand_dims(image_4c, axis=0)
        image_4c = np.ascontiguousarray(image_4c, dtype=np.float32)  # contiguous
        image_4c /= 255.
        
        return image_4c, image
    
    def postprocess(self, preds, img, org_img, conf_thres, iou_thres, classes=None):
        """The Non-max-suppression function return 

        Args:
            preds (np.array): _description_
            img (np.array): _description_
            org_img (np.array): _description_
            conf_thres (float): _description_
            iou_thres (float): _description_
            classes (int, optional): List id classes from COCO. Defaults to None.

        Returns:
            list: The list of np.array[bbox, score, segmentation]
            
            Ex: The example below get output is an np.array which only return person class (class=0)
            
            tensor_image, org_im = preprocess(image)
            preds = model(tensor_image)
            result = postprocess(preds, tensor_image, org_img, 
                                conf_thres, iou_thres, classes=0)
        """
        p = non_max_suppression(preds[0], conf_thres, iou_thres, classes=classes)
        
        for i, pred in enumerate(p):  # per image
            shape = org_img.shape
            results = []
            proto = preds[1]  
            if not len(preds):
                results.append([[], [], []])  # save empty boxes
                continue
            masks = process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:])  # HWC
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            results.append([pred[:, :6], masks, shape[:2]])
        return results
    
    