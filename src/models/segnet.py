
import os
import cv2
import numpy as np
import onnxruntime as ort

from src import BaseInference
from src.util import (
    non_max_suppression,
    process_mask,
    scale_boxes,
    gen_color, 
    vis_result,
    letterbox
)

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


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
        self.load_model(model_path)
        
    def load_model(self, model_path):
        self.model = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        
        self.inp_name = self.model.get_inputs()[0].name
        self.opt_name = [
            self.model.get_outputs()[0].name,
            self.model.get_outputs()[1].name,
            self.model.get_outputs()[2].name,
            self.model.get_outputs()[3].name,
            self.model.get_outputs()[4].name
        ] 
        _, _, w, h, = self.model.get_inputs()[0].shape
        self.model_inpsize = (w,h)
        
    def inference(self, image: np.array, saved_path=None, conf_thres=0.7, iou_thres=0.45, classes=None):
        image, org_img = self.preprocess(image)
        
        result = self.model.run(
            self.opt_name, {self.inp_name: image.astype("float32")}
            )
        
        pred_img = [result[0], result[4]]
        results = self.postprocess(pred_img, image, org_img, conf_thres, iou_thres, classes)[0]
        
        if isinstance(results[1], np.ndarray):
            color = gen_color(len(CLASSES))
            _, mask_img, vis_img = vis_result(org_img, results, color, CLASSES)
        else:
            print("No segmentation result")
            return [], []
        
        if saved_path is not None:
            os.makedirs(saved_path, exist_ok=True)
            cv2.imwrite(f"./{saved_path}/mask_image.jpg", mask_img)
            cv2.imwrite(f"./{saved_path}/visual_image.jpg", vis_img)
            print('--> Save inference result')
        
        return mask_img, vis_img
    
    def preprocess(self, im:np.array) -> list:
        """ Preprocessing function with reshape and normalize input

        Args:
            im (np.array, optional): input image
            new_shape (tuple, optional): new shape to resize. Defaults to (640, 640).
            color (tuple, optional): _description_. Defaults to (114, 114, 114).
            scaleup (bool, optional): resize small to large input size. Defaults to True.

        Returns:
            im: image after normalize and resize
            r: scale ratio between original and new shape 
            dw, dh: padding follow by yolo processing
        """
        image_3c = im

        # Convert the image_3c color space from BGR to RGB
        image_3c = cv2.cvtColor(image_3c, cv2.COLOR_BGR2RGB)
        image_3c, _, _ = letterbox(image_3c, new_shape=list(self.model_inpsize), auto=False)
        
        image_4c = np.array(image_3c) / 255.0
        image_4c = image_4c.transpose((2, 0, 1))
        image_4c = np.expand_dims(image_4c, axis=0).astype(np.float32)
        image_4c = np.ascontiguousarray(image_4c)  # contiguous
        
        return image_4c, image_3c
    
    def postprocess(self, preds, img, org_img, conf_thres, iou_thres, classes=None):
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
    
    