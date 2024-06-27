import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import cv2
import numpy as np
from src import BaseInference
from src.yolo.yolo_utils import letterbox

  
class YoloDet(BaseInference):
    def __init__(self, model_path):
        super().__init__(model_path)

    def predict(self, img: np.array, img_size=(640, 640), det_thres=0.7, get_layer='face') -> list:
        """ Execute the main process

        Args:
            img (np.array): _description_
            test_size (tuple, optional): _description_. Defaults to (640, 640).
            det_thres (float, optional): _description_. Defaults to 0.6.
            get_layer (_type_, optional): _description_. Defaults to None.

        Returns:
            bbox: xyxy object
            score: bbox score of object detection
            label: both face = head = 0 (the same class name)
            kpts: if get_layer == 'face' return keypoints else None
        """
        # preprocess input
        tensor_img, ratio, dwdh = self.preprocess(img, img_size)
        
        # model prediction
        outputs = self.sess.run(self.output_names, dict(zip(self.input_names, [tensor_img])))
        
        pred = outputs[1] if get_layer == 'face' else outputs[0]
   
        # postprocess output
        bboxes, scores, labels, kpts = self.postprocess(pred, ratio, dwdh, det_thres, get_layer)
        
        return bboxes, scores, labels, kpts
        
    def preprocess(self, im:np.array, img_size=(640, 640)) -> list:
        """ Preprocessing function with reshape and normalize input

        Args:
            im (np.array, optional): input image
            new_shape (tuple, optional): new shape to resize. Defaults to (640, 640).

        Returns:
            im: image after normalize and resize
            r: scale ratio between original and new shape 
            dw, dh: padding follow by yolo processing
        """
        image = im
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, ratio, (dw, dh) = letterbox(image, new_shape=img_size)
        
        image = np.array(image) / 255.
        image = np.expand_dims(image.transpose((2, 0, 1)), axis=0)
        image = np.ascontiguousarray(image, dtype=np.float32)
        
        return image, ratio, (dw, dh)
    
    def postprocess(self, pred, ratio, dwdh, det_thres = 0.7, get_layer=None):
        """Processing output model match output format

        Args:
            pred (array): predict output model 
                base_opt: batch_index, xmin, ymin, xmax, ymax, bbox_label, bbox_score
                - pred w/o keypoint
                    pred[batch, 7]: base_opt
                - pred with keypoint
                    pred[batch, 7 + kpts]: base_opt,
                                            x_keypoint1, y_keypoint1, keypoint1_score,
                                            x_keypoint2, y_keypoint2, keypoint2_score,
                                            ...
                                            x_keypoint, y_keypoint2, keypoint2_score,
            ratio (float, optional): 
            dwdh (float, optional): 
            det_thres (float, optional): _description_. Defaults to 0.7.
            get_layer (str, optional): get detection output layer if the ouput has:
                                        3 items [head, face, body]
                                        2 items [face, head].

        Returns:
            [bbox, score, class_name, keypoints]
        """
        assert get_layer != None, f'get_layer is not None'
        
        if isinstance(pred, list):
            pred = np.array(pred)
            
        pred = pred[pred[:, 6] > det_thres] # get sample higher than threshold
        
        padding = dwdh*2
        det_bboxes, det_scores, det_labels  = pred[:,1:5], pred[:,6], pred[:, 5]
        kpts = pred[:, 7:] if pred.shape[1] > 6 else None
        det_bboxes = (det_bboxes[:, 0::] - np.array(padding)) / ratio
        
        if kpts is not None:
            kpts[:,0::3] = (kpts[:,0::3] - np.array(padding[0])) / ratio
            kpts[:,1::3] = (kpts[:,1::3]- np.array(padding[1])) / ratio

        return det_bboxes, det_scores, det_labels, kpts