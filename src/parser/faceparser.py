import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import cv2
import numpy as np 
from transformers import SegformerImageProcessor

from src import BaseInference



PARSER_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0],
    [255, 0, 85], [255, 0, 170],
    [0, 255, 0], [85, 255, 0], [170, 255, 0],
    [0, 255, 85], [0, 255, 170],
    [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [0, 85, 255], [0, 170, 255],
    [255, 255, 0], [255, 255, 85], [255, 255, 170],
    [255, 0, 255], [255, 85, 255], [255, 170, 255],
    [0, 255, 255], [85, 255, 255], [170, 255, 255]
]


class ParserNet(BaseInference):
    def __init__(self, model_path, processor="jonathandinu/face-parsing"):
        super().__init__(model_path)
        
        if isinstance(processor, str):
            self.processor = SegformerImageProcessor.from_pretrained(processor)
        else:
            self.processor = processor
            
    def predict(self, image: np.array):
        h, w, _ = image.shape
        inp_tensor = self.preprocess(image)

        logits = self.sess.run(self.output_names, dict(zip(self.input_names, inp_tensor)))
        logits = logits[0].squeeze().argmax(0)
        logits = self.postprocess(logits, org_img=(w,h))
        
        return logits

    def preprocess(self, image):
        tensor = self.processor(images=image, return_tensors="np")
        tensor = np.expand_dims(tensor['pixel_values'], 0)
        
        return tensor
    
    def postprocess(self, parsing_anno, org_img):
        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, org_img, interpolation=cv2.INTER_NEAREST)
        
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

        num_of_class = np.max(vis_parsing_anno)
        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1], :] = PARSER_COLORS[pi]
        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        
        return vis_parsing_anno_color
        
        
if __name__ == '__main__':
    model = ParserNet('weights/model_quantized.onnx')
    
    img = cv2.imread('samples/testface2.jpg')[:,:,::-1]
    
    labels = model.predict(img)
    
    cv2.imwrite('result.jpg', labels)
    
    print("Type: ", labels.shape)