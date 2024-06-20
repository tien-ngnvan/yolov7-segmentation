import os
import cv2
from src import DetectBase, YoloSeg


def test_seg():
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


def test_detect():
    image = cv2.imread(os.path.join('samples', 'bus.jpg'))
    
    # load model
    model = DetectBase('weights/yolov7-tiny-v0.onnx')
    
    bboxes, scores, labels, kpts = model.inference(image, det_thres=0.5, get_layer='head')
    
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
    

if __name__ == '__main__':
    test_seg()
    test_detect()