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
    test_detect('samples/testface.jpg', 'output/testface.jpg', model)
    
    ### Segmentation
    model = YoloSeg('weights/yolov7-seg-480-640.onnx')
    test_seg("samples/bus.jpg", model)

