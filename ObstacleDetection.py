import cv2 as cv
import numpy as np
from ultralytics import YOLO
class ObstacleDetection:
    def __init__(self):
        self.model = YOLO('yolo11n.pt')
        self.obstacle_class_id = [0,2,3,5,7]  # person, car, bicycle, bus, truck
        self.obj = {}

    def obstacle_map(self, img):
        self.results = self.model(img, classes=self.obstacle_class_id, imgsz=320, conf=0.4)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for result in self.results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id in self.obstacle_class_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cv.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                    self.obj[self.model.names[cls_id]] = {
                        "conf": conf,
                        "cords": [x1, y1, x2, y2]
                    }
        return mask
    
    def obstacle_vis(self, img):
        for cls in self.obj:
            x1, y1, x2, y2 = self.obj[cls]["cords"]
            cv.rectangle(img, (x1, y1), (x2, y2), (0,0, 255), 2)
            label = f"{cls} {self.obj[cls]["conf"]:.2f}"
            cv.putText(img, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return img