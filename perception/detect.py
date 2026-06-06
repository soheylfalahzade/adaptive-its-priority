import os
import logging
from typing import Dict, Any, Tuple, List
import cv2
import numpy as np

# بررسی نصب بودن کتابخانه اولترالیتیکس (YOLO)
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# تنظیمات سیستم لاگ برای مانیتورینگ وضعیت سیستم
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PerceptionEngine")


class YOLOPerceptionEngine:
    """A high-performance object detection pipeline for Intelligent Transportation Systems (ITS).
    
    Loads YOLOv11/v8 models to extract real-time vehicle counts, queue states, and 
    emergency vehicle (EV) presence metrics from intersection surveillance video streams.
    """
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        """Initializes the perception engine.
        
        Args:
            model_path (str): Path to the serialized YOLO weights file (.pt).
            conf_threshold (float): Minimum confidence threshold for detections.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        
        if YOLO is None:
            logger.error("Ultralytics library is not installed. Please run 'pip install ultralytics'.")
            return
            
        # بررسی وجود داشتن فایل وزن‌های مدل روی دیسک
        if not os.path.exists(model_path):
            logger.warning(f"Weights file not found at local path: '{model_path}'. Model will be initialized empty.")
        else:
            try:
                self.model = YOLO(model_path)
                logger.info(f"Successfully loaded YOLO model from '{model_path}'")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")

    def process_frame(self, frame: np.ndarray, ev_class_id: int = 0) -> Dict[str, Any]:
        """Processes a single video frame to extract structural state variables.
        
        Args:
            frame (np.ndarray): The input image/frame from the CCTV stream.
            ev_class_id (int): The target YOLO class ID representing emergency vehicles (Ambulance).
            
        Returns:
            Dict[str, Any]: Extracted states corresponding directly to the mathematical State Space (S_t).
        """
        # پیش‌فرض‌های حالت اولیه در صورت لود نشدن مدل یا فریم نامعتبر
        state_space = {
            "vehicle_count": 0,
            "queue_density": 0.0,
            "ev_detected": False,
            "ev_distance_simulated": 999.0  # مقدار دیفالت به معنای نبود آمبولانس
        }
        
        if self.model is None or frame is None:
            return state_space

        try:
            # اجرای استنتاج روی تصویر ورودی با آستانه اطمینان تعریف شده
            results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
            boxes = results.boxes
            
            total_vehicles = 0
            ev_found = False
            
            # شمارش کلاس‌های مختلف وسایل نقلیه
            for box in boxes:
                class_id = int(box.cls[0].item())
                
                # کلاس‌های استاندارد خودرو در مجموعه داده COCO (2: car, 5: bus, 7: truck)
                if class_id in [2, 5, 7]:
                    total_vehicles += 1
                
                # بررسی وجود آمبولانس/خودروی اورژانس
                if class_id == ev_class_id:
                    ev_found = True
                    # شبیه‌سازی فاصله آمبولانس بر اساس اندازه جعبه دوبعدی (BBox Size) در تصویر هوایی
                    # در کارهای آیندهون، این مقدار به صورت فیزیکی از سنسورهای کارلا خوانده می‌شود.
                    bbox_height = box.xyxy[0][3].item() - box.xyxy[0][1].item()
                    state_space["ev_distance_simulated"] = round(1000.0 / (bbox_height + 1e-5), 2)
            
            # تخصیص متغیرهای حالت استخراج شده
            state_space["vehicle_count"] = total_vehicles
            state_space["ev_detected"] = ev_found
            # تراکم ترافیک به عنوان یک فاکتور نرمالایز شده ساده (تعداد خودروها به ماکزیمم ظرفیت لاین)
            state_space["queue_density"] = round(min(total_vehicles / 15.0, 1.0), 2)
            
            logger.debug(f"Frame processed: Vehicles={total_vehicles}, EV_Found={ev_found}")
            
        except Exception as e:
            logger.error(f"Error during frame processing: {e}")
            
        return state_space


# یک اسکریپت ساده برای اجرای تست مستقل فایل
if __name__ == "__main__":
    # ایجاد یک فریم نمونه فرضی مشکی برای تست سلامت اجرای کد بدون نیاز به ویدیو
    sample_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # راه‌اندازی با وزن‌های فرضی
    engine = YOLOPerceptionEngine(model_path="yolo11m.pt")
    extracted_state = engine.process_frame(sample_frame)
    print("Test Frame Processing Output State:")
    print(extracted_state)