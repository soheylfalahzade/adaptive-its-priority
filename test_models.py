import os
from ultralytics import YOLO
import cv2

# لیست تمام مدل‌های پیدا شده روی سیستم تو برای مقایسه
models = [
    "yolo11m.pt",
    "/home/soheil79/Portfolio/01_Smart_Traffic_Vision/yolov8n.pt",
    "/home/soheil79/Portfolio/Traffic_Project/runs/detect/train/weights/best.pt",
    "/home/soheil79/Portfolio/Traffic_Project/runs/detect/runs/train/robust_ambulance_model5/weights/best.pt",
    "/home/soheil79/Portfolio/Traffic_Project/runs/detect/runs/train/robust_ambulance_model4/weights/best.pt",
    "/home/soheil79/Portfolio/Traffic_Project/runs/detect/runs/detect/robust_ambulance_model-3/weights/best.pt",
    "/home/soheil79/Portfolio/Traffic_Project/runs/detect/runs/detect/robust_ambulance_model-4/weights/best.pt",
    "/home/soheil79/Portfolio/Traffic_Project/weights/yolov8x.pt",
    "/home/soheil79/Portfolio/Traffic_Project/weights/ambulance_sim_v1.pt",
    "/home/soheil79/Portfolio/Traffic_Project/weights/best_custom_yolo.pt"
]

image_path = "cctv_ambulance_test.jpg"

if not os.path.exists(image_path):
    print(f"Error: {image_path} not found in current folder!")
    exit()

img = cv2.imread(image_path)

print("\n" + "="*95)
print(f"{'Model Path (Location on Disk)':<65} | {'Detections in Test Image':<30}")
print("="*95)

# تست خودکار تک‌تک مدل‌ها روی CPU
for path in models:
    if not os.path.exists(path):
        continue
    try:
        model = YOLO(path)
        # اجرای استنتاج روی CPU برای تست سریع
        results = model(img, conf=0.25, verbose=False, device="cpu")[0]
        detections = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            name = model.names[cls_id]
            conf = float(box.conf[0].item())
            detections.append(f"{name} ({conf:.2f})")
        
        det_str = ", ".join(detections) if detections else "No objects detected"
        short_path = path.replace("/home/soheil79/Portfolio/", ".../")
        print(f"{short_path:<65} | {det_str:<30}")
        
    except Exception as e:
        short_path = path.replace("/home/soheil79/Portfolio/", ".../")
        print(f"{short_path:<65} | Error during test: {e}")
        
print("="*95 + "\n")