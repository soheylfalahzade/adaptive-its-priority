import os
import sys
import time
import argparse
import logging
from typing import Optional

# اضافه کردن مسیر پوشه‌های پروژه به پایتون جهت لود کردن کلاس‌های قبلی
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.detect import YOLOPerceptionEngine
from control.adaptive_policy import AdaptiveTrafficController

# تنظیمات لاگ سیستم برای نمایش خروجی شبیه‌سازی دمو
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CoSimulationBridge")

# تلاش برای لود کردن کتابخانه‌های شبیه‌سازها (در صورت نصب بودن مسیرهای آن‌ها در سیستم)
try:
    import traci
except ImportError:
    traci = None

try:
    import carla
except ImportError:
    carla = None


class CoSimulationBridge:
    """Manages the real-time sync loop between CARLA rendering and SUMO physics.
    
    Acts as the main orchestrator applying YOLO detections to adaptive traffic policies.
    """
    
    def __init__(self, sumo_cfg_path: Optional[str] = None, carla_host: str = "localhost", carla_port: int = 2000):
        self.sumo_cfg_path = sumo_cfg_path
        self.carla_host = carla_host
        self.carla_port = carla_port
        
        # مقداردهی به موتورهای پردازش و کنترل که در مراحل قبلی ساختیم
        self.perception_engine = YOLOPerceptionEngine(model_path="yolo11m.pt")
        self.traffic_controller = AdaptiveTrafficController(num_lanes=4)
        
    def run_simulation_demo(self, steps: int = 20):
        """Runs a real-time console demo simulating an ambulance approaching an intersection.
        
        This allows testing the complete pipeline (Perception -> Control -> Action)
        instantly without booting up heavy simulator servers.
        """
        logger.info("=" * 60)
        logger.info("STARTING CO-SIMULATION TELEMETRY DEMO (MOCK MODE)")
        logger.info("=" * 60)
        
        # تعریف سناریو: آمبولانس از فاصله ۲۶۰ متری لاین ۲ شروع به حرکت می‌کند
        ev_distance = 260.0
        ev_detected = True
        current_phase = 0  # فاز فعلی چراغ (۰: همه لاین‌ها عادی، ۱: اولویت آمبولانس)
        
        # طول صف‌های فرضی لاین‌ها
        lane_queues = [0.1, 0.4, 0.15, 0.2]
        
        for step in range(1, steps + 1):
            time.sleep(0.5)  # شبیه‌سازی فاصله زمانی نیم ثانیه‌ای بین فریم‌ها
            
            # ۱. پیشروی آمبولانس در شبیه‌ساز (کاهش فاصله با تقاطع)
            if ev_detected:
                ev_distance -= 15.0  # آمبولانس با سرعت ۳۰ متر بر ثانیه نزدیک می‌شود
                if ev_distance <= 5.0:
                    logger.info("[Simulator] Ambulance cleared the intersection stop-line.")
                    ev_detected = False
                    ev_distance = 999.0
                    current_phase = 0  # بازگشت چراغ به حالت عادی
                    lane_queues = [0.1, 0.1, 0.1, 0.1]
            
            # ۲. شبیه‌سازی رشد صف در سایر لاین‌ها در طول زمان
            for i in range(len(lane_queues)):
                if not (ev_detected and i == 1):  # لاین‌های دیگر ترافیک جمع می‌شود
                    lane_queues[i] = round(min(lane_queues[i] + 0.04, 1.0), 2)
                    
            # ۳. پردازش تصویر (بخش بینایی ماشین): شبیه‌سازی تصویر دوربین CCTV
            # در شبیه‌ساز واقعی، فریم تصویر از CARLA گرفته شده و به تابع زیر فرستاده می‌شود:
            # frame_data = get_carla_camera_frame()
            # outputs = self.perception_engine.process_frame(frame_data)
            
            logger.info(f"\n--- TIME STEP {step} ---")
            logger.info(f"[Perception Sensors] Monitoring Lane Queues: {lane_queues}")
            if ev_detected:
                logger.info(f"[Perception Sensors] WARNING: Emergency Vehicle Detected at {round(ev_distance, 1)}m!")
                
            # ۴. موتور تصمیم‌گیری ریاضی (بخش کنترل تطبیقی)
            action, expected_reward = self.traffic_controller.select_action(
                current_queues=lane_queues,
                ev_distance=ev_distance,
                ev_detected=ev_detected,
                current_phase=current_phase
            )
            
            # ۵. اعمال تصمیم به شبیه‌ساز ترافیک واقعی SUMO
            if action == 1:
                current_phase = 1  # تغییر چراغ به فاز اولویت
                logger.info(f"[Control Action] Action triggered: Changing Traffic Signal Phase to prioritze Lane 1.")
                # در سناریوی واقعی سومو این دستور اجرا می‌شود:
                # if traci: traci.trafficlight.setPhase("intersection_id", current_phase)
                if ev_detected:
                    lane_queues[1] = max(0.0, round(lane_queues[1] - 0.25, 2))  # باز شدن صف لاین آمبولانس
            else:
                logger.info(f"[Control Action] Action: Maintain Current Phase Configuration.")
                
            logger.info(f"[Reward Optimization] Instantaneous System Reward R_t: {expected_reward}")
            
        logger.info("=" * 60)
        logger.info("DEMO SIMULATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARLA-SUMO Co-Simulation Bridge")
    parser.add_argument("--demo", action="store_true", help="Run the telemetry console demo")
    args = parser.parse_args()
    
    # اگر دستور اجرای دمو داده شد یا شبیه‌سازها به مسیر پایتون متصل نبودند
    if args.demo or (traci is None) or (carla is None):
        if not args.demo:
            logger.warning("CARLA or SUMO libraries not found in Python path. Running in Telemetry Demo mode...")
        bridge = CoSimulationBridge()
        bridge.run_simulation_demo(steps=15)
    else:
        logger.info("CARLA and SUMO libraries detected. Ready for actual simulator coupling on localhost:2000.")
        # در فازهای بعدی اسکریپت اتصال واقعی را در این بخش پیاده‌سازی می‌کنیم.
        bridge = CoSimulationBridge(sumo_cfg_path="sumo_network/intersection.sumocfg")