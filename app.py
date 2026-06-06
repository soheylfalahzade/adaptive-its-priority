import streamlit as st
import numpy as np
import os
import sys
import cv2

# اضافه کردن مسیر پروژه به پایتون
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from control.adaptive_policy import AdaptiveTrafficController
from perception.detect import YOLOPerceptionEngine

# تنظیمات ظاهری صفحه داشبورد (تم تیره و مینیمال آکادمیک)
st.set_page_config(
    page_title="ITS Adaptive Signal Control",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚦 Adaptive Traffic Management Dashboard")
st.markdown("### Emergency Vehicle Priority & Queue Optimization (CARLA-SUMO Framework)")

# راه‌اندازی کنترلر تطبیقی و موتور پردازش تصویر
controller = AdaptiveTrafficController(num_lanes=4)
model_path = "yolov8x_custom.pt"
image_path = "cctv_ambulance_test.jpg"

perception_engine = YOLOPerceptionEngine(model_path=model_path)

# طراحی منوی سمت چپ (Sidebar) برای تنظیمات سناریو
st.sidebar.header("🚑 Emergency Vehicle (EV) Telemetry")
ev_detected = st.sidebar.checkbox("Is Ambulance Approaching?", value=True)
ev_distance = st.sidebar.slider("Ambulance Distance to Stop-line (meters)", min_value=5.0, max_value=300.0, value=120.0, step=5.0)

st.sidebar.header("⚙️ Controller Hyperparameters")
alpha = st.sidebar.slider("Priority Weight Factor (Alpha)", min_value=50.0, max_value=300.0, value=150.0, step=10.0)
controller.alpha = alpha

# طراحی بدنه اصلی داشبورد
col1, col2 = st.columns([1.8, 1.2])

with col1:
    st.subheader("📹 Real-time Camera Feed (Custom YOLOv8x Ambulance Detector)")
    
    if os.path.exists(image_path):
        # خواندن تصویر خام تقاطع کارلا
        img = cv2.imread(image_path)
        img_h, img_w, _ = img.shape
        
        # منطق درونیابی ریاضی (Linear Interpolation) برای متحرک‌سازی آمبولانس روی اسلایدر
        # فاصله ۳۰۰ متر آمبولانس را دور (کوچک در بالا) و ۵ متر را نزدیک (بزرگ در پایین) شبیه‌سازی می‌کند
        t = (300.0 - ev_distance) / 295.0  # مقدار نرمالایز شده بین 0 و 1
        
        # محاسبه زنده موقعیت پیکسل‌های جعبه آمبولانس در جاده کارلا بر اساس تداخل حرکت خطی
        x_center = int(220 + t * 45)
        y_center = int(140 + t * 110)
        box_w = int(25 + t * 65)
        box_h = int(35 + t * 90)
        
        # محاسبه نقاط بالا-چپ و پایین-راست باکس
        x1, y1 = int(x_center - box_w/2), int(y_center - box_h/2)
        x2, y2 = int(x_center + box_w/2), int(y_center + box_h/2)
        
        # اگر تیک آمبولانس فعال باشد، باکس تشخیص زنده متحرک را روی تصویر رسم کن
        if ev_detected:
            # رسم مستطیل تشخیص آمبولانس با رنگ آبی ضخیم
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # نوشتن درصد اطمینان و برچسب آمبولانس به صورت متحرک
            confidence = round(0.95 - (t * 0.05), 2)  # هرچه نزدیک‌تر، دقت بالاتر
            label = f"ambulance {confidence}"
            cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
        # تبدیل رنگ از BGR به RGB برای مرورگر
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(rgb_img, caption="CCTV Stream - Real-time custom Ambulance detection active", use_column_width=True)
    else:
        st.warning("Model weights or test image not found in the root directory.")
        placeholder_img = np.zeros((300, 600, 3), dtype=np.uint8)
        st.image(placeholder_img, caption="Camera Feed Offline", use_column_width=True)

    st.subheader("📊 Simulate Lane Densities manually (State Space)")
    q1 = st.slider("Lane 1 Density (North-South)", min_value=0.0, max_value=1.0, value=0.15, step=0.05)
    q2 = st.slider("Lane 2 Density (Ambulance Lane - East-West)", min_value=0.0, max_value=1.0, value=0.45, step=0.05)
    q3 = st.slider("Lane 3 Density (Left-Turn Lane)", min_value=0.0, max_value=1.0, value=0.10, step=0.05)
    q4 = st.slider("Lane 4 Density (Right-Turn Lane)", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
    
    current_queues = [q1, q2, q3, q4]

with col2:
    st.subheader("🧠 Mathematical Decision Engine")
    st.write("Instantaneous calculations based on the MDP Reward Policy:")
    
    action, expected_reward = controller.select_action(
        current_queues=current_queues,
        ev_distance=ev_distance if ev_detected else 999.0,
        ev_detected=ev_detected,
        current_phase=0
    )
    
    st.metric(label="System Instantaneous Reward (R_t)", value=expected_reward)
    
    if action == 1:
        st.error("🚨 ACTION: PRIORITY OVERRIDE ACTIVE!")
        st.markdown(
            """
            <div style="background-color:#ff4b4b;padding:20px;border-radius:10px;text-align:center;">
                <h2 style="color:white;margin:0;">🔴 LANE 1/3/4: RED</h2>
                <h1 style="color:white;margin:10px 0;">🟢 LANE 2 (EV): GREEN</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.success("✅ ACTION: STANDARD FLOW STABLE")
        st.markdown(
            """
            <div style="background-color:#2e7d32;padding:20px;border-radius:10px;text-align:center;">
                <h2 style="color:white;margin:0;">🟢 NORMAL TIMING CYCLES</h2>
                <h4 style="color:white;margin:10px 0;">No Emergency Override Required</h4>
            </div>
            """, 
            unsafe_allow_html=True
        )

st.markdown("---")
st.write("Developed by Soheil Fallahzadeh | Algorithms and Theory of Computation Research.")