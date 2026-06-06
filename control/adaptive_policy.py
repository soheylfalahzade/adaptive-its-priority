import logging
from typing import List, Tuple, Dict

# تنظیمات سیستم لاگ برای مانیتورینگ تصمیم‌گیری‌های هوشمند چراغ راهنمایی
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ControlEngine")


class AdaptiveTrafficController:
    """Adaptive Traffic Signal Controller implementing the mathematical queue-minimization policy.
    
    This class directly translates the MDP reward function (R_t) and State Space (S_t) 
    defined in the README into executable Python control logic for real-time intersection phase transitions.
    """
    
    def __init__(self, num_lanes: int, alpha: float = 150.0, epsilon: float = 0.1):
        """Initializes the adaptive decision engine.
        
        Args:
            num_lanes (int): Total number of lanes monitored at the intersection.
            alpha (float): Hyperparameter (prioritization weight) for the Emergency Vehicle (EV).
            epsilon (float): A small smoothing constant to prevent division-by-zero.
        """
        self.num_lanes = num_lanes
        self.alpha = alpha
        self.epsilon = epsilon
        
        # تخصیص وزن پیش‌فرض به لاین‌ها (w_i در فرمول ریاضی) برای حفظ عدالت ترافیکی
        self.lane_weights = [1.5 for _ in range(num_lanes)]
        
    def calculate_reward(self, queue_states: List[float], ev_distance: float, ev_detected: bool) -> float:
        """Computes the instantaneous control Reward (R_t) as defined in the system LaTeX specs.
        
        Formula implemented:
            R_t = - ( Sum( w_i * Q_i^2 ) + alpha * (1 / (D_ev + epsilon)) * I(EV) )
            
        Args:
            queue_states (List[float]): Normalized queue densities Q_i for each lane (values between 0 and 1).
            ev_distance (float): Distance of the approaching EV to the stop-line.
            ev_detected (bool): Boolean indicator I(EV) showing if an EV is present.
            
        Returns:
            float: Calculated reward value (negative penalty value, where closer to 0 is better).
        """
        # بخش اول فرمول: جریمه سنگین برای طول صف‌های بزرگ (تلاش برای صفر کردن صف‌ها)
        queue_penalty = 0.0
        for i, q_density in enumerate(queue_states):
            if i < len(self.lane_weights):
                queue_penalty += self.lane_weights[i] * (q_density ** 2)
                
        # بخش دوم فرمول: جریمه آمبولانس (با نزدیک شدن آمبولانس به تقاطع، این جریمه به شدت بزرگتر می‌شود)
        ev_penalty = 0.0
        if ev_detected and ev_distance < 300.0:  # اگر آمبولانس در محدوده ۳۰۰ متری باشد
            ev_penalty = self.alpha * (1.0 / (ev_distance + self.epsilon))
            
        # مجموع تابع پاداش (منفی بودن آن برای مینیمم‌سازی جریمه‌هاست)
        total_reward = -(queue_penalty + ev_penalty)
        return round(total_reward, 4)

    def select_action(self, current_queues: List[float], ev_distance: float, ev_detected: bool, 
                      current_phase: int) -> Tuple[int, float]:
        """Decides the optimal action (A_t) to take: maintain phase (a=0) or trigger transition (a=1).
        
        Args:
            current_queues (List[float]): Current list of lane queue densities.
            ev_distance (float): Measured distance of the EV.
            ev_detected (bool): Presence indicator of the EV.
            current_phase (int): Active traffic signal phase index.
            
        Returns:
            Tuple[int, float]: (action, expected_reward) where action is 0 (keep) or 1 (transition).
        """
        # محاسبه پاداش حالت فعلی (اگر چراغ را تغییر ندهیم)
        current_reward = self.calculate_reward(current_queues, ev_distance, ev_detected)
        
        action = 0  # پیش‌فرض: حفظ فاز سبز فعلی
        expected_reward = current_reward
        
        # منطق تصمیم‌گیری تطبیقی:
        # اگر آمبولانس تشخیص داده شود و فاصله آن نزدیک باشد، باید سریعاً لاینش سبز شود (اکشن تغییر فاز)
        if ev_detected and ev_distance < 150.0:
            logger.info(f"EV Detected at {ev_distance}m! Initiating priority signal override (Action = 1).")
            action = 1
            # پاداش بعد از سبز شدن لاین آمبولانس (فاصله فرضی صفر یا بسیار دور می‌شود چون صف مسیرش باز می‌شود)
            expected_reward = self.calculate_reward(current_queues, ev_distance=999.0, ev_detected=False)
        else:
            # اگر آمبولانسی نبود، تصمیم‌گیری بر اساس طول صف‌ها انجام می‌شود
            max_queue = max(current_queues) if current_queues else 0
            if max_queue > 0.8:  # اگر صف یکی از لاین‌ها از ۸۰ درصد ظرفیت گذشت
                logger.info(f"Queue threshold exceeded ({max_queue}). Triggering adaptive clearing phase (Action = 1).")
                action = 1
                # شبیه‌سازی کاهش صف پس از تغییر فاز چراغ
                simulated_queues = [q * 0.3 if q == max_queue else q for q in current_queues]
                expected_reward = self.calculate_reward(simulated_queues, ev_distance, ev_detected)
            else:
                logger.debug("Traffic flow stable. Maintaining active phase config.")
                action = 0
                
        return action, expected_reward


# اسکریپت ساده برای تست محلی سلامت توابع ریاضی بالا
if __name__ == "__main__":
    controller = AdaptiveTrafficController(num_lanes=4)
    
    # سناریو ۱: ترافیک عادی و روان، بدون آمبولانس
    queues_normal = [0.2, 0.3, 0.1, 0.25]
    action, reward = controller.select_action(queues_normal, ev_distance=999.0, ev_detected=False, current_phase=1)
    print(f"Scenario 1 (Normal) -> Action: {action} (Keep Phase), Expected Reward: {reward}")
    
    # سناریو ۲: حضور آمبولانس در فاصله ۲۰ متری تقاطع
    queues_heavy = [0.2, 0.4, 0.1, 0.3]
    action_ev, reward_ev = controller.select_action(queues_heavy, ev_distance=20.0, ev_detected=True, current_phase=1)
    print(f"Scenario 2 (Ambulance Close) -> Action: {action_ev} (Trigger Green Light!), Expected Reward: {reward_ev}")