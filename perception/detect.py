import os
import logging
from typing import Dict, Any
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PerceptionEngine")

# COCO class IDs for generic vehicles (used by the general-purpose detector)
VEHICLE_CLASS_IDS = [2, 5, 7]  # car, bus, truck

# Index of the "ambulance" class in OUR fine-tuned classifier's own label space
# (NOT a COCO class ID -- this classifier only knows two classes: ambulance / noambulance)
AMBULANCE_CLASSIFIER_LABEL = "ambulance"


class YOLOPerceptionEngine:
    """Two-stage perception pipeline for Intelligent Transportation Systems (ITS).

    Stage 1 (Detection): a general-purpose YOLO model (pretrained on COCO) locates
    all vehicles in the frame.
    Stage 2 (Classification): each detected vehicle is cropped and passed through a
    fine-tuned binary classifier (ambulance vs. non-ambulance) to determine whether
    it is an emergency vehicle.

    This two-stage design exists because COCO's default classes do NOT include an
    "ambulance" category -- a fine-tuned classifier is required to make that
    distinction, which is what `classifier_model_path` provides.
    """

    def __init__(
        self,
        detector_model_path: str,
        classifier_model_path: str,
        conf_threshold: float = 0.25,
        classifier_conf_threshold: float = 0.5,
    ):
        """Initializes the two-stage perception engine.

        Args:
            detector_model_path (str): Path to the general-purpose YOLO detection weights (.pt).
            classifier_model_path (str): Path to the fine-tuned ambulance/non-ambulance
                classifier weights (.pt), produced by train_ambulance_classifier.py.
            conf_threshold (float): Minimum confidence for the detection stage.
            classifier_conf_threshold (float): Minimum confidence for the ambulance
                classification stage (applied to the "ambulance" class probability).
        """
        self.conf_threshold = conf_threshold
        self.classifier_conf_threshold = classifier_conf_threshold
        self.detector = None
        self.classifier = None

        if YOLO is None:
            logger.error("Ultralytics library is not installed. Please run 'pip install ultralytics'.")
            return

        self.detector = self._load_model(detector_model_path, "detector")
        self.classifier = self._load_model(classifier_model_path, "classifier")

    @staticmethod
    def _load_model(model_path: str, label: str):
        if not os.path.exists(model_path):
            logger.warning(f"{label.capitalize()} weights not found at '{model_path}'. This stage will be disabled.")
            return None
        try:
            model = YOLO(model_path)
            logger.info(f"Successfully loaded {label} model from '{model_path}'")
            return model
        except Exception as e:
            logger.error(f"Failed to load {label} model: {e}")
            return None

    def _classify_crop(self, crop: np.ndarray) -> bool:
        """Runs the fine-tuned classifier on a single cropped vehicle image.

        Returns:
            bool: True if classified as "ambulance" above the confidence threshold.
        """
        if self.classifier is None or crop.size == 0:
            return False

        try:
            result = self.classifier(crop, verbose=False)[0]
            # Ultralytics classification results expose `.names` (id->label) and `.probs` (per-class confidence)
            top_class_id = int(result.probs.top1)
            top_class_conf = float(result.probs.top1conf)
            top_class_name = result.names[top_class_id]

            return top_class_name == AMBULANCE_CLASSIFIER_LABEL and top_class_conf >= self.classifier_conf_threshold
        except Exception as e:
            logger.error(f"Error during classification of cropped vehicle: {e}")
            return False

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Processes a single video frame through the full detect-then-classify pipeline.

        Returns:
            Dict[str, Any]: Extracted state variables corresponding to the mathematical
            State Space (S_t) used by the adaptive control policy.
        """
        state_space = {
            "vehicle_count": 0,
            "queue_density": 0.0,
            "ev_detected": False,
            "ev_distance_simulated": 999.0,  # placeholder until real distance sensing (e.g. from CARLA depth) is wired in
        }

        if self.detector is None or frame is None:
            return state_space

        try:
            results = self.detector(frame, conf=self.conf_threshold, verbose=False)[0]
            boxes = results.boxes

            total_vehicles = 0
            ev_found = False
            ev_bbox_height = None

            for box in boxes:
                class_id = int(box.cls[0].item())
                if class_id not in VEHICLE_CLASS_IDS:
                    continue

                total_vehicles += 1

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]

                if self._classify_crop(crop):
                    ev_found = True
                    ev_bbox_height = y2 - y1  # used as a rough proxy for distance until real sensing is added

            state_space["vehicle_count"] = total_vehicles
            state_space["ev_detected"] = ev_found
            state_space["queue_density"] = round(min(total_vehicles / 15.0, 1.0), 2)

            if ev_found and ev_bbox_height:
                # NOTE: this is a simple heuristic proxy (larger bbox = closer vehicle),
                # not a calibrated real-world distance measurement. Replace with actual
                # depth/telemetry once integrated with CARLA.
                state_space["ev_distance_simulated"] = round(1000.0 / (ev_bbox_height + 1e-5), 2)

            logger.debug(f"Frame processed: Vehicles={total_vehicles}, EV_Found={ev_found}")

        except Exception as e:
            logger.error(f"Error during frame processing: {e}")

        return state_space


if __name__ == "__main__":
    sample_frame = np.zeros((640, 640, 3), dtype=np.uint8)

    engine = YOLOPerceptionEngine(
        detector_model_path="yolo11m.pt",
        classifier_model_path="perception/weights/ambulance_classifier.pt",
    )
    extracted_state = engine.process_frame(sample_frame)
    print("Test Frame Processing Output State:")
    print(extracted_state)
