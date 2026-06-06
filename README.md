# Adaptive Traffic Signal Management for Emergency Vehicle Priority

An end-to-end intelligent transportation framework that dynamically prioritizes emergency vehicles (such as ambulances) at complex urban intersections. This project integrates 3D physical environment simulation, microscopic traffic flow, and deep learning perception in a closed-loop control system.

---

## 🚀 Key Performance Indicators (KPIs)
*   **32% Reduction** in emergency vehicle queue wait times under peak traffic flow conditions.
*   **98.4% mAP** on custom-trained YOLOv11m for real-time emergency vehicle identification.
*   **45+ FPS** inference latency on edge-grade computing setups utilizing TensorRT optimization.

---

## 🛠️ System Architecture & Workflow
The framework operates as a closed-loop control system connecting perception, traffic simulation, and adaptive logic:

          [ CARLA Simulator ] (CCTV Feed) ──> [ YOLOv11 Perception ] (Ambulance Detection)
                  ▲                                          │
                  │ (20Hz Sync via TraCI)                    ▼
          [ SUMO Traffic Flow ] <── [ Adaptive Traffic Logic ] (Phase Overrides)

1.  **3D Simulation (CARLA):** Renders high-fidelity urban environments and feeds virtual CCTV camera streams.
2.  **Perception Node (YOLOv11):** Detects and tracks incoming emergency vehicles from the CCTV feed in real-time.
3.  **Control Node (SUMO + TraCI):** Executes a custom queue-aware green-wave logic, overriding standard signal phases to clear the path.

---

## 📁 Repository Structure
*   `perception/`: Custom scripts for YOLOv11 object detection, frame grabbers, and inference optimization.
*   `control/`: Dynamic traffic signal control algorithms and queue management policies.
*   `simulation/`: Bridge scripts to coordinate and sync clock steps between CARLA and SUMO.
*   `test_models.py`: Benchmarking scripts to evaluate different custom-trained YOLO weights.

---

## ⚡ Quick Start (Run Co-Simulation)

### 1. Prerequisites
Ensure you have the following installed and configured on Ubuntu 22.04 / WSL2:
*   CARLA Simulator (0.9.15+)
*   SUMO (Simulation of Urban MObility)
*   PyTorch (with CUDA support)

### 2. Run Pipeline
First, launch the CARLA server, then start the synchronization bridge and control loops:

Activate your virtual environment:
$ source .venv/bin/activate

Execute the main orchestrator:
$ python app.py --carla-port 2000 --sumo-cfg simulation/map.sumocfg --model yolo11m.pt