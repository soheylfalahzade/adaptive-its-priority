# Adaptive Traffic Management for Emergency Vehicle Priority

This repository contains the official implementation of an Intelligent Transportation System (ITS) designed to prioritize emergency vehicles (EVs) at urban intersections. The framework bridges high-angle visual perception with microscopic traffic physics by coupling **CARLA** and **SUMO** simulators.

---

## 📌 System Architecture Overview

The system operates in a closed-loop co-simulation environment consisting of three core modules:

* **Perception Module (CARLA):** High-angle surveillance camera stream processing using an optimized YOLOv11m network.
* **Co-Simulation Bridge (TraCI):** Dual-simulator synchronization translating visual coordinates into spatial SUMO node states in real-time.
* **Control Module (Adaptive Decision Engine):** A mathematical control policy optimizing traffic light phase transitions based on vehicle queues and EV proximity.

### 🔄 Data Flow

```text
[CARLA CCTV Stream] 
       │
       ▼ (Image Coordinates)
[YOLOv11m Object Detection]
       │
       ▼ (Queue State & EV Distance)
[TraCI Python Bridge]
       │
       ▼ (Phase Configuration)
[SUMO Traffic Physics Engine]
🛠️ Project Structure
adaptive-its-priority/
├── assets/                  # Architecture diagrams and performance charts
├── perception/              # Object detection & auto-labeling pipeline (YOLOv11m)
│   ├── dataset/             # Scripts for synthetic data generation in CARLA
│   └── detect.py            # Real-time inference wrapper
├── simulation/              # Dual-Simulation Environments
│   ├── carla_env/           # Client scripts, camera sensors setup
│   ├── sumo_network/        # Network definitions, demand generation
│   └── traci_bridge.py      # Microscopic Co-simulation sync bridge
├── control/                 # Decision & Optimization Models
│   └── adaptive_policy.py   # Code implementation of the mathematical Reward Policy
└── README.md                # Technical Documentation
---

## 🧠 Mathematical Formulation of Control Policy

The intersection control logic is formulated as an **Adaptive Queue-Length Minimization Problem** with priority constraints. We model the junction state as a Markov Decision Process (MDP) defined by the tuple `(S, A, P, R, gamma)`.

### 1. State Space (`S_t`)

At any time step `t`, the system state `S_t` is represented by the following vector:

```math
S_t = \{ Q_i, D_{ev}, V_{ev}, \Phi_t \}
```

Where:
* **`Q_i`** is the normalized queue length (vehicle density) of lane `i` for `N` incoming lanes.
* **`D_ev`** is the longitudinal distance of the approaching Emergency Vehicle (EV) to the intersection stop-line. If no EV is present, `D_ev = D_max`.
* **`V_ev`** is the current velocity of the approaching Emergency Vehicle.
* **`Phi_t`** is the active traffic signal phase index out of `K` possible phase configurations.

### 2. Action Space (`A_t`)

The control action `A_t` at step `t` dictates the signal state change:

```math
A_t = a \in \{0, 1\}
```

Where:
* **`a = 0`** means maintain the current green phase configuration (no transition).
* **`a = 1`** means initiate a transition to the optimal green phase to clear the path for the EV or minimize general queues.

### 3. Reward Function (`R_t`)

To guarantee safety, reduce emergency response delays, and maintain secondary traffic flow, the objective function (Reward `R_t`) penalized queues and prioritizing the EV is defined as:

```math
R_t = - \left( \sum_{i=1}^{N} w_i Q_i^2 + \alpha \cdot \frac{1}{D_{ev} + \epsilon} \cdot \mathbb{I}(EV) \right)
```

Where:
* **`w_i`** is the dynamic waiting-time weight assigned to lane `i` to prevent starvation of non-priority traffic.
* **`alpha`** is the prioritization weight factor (`alpha >> w_i`) triggered when an EV is detected.
* **`epsilon`** is a small positive smoothing constant to prevent division-by-zero when the EV reaches the stop-line.
* **`I(EV) in {0, 1}`** is an indicator function that outputs `1` if an emergency vehicle is detected within the perception range, and `0` otherwise.
---

## 📊 Evaluation Metrics

The framework is evaluated using standardized intelligent transportation metrics to measure both throughput and emergency response efficiency:

### 1. Emergency Vehicle Delay Reduction (EVDR)

This metric quantifies the percentage of time saved for the emergency vehicle compared to standard fixed-time or basic inductive-loop control systems:

```math
\text{EVDR} = \frac{\bar{T}_{\text{baseline}}^{\text{ev}} - \bar{T}_{\text{adaptive}}^{\text{ev}}}{\bar{T}_{\text{baseline}}^{\text{ev}}} \times 100\%
```

Where:
* **`T_baseline`** is the average travel time of the EV under the baseline traffic control policy.
* **`T_adaptive`** is the average travel time of the EV under our proposed adaptive model.

### 2. Average Intersection Throughput (C)

Measured in vehicles per hour (veh/h), calculating the total number of cleared vehicles across all lanes:

```math
C = \frac{\sum_{i=1}^{N} V_i}{\Delta t}
```

### 3. Queue Length Variance

Assessing the stability and fairness of the system to ensure non-priority lanes do not suffer from extreme starvation:

```math
\sigma^2_q = \frac{1}{N} \sum_{i=1}^{N} (Q_i - \bar{Q})^2
```