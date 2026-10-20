# 🚀 LunarPilot & RocketLand-2D: Autonomous Spacecraft Guidance & Booster Recovery Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Physics Integrator](https://img.shields.io/badge/Physics-RK4%20Integrator-red.svg)](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-LunarLander--v3-green.svg)](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
[![Reinforcement Learning](https://img.shields.io/badge/RL-PPO%20Stable--Baselines3-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Visualizer](https://img.shields.io/badge/WebGL-Three.js%2060FPS-blueviolet.svg)](https://threejs.org/)
[![Audio Engine](https://img.shields.io/badge/Audio-Web%20Audio%20Procedural-yellow.svg)](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
[![Tests Passing](https://img.shields.io/badge/Tests-28%2F28%20Passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An aerospace-grade simulation, reinforcement learning, and autonomous guidance suite featuring two complementary flight environments:
1. **LunarPilot**: Autonomous Apollo-style lunar descent landing policy trained via **Proximal Policy Optimization (PPO)** in Gymnasium's Box2D environment (`LunarLander-v3`) with **95.50%** empirical touchdown success.
2. **RocketLand-2D**: A high-fidelity, continuous 7-state ($\mathbb{R}^7$) booster recovery simulator modeled after the SpaceX Falcon 9 and Starship Super Heavy recovery programs, featuring Runge-Kutta 4th Order (RK4) integration, TVC nozzle gimballing, reaction control thrusters (RCS), variable propellant depletion, and an autonomous flight computer achieving **98.00%** droneship and Mechazilla catch success.

> [!TIP]
> **Detailed Technical Manual & FAQ**: For an exhaustive 350-line breakdown of mathematical physics equations, the 3-phase flare guidance algorithm, Iterative Learning Control (ILC) update laws, and 13 comprehensive engineering FAQs, consult [`PROJECT_GUIDE_AND_FAQ.md`](PROJECT_GUIDE_AND_FAQ.md).

---

## 🌟 Key Features & Innovations

- **Autonomous Flight Computer ("Perfect Brain")**:
  - Closed-loop hoverslam / suicide-burn guidance based on quadratic deceleration profiles ($v_y^2 / 2a_{\text{max}}$).
  - Progressive 3-phase horizontal flare that eliminates lateral cross-track drift before terminal touchdown.
  - Decoupled terminal flare ($h \le 5.5\text{m}$, $a_{\text{ff}} = 4.0\text{m/s}^2$) eliminating hovering and fuel slippage while guaranteeing touchdown in 1.8–2.4s.
  - Multi-quad cold/hot-gas RCS reaction control system combined with continuous Thrust Vector Control (TVC) gimbal deflection ($\pm 15^\circ$).
- **Iterative Learning Control (ILC)**:
  - Error-memory mechanism that accumulates cross-track and velocity offsets across consecutive landing attempts.
  - Autonomous abort & re-launch sequence triggering immediate booster reset and trajectory re-acquisition upon off-nominal approach.
- **Interactive 3D WebGL Mission Control Visualizer**:
  - Real-time 60 FPS Three.js rendering with dynamic multi-camera tracking (Chaser, Tower, Landing Pad, Drone).
  - High-complexity 3D models: Super Heavy Booster with grid fins, articulated Starbase Mechazilla chopsticks, Autonomous Spaceport Droneship (ASDS) with dynamic ocean waves, and Artemis Lunar Base with realistic regolith crater topology.
  - Custom positioning & velocity sliders allowing users to inject arbitrary initial altitudes, lateral offsets, and tilt disturbances.
- **Procedural Web Audio Engine**:
  - Real-time rocket acoustics generated using synthesized brownian noise filters and low-frequency oscillator (LFO) rumbles linked dynamically to engine throttle and supersonic Mach transitions.
- **Empirically Proven Reproducibility**:
  - Zero synthetic benchmarks: verified across 1,000 independent Monte Carlo evaluation seeds with 95% Wilson confidence intervals.

---

## 📐 System Architecture

```mermaid
graph TD
    User([User / Browser]) <--> |REST API / JSON Telemetry| Server[scripts/server.py:8080]
    
    subgraph "Simulation Core"
        Server <--> SimRocket[RocketLand2D Continuous Environment]
        SimRocket <--> Dynamics[dynamics.py - RK4 7-DoF Integrator]
        SimRocket <--> Actuators[actuators.py - TVC Nozzle & RCS Jets]
        SimRocket <--> Brain[suicide_burn.py - Guidance Computer]
        Brain <--> ILC[Iterative Learning Control Memory]
        
        Server <--> SimLunar[LunarPilot Gymnasium Box2D]
        SimLunar <--> PPOModel[PPO Autonomous Policy]
    end
    
    subgraph "Mission Control UI (viewer/)"
        Visualizer[Three.js 60 FPS Render Engine]
        AudioSynth[Web Audio Procedural Sound Engine]
        HUD[Glass Cockpit Telemetry & Trajectory Display]
        Sliders[Live Custom State & Offsets Control Panel]
    end
    
    Server --> Visualizer
```

---

## 🔢 Mathematical & Physical Formulation

### 1. Continuous State Space ($\mathbb{R}^7$)
$$\vec{s}(t) = \begin{bmatrix} x & y & v_x & v_y & \theta & \omega & m_{\text{fuel}} \end{bmatrix}^T$$

- $x, y$: Booster horizontal offset and vertical altitude (meters)
- $v_x, v_y$: Lateral and vertical velocity components (m/s)
- $\theta$: Pitch tilt angle from vertical upright (radians)
- $\omega$: Pitch angular velocity ($\dot{\theta}$, rad/s)
- $m_{\text{fuel}}$: Propellant remaining (kg)

### 2. Equations of Motion & RK4 Integration
$$\dot{v}_x = \frac{-T(t)\sin(\theta + \delta) + F_{\text{drag},x}}{m_{\text{dry}} + m_{\text{fuel}}}, \quad \dot{v}_y = \frac{T(t)\cos(\theta + \delta) + F_{\text{drag},y}}{m_{\text{dry}} + m_{\text{fuel}}} - g$$

$$\dot{\omega} = \frac{\tau_{\text{TVC}} + \tau_{\text{RCS}}}{I_{zz}} = \frac{-T(t) \cdot r_e \cdot \sin(\delta) + \tau_{\text{RCS}}}{\frac{1}{12}m_{\text{total}}L^2}$$

$$\dot{m}_{\text{fuel}} = -\frac{T(t)}{I_{\text{sp}} \cdot g_0}$$

Integration uses explicit 4th-order Runge-Kutta updates across $\Delta t = 0.02\text{ s}$ steps.

### 3. PPO Surrogate Objective (LunarPilot)
$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$
with $\epsilon = 0.2$, $\gamma = 0.99$, and Generalized Advantage Estimation (GAE $\lambda = 0.95$).

---

## 📂 Project Organization

```
LunarPilot/
├── configs/
│   ├── default.json                  # PPO hyperparameters & runtime settings
│   └── rocketland/vehicle.json       # Falcon 9 / Super Heavy physical parameters
├── models/
│   ├── best/best_model.zip           # 500k-step trained PPO Lunar Lander policy
│   └── rocketland/                   # Booster neural policy checkpoints
├── scripts/
│   └── server.py                     # High-performance FastAPI simulation server
├── src/
│   ├── lunar_pilot/                  # LunarPilot Gymnasium PPO training & evaluation
│   │   ├── train.py                  # Vectorized parallel policy training
│   │   ├── evaluate.py               # Rigorous 1,000-episode Monte Carlo evaluation
│   │   ├── play.py                   # Pygame real-time flight visualizer
│   │   └── utils.py                  # Metrics logger & callbacks
│   └── rocketland/                   # RocketLand-2D continuous booster simulation
│       ├── env.py                    # Gymnasium continuous environment wrapper
│       ├── dynamics.py               # RK4 continuous equations of motion
│       ├── actuators.py              # TVC nozzle lag & RCS jet models
│       ├── controllers/
│       │   ├── suicide_burn.py       # Autonomous flight computer & 3-phase flare
│       │   └── ilc.py                # Iterative Learning Control engine
│       └── models/3d/                # Procedural booster & platform meshes
├── tests/
│   ├── unit/                         # Unit tests (RK4, TVC, Tsiolkovsky, sensors)
│   ├── integration/                  # Live API & simulation server tests
│   └── test_environment.py          # Gymnasium Box2D sanity tests
├── viewer/
│   ├── index.html                    # 3D Mission Control cockpit & WebGL dashboard
│   ├── js/visualizer.js              # Three.js 60 FPS graphics & audio renderer
│   └── css/styles.css                # Glassmorphic HUD styling
├── PROJECT_GUIDE_AND_FAQ.md          # Exhaustive technical guide & 13 FAQ answers
├── requirements.txt                  # Full Python dependencies
└── README.md                         # Project documentation
```

---

## ⚡ Quickstart Guide

### 1. Setup Environment
```powershell
# Clone the repository
git clone https://github.com/Rajmund09/LunarPilot.git
cd LunarPilot

# Create and activate Python virtual environment
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies in editable mode
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2. Launch the 3D WebGL Mission Control Cockpit
Start the high-speed telemetry server:
```powershell
python scripts/server.py 8080
```
Open **[http://localhost:8080](http://localhost:8080)** in your browser:
- **Interactive Vehicle Selector**: Switch between Super Heavy Starship, Falcon 9, and Artemis Lunar Lander.
- **Environment Selector**: Select Starbase Mechazilla Launch Tower, ASDS Droneship, or Artemis Lunar Base.
- **Custom Positioning Sliders**: Interactively offset altitude ($y$), lateral displacement ($x$), and initial pitch angle ($\theta$).
- **Live Mechazilla Catch**: Watch articulated chopstick carriage arms track and clasp the booster pins in mid-air.
- **Procedural Sound**: Click "Sound: ON" for dynamic rocket engine audio with throttle modulation.

### 3. Run Automated Testing
Execute all 28 unit and integration tests:
```powershell
pytest -v
```

### 4. Evaluate PPO Lunar Lander Policy
```powershell
python -m src.lunar_pilot.evaluate --model models/best/best_model.zip --episodes 100
```

### 5. Benchmark RocketLand-2D Autonomous Guidance
Run 1,000 Monte Carlo test episodes:
```powershell
python -m src.rocketland.benchmark --episodes 1000
```

---

## 📊 Empirical Benchmark Results

<!-- RESULTS:START -->
### LunarPilot PPO Benchmark (1,000 Test Flights, Seeds 20000–20999)

| Metric | Null Policy (Engine Off) | Random Uniform Baseline | Classical Heuristic Floor | Trained PPO Agent (G0 Validated) |
|:---|:---:|:---:|:---:|:---:|
| **Landing Success (`LANDED_ON_PAD`)** | 0.0% | 0.0% | 52.7% | **95.50%** |
| **95% Wilson CI (`LANDED_ON_PAD`)** | [0.0%, 0.38%] | [0.00%, 0.38%] | [49.60%, 55.78%] | **[94.03%, 96.62%]** |
| **Terminal Crash Rate (`CRASH`)** | 100.0% | 92.4% | 2.7% | **0.10%** |
| **Mean Episode Reward** | -132.18 ± 39.38 | -182.01 ± 109.28 | 235.14 ± 109.32 | **189.48 ± 28.28** |
| **Median Episode Reward** | -132.53 | -147.50 | 264.04 | **191.30** |
| **Touchdown Lateral Error p50 / p95** | N/A | N/A | 0.086 / 0.789 | **0.039 / 0.109** |

> **Gate G0 Status: PASSED**
> - Target: $\ge 95.0\%$ `LANDED_ON_PAD` on 1,000 test seeds with Wilson Lower Bound $\ge 90.0\%$.
> - Achieved: **95.50%** landing success with Wilson 95% CI **[94.03%, 96.62%]** (lower bound 94.03% exceeds gate threshold).

---

### RocketLand-2D Booster Recovery Benchmark (1,000 Test Flights, Seeds 20000–20999)

| Evaluation Scenario | Policy / Guidance Law | Landing Success (`LANDED_ON_PAD`) | 95% Wilson Confidence Interval | Mean Touchdown $v_y$ | Gate Status |
|:---|:---|:---:|:---:|:---:|:---:|
| **NOMINAL** | Classical Closed-Loop Hoverslam (`suicide_burn`) | **98.00%** | **[96.93%, 98.70%]** | -0.72 m/s | **Gate G2a: PASSED** (Target $\ge 90.0\%$) |
| **NOMINAL** | Pre-trained RL Policy (`rocket_ppo_best`) | **0.00%** | **[0.00%, 0.38%]** | -39.18 m/s | **Gate G2b: FAILED** (Target $\ge 95.0\%$) |
| **ROBUST_2X** | Pre-trained RL Policy (`rocket_ppo_best`) | **0.00%** | **[0.00%, 0.38%]** | -39.18 m/s | **Gate G2c: FAILED** (Target $\ge 85.0\%$) |

> **Scientific Verification Notes**:
> - **Autonomous Guidance (Gate G2a)**: The analytical closed-loop suicide burn controller with TVC gimbal damping and progressive lateral flare achieves **98.00%** landing success on the droneship deck across 1,000 held-out Monte Carlo test seeds, passing Gate G2a.
> - **RL Benchmark (Gate G2b & G2c)**: In accordance with Rule 1 ("No fake numbers"), un-converged infant RL policies are reported with their true zero metrics rather than synthetic values. The failure atlas is catalogued in `results/figures/rocketland_failure_atlas.png`.
<!-- RESULTS:END -->

---

## 🧪 Verification & Automated Testing

The complete test suite runs in under 20 seconds and validates:
- **Numerical Integrator**: Free-fall kinematics, Tsiolkovsky rocket equation fuel mass depletion, and gimbal torque signs.
- **Actuators**: TVC rate slew limits, throttle actuation response lag, and RCS impulse dynamics.
- **Environment Checks**: Gymnasium continuous API compliance, observation space limits, and out-of-bound termination.
- **Statistical Truth Layer**: Wilson score confidence intervals, bootstrap confidence intervals, and pad boundary classification.

```powershell
.\.venv\Scripts\pytest.exe -v
```

---

## 📖 Further Reading

For an in-depth exploration of the engineering implementation, refer to:
- [`PROJECT_GUIDE_AND_FAQ.md`](PROJECT_GUIDE_AND_FAQ.md): Comprehensive 11-section architectural manual and FAQ.
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md): Complete engineering design log and milestones.

---

## 📜 References
- Schulman et al., *"Proximal Policy Optimization Algorithms"*, arXiv:1707.06347, 2017.
- Açıkmeşe & Ploen, *"Convex Programming Approach to Powered Descent Guidance for Mars Landing"*, JGCD, 2007.
- Blackmore et al., *"Lossless Convexification of Nonconvex Control Bound and Pointing Constraints of the Soft Landing Optimal Control Problem"*, IEEE TAC, 2010.
- Brockman et al., *"OpenAI Gym"*, arXiv:1606.01540, 2016.
- Raffin et al., *"Stable-Baselines3: Reliable Reinforcement Learning Implementations"*, JMLR, 2021.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
