# 🚀 LunarPilot: Autonomous Lunar Landing via PPO

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-LunarLander--v3-green.svg)](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black-black.svg)](https://github.com/psf/black)

A production-grade Reinforcement Learning (RL) engineering project that trains, benchmarks, and visualizes an autonomous lunar landing policy using **Proximal Policy Optimization (PPO)** in Gymnasium's Box2D `LunarLander-v3` environment.

---

## 🌟 Key Highlights

- **Reproducible Pipeline**: Decoupled hyperparameters in `configs/default.json` with multi-seed determinism across disjoint train, validation, and test seed ranges.
- **Empirical Baselines**: Rigorous 1,000-episode benchmarking against Null Policy, Random Uniform Policy, and Classical Heuristic Policy.
- **Continuous Telemetry**: Automated logging of policy entropy, value loss, KL divergence, landing outcome rates, and snapshot replay trajectories.
- **Physical Outcome Classification**: Ground truth event classification (`LANDED_ON_PAD`, `LANDED_OFF_PAD`, `CRASH`, `TIMEOUT`, `OUT_OF_BOUNDS`) directly from simulator physics.
- **Publication-Ready Figures**: Automated generation of comparative outcome distributions and learning curves.
- **Interactive Demonstrator & API**: Real-time telemetry streaming and web dashboard.
- **Automated Test Suite**: Unit and integration testing via `pytest` with 100% CI compliance.

---

## 📐 Architecture & Mathematical Formulation

The agent solves a continuous state, discrete action Markov Decision Process (MDP):

### 1. State Space ($\mathbb{R}^8$)
| Dimension | Variable | Description |
|:---:|:---|:---|
| `0` | $x$ | Horizontal coordinate relative to landing site $(x=0)$ |
| `1` | $y$ | Altitude relative to lunar surface $(y=0)$ |
| `2` | $v_x$ | Lateral horizontal velocity |
| `3` | $v_y$ | Vertical velocity (descent speed) |
| `4` | $\theta$ | Lander tilt angle in radians ($0 = \text{upright}$) |
| `5` | $\omega$ | Angular rotational velocity |
| `6` | $c_L$ | Left leg ground contact boolean sensor |
| `7` | $c_R$ | Right leg ground contact boolean sensor |

### 2. Action Space ($\mathcal{A} \in \{0, 1, 2, 3\}$)
- `0`: Idle / Coasting (No thrust)
- `1`: Fire Left Engine (Induces counter-clockwise torque)
- `2`: Fire Main Thruster (Provides positive vertical acceleration)
- `3`: Fire Right Engine (Induces clockwise torque)

### 3. PPO Clipped Surrogate Objective
$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$
where $\epsilon = 0.2$, with Generalized Advantage Estimation (GAE, $\gamma = 0.99, \lambda = 0.95$).

---

## 📂 Project Structure

```
LunarPilot/
├── configs/
│   └── default.json             # Hyperparameters & runtime configs
├── models/
│   ├── best/                    # Best checkpoint by validation reward
│   └── checkpoints/             # Periodic snapshot weights
├── logs/
│   ├── evaluations/             # Structured evaluation histories
│   └── tensorboard/             # TensorBoard event telemetry
├── results/
│   ├── figures/                 # Learning curves & outcome plots
│   └── metrics/                 # Benchmark metric JSONs
├── src/
│   └── lunar_pilot/
│       ├── __init__.py          # Package initialization
│       ├── train.py             # Vectorized training pipeline
│       ├── evaluate.py          # 100-episode scientific evaluation
│       ├── play.py              # Live playback & telemetry visualizer
│       ├── plot_results.py      # Publication chart generation
│       └── utils.py             # Custom metrics callbacks & helpers
├── tests/
│   ├── test_environment.py     # Environment dynamics and space tests
│   └── test_pipeline.py        # Config, model forward-pass & save tests
├── baseline.py                  # Random policy benchmark script
├── verify_environment.py        # Environment sanity check script
├── requirements.txt             # Project dependencies
├── pyproject.toml               # Package definition
├── IMPLEMENTATION_PLAN.md       # Comprehensive engineering plan
└── README.md                    # Project documentation
```

---

## ⚡ Quickstart Guide

### 1. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/Rajmund09/LunarPilot.git
cd LunarPilot

# Create and activate virtual environment (Python 3.10 recommended)
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2. Verify Simulator & Box2D Bindings
```powershell
python verify_environment.py
```

### 3. Establish Random Baseline Benchmark
Record 100 episodes of random exploration to set an empirical baseline:
```powershell
python baseline.py
```
*Results saved to `results/metrics/baseline_metrics.json`.*

### 4. Train the PPO Policy
Launch parallel training with TensorBoard logging and periodic evaluation:
```powershell
python train.py --config configs/default.json
```

In a separate terminal, launch TensorBoard:
```powershell
tensorboard --logdir logs/tensorboard
```
*Open [http://localhost:6006](http://localhost:6006) to monitor reward, policy entropy, value loss, and success rate live.*

### 5. Evaluate the Trained Model
Benchmark the trained policy across 100 test episodes with deterministic evaluation:
```powershell
python evaluate.py --model models/best/best_model.zip --episodes 100
```
*Outputs an empirical side-by-side comparison against the random baseline.*

### 6. Generate Figures & Learning Curves
```powershell
python plot_results.py
```
*High-resolution PNGs are saved to `results/figures/`.*

### 7. Watch Live Agent Demonstration
Run real-time inference with human rendering and live telemetry:
```powershell
python play.py --model models/best/best_model.zip --episodes 5
```

---

## 📊 Empirical Benchmark Results

<!-- RESULTS:START -->
### Verified Benchmark Results (1000 Test Flights, Seeds 20000–20999)

| Metric | Null Policy (Engine Off) | Random Uniform Baseline | Classical Heuristic Floor | Trained PPO Agent (G0 Validated) |
|:---|:---:|:---:|:---:|:---:|
| **Landing Success (`LANDED_ON_PAD`)** | 0.0% | 0.0% | 52.7% | **95.50%** |
| **95% Wilson CI (`LANDED_ON_PAD`)** | [0.0%, 0.38%] | [0.00%, 0.38%] | [49.60%, 55.78%] | **[94.03%, 96.62%]** |
| **Terminal Crash Rate (`CRASH`)** | 100.0% | 92.4% | 2.7% | **0.10%** |
| **Mean Episode Reward** | -132.18 ± 39.38 | -182.01 ± 109.28 | 235.14 ± 109.32 | **189.48 ± 28.28** |
| **Median Episode Reward** | -132.53 | -147.50 | 264.04 | **191.30** |
| **Touchdown Lateral Error p50 / p95** | N/A | N/A | 0.086 / 0.789 | **0.039 / 0.109** |

> **Gate G0 Status: PASSED**
> - Target: ≥95.0% `LANDED_ON_PAD` on 1,000 test seeds with Wilson Lower Bound ≥90.0%.
> - Achieved: **95.50%** landing success with Wilson 95% CI **[94.03%, 96.62%]** (lower bound 94.03% exceeds gate threshold).

<!-- RESULTS:END -->
## 🧪 Automated Testing

Execute the test suite to verify simulator stability and model integrity:
```powershell
pytest -v
```

---

## 📜 References
- Schulman et al., *"Proximal Policy Optimization Algorithms"*, arXiv:1707.06347, 2017.
- Brockman et al., *"OpenAI Gym"*, arXiv:1606.01540, 2016.
- Raffin et al., *"Stable-Baselines3: Reliable Reinforcement Learning Implementations"*, JMLR, 2021.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
