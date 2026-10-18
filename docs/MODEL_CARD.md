# Model Card — LunarPilot & RocketLand-2D

## 1. Model Details
* **Developed by**: LunarPilot Autonomous Aerospace Guidance Project
* **Model Date**: October 2026
* **Model Versions**:
  * `models/lander_ppo/lander_ppo_best.zip` (LunarLander-v3 PPO agent, SHA-256: `785d03823466144e5436eebe33e5c94d01db908ae55db4dbdbbe1c6ff992336d`)
  * `models/rocketland_ppo/rocket_ppo_best.zip` (RocketLand-2D Continuous PPO agent)
* **Model Type**: Deep Reinforcement Learning (Proximal Policy Optimization - PPO) via Stable-Baselines3, supplemented with Classical Non-linear Suicide Burn (Hoverslam) Guidance controllers.

---

## 2. Intended Use & Scope
* **Primary Intended Use**:
  * Precision vertical recovery and terminal guidance of orbital booster stages on autonomous droneship landing pads.
  * Rigorous empirical benchmarking and verification of reinforcement learning versus classical aerospace optimal control laws.
* **Out-of-Scope Uses**:
  * Flight-critical embedded deployment on orbital launch vehicles without full 6-DOF hardware-in-the-loop (HIL) qualification and safety-critical formal verification.

---

## 3. Simulation Fidelity & Physics Basis

### Fidelity Level 2: Non-Linear Planar Atmospheric Dynamics
* **Equations of Motion**: 3-DOF planar rigid body translation and pitch rotation integrated via 4th-order Runge-Kutta (RK4) with adaptive substepping ($dt_{\text{flight}} = 0.02\text{s}$, $dt_{\text{contact}} = 0.002\text{s}$).
* **Aerodynamics**: Non-linear drag resolved into body axial and lateral crossflow components under exponential barometric atmosphere ($\rho(y) = 1.225 \exp(-y / 8500)$).
* **Propulsion & TVC**:
  * Throttled single-engine propulsion ($T_{\text{max}} = 845\text{ kN}$, minimum throttle $40\% = 338\text{ kN}$).
  * First-order spool-up lag ($\tau = 0.1\text{s}$) with throttle rate limits.
  * Thrust vector control (TVC) engine gimbal ($\pm 5.0^\circ$ at $20^\circ/\text{s}$ slew limit).
  * Minimum thrust-to-weight ratio $>1.0$ at terminal mass ($W \approx 270\text{ kN}$), requiring closed-loop hoverslam suicide burns (no static hover possible).
* **Landing Gear & Surface Interaction**:
  * Dual-point compliant spring-damper contact mechanics ($k = 3.0\times 10^6\text{ N/m}$, $c = 8.0\times 10^4\text{ N}\cdot\text{s/m}$) with Coulomb sliding friction ($\mu = 0.6$).
  * Droneship deck envelope: $50\text{ m}$ span ($|x| \le 25\text{ m}$).

---

## 4. Validity Envelope & Operational Bounds

The policy and classical guidance controllers are validated across the following envelope:

| Parameter | Minimum | Maximum | Unit |
|:---|:---:|:---:|:---:|
| Initial Altitude ($y_0$) | $800.0$ | $1,800.0$ | $\text{m}$ |
| Downrange Offset ($x_0$) | $-300.0$ | $+300.0$ | $\text{m}$ |
| Vertical Descent Velocity ($v_{y,0}$) | $-120.0$ | $-40.0$ | $\text{m/s}$ |
| Horizontal Velocity ($v_{x,0}$) | $-30.0$ | $+30.0$ | $\text{m/s}$ |
| Initial Pitch Angle ($\theta_0$) | $-15.0$ | $+15.0$ | $\text{deg}$ |
| Propellant Reserve ($m_{\text{prop}}$) | $2,500.0$ | $4,000.0$ | $\text{kg}$ |
| Sustained Crosswind ($v_{\text{wind}}$) | $0.0$ | $12.0$ | $\text{m/s}$ |
| Stochastic Wind Gusts ($\sigma_{\text{gust}}$) | $0.0$ | $2.5$ | $\text{m/s}$ |

---

## 5. Known Failure Modes & Boundaries

Evaluated across 1,000 held-out Monte Carlo test seeds (`20000 - 20999`) and visualized in `results/figures/rocketland_failure_atlas.png`:

1. **Energy Depletion / Exhaustion Crash**:
   * Occurs when initial entry conditions (high kinetic energy combined with low altitude or low initial propellant mass $<2600\text{ kg}$) violate the physical stopping distance envelope. Fuel runs dry prior to $y=0$.
2. **Dynamic Crosswind Drift (`LANDED_OFF_PAD`)**:
   * Under extreme crosswinds ($>10\text{ m/s}$), aerodynamic drag forces exceed the lateral component of engine thrust available within the $\pm 5^\circ$ TVC gimbal limit, causing touchdown beyond the $25\text{m}$ droneship deck boundary.
3. **Hard Landing (`HARD_LANDING`)**:
   * Occurs if the engine throttles to minimum ($40\%$) and the hoverslam trajectory encounters sudden downward windshear near contact, exceeding the structural $2.0\text{ m/s}$ vertical limit.
4. **Attitude Tip-Over (`TIPPED`)**:
   * Occurs if residual horizontal shear velocity at contact exceeds $1.0\text{ m/s}$, creating a destabilizing trip moment about the leading landing leg footpad.

---

## 6. Verification & Evaluation Protocol

* **Seed Partitions**:
  * Training Seeds: `0 - 9999`
  * Validation Seeds: `10000 - 10999`
  * Held-out Test Seeds: `20000 - 20999` (used for all official gate benchmarks)
* **Wilson Score Interval**: All success rates reported with exact two-sided 95% Wilson confidence intervals ($z=1.96$):
  $$CI_{95\%} = \frac{p + \frac{z^2}{2n} \pm z\sqrt{\frac{p(1-p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$
* **Reproduction**:
  ```bash
  # Evaluate Gate G0 (LunarLander PPO)
  python scripts/evaluate.py --model models/lander_ppo/lander_ppo_best.zip --seeds 20000:21000

  # Evaluate Gate G2a/b/c & generate Failure Atlas
  python scripts/run_rocket_gates.py

  # Deterministically update README results
  python scripts/make_report.py
  ```
