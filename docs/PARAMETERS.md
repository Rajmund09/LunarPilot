# RocketLand-2D & LunarPilot Parameter Reference

This document provides the definitive physical constants, aerodynamics, propulsion parameters, coordinate conventions, and disturbance specifications for the **RocketLand-2D** booster landing simulation and **LunarLander-v3** benchmarking environment.

---

## 1. Coordinate Frames & Conventions

### Frame Definition
* **Coordinate System**: 2D Cartesian right-handed frame:
  * $+x$: Downrange horizontal position (East / Right, meters).
  * $+y$: Altitude above mean sea surface / landing deck (Up, meters).
* **Vehicle Attitude ($\theta$)**:
  * Angle from vertical $+y$ axis in radians / degrees.
  * Counter-Clockwise (CCW) is positive ($\theta > 0$ tilts left / nose-west).
  * Body axis vector: $\mathbf{u} = (-\sin\theta, \cos\theta)$.
* **Thrust Vector Angle ($\delta$)**:
  * CCW gimbal rotation of the engine thrust vector relative to the body centerline axis.
  * Gimbal range: $\delta \in [-5.0^\circ, +5.0^\circ]$ ($\pm 0.087266\text{ rad}$).
  * Total thrust vector direction in world frame: $\theta + \delta$.
  * Engine torque about vehicle Center of Mass (CoM): $\tau_T = -T \cdot r_e \cdot \sin(\delta)$.
  * *Sign verification*: A positive gimbal $\delta > 0$ produces a negative (clockwise) torque, rotating the rocket clockwise toward the right.

---

## 2. Vehicle Physical Constants (RocketLand-2D)

Values are calibrated to a Falcon 9 / Starship-class orbital first-stage booster landing under terminal hoverslam conditions (`configs/rocketland/vehicle.json`).

| Parameter | Symbol | Value | Units | Provenance / Assumption |
|:---|:---:|:---:|:---:|:---|
| **Dry Mass** | $m_{\text{dry}}$ | $25,000.0$ | $\text{kg}$ | Calibrated booster recovery empty weight |
| **Propellant Mass (Min)** | $m_{\text{prop,min}}$ | $2,500.0$ | $\text{kg}$ | Landing burn reserve floor |
| **Propellant Mass (Max)** | $m_{\text{prop,max}}$ | $4,000.0$ | $\text{kg}$ | Entry burn cutoff reserve ceiling |
| **Booster Length** | $L$ | $40.0$ | $\text{m}$ | Booster cylindrical fuselage height |
| **Booster Diameter** | $D$ | $3.7$ | $\text{m}$ | Fuselage cross-sectional diameter |
| **Center of Engine to CoM** | $r_e$ | $18.0$ | $\text{m}$ | Distance from gimbal pivot to CoM |
| **Pitch Moment of Inertia** | $I_{zz}$ | $\approx \frac{1}{12} m L^2$ | $\text{kg}\cdot\text{m}^2$ | Slender cylinder approximation: $\sim 3.6\times 10^6\text{ kg}\cdot\text{m}^2$ |
| **Standard Gravitational Accel** | $g$ | $9.81$ | $\text{m/s}^2$ | Terrestrial gravity at surface |
| **Standard Gravity Constant** | $g_0$ | $9.80665$ | $\text{m/s}^2$ | Rocket equation $I_{\text{sp}}$ conversion standard |

---

## 3. Propulsion & Control Limits

| Parameter | Symbol | Value | Units | Description |
|:---|:---:|:---:|:---:|:---|
| **Maximum Thrust** | $T_{\text{max}}$ | $845,000.0$ | $\text{N}$ ($845\text{ kN}$) | Single Merlin 1D-class sea-level engine |
| **Minimum Throttle** | $\text{thro}_{\text{min}}$ | $0.40$ ($40\%$) | $-$ | Minimum turbopump throttle limit |
| **Minimum Thrust** | $T_{\text{min}}$ | $338,000.0$ | $\text{N}$ ($338\text{ kN}$) | $40\% \times 845\text{ kN}$ |
| **Specific Impulse** | $I_{\text{sp}}$ | $282.0$ | $\text{s}$ | Sea-level RP-1 / LOX combustion efficiency |
| **Gimbal Range** | $\delta_{\text{max}}$ | $\pm 5.0$ | $\text{deg}$ | Hydraulic TVC actuator physical stroke limit |
| **Gimbal Rate Limit** | $\dot{\delta}_{\text{max}}$ | $20.0$ | $\text{deg/s}$ | Hydraulic slew rate limit |
| **Throttle Slew Time** | $\dot{u}_{\text{thro}}$ | $2.0$ | $\text{s}^{-1}$ | Full throttle swing limit ($0 \to 1$ in $0.5\text{s}$) |
| **Actuator First-Order Lag** | $\tau_{\text{thro}}$ | $0.10$ | $\text{s}$ | Turbopump spool-up response time constant |
| **RCS Max Control Torque** | $\tau_{\text{RCS}}$ | $50,000.0$ | $\text{N}\cdot\text{m}$ | Cold-gas thrusters for attitude stabilization |

> [!IMPORTANT]
> **Fundamental Suicide Burn (Hoverslam) Physics**:
> At landing mass ($m \approx 27,000 - 29,000\text{ kg}$), the vehicle's gravity weight is $W = m \cdot g \approx 265 - 285\text{ kN}$.
> Since the minimum engine thrust is $T_{\text{min}} = 338\text{ kN}$, the minimum thrust-to-weight ratio is:
> $$\text{TWR}_{\text{min}} = \frac{338\text{ kN}}{285\text{ kN}} \approx 1.18 > 1.0$$
> **The booster cannot hover.** If ignited too early, it will decelerate to zero and accelerate back upwards into the sky before running out of fuel. If ignited too late, it will crash into the droneship. It must perform an exact closed-loop hoverslam suicide burn.

---

## 4. Atmospheric Aerodynamics

Aerodynamic drag is computed using an exponential barometric density model:
$$\rho(y) = \rho_0 \exp\left(-\frac{y}{H}\right)$$
Relative aerodynamic velocity:
$$\mathbf{v}_{\text{rel}} = \mathbf{v} - \mathbf{v}_{\text{wind}}$$

| Parameter | Symbol | Value | Units | Description |
|:---|:---:|:---:|:---:|:---|
| **Sea-Level Air Density** | $\rho_0$ | $1.225$ | $\text{kg/m}^3$ | ISA standard atmosphere at $y=0$ |
| **Atmospheric Scale Height** | $H$ | $8,500.0$ | $\text{m}$ | Barometric decay scale height |
| **Axial Drag Coefficient** | $C_{d,\text{axial}}$ | $0.80$ | $-$ | Blunt base/engine bell drag coefficient |
| **Lateral Drag Coefficient** | $C_{d,\text{lateral}}$ | $1.50$ | $-$ | Cylinder crossflow drag coefficient |
| **Axial Reference Area** | $A_{\text{axial}}$ | $\frac{\pi D^2}{4} \approx 10.75$ | $\text{m}^2$ | Booster circular cross-section |
| **Lateral Reference Area** | $A_{\text{lateral}}$ | $L \cdot D \approx 148.0$ | $\text{m}^2$ | Booster cylindrical projection |

---

## 5. Droneship Landing Deck & Contact Mechanics

Landing contact is simulated using a two-point compliant landing leg model (starboard and port footpads at span $w_{\text{gear}} = 12.0\text{ m}$).

| Parameter | Symbol | Value | Units | Description |
|:---|:---:|:---:|:---:|:---|
| **Deck Center** | $x_{\text{pad}}$ | $0.0$ | $\text{m}$ | Droneship center coordinate |
| **Deck Half-Width** | $w_{\text{pad}}/2$ | $25.0$ | $\text{m}$ | Autonomous Spaceport Droneship (ASDS) $50\text{m}$ deck |
| **Contact Spring Stiffness** | $k_{\text{contact}}$ | $3.0\times 10^6$ | $\text{N/m}$ | Compliant deck / crush core spring constant |
| **Contact Damping** | $c_{\text{contact}}$ | $8.0\times 10^4$ | $\text{N}\cdot\text{s/m}$ | Hydraulic landing strut damper |
| **Coulomb Friction Coeff** | $\mu$ | $0.60$ | $-$ | Steel grid deck non-skid friction coefficient |

### Touchdown Envelope (Pass Criteria)
To be classified as `LANDED_ON_PAD` (`is_success = True`), all four criteria must simultaneously hold at contact:
1. **Pad Clearance**: $|x_{\text{touchdown}}| \le 25.0\text{ m}$ (within ASDS deck boundary).
2. **Vertical Impact Velocity**: $|v_y| \le 2.0\text{ m/s}$ (below structural crush core limit).
3. **Lateral Shear Velocity**: $|v_x| \le 1.0\text{ m/s}$ (prevents gear trip and roll-over).
4. **Attitude Tilt**: $|\theta| \le 5.0^\circ$ (within upright vertical stability cone).

---

## 6. Disturbance Realism Models

### A. Boundary Layer & Stochastic Wind
- **Altitude Wind Profile**:
  $$v_{\text{wind,mean}}(y) = v_{\text{base}} \cdot \left(\frac{\max(y, 1.0)}{100.0}\right)^{0.143}$$
- **Ornstein-Uhlenbeck Gust Process**:
  $$dv_g(t) = -\frac{1}{\tau_g} v_g(t) dt + \sigma_g \sqrt{\frac{2}{\tau_g}} dW(t)$$
  where $\tau_g = 2.0\text{ s}$, $\sigma_g = \text{wind\_gust\_std}$.

### B. Sensor Suite Noise & Delay
- **IMU Gyroscope**: Gaussian zero-mean noise $\sigma_\omega = 0.005\text{ rad/s}$ ($0.28^\circ\text{/s}$), bias drift $b_\omega = 0.001\text{ rad/s}$.
- **Altimeter / Radar**: Gaussian noise $\sigma_y = 0.5\text{ m}$, bias $b_y = 0.1\text{ m}$.
- **Downrange GPS / Radar**: Gaussian noise $\sigma_x = 0.3\text{ m}$, bias $b_x = 0.05\text{ m}$.
- **Transport Latency Buffer**: 1 control cycle ($100\text{ ms}$) discrete state delay.
