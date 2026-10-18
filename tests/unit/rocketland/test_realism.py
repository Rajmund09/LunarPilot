import numpy as np

from rocketland.config import RocketConfig
from rocketland.sensors import SensorSuite
from rocketland.wind import OrnsteinUhlenbeckWind


def test_ornstein_uhlenbeck_wind():
    wind = OrnsteinUhlenbeckWind(base_wind=5.0, gust_std=1.0, seed=42)
    w0 = wind.step(y=10.0, dt=0.05)
    w100 = wind.step(y=100.0, dt=0.05)
    assert np.isfinite(w0)
    assert np.isfinite(w100)
    # Reset restores gust
    wind.reset(seed=42)
    assert wind.gust == 0.0


def test_sensor_suite_bias_and_noise():
    cfg = RocketConfig()
    suite = SensorSuite(cfg, noise_std=0.02, bias_std=0.01, delay_steps=2, seed=123)
    state = np.array([0.0, 500.0, 0.0, -40.0, 0.0, 0.0, 3000.0])
    contacts = (False, False)

    obs = suite.observe(state, throttle=0.5, gimbal_deg=0.0, contacts=contacts)
    assert obs.shape == (12,)
    assert np.all(np.isfinite(obs))
    # Contacts should remain exact binary 0.0
    assert obs[10] == 0.0
    assert obs[11] == 0.0
