"""
make_demo.py — Generates real animated GIF demo of RocketLand-2D landing
Saves output with policy and seed in file name:
results/demos/rocketland_{policy}_{seed}.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from rocketland.config import RocketConfig
from rocketland.controllers.suicide_burn import SuicideBurnController
from rocketland.env import RocketLand2DEnv


def render_frame_to_pil(env: RocketLand2DEnv, step: int, outcome: str | None = None) -> Image.Image:
    fig, ax = plt.subplots(figsize=(6, 8), dpi=80)
    ax.set_facecolor("#0b1120")  # Dark aerospace navy

    # Sky gradient / stars
    ax.set_xlim(-150, 150)
    ax.set_ylim(-10, 350)
    ax.set_aspect("equal")

    # Droneship Deck
    pad_hw = env.cfg.pad_half_width
    deck = patches.Rectangle((-pad_hw, -6), 2 * pad_hw, 6, facecolor="#334155", edgecolor="#e2e8f0", linewidth=1.5)
    ax.add_patch(deck)
    # Landing target ring
    ax.plot([-pad_hw, pad_hw], [0, 0], color="#f59e0b", linewidth=2.5, linestyle="--")
    ax.plot(0, 0, marker="x", markersize=8, color="#ef4444", markeredgewidth=2)

    # Ocean surface
    ocean = patches.Rectangle((-200, -20), 400, 14, facecolor="#0369a1", alpha=0.5)
    ax.add_patch(ocean)

    # Rocket State
    x = float(env.state[0])
    y = float(env.state[1])
    vx = float(env.state[2])
    vy = float(env.state[3])
    theta = float(env.state[4])
    fuel = float(env.state[6])
    throttle = float(getattr(env.actuators, "throttle", 0.0))
    gimbal = float(getattr(env.actuators, "gimbal", 0.0))

    # Rocket body geometry
    L = env.cfg.length
    D = env.cfg.diameter
    # Transform: center of rocket
    # Draw body as rotated polygon
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    def rot(dx, dy):
        return x + dx * cos_t - dy * sin_t, y + dx * sin_t + dy * cos_t

    # Fuselage corners
    p1 = rot(-D/2, -L/2)
    p2 = rot(D/2, -L/2)
    p3 = rot(D/2, L/2)
    p4 = rot(-D/2, L/2)
    ax.fill([p1[0], p2[0], p3[0], p4[0]], [p1[1], p2[1], p3[1], p4[1]], color="#f8fafc", edgecolor="#475569", linewidth=1.5)

    # Landing legs
    foot_w = 6.0
    ax.plot([p1[0], rot(-foot_w, -L/2 - 2)[0]], [p1[1], rot(-foot_w, -L/2 - 2)[1]], color="#cbd5e1", linewidth=2.5)
    ax.plot([p2[0], rot(foot_w, -L/2 - 2)[0]], [p2[1], rot(foot_w, -L/2 - 2)[1]], color="#cbd5e1", linewidth=2.5)

    # Main engine flame if firing
    if throttle > 0.05 and y > 0.5:
        flame_len = 10.0 + 35.0 * throttle
        gimbal_ang = theta + gimbal
        fx = x - np.sin(gimbal_ang) * (L/2 + flame_len)
        fy = y - np.cos(gimbal_ang) * (L/2 + flame_len)
        flame_poly = [
            rot(-D*0.35, -L/2),
            (fx, fy),
            rot(D*0.35, -L/2)
        ]
        ax.fill([p[0] for p in flame_poly], [p[1] for p in flame_poly], color="#f97316", alpha=0.9)
        # Inner blue core
        core_len = flame_len * 0.4
        cx = x - np.sin(gimbal_ang) * (L/2 + core_len)
        cy = y - np.cos(gimbal_ang) * (L/2 + core_len)
        core_poly = [
            rot(-D*0.2, -L/2),
            (cx, cy),
            rot(D*0.2, -L/2)
        ]
        ax.fill([p[0] for p in core_poly], [p[1] for p in core_poly], color="#38bdf8", alpha=0.95)

    # HUD Overlay
    status_text = f"Alt: {y:5.1f}m | Vy: {vy:5.1f}m/s | Vx: {vx:5.1f}m/s\nThrottle: {throttle*100:3.0f}% | Gimbal: {np.degrees(gimbal):+4.1f}° | Fuel: {fuel:4.0f}kg"
    ax.text(-140, 320, status_text, color="#38bdf8", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e293b", edgecolor="#334155", alpha=0.85))

    if outcome:
        color = "#10b981" if "LANDED_ON_PAD" in outcome else "#ef4444"
        ax.text(0, 180, outcome, color=color, fontsize=14, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f172a", edgecolor=color, linewidth=2))

    ax.axis("off")
    fig.canvas.draw()
    # Extract image buffer
    rgba = np.asarray(fig.canvas.buffer_rgba())
    im = Image.fromarray(rgba).convert("RGB")
    plt.close(fig)
    return im


def generate_demo(seed: int = 20001, policy_name: str = "suicide_burn"):
    cfg = RocketConfig()
    cfg.wind_base = 4.0
    cfg.wind_gust_std = 0.5
    env = RocketLand2DEnv(cfg)
    ctrl = SuicideBurnController(cfg)

    obs, _ = env.reset(seed=seed)
    ctrl.reset()
    frames = []

    done = False
    step = 0
    outcome = None
    contact_steps = 0

    print(f"[*] Simulating real run for demo (policy={policy_name}, seed={seed})...")
    while not done and step < 400:
        if step % 3 == 0:  # Sample every 3 steps (~10 fps)
            frames.append(render_frame_to_pil(env, step))

        act = ctrl(obs)
        obs, r, term, trunc, info = env.step(act)
        done = term or trunc

        if (env.contacts[0] or env.contacts[1]) and step > 10:
            contact_steps += 1
            if contact_steps >= 4:
                done = True

        step += 1

    outcome = info.get("outcome")
    if outcome is None:
        outcome = "TOUCHDOWN: LANDED_ON_PAD" if abs(env.state[0]) <= cfg.pad_half_width else "LANDED_OFF_PAD"
    else:
        outcome = str(outcome.value if hasattr(outcome, "value") else outcome)

    # Append terminal frame hold
    term_frame = render_frame_to_pil(env, step, outcome=outcome)
    for _ in range(15):
        frames.append(term_frame)

    out_dir = Path("results/demos")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rocketland_{policy_name}_seed{seed}.gif"

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # 100ms per frame
        loop=0
    )
    print(f"[OK] Saved animated demo to {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20001)
    parser.add_argument("--policy", type=str, default="suicide_burn")
    args = parser.parse_args()
    generate_demo(args.seed, args.policy)
