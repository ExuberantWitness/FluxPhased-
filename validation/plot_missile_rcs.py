"""Plot cruise missile aspect RCS + Swerling fluctuation."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Aspect RCS model (matches vec_missile.py) ---
rcs_nose = -5.0   # dBsm
rcs_side = 12.0   # dBsm
rcs_tail = 3.0    # dBsm

a = (rcs_nose + rcs_tail) / 2.0 - rcs_side
b = (rcs_tail - rcs_nose) / 2.0
d = rcs_side

angles_deg = np.linspace(0, 360, 721)
angles_rad = np.deg2rad(angles_deg)

# cos_aspect: -1=nose-on (missile heading toward radar), +1=tail-on
# aspect_angle from missile nose: 0°=nose, 90°=broadside, 180°=tail
aspect_from_nose = angles_rad  # 0 to 2π
cos_c = np.cos(aspect_from_nose)

# For angles > 180°, mirror (RCS is symmetric about missile axis in yaw)
# Use |cos| for the base, but split nose/tail
cos_c_wrapped = np.where(angles_deg <= 180, -cos_c, cos_c)
# Redefine: c = -cos(aspect_from_nose), so c=-1 at nose (0°), c=+1 at tail (180°)
# But we want full 360°, so:
# 0° (nose): c = -cos(0) = -1
# 90° (side): c = -cos(90°) = 0
# 180° (tail): c = -cos(180°) = +1
# 270° (side): c = -cos(270°) = 0
c_full = -np.cos(angles_rad)

rcs_db = a * c_full**2 + b * c_full + d

# --- Plot 1: Polar RCS pattern ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7), subplot_kw={"projection": "polar"})

# Left: dBsm
ax1 = axes[0]
ax1.plot(angles_rad, rcs_db, "b-", linewidth=2)
ax1.set_theta_zero_location("N")
ax1.set_theta_direction(-1)
ax1.set_rlabel_position(135)
ax1.set_title("Missile RCS vs Aspect Angle (dBsm)", pad=20, fontsize=13)
ax1.set_rticks([-5, 0, 5, 10, 12])
ax1.set_rlim(-10, 15)
ax1.annotate("Nose\n-5 dBsm", xy=(0, -5), fontsize=9, ha="center", color="red",
             xytext=(0.3, -9), arrowprops=dict(arrowstyle="->", color="red"))
ax1.annotate("Side\n12 dBsm", xy=(np.pi/2, 12), fontsize=9, ha="center", color="green",
             xytext=(np.pi/2+0.4, 14))
ax1.annotate("Tail\n3 dBsm", xy=(np.pi, 3), fontsize=9, ha="center", color="orange",
             xytext=(np.pi+0.3, 7), arrowprops=dict(arrowstyle="->", color="orange"))

# Right: linear m²
rcs_linear = 10.0 ** (rcs_db / 10.0)
ax2 = axes[1]
ax2.plot(angles_rad, rcs_linear, "r-", linewidth=2)
ax2.set_theta_zero_location("N")
ax2.set_theta_direction(-1)
ax2.set_rlabel_position(135)
ax2.set_title("Missile RCS vs Aspect Angle (m²)", pad=20, fontsize=13)
ax2.annotate("Nose\n0.3 m²", xy=(0, 0.3), fontsize=9, ha="center", color="blue",
             xytext=(0.3, 3))
ax2.annotate("Side\n16 m²", xy=(np.pi/2, 16), fontsize=9, ha="center", color="green",
             xytext=(np.pi/2+0.4, 18))
ax2.annotate("Tail\n2 m²", xy=(np.pi, 2), fontsize=9, ha="center", color="orange",
             xytext=(np.pi+0.3, 5))

plt.tight_layout()
plt.savefig("validation/figures/18_missile_rcs_polar.png", dpi=150, bbox_inches="tight")
print("Saved: validation/figures/18_missile_rcs_polar.png")

# --- Plot 2: Swerling fluctuation time series ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
np.random.seed(42)
n_samples = 200

for idx, (model, name) in enumerate([
    (0, "Swerling 0 (none)"),
    (1, "Swerling 1 (slow, exp)"),
    (3, "Swerling 3 (slow, χ²(4))"),
    (4, "Swerling 4 (fast, χ²(4))"),
]):
    ax = axes[idx // 2, idx % 2]
    rcs_avg_db = 10.0  # average RCS in dBsm

    if model == 0:
        rcs_samples = np.ones(n_samples) * 10.0 ** (rcs_avg_db / 10.0)
    else:
        rcs_samples = []
        for i in range(n_samples):
            if model in (1, 2):
                u = np.random.uniform(0, 1)
                sigma = -np.log(max(u, 1e-10))
            elif model in (3, 4):
                u1 = np.random.uniform(0, 1)
                u2 = np.random.uniform(0, 1)
                sigma = (-np.log(max(u1, 1e-10)) - np.log(max(u2, 1e-10))) / 2.0
            rcs_samples.append(sigma * 10.0 ** (rcs_avg_db / 10.0))
        rcs_samples = np.array(rcs_samples)

    ax.plot(rcs_samples, linewidth=0.8, alpha=0.8)
    ax.axhline(10.0 ** (rcs_avg_db / 10.0), color="red", linestyle="--", label=f"Mean ({rcs_avg_db} dBsm)")
    ax.set_title(name, fontsize=12)
    ax.set_xlabel("Pulse / CPI index")
    ax.set_ylabel("RCS (m²)")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle("Swerling RCS Fluctuation Models (Cruise Missile, σ_avg = 10 dBsm)", fontsize=14)
plt.tight_layout()
plt.savefig("validation/figures/19_missile_swerling.png", dpi=150, bbox_inches="tight")
print("Saved: validation/figures/19_missile_swerling.png")

print("Done.")
