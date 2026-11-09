"""
step1_fk_plot.py  --  load the URDF, run forward kinematics, plot the robot.

Run:   python step1_fk_plot.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from robot_model import RobotModel

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "biped_simple_v1_2.urdf")

model = RobotModel(URDF)
print(model.summary())


def draw(ax, q, title):
    F = model.forward_kinematics(q)
    P = lambda n: F[n][1]
    # skeleton: one line per joint (parent origin -> child origin), links in gray
    for j in model.joints:
        a, b = P(j.parent), P(j.child)
        ax.plot(*zip(a, b), "-", color="0.55", lw=3, zorder=1)
    # pelvis bar and feet (heel -> toe)
    ax.plot(*zip(P("left_hip_roll_link"), P("right_hip_roll_link")), "-", color="k", lw=5)
    for side in ("left", "right"):
        ax.plot(*zip(P(f"{side}_heel"), P(f"{side}_toe")), "-", color="tab:orange", lw=4)

    # joints (blue): a dot at each joint position and a short arrow along its world axis.
    # Hip and ankle have two joints at the same point; the arrows show which is which
    # (roll axis points forward along x, pitch axis sideways along y).
    for j in model.actuated:
        R_c, p_j = F[j.child]                 # joint frame = child link frame (Step 2)
        a = R_c @ j.axis                      # joint axis in world (same as in the Jacobian)
        ax.scatter(*p_j, color="tab:blue", s=45, edgecolors="navy", depthshade=False, zorder=5)
        ax.quiver(*p_j, *(0.09 * a), color="tab:blue", lw=1.5, arrow_length_ratio=0.35)

    # link CoMs and total CoM
    for name, link in model.links.items():
        if link.mass > 0.01:
            R, p = F[name]
            ax.scatter(*(p + R @ link.com), color="tab:red", s=15)
    com = model.center_of_mass(q, F)
    ax.scatter(*com, color="k", s=60, marker="x")
    ax.plot([com[0], com[0]], [com[1], com[1]], [com[2], 0], ":", color="gray")

    # legend (proxy artists so each item appears once)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color="0.55", lw=3, label="link"),
        Line2D([], [], marker="o", ls="", color="tab:blue", markeredgecolor="navy", label="joint (+ axis)"),
        Line2D([], [], marker="o", ls="", color="tab:red", ms=4, label="link CoM"),
        Line2D([], [], marker="x", ls="", color="k", label="robot CoM"),
        Line2D([], [], color="tab:orange", lw=4, label="foot (heel-toe)")],
        loc="upper left", fontsize=7, framealpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5); ax.set_zlim(0, 1.0)
    ax.set_box_aspect((1, 1, 1))


# Pose 1: zero joint angles, pelvis placed so the toes/heels touch z = 0
q0 = model.neutral_q()
F0 = model.forward_kinematics(q0)
foot_z = F0["left_heel"][1][2]           # heel height when pelvis is at z = 0
q0[2] = -foot_z
print("\nZero pose: pelvis height = %.3f m" % q0[2])
for n in ("left_thigh", "left_shank", "left_foot", "left_heel", "left_toe", "right_toe"):
    print(f"  {n:12s} at {np.round(model.frame_position(q0, n), 4)}")
print("  CoM        at", np.round(model.center_of_mass(q0), 4))

# Pose 2: crouch (hip -0.4, knee +0.8, ankle -0.4) and pelvis tilted 0.2 rad roll
q1 = model.neutral_q(base_height=0.75)
for side in ("left", "right"):
    q1[model.joint_index(f"{side}_hip_pitch_joint")] = -0.4
    q1[model.joint_index(f"{side}_knee_joint")] = 0.8
    q1[model.joint_index(f"{side}_ankle_pitch_joint")] = -0.4
q1[3:6] = [0.2, 0.0, 0.0]   # pelvis (roll, pitch, yaw) in rad

fig = plt.figure(figsize=(11, 5))
draw(fig.add_subplot(121, projection="3d"), q0, "Zero pose")
draw(fig.add_subplot(122, projection="3d"), q1, "Crouch + pelvis roll 0.2 rad")
plt.tight_layout()
out = os.path.join(HERE, "step1_poses.png")
plt.savefig(out, dpi=110)
print("\nSaved", out)
plt.show()
