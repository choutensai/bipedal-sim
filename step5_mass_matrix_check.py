"""
step5_mass_matrix_check.py  --  tests for guide STEP 5 (mass matrix).

    M(q) = sum_i [ m_i Jv_i^T Jv_i + Jw_i^T I_i^W Jw_i ]        T = 1/2 q_dot^T M q_dot

What is checked (tests (a)-(e) of Step 5 in the guide)
    (a) M is symmetric
    (b) M is positive definite (every motion has positive kinetic energy)
    (c) the base-translation block M[0:3,0:3] = total mass * I
    (d) 1/2 q_dot^T M q_dot = kinetic energy summed link by link (computed without M)
    (e) one entry checked by hand: the knee, at the zero pose

Run:   python step5_mass_matrix_check.py
"""
import os
import numpy as np
from robot_model import RobotModel

HERE = os.path.dirname(os.path.abspath(__file__))
model = RobotModel(os.path.join(HERE, "biped_simple_v1_2.urdf"))
rng = np.random.default_rng(0)                     # fixed seed: same "random" poses every run
np.set_printoptions(precision=4, suppress=True, linewidth=110)

results = []


def check(name, ok, detail):
    """Print one PASS/FAIL line and remember the result."""
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:38s} {detail}")


# -------------------------------------------------------------------------------------
#  Random test poses
# -------------------------------------------------------------------------------------
def random_q():
    """A random but valid configuration (pitch kept away from the gimbal-lock angle)."""
    q = model.neutral_q(base_height=0.8)
    q[0:3] += rng.uniform(-0.2, 0.2, 3)                # pelvis position
    q[3:6] = [rng.uniform(-np.pi, np.pi),              # roll
              rng.uniform(-1.2, 1.2),                  # pitch, |pitch| < 69 deg
              rng.uniform(-np.pi, np.pi)]              # yaw
    for j in model.actuated:
        q[6 + j.q_index] = rng.uniform(j.lower, j.upper)   # joint angles within URDF limits
    return q


# =====================================================================================
#  STEP 5 CHECK (a)-(d) -- properties that must hold at ANY pose
# =====================================================================================
print("STEP 5  mass matrix, 20 random poses")
worst_sym, worst_eig, worst_mass, worst_T = 0.0, np.inf, 0.0, 0.0
for _ in range(20):
    q = random_q()
    qd = rng.normal(size=model.nv)                     # random velocity for test (d)
    M = model.mass_matrix(q)

    # (a) symmetric: M_ij = M_ji
    worst_sym = max(worst_sym, np.abs(M - M.T).max())
    # (b) positive definite: smallest eigenvalue > 0
    worst_eig = min(worst_eig, np.linalg.eigvalsh(M).min())
    # (c) moving the whole robot by dx needs force m*a in every direction -> M[0:3,0:3] = m*I
    worst_mass = max(worst_mass, np.abs(M[0:3, 0:3] - model.total_mass * np.eye(3)).max())
    # (d) same kinetic energy two ways: from M, and link by link
    worst_T = max(worst_T, abs(0.5 * qd @ M @ qd - model.kinetic_energy(q, qd)))

check("(a) symmetric", worst_sym < 1e-12, f"max |M - M^T| = {worst_sym:.1e}")
check("(b) positive definite", worst_eig > 0, f"smallest eigenvalue = {worst_eig:.2e}")
check("(c) M[0:3,0:3] = total mass * I", worst_mass < 1e-10,
      f"error = {worst_mass:.1e}  (m = {model.total_mass:.3f} kg)")
check("(d) 0.5 qd^T M qd = sum of link T", worst_T < 1e-9, f"error = {worst_T:.1e}")


# =====================================================================================
#  STEP 5 CHECK (e) -- hand calculation of the knee entry at the zero pose
#
#  With the robot standing straight, turning only the knee rotates everything below it
#  (shank, ankle dummy link, foot) about the knee's y axis.  So M[knee, knee] is the
#  moment of inertia of those bodies about that axis (parallel-axis theorem):
#      sum over bodies of   I_yy(about own CoM) + m * (distance from the knee axis)^2
# =====================================================================================
q0 = model.neutral_q(base_height=0.85)
M0 = model.mass_matrix(q0)
k = model.joint_index("left_knee_joint")
shank = model.links["left_shank"]
dummy = model.links["left_ankle_pitch_link"]
foot = model.links["left_foot"]
hand = (shank.inertia[1, 1] + shank.mass * 0.2**2                  # shank CoM 0.2 m below the knee
        + dummy.inertia[1, 1] + dummy.mass * 0.4**2                 # 0.001 kg dummy at the ankle, 0.4 m below
        + foot.inertia[1, 1] + foot.mass * (0.05**2 + 0.45**2))     # foot CoM 0.45 m below, 0.05 m ahead
check("(e) knee entry vs hand calculation", abs(M0[k, k] - hand) < 1e-9,
      f"M = {M0[k, k]:.4f}, hand = {hand:.4f}")


# -------------------------------------------------------------------------------------
#  For reference: diagonal of M at the zero pose
#  (how 'heavy' each coordinate is; ankle entries are ~700x smaller than hips -> M is
#   poorly conditioned, which is why we will use np.linalg.solve, never inv(M))
# -------------------------------------------------------------------------------------
print("\nZero pose, diagonal of M:")
labels = ["x", "y", "z", "roll", "pitch", "yaw"] + [n.replace("_joint", "") for n in model.joint_names]
for lab, val in zip(labels, np.diag(M0)):
    print(f"    {lab:22s} {val:9.4f}")

print("\nALL PASS" if all(results) else "\nSOME TESTS FAILED")
