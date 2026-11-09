"""
step2_jacobian_check.py  --  tests for guide STEP 3 (base orientation) and STEP 4 (Jacobians).

(The file keeps its old name "step2"; in the guide these are Steps 3 and 4.)

What is checked
    STEP 3  E(theta):  omega computed as E @ Euler-rates must equal the rotation rate measured
                       numerically from R(theta);  det E = cos(pitch) -> 0 at 90 deg.
    STEP 4  Jacobians: every column of frame_jacobian / com_jacobian must equal a finite
                       difference of forward kinematics, plus one case you can check by hand.

Idea of the finite-difference test
    Move the robot a tiny bit along one velocity direction e_k:  q_eps = q + eps * e_k
    and measure how a point / frame moved:
        linear  column k  ~=  (x(q_eps) - x(q)) / eps
        angular column k  ~=  log( R(q_eps) @ R(q)^T ) / eps      (small rotation -> rotation vector)
    Both must match column k of the analytic Jacobian (error ~ eps, i.e. ~1e-7).

Run:   python step2_jacobian_check.py
"""
import os
import numpy as np
from robot_model import RobotModel, R_to_rotvec, rpy_to_R, euler_rate_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
model = RobotModel(os.path.join(HERE, "biped_simple_v1_2.urdf"))
rng = np.random.default_rng(0)                     # fixed seed: same "random" poses every run

results = []


def check(name, ok, detail):
    """Print one PASS/FAIL line and remember the result."""
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:40s} {detail}")


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
#  STEP 3 CHECK -- E(theta): Euler-angle rates -> angular velocity
# =====================================================================================
print("STEP 3  base orientation")
worst_E = 0.0
for _ in range(10):
    q = random_q()
    rates = rng.normal(size=3)                         # random (roll_dot, pitch_dot, yaw_dot)
    eps = 1e-7
    R0 = rpy_to_R(q[3:6])                              # orientation now
    R1 = rpy_to_R(q[3:6] + rates * eps)                # orientation a moment later
    omega_num = R_to_rotvec(R1 @ R0.T) / eps           # measured angular velocity (world frame)
    omega_E = euler_rate_matrix(q[3:6]) @ rates        # predicted: omega = E(theta) theta_dot
    worst_E = max(worst_E, np.abs(omega_num - omega_E).max())
check("omega = E(theta) @ Euler rates", worst_E < 1e-5, f"max error = {worst_E:.1e}")

# det E = cos(pitch): shrinks toward 0 as pitch -> 90 deg (gimbal lock)
dets = [np.linalg.det(euler_rate_matrix([0, np.deg2rad(a), 0])) for a in (0, 60, 89)]
check("det E = cos(pitch)", np.allclose(dets, np.cos(np.deg2rad([0, 60, 89]))),
      f"det E at 0/60/89 deg = {dets[0]:.3f} / {dets[1]:.3f} / {dets[2]:.4f}  (-> 0 at 90 deg)")


# =====================================================================================
#  STEP 4 CHECK -- Jacobians against finite differences
# =====================================================================================
def numeric_jacobian(q, link, local_point, eps=1e-6):
    """Finite-difference Jacobian of a point on `link`, one column per entry of q_dot."""
    F0 = model.forward_kinematics(q)
    R0, p0 = F0[link]
    x0 = p0 + R0 @ local_point                         # the point now
    J = np.zeros((6, model.nv))
    for k in range(model.nv):
        dv = np.zeros(model.nv)
        dv[k] = 1.0                                    # move only coordinate k
        F1 = model.forward_kinematics(model.integrate(q, dv, eps))
        R1, p1 = F1[link]
        J[0:3, k] = (p1 + R1 @ local_point - x0) / eps   # how far the point moved / eps
        J[3:6, k] = R_to_rotvec(R1 @ R0.T) / eps         # how much the link turned / eps
    return J


print("\nSTEP 4  Jacobians")
# points to test: frame origins, a sole corner (contact point), a link CoM, a point on the pelvis
tests = [("left_toe",    np.zeros(3)),
         ("right_heel",  np.zeros(3)),
         ("left_foot",   np.array([0.15, 0.04, -0.075])),   # a sole corner
         ("right_shank", np.array([0, 0, -0.2])),           # shank CoM
         ("pelvis",      np.array([0, 0.1, 0]))]            # left hip point

worst = {link: 0.0 for link, _ in tests}
for _ in range(5):                                     # 5 random poses
    q = random_q()
    for link, pt in tests:
        Ja = model.frame_jacobian(q, link, pt)         # analytic (Step 4 table)
        Jn = numeric_jacobian(q, link, pt)             # measured
        worst[link] = max(worst[link], np.abs(Ja - Jn).max())
for link, pt in tests:
    check(f"frame_jacobian: {link} {np.round(pt, 3)}", worst[link] < 1e-5, f"max error = {worst[link]:.1e}")

# CoM Jacobian: CoM velocity predicted by J_CoM @ v vs CoM displacement over a tiny step
q = random_q()
v = rng.normal(size=model.nv)
eps = 1e-6
com_rate_numeric = (model.center_of_mass(model.integrate(q, v, eps)) - model.center_of_mass(q)) / eps
com_rate_analytic = model.com_jacobian(q) @ v
err_com = np.abs(com_rate_numeric - com_rate_analytic).max()
check("com_jacobian", err_com < 1e-5, f"max error = {err_com:.1e}")

# Hand check: standing straight, only the left knee turning at +1 rad/s.
#   The ankle is 0.4 m below the knee; turning about +y swings it backward:
#   a x (x - p_knee) = (0,1,0) x (0,0,-0.4) = (-0.4, 0, 0).  The foot turns with omega = (0,1,0).
q0 = model.neutral_q(base_height=0.85)
v0 = np.zeros(model.nv)
v0[model.joint_index("left_knee_joint")] = 1.0
lin, ang = model.frame_velocity(q0, v0, "left_foot")
check("hand check: left knee at +1 rad/s",
      np.allclose(lin, [-0.4, 0, 0]) and np.allclose(ang, [0, 1, 0]),
      f"ankle velocity {np.round(lin, 4)}, foot omega {np.round(ang, 4)}")

print("\nALL PASS" if all(results) else "\nSOME TESTS FAILED")
