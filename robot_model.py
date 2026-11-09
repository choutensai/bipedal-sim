"""
robot_model.py  --  floating-base biped model: URDF loader, kinematics, Jacobians, mass matrix.

Follows "biped_dynamics_step_by_step_guide.pdf". Each section below is marked with the
guide step it implements:

    STEP 0  conventions (this docstring)
    STEP 1  kinematic tree ............ RobotModel.__init__
    STEP 2  forward kinematics ........ forward_kinematics, center_of_mass
    STEP 3  base orientation .......... rpy_to_R, euler_rate_matrix, integrate
    STEP 4  Jacobians ................. frame_jacobian, com_jacobian
    STEP 5  mass matrix ............... mass_matrix, kinetic_energy

STEP 0 -- CONVENTIONS
World frame: X forward, Y left, Z up.  Units: m, kg, rad, s.

Configuration vector q (nq = 6 + n_joints = 16 for this biped):
    q = [ px, py, pz,            # p_b     : pelvis position in world          q[0:3]
          roll, pitch, yaw,      # theta_b : pelvis orientation (Euler angles)  q[3:6]
                                 #           R = Rz(yaw) @ Ry(pitch) @ Rx(roll)  (URDF convention)
          joint_1 ... joint_n ]  # q_j     : joint angles, in URDF file order   q[6:]

Velocity vector v (nv = nq): with Euler angles, v is simply the time derivative of q:
    v = q_dot = [ px_dot, py_dot, pz_dot, roll_dot, pitch_dot, yaw_dot, joint_rates... ]

NOTE: roll_dot/pitch_dot/yaw_dot are NOT the angular velocity.
      omega_world = E(theta_b) @ [roll_dot, pitch_dot, yaw_dot]   (see euler_rate_matrix)
      E is singular at pitch = +-90 deg (gimbal lock).
"""
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
import numpy as np


# =====================================================================================
#  MATH HELPERS  (rotation formulas used by Steps 2-4)
# =====================================================================================

# ---- STEP 3: rotation matrix of the pelvis from its Euler angles ---------------------
def rpy_to_R(rpy):
    """URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    Used for the pelvis orientation q[3:6] and for the fixed <origin rpy="..."> of joints."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])      # rotation about x (roll)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])      # rotation about y (pitch)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])      # rotation about z (yaw)
    return Rz @ Ry @ Rx                                         # order matters: yaw * pitch * roll


# ---- STEP 3: Euler-angle rates -> angular velocity -----------------------------------
def euler_rate_matrix(rpy):
    """E(theta) such that  omega_world = E @ [roll_dot, pitch_dot, yaw_dot].

    Each Euler rate turns the body about its own current axis; E's columns are those axes
    expressed in world coordinates:
        column 1: roll axis   = Rz @ Ry @ e_x
        column 2: pitch axis  = Rz @ e_y
        column 3: yaw axis    = e_z
    det(E) = cos(pitch)  ->  singular at pitch = +-pi/2 (gimbal lock)."""
    _, p, y = rpy                                  # roll does not appear in E
    cp, sp, cy, sy = np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, -sy, 0.0],
                     [sy * cp,  cy, 0.0],
                     [-sp,     0.0, 1.0]])


GIMBAL_WARN = np.deg2rad(75)   # integrate() warns when |pitch| exceeds this


# ---- STEP 2: rotation of a revolute joint (Rodrigues formula) ------------------------
def axis_angle_to_R(axis, angle):
    """Rotation by `angle` about unit vector `axis`:
        R = I + sin(angle) [a]x + (1 - cos(angle)) [a]x^2"""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)                      # make sure the axis is a unit vector
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])   # K = [a]x
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


# ---- STEP 4: cross-product matrix ----------------------------------------------------
def skew(w):
    """[w]x :  skew(w) @ v == np.cross(w, v)."""
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


# ---- used by the Jacobian test (step2_jacobian_check.py) -----------------------------
def R_to_rotvec(R):
    """Inverse of Rodrigues: rotation matrix -> axis * angle (the 'log' map).
    The test uses it to measure how much a frame rotated between two poses."""
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)  # cos(angle) from trace(R) = 1 + 2 cos(angle)
    angle = np.arccos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])   # = 2 sin(angle) * axis
    if angle < 1e-8:
        return 0.5 * w                             # small-angle limit
    return angle / (2 * np.sin(angle)) * w


# ---- not used in the Euler-angle version (kept for a possible switch to quaternions) --
def quat_to_R(quat):
    """quat = [w, x, y, z] (normalized inside)."""
    w, x, y, z = np.asarray(quat, float) / np.linalg.norm(quat)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]])


def quat_mul(a, b):
    """Hamilton product a (x) b, quaternions as [w, x, y, z]."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def rotvec_to_quat(phi):
    """Rotation vector (axis * angle) -> quaternion."""
    angle = np.linalg.norm(phi)
    if angle < 1e-12:
        return np.array([1.0, *(0.5 * np.asarray(phi))])
    axis = np.asarray(phi) / angle
    return np.array([np.cos(angle / 2), *(np.sin(angle / 2) * axis)])


# ---- URDF text helper ----------------------------------------------------------------
def _vec(text, default=(0.0, 0.0, 0.0)):
    """'0 0.1 0'  ->  np.array([0, 0.1, 0])"""
    return np.array([float(s) for s in text.split()]) if text else np.array(default, float)


# =====================================================================================
#  STEP 1 data containers: what we store per link and per joint
# =====================================================================================
@dataclass
class Link:
    name: str
    mass: float = 0.0                                                      # m_i
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))           # c_i, in link frame
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))  # I_i^body, about CoM, link axes


@dataclass
class Joint:
    name: str
    type: str             # 'revolute', 'continuous', 'prismatic', 'fixed'
    parent: str           # parent link name
    child: str            # child link name
    xyz: np.ndarray       # d_j      : joint position in the parent link frame (URDF <origin xyz>)
    R_origin: np.ndarray  # R_{o,j}  : joint frame rotation w.r.t. parent       (URDF <origin rpy>)
    axis: np.ndarray      # a_hat_j  : joint axis in the joint frame            (URDF <axis>)
    lower: float = -np.inf
    upper: float = np.inf
    effort: float = np.inf
    q_index: int = -1     # k: this joint's angle is q[6 + k];  -1 for fixed joints


# =====================================================================================
#  THE MODEL
# =====================================================================================
class RobotModel:

    # =================================================================================
    #  STEP 1 -- KINEMATIC TREE: read the URDF into links and joints rooted at the pelvis
    # =================================================================================
    def __init__(self, urdf_path):
        root = ET.parse(urdf_path).getroot()
        self.name = root.get("name")

        # (1a) links: mass, CoM and inertia from each <inertial>
        self.links = {}
        for el in root.findall("link"):
            link = Link(el.get("name"))
            ine = el.find("inertial")
            if ine is not None:                    # heel/toe markers have no <inertial>: mass stays 0
                o = ine.find("origin")
                if o is not None:
                    link.com = _vec(o.get("xyz"))
                    if o.get("rpy") and np.any(_vec(o.get("rpy"))):
                        raise NotImplementedError(f"rotated inertial frame in {link.name}")
                link.mass = float(ine.find("mass").get("value"))
                I = ine.find("inertia").attrib
                g = lambda k: float(I.get(k, 0.0))
                link.inertia = np.array([[g("ixx"), g("ixy"), g("ixz")],   # symmetric 3x3 tensor
                                         [g("ixy"), g("iyy"), g("iyz")],
                                         [g("ixz"), g("iyz"), g("izz")]])
            self.links[link.name] = link

        # (1b) joints: parent/child, fixed origin offset, axis, limits (unordered for now)
        joints = []
        for el in root.findall("joint"):
            o = el.find("origin")
            xyz = _vec(o.get("xyz")) if o is not None else np.zeros(3)
            rpy = _vec(o.get("rpy")) if o is not None else np.zeros(3)
            ax = el.find("axis")
            j = Joint(name=el.get("name"), type=el.get("type"),
                      parent=el.find("parent").get("link"),
                      child=el.find("child").get("link"),
                      xyz=xyz, R_origin=rpy_to_R(rpy),
                      axis=_vec(ax.get("xyz")) if ax is not None else np.array([1.0, 0, 0]))
            lim = el.find("limit")
            if lim is not None:
                j.lower = float(lim.get("lower", -np.inf))
                j.upper = float(lim.get("upper", np.inf))
                j.effort = float(lim.get("effort", np.inf))
            joints.append(j)

        # (1c) root link = the one that is never a child  ->  'pelvis'
        children = {j.child for j in joints}
        roots = [n for n in self.links if n not in children]
        assert len(roots) == 1, f"expected one root link, got {roots}"
        self.root = roots[0]

        # (1d) order joints parent-before-child (breadth-first from the root).
        #      Forward kinematics needs every parent frame before its children.
        self.joints, frontier = [], [self.root]
        while frontier:
            nxt = []
            for parent in frontier:
                for j in joints:
                    if j.parent == parent:
                        self.joints.append(j)
                        nxt.append(j.child)
            frontier = nxt

        # (1e) number the actuated joints in URDF file order: joint k  ->  q[6 + k]
        k = 0
        for j in joints:
            if j.type in ("revolute", "continuous", "prismatic"):
                j.q_index = k
                k += 1
        self.actuated = sorted([j for j in joints if j.q_index >= 0], key=lambda j: j.q_index)
        self.joint_names = [j.name for j in self.actuated]
        self.n_joints = k                          # n = 10
        self.nq = 6 + k                            # 6 base coordinates + n joints = 16
        self.nv = 6 + k                            # v = q_dot has the same size
        self.total_mass = sum(l.mass for l in self.links.values())   # m = 24.004 kg

        # (1f) ancestor joints of every link: the actuated joints on its path to the pelvis.
        #      Only these joints can move the link, so only they get Jacobian columns (Step 4).
        parent_joint = {j.child: j for j in self.joints}
        self.supporting_joints = {}
        for name in self.links:
            chain, cur = [], name
            while cur in parent_joint:             # walk up until we reach the pelvis
                j = parent_joint[cur]
                if j.q_index >= 0:                 # skip fixed joints (heel/toe)
                    chain.append(j)
                cur = j.parent
            self.supporting_joints[name] = chain[::-1]   # root -> link order

    # ---- configuration helpers ------------------------------------------------------
    def neutral_q(self, base_height=0.0):
        """All zeros (upright pelvis, straight legs) with the pelvis at the given height."""
        q = np.zeros(self.nq)
        q[2] = base_height
        return q

    def joint_index(self, name):
        """Index of a joint angle inside q, e.g. joint_index('left_knee_joint') -> 8."""
        return 6 + self.joint_names.index(name)

    # =================================================================================
    #  STEP 2 -- FORWARD KINEMATICS: world pose (R_i, o_i) of every link frame
    # =================================================================================
    def forward_kinematics(self, q):
        """Return {link_name: (R, o)}: world rotation and origin of every link frame.

        For each joint (parents first):
            R_joint = R_parent @ R_origin           o_joint = o_parent + R_parent @ d    (fixed URDF offset)
            R_child = R_joint  @ Rot(axis, q_j)     o_child = o_joint                     (joint motion)
        """
        q = np.asarray(q, float)
        # pelvis pose comes straight from q: orientation via Step 3, position q[0:3]
        frames = {self.root: (rpy_to_R(q[3:6]), q[0:3].copy())}
        for j in self.joints:                      # ordered parent-before-child (Step 1d)
            R_p, p_p = frames[j.parent]
            # fixed mounting offset from the URDF <origin>
            R_j = R_p @ j.R_origin
            p_j = p_p + R_p @ j.xyz                # offset is given in parent axes -> rotate it first
            # joint motion, multiplied on the RIGHT = rotation about the axis in the joint's own frame
            if j.type in ("revolute", "continuous"):
                R_j = R_j @ axis_angle_to_R(j.axis, q[6 + j.q_index])
            elif j.type == "prismatic":
                p_j = p_j + R_j @ (j.axis * q[6 + j.q_index])
            frames[j.child] = (R_j, p_j)           # fixed joints: child frame = joint frame
        return frames

    def frame_position(self, q, link_name, frames=None):
        """World position of a link frame origin (e.g. 'left_toe')."""
        frames = frames or self.forward_kinematics(q)
        return frames[link_name][1]

    def center_of_mass(self, q, frames=None):
        """p_CoM = (1/m) sum_i m_i (o_i + R_i c_i)."""
        frames = frames or self.forward_kinematics(q)
        s = np.zeros(3)
        for name, link in self.links.items():
            R, p = frames[name]
            s += link.mass * (p + R @ link.com)    # link CoM in world, weighted by mass
        return s / self.total_mass

    # =================================================================================
    #  STEP 3 -- BASE ORIENTATION: angular velocity, gimbal check, time stepping
    #  (rpy_to_R and euler_rate_matrix are at the top of the file)
    #
    #  Velocity convention (Euler-angle version): v = q_dot
    #    v[0:3] = pelvis linear velocity (world frame)
    #    v[3:6] = Euler-angle rates (roll_dot, pitch_dot, yaw_dot)  -- NOT omega
    #    v[6:]  = joint rates, same order as q[6:]
    # =================================================================================
    def base_angular_velocity(self, q, v):
        """Physical angular velocity of the pelvis (world frame): omega = E(theta) theta_dot."""
        return euler_rate_matrix(q[3:6]) @ v[3:6]

    def check_gimbal(self, q):
        """True if the pelvis pitch q[4] is near +-90 deg, where det E = cos(pitch) -> 0."""
        return abs(np.cos(q[4])) < np.cos(GIMBAL_WARN)

    def integrate(self, q, v, dt):
        """One time step.  Euler angles: q_dot = v, so it is a plain vector update."""
        q_new = np.asarray(q, float) + np.asarray(v, float) * dt
        if self.check_gimbal(q_new):
            import warnings
            warnings.warn(f"pelvis pitch = {np.rad2deg(q_new[4]):.1f} deg: "
                          "near gimbal lock (E(theta) singular at +-90 deg)", RuntimeWarning)
        return q_new

    # =================================================================================
    #  STEP 4 -- JACOBIANS:  [point velocity; link angular velocity] = J @ q_dot
    # =================================================================================
    def frame_jacobian(self, q, link_name, local_point=(0, 0, 0), frames=None):
        """6 x nv Jacobian of a point fixed on a link (world frame):
             J[0:3] = J_v : linear velocity of the point
             J[3:6] = J_w : angular velocity of the link
        local_point is in the link's own frame: link.com for the CoM, a sole corner for contact.

        Every column follows from  x_dot = o_dot + omega x (x - o).  Column k is the motion
        caused by q_dot[k] alone (all other entries zero)."""
        frames = frames or self.forward_kinematics(q)
        R, p_link = frames[link_name]
        p = p_link + R @ np.asarray(local_point, float)   # the point x, in world
        p_base = frames[self.root][1]                     # pelvis origin p_b

        J = np.zeros((6, self.nv))

        # columns 0-2, base translation: every point moves with the pelvis, nothing turns
        J[0:3, 0:3] = np.eye(3)

        # columns 3-5, base rotation:  omega = E(theta) theta_dot,
        #   point velocity = omega x (x - p_b) = -[x - p_b]x @ omega
        #   ->  J_v = -[x - p_b]x @ E,    J_w = E
        E = euler_rate_matrix(q[3:6])
        J[0:3, 3:6] = -skew(p - p_base) @ E
        J[3:6, 3:6] = E

        # columns 6+k, joints between the pelvis and this link (all others stay 0)
        for j in self.supporting_joints[link_name]:
            R_c, p_j = frames[j.child]        # joint frame = child link frame; its origin is the joint point p_j
            a = R_c @ j.axis                  # joint axis in world: a_j
            col = 6 + j.q_index
            if j.type in ("revolute", "continuous"):
                J[0:3, col] = np.cross(a, p - p_j)   # turning about a_j at p_j moves x by a_j x (x - p_j)
                J[3:6, col] = a                      # and adds a_j to the link's angular velocity
            else:                             # prismatic: slides along a_j, no rotation
                J[0:3, col] = a
        return J

    def frame_velocity(self, q, v, link_name, local_point=(0, 0, 0)):
        """(linear velocity of the point, angular velocity of the link) = J @ v."""
        Jv = self.frame_jacobian(q, link_name, local_point) @ v
        return Jv[0:3], Jv[3:6]

    def com_jacobian(self, q, frames=None):
        """J_CoM = (1/m) sum_i m_i J_v_i (at each link's CoM):  CoM velocity = J_CoM @ v."""
        frames = frames or self.forward_kinematics(q)
        Jc = np.zeros((3, self.nv))
        for name, link in self.links.items():
            if link.mass > 0:
                Jc += link.mass * self.frame_jacobian(q, name, link.com, frames)[0:3]
        return Jc / self.total_mass

    # =================================================================================
    #  STEP 5 -- MASS MATRIX:  T = 1/2 q_dot^T M(q) q_dot
    #
    #  Kinetic energy of link i = translation of its CoM + rotation about its CoM:
    #     T_i = 1/2 m_i v_i^T v_i + 1/2 w_i^T I_i^W w_i,   v_i = J_v_i q_dot,  w_i = J_w_i q_dot
    #  =>  M(q) = sum_i [ m_i J_v_i^T J_v_i  +  J_w_i^T I_i^W J_w_i ]
    # =================================================================================
    def link_world_inertia(self, link_name, frames):
        """I_i^W = R_i I_i^body R_i^T : inertia about the link CoM, in world axes.
        (The URDF gives it in link axes; omega is in world axes, so rotate it to match.)"""
        R = frames[link_name][0]
        return R @ self.links[link_name].inertia @ R.T

    def mass_matrix(self, q, frames=None):
        """M(q) = sum_i [ m_i Jv_i^T Jv_i + Jw_i^T I_i^W Jw_i ]   (guide, Step 5)."""
        frames = frames or self.forward_kinematics(q)      # forward kinematics once, reused below
        M = np.zeros((self.nv, self.nv))
        for name, link in self.links.items():
            if link.mass <= 0:
                continue                                   # heel/toe marker frames have no mass
            # Jacobian at the link's CENTER OF MASS (link.com), not at its frame origin
            J = self.frame_jacobian(q, name, link.com, frames)
            Jv, Jw = J[0:3], J[3:6]
            IW = self.link_world_inertia(name, frames)
            M += link.mass * Jv.T @ Jv + Jw.T @ IW @ Jw    # translation term + rotation term
        return 0.5 * (M + M.T)                             # remove round-off asymmetry

    def kinetic_energy(self, q, qd, frames=None):
        """T computed link by link, WITHOUT M. Used to test that 1/2 qd^T M qd gives the same."""
        frames = frames or self.forward_kinematics(q)
        T = 0.0
        for name, link in self.links.items():
            if link.mass <= 0:
                continue
            J = self.frame_jacobian(q, name, link.com, frames)
            v, w = J[0:3] @ qd, J[3:6] @ qd                # this link's CoM velocity and omega
            T += 0.5 * link.mass * v @ v + 0.5 * w @ self.link_world_inertia(name, frames) @ w
        return T

    # ---- printing ---------------------------------------------------------------------
    def summary(self):
        lines = [f"Robot '{self.name}': root link '{self.root}', "
                 f"{self.n_joints} actuated joints, nq={self.nq}, nv={self.nv}, "
                 f"total mass={self.total_mass:.3f} kg"]
        for j in self.actuated:
            lines.append(f"  q[{6 + j.q_index:2d}]  {j.name:24s} axis={j.axis}  "
                         f"limits=[{j.lower:+.2f}, {j.upper:+.2f}]")
        return "\n".join(lines)
