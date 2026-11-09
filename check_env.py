"""
check_env.py  --  verify the Python environment before running the biped code.

Checks: Python version, required packages and their minimum versions, which environment
is active, and that the project files load (URDF + robot_model).

Run:   python check_env.py
"""
import os
import sys
import importlib

HERE = os.path.dirname(os.path.abspath(__file__))

REQUIRED = {"numpy": "1.24", "matplotlib": "3.7"}       # keep in sync with requirements.txt
OPTIONAL = {"scipy": "1.10", "pinocchio": "2.7"}        # used in later steps only
MIN_PYTHON = (3, 10)
EXPECTED_ENV = "bipedal_env"                            # name in environment.yml

ok = True


def version_tuple(v):
    """'2.5.3' -> (2, 5, 3); ignores suffixes like 'rc1'."""
    out = []
    for part in v.split(".")[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def line(status, name, detail):
    print(f"  [{status:4s}] {name:14s} {detail}")


# -------------------------------------------------------------------------------------
print("Python")
py = sys.version_info
if py[:2] >= MIN_PYTHON:
    line("OK", "python", f"{py.major}.{py.minor}.{py.micro}  ({sys.executable})")
else:
    line("FAIL", "python", f"{py.major}.{py.minor} found, need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")
    ok = False

# which environment is active (conda or venv)
conda_env = os.environ.get("CONDA_DEFAULT_ENV")
venv = os.environ.get("VIRTUAL_ENV")
if conda_env:
    status = "OK" if conda_env == EXPECTED_ENV else "WARN"
    line(status, "conda env", f"'{conda_env}'" + ("" if status == "OK" else
         f"  (expected '{EXPECTED_ENV}': conda activate {EXPECTED_ENV})"))
elif venv:
    line("OK", "venv", venv)
else:
    line("WARN", "environment", "no conda env or venv active -- using the system Python")

# -------------------------------------------------------------------------------------
print("\nRequired packages")
for name, minimum in REQUIRED.items():
    try:
        mod = importlib.import_module(name)
        v = mod.__version__
        if version_tuple(v) >= version_tuple(minimum):
            line("OK", name, v)
        else:
            line("FAIL", name, f"{v} found, need >= {minimum}  (pip install -U {name})")
            ok = False
    except ImportError:
        line("FAIL", name, f"not installed  (pip install -r requirements.txt)")
        ok = False

print("\nOptional packages (later steps)")
for name, minimum in OPTIONAL.items():
    try:
        mod = importlib.import_module(name)
        line("OK", name, getattr(mod, "__version__", "installed"))
    except ImportError:
        line("--", name, "not installed (not needed yet)")

# -------------------------------------------------------------------------------------
print("\nProject files")
urdf = os.path.join(HERE, "biped_simple_v1_2.urdf")
if os.path.exists(urdf):
    line("OK", "URDF", os.path.basename(urdf))
else:
    line("FAIL", "URDF", f"{os.path.basename(urdf)} not found next to this script")
    ok = False

if ok:
    try:
        sys.path.insert(0, HERE)
        from robot_model import RobotModel
        m = RobotModel(urdf)
        good = (m.nq == 16 and m.n_joints == 10 and abs(m.total_mass - 24.004) < 1e-9)
        line("OK" if good else "FAIL", "robot_model",
             f"loads: {m.n_joints} joints, nq = {m.nq}, mass = {m.total_mass:.3f} kg")
        ok = ok and good
    except Exception as e:                              # show the real error, do not hide it
        line("FAIL", "robot_model", f"{type(e).__name__}: {e}")
        ok = False

print("\nENVIRONMENT OK" if ok else "\nENVIRONMENT HAS PROBLEMS (see FAIL lines above)")
sys.exit(0 if ok else 1)
