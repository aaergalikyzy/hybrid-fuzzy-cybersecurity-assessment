"""Hybrid fuzzy inference model for national cybersecurity state assessment.
Reference implementation (Python 3, NumPy). Deterministic; no random components
except the Monte Carlo sensitivity analysis, which uses a fixed seed.
"""
import numpy as np

STEP = 0.001                       # discretisation step of the output universe
Z = np.round(np.arange(0, 1 + STEP / 2, STEP), 6)   # 1001 points on [0, 1]

GROUPS = ["CII", "IR", "Policy", "R&D", "Education", "Crisis", "Cybercrime",
          "Awareness", "PersonalData", "Military", "Cooperation", "Enablers"]
# Fishburn rank order (1 = most important) = order of GROUPS above

def fishburn(n):
    return np.array([2 * (n - i + 1) / (n * (n + 1)) for i in range(1, n + 1)])

def trap(x, a, b, c, d):
    x = np.asarray(x, float)
    y = np.where((x >= b) & (x <= c), 1.0, 0.0)
    if b > a:
        y = np.where((x > a) & (x < b), (x - a) / (b - a), y)
    if d > c:
        y = np.where((x > c) & (x < d), (d - x) / (d - c), y)
    return y

# ---- membership functions (strong uniform fuzzy partitions) ----
IN_MF = {"L": (0.0, 0.0, 0.2, 0.4), "M": (0.2, 0.4, 0.6, 0.8), "H": (0.6, 0.8, 1.0, 1.0)}
OUT_MF = {"VL": (0.0, 0.0, 0.12, 0.22), "L": (0.12, 0.22, 0.34, 0.44),
          "M": (0.34, 0.44, 0.56, 0.66), "H": (0.56, 0.66, 0.78, 0.88),
          "VH": (0.78, 0.88, 1.0, 1.0)}
OUT_ORDER = ["VL", "L", "M", "H", "VH"]
# rule base: consequent index = idx(CS) - idx(Y1) + 2  (complete, consistent, monotone)
LV = ["L", "M", "H"]
RULES = {(cs, y): OUT_ORDER[LV.index(cs) - LV.index(y) + 2] for cs in LV for y in LV}

def fuzzify(x, mf=IN_MF):
    return {k: float(trap(x, *p)) for k, p in mf.items()}

def mamdani(cs, y1, in_cs=IN_MF, in_y=IN_MF, out=OUT_MF, rules=RULES, details=False):
    mcs, my = fuzzify(cs, in_cs), fuzzify(y1, in_y)
    agg = np.zeros_like(Z)
    act = {}
    for (a, b), c in rules.items():
        w = min(mcs[a], my[b])                     # AND = min
        if w > 0:
            act[(a, b, c)] = w
            agg = np.maximum(agg, np.minimum(w, trap(Z, *out[c])))  # implication min, aggregation max
    val = float((Z * agg).sum() / agg.sum())       # discrete centroid
    return (val, mcs, my, act) if details else val

def sugeno(cs, y1, in_cs=IN_MF, in_y=IN_MF, rules=RULES):
    centres = {k: (p[1] + p[2]) / 2 for k, p in OUT_MF.items()}
    centres["VL"], centres["VH"] = 0.06, 0.94
    mcs, my = fuzzify(cs, in_cs), fuzzify(y1, in_y)
    num = den = 0.0
    for (a, b), c in rules.items():
        w = min(mcs[a], my[b]); num += w * centres[c]; den += w
    return num / den

def label(v):
    m = {k: float(trap(v, *p)) for k, p in OUT_MF.items()}
    return max(m, key=m.get)

def baseline_cs(x, w=None):
    x = np.asarray(x, float)
    w = fishburn(len(x)) if w is None else np.asarray(w)
    return float(w @ x)

def threat_components(x, nri, n_inc=None, n_max=None):
    """C1 exposure (1-NRI), C2 weakness of protection, C3 weakness of incident/crisis management,
    C4 incident intensity N/N_max (only if incident statistics are available)."""
    d = dict(zip(GROUPS, x))
    c = [1 - nri, 1 - (d["CII"] + d["IR"] + d["Enablers"]) / 3,
         ((1 - d["IR"]) + (1 - d["Crisis"]) + (1 - d["Cybercrime"])) / 3]
    if n_inc is not None:
        c.append(min(1.0, n_inc / n_max))
    return np.array(c)

def threat(x, nri, a=None, n_inc=None, n_max=None):
    c = threat_components(x, nri, n_inc, n_max)
    a = np.ones(len(c)) / len(c) if a is None else np.asarray(a, float)   # equal weights by default
    return float(a @ c)

def dynamics(y_now, y_prev):
    d = y_now - y_prev
    m = abs(d)
    cls = "low" if m < 0.1 else "moderate" if m < 0.3 else "high" if m < 0.6 else "critical"
    return d, cls

def csi(cs_final, d):
    return cs_final * (1 - abs(d))
