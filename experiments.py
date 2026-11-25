from pathlib import Path
HERE = Path(__file__).resolve().parent
import json, time, itertools, platform
import numpy as np
from scipy import stats
from fis import *
from data import PTS, X, INCIDENTS, INCIDENTS_2026_ANNUALISED, INCIDENTS_2026_PARTIAL, N_MAX, NRI_KZ

R = {}
# ---------------- 1. Base case (Kazakhstan, Feb 2025) ----------------
x, nri, _ = X("KZ-2025")
N_BASE = INCIDENTS[2024]   # latest complete year before the Feb 2025 NCSI version
cs = round(baseline_cs(x), 3); comp = threat_components(x, nri, N_BASE, N_MAX); y1 = round(float(comp.mean()), 3)
val, mcs, my, act = mamdani(cs, y1, details=True)
R["base"] = dict(CS=cs, C=comp.tolist(), Y1=y1, mu_cs=mcs, mu_y=my,
                 act={f"{a}-{b}->{c}": w for (a, b, c), w in act.items()},
                 CSfinal=val, label=label(val), linear_mult=cs * (1 - y1),
                 linear_sub=max(cs - y1, 0), sugeno=sugeno(cs, y1), N=N_BASE, NMAX=N_MAX,
                 Y1_3comp=round(float(comp[:3].mean()), 3))

# ---------------- 2. Methods on all observations ----------------
obs = ["KZ-2025", "KZ-2026", "USA", "Germany", "Japan", "Estonia", "Uzbekistan", "Kyrgyzstan"]
Xm = np.array([X(o)[0] for o in obs]); NRI = np.array([X(o)[1] for o in obs])
NCSI = np.array([X(o)[2] for o in obs])

def entropy_w(M):
    P = (M + 1e-9) / (M + 1e-9).sum(0)
    e = -(P * np.log(P)).sum(0) / np.log(len(M))
    d = 1 - e; return d / d.sum()

def ahp_w(n):  # pairwise matrix from the same expert ranking, Saaty 1-9 scale
    r = np.arange(1, n + 1)
    A = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            s = 1 + (r[j] - r[i]) * 8 / (n - 1)
            A[i, j] = s if s >= 1 else 1 / (1 + (r[i] - r[j]) * 8 / (n - 1))
    ev, V = np.linalg.eig(A); k = np.argmax(ev.real); w = np.abs(V[:, k].real); w /= w.sum()
    ci = (ev.real[k] - n) / (n - 1); cr = ci / 1.54
    return w, cr

def topsis(M, w, benefit):
    N = M / np.sqrt((M ** 2).sum(0)); V = N * w
    best = np.where(benefit, V.max(0), V.min(0)); worst = np.where(benefit, V.min(0), V.max(0))
    dp = np.sqrt(((V - best) ** 2).sum(1)); dm = np.sqrt(((V - worst) ** 2).sum(1))
    return dm / (dp + dm)

w_f = fishburn(12); w_eq = np.ones(12) / 12; w_ahp, cr = ahp_w(12)
Y = np.round([threat(xx, n) for xx, n in zip(Xm, NRI)], 3)
CSf = np.round(Xm @ w_f, 3); CSe = Xm @ w_eq; CSa = Xm @ w_ahp
Ment = np.column_stack([Xm, Y]); w_ent = entropy_w(np.column_stack([Xm, 1 - Y]))
methods = {
    "Proposed (Fishburn + Mamdani)": [mamdani(c, y) for c, y in zip(CSf, Y)],
    "A1: equal weights + Mamdani (no Fishburn)": [mamdani(c, y) for c, y in zip(CSe, Y)],
    "A2: Fishburn + linear CS(1-Y1) (no fuzzy)": list(CSf * (1 - Y)),
    "A3: Fishburn + Mamdani, threat ignored (CS only)": list(CSf),
    "B1: linear subtractive CS-Y1": list(np.clip(CSf - Y, 0, 1)),
    "B2: AHP weights + linear CS(1-Y1)": list(CSa * (1 - Y)),
    "B3: Entropy-TOPSIS (12 indicators + Y1 as cost)": list(topsis(Ment, w_ent, np.r_[np.ones(12, bool), False])),
    "B4: Fishburn + Sugeno (zero-order TSK)": [sugeno(c, y) for c, y in zip(CSf, Y)],
}
ref = NCSI  # official NCSI scores: convergent-validity reference
tab = {}
for m, v in methods.items():
    v = np.array(v); tau, p = stats.kendalltau(v, ref); rho, pr = stats.spearmanr(v, ref)
    tab[m] = dict(values=dict(zip(obs, v.round(3).tolist())), tau=tau, p_tau=p, rho=rho, p_rho=pr,
                  spread=float(v.max() - v.min()), sd=float(v.std(ddof=1)))
R["methods"] = tab; R["ahp_cr"] = cr; R["Y_all"] = dict(zip(obs, Y.round(3).tolist()))
R["CS_all"] = dict(zip(obs, CSf.round(3).tolist()))
R["labels_all"] = {o: label(v) for o, v in zip(obs, methods["Proposed (Fishburn + Mamdani)"])}
R["ncsi"] = dict(zip(obs, NCSI.tolist()))

# ---------------- 3. Dynamic case ----------------
x2, nri2, _ = X("KZ-2026"); cs2 = round(baseline_cs(x2), 3)
# 3a annual series 2021-2026 (KZ-CERT incidents + NRI by edition; NCSI 3.0 capability data from Feb 2025 for 2021-2024)
xs25, _, _ = X("KZ-2025-11")
ann = []; prev = None
for yr in range(2021, 2027):
    if yr <= 2025:
        xx = x if yr <= 2024 else xs25; nn = NRI_KZ[yr]; ninc = INCIDENTS[yr]
    else:
        xx = x2; nn = NRI_KZ[2025]; ninc = INCIDENTS_2026_ANNUALISED
    comp_y = threat_components(xx, nn, ninc, N_MAX); c_ = round(baseline_cs(xx), 3); y_ = round(float(comp_y.mean()), 3)
    f_ = mamdani(c_, y_); dd = None if prev is None else round(y_ - prev, 3)
    ann.append(dict(year=yr, N=ninc, NRI=nn, C=[round(v, 3) for v in comp_y], CS=c_, Y1=y_, D=dd,
                    cls=None if dd is None else dynamics(y_, prev)[1], CSfinal=f_, label=label(f_),
                    CSI=f_ if dd is None else csi(f_, dd), linear=c_ * (1 - y_)))
    prev = y_
R["dyn_annual"] = ann
# entropy-weight check on complete years 2021-2025
Cm = np.array([r["C"] for r in ann[:5]])
P = Cm / Cm.sum(0); e = -(P * np.log(P)).sum(0) / np.log(len(Cm)); wE = (1 - e) / (1 - e).sum()
entr = []; prev = None
for r in ann:
    y_ = round(float(np.array(r["C"]) @ wE), 3); f_ = mamdani(r["CS"], y_); dd = None if prev is None else round(y_ - prev, 3)
    entr.append(dict(year=r["year"], Y1=y_, D=dd, CSfinal=f_, CSI=f_ if dd is None else csi(f_, dd))); prev = y_
R["dyn_entropy"] = dict(w=wE.tolist(), series=entr,
                        base=dict(Y1=round(float(comp @ wE), 3), CSf=mamdani(cs, round(float(comp @ wE), 3))))
# 3b NCSI-version series without incidents (C1-C3 only)
SEQ = [("KZ-2025", "14 Feb 2025"), ("KZ-2025-09", "15 Sep 2025"), ("KZ-2025-11", "30 Nov 2025"),
       ("KZ-2026-07", "6 Jul 2026"), ("KZ-2026-08", "14 Aug 2026"), ("KZ-2026", "3 Nov 2026")]
series = []; prev = None
for o, date in SEQ:
    xx, nn, sc_ = X(o); c_ = round(baseline_cs(xx), 3); y_ = round(threat(xx, nn), 3); f_ = mamdani(c_, y_)
    dd = None if prev is None else round(y_ - prev, 3)
    series.append(dict(obs=o, date=date, ncsi=sc_, CS=c_, Y1=y_, D=dd, CSfinal=f_)); prev = y_
R["dyn_series"] = series
# 3c stress scenario (synthetic): CS fixed at Nov 2026 level, starts from the real 2026 threat level
y26 = ann[-1]["Y1"]
scen_y = [y26, 0.56, 0.68, 0.85, 0.50]
rows = []
for t, yy in enumerate(scen_y):
    dd, cc = (0.0, "-") if t == 0 else dynamics(yy, scen_y[t - 1])
    dd = round(dd, 3); ff = mamdani(cs2, yy)
    rows.append(dict(t=t, Y1=yy, D=dd, cls=cc, CSfinal=ff, label=label(ff), CSI=csi(ff, dd), linear=cs2 * (1 - yy)))
R["dyn_scen"] = rows

# ---------------- 4. Sensitivity analysis (Kazakhstan 2025) ----------------
rng = np.random.default_rng(42)
# 4a Y1 weights: uniform over simplex
A = rng.dirichlet([1, 1, 1, 1], 10000)
ys = A @ comp; fs = np.array([mamdani(cs, v) for v in ys])
R["sa_y1w"] = dict(y_min=ys.min(), y_max=ys.max(), y_mean=ys.mean(), f_min=fs.min(), f_max=fs.max(),
                   f_mean=fs.mean(), f_sd=fs.std(), ci=np.percentile(fs, [2.5, 97.5]).tolist(),
                   p_high=float(np.mean([label(v) == "H" for v in fs])),
                   old_w=dict(Y1=round(float(np.array([.5, 1/6, 1/3]) @ comp[:3]), 3), CSf=mamdani(cs, round(float(np.array([.5, 1/6, 1/3]) @ comp[:3]), 3))),
                   reviewer_w=dict(Y1=round(float(np.array([.5, .3, .2]) @ comp[:3]), 3), CSf=mamdani(cs, round(float(np.array([.5, .3, .2]) @ comp[:3]), 3))),
                   no_c4=dict(Y1=round(float(comp[:3].mean()), 3), CSf=mamdani(cs, round(float(comp[:3].mean()), 3))))
# OAT: vary each alpha_k in [0.1,0.8], others rescaled proportionally
oat = {}
for k in range(4):
    out = []
    for a in np.linspace(0.05, 0.7, 14):
        w = np.ones(4) / 4; rest = np.delete(w, k); rest = rest / rest.sum() * (1 - a)
        w = np.insert(rest, k, a); yy = w @ comp; out.append((round(a, 2), yy, mamdani(cs, yy)))
    oat[f"alpha{k+1}"] = out
R["sa_oat"] = oat
ls = cs * (1 - ys)
R["sa_y1w"]["linear"] = dict(mean=ls.mean(), sd=ls.std(), cv=ls.std()/ls.mean(), ci=np.percentile(ls,[2.5,97.5]).tolist())
R["sa_y1w"]["cv"] = fs.std()/fs.mean()
# 4b MF breakpoints: each input breakpoint jittered U(-0.05, 0.05), monotone order kept
def jitter(mf, s):
    out = {}
    for k, p in mf.items():
        q = np.clip(np.array(p) + rng.uniform(-s, s, 4) * (np.array(p) % 1 != 0), 0, 1)
        out[k] = tuple(np.sort(q))
    return out
fm = []
for _ in range(3000):
    fm.append(mamdani(cs, y1, in_cs=jitter(IN_MF, .05), in_y=jitter(IN_MF, .05), out=jitter(OUT_MF, .05)))
fm = np.array(fm)
R["sa_mf"] = dict(mean=fm.mean(), sd=fm.std(), ci=np.percentile(fm, [2.5, 97.5]).tolist(),
                  min=fm.min(), max=fm.max(), p_high=float(np.mean([label(v) == "H" for v in fm])))
# 4c Fishburn ranking: random permutations vs top-3 fixed
perm = [baseline_cs(x, fishburn(12)[rng.permutation(12)]) for _ in range(5000)]
pf = np.array([mamdani(c, y1) for c in perm]); pl = np.array(perm) * (1 - y1)
R["sa_rank_cmp"] = dict(f_mean=pf.mean(), f_sd=pf.std(), f_cv=pf.std()/pf.mean(), l_mean=pl.mean(), l_sd=pl.std(), l_cv=pl.std()/pl.mean(),
    p_high=float(np.mean([label(v)=="H" for v in pf])))
R["sa_rank"] = dict(min=min(perm), max=max(perm), mean=float(np.mean(perm)),
                    ci=np.percentile(perm, [2.5, 97.5]).tolist(),
                    f_ci=[mamdani(c, y1) for c in np.percentile(perm, [2.5, 97.5])])
# 4d rule perturbation: each consequent shifted by +-1 level
rp = []
for key in RULES:
    for s in (-1, 1):
        i = OUT_ORDER.index(RULES[key]) + s
        if 0 <= i < 5:
            r2 = dict(RULES); r2[key] = OUT_ORDER[i]; rp.append((key, s, mamdani(cs, y1, rules=r2)))
R["sa_rules"] = [(f"{k[0]}-{k[1]}", s, v) for k, s, v in rp]

# ---------------- 5. Rule-base properties ----------------
grid = np.linspace(0, 1, 101)
S = np.array([[mamdani(a, b) for b in grid] for a in grid])
R["rb"] = dict(complete=bool(all(sum(min(fuzzify(a)[p], fuzzify(b)[q]) for p in LV for q in LV) > 0
                                  for a in grid for b in grid)),
               mono_cs=bool(np.all(np.diff(S, axis=0) >= -1e-9)),
               mono_y=bool(np.all(np.diff(S, axis=1) <= 1e-9)),
               fmin=S.min(), fmax=S.max())
np.save(HERE / "surface.npy", S)

# ---------------- 6. Run time ----------------
t0 = time.perf_counter(); N = 2000
for _ in range(N):
    c = baseline_cs(x); yy = threat(x, nri, n_inc=N_BASE, n_max=N_MAX); mamdani(c, yy)
R["timing_ms"] = (time.perf_counter() - t0) / N * 1000
R["env"] = dict(python=platform.python_version(), numpy=np.__version__, cpu=platform.processor() or platform.machine())

json.dump(R, open(HERE / "results.json", "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))[:9000])
