from pathlib import Path
HERE = Path(__file__).resolve().parent
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from fis import *

R = json.load(open(HERE / "results.json"))
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
                     "legend.frameon": False})
OUT = str(HERE / "fig") + "/"
import os; os.makedirs(OUT, exist_ok=True)
NAMES = {"VL": "Very low", "L": "Low", "M": "Medium", "H": "High", "VH": "Very high"}

def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6); ax.set_axisbelow(True)

# Fig 3: input MFs (CS and Y1 share the partition)
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
for ax, var, pt in zip(axs, ["CS", "$Y_1$"], [0.785, 0.386]):
    for i, (k, p) in enumerate(IN_MF.items()):
        ax.plot(Z, trap(Z, *p), lw=2, color=C[i], label=NAMES[k])
        ax.text((p[1] + p[2]) / 2, 1.05, NAMES[k], ha="center", color=INK, fontsize=8)
    ax.axvline(pt, color=INK2, ls="--", lw=1)
    ax.text(pt + 0.01, 0.55, f"{var} = {pt}", fontsize=8, color=INK)
    ax.set_xlabel(var); ax.set_xlim(0, 1); ax.set_ylim(0, 1.15); grid(ax)
axs[0].set_ylabel("Membership degree")
fig.tight_layout(); fig.savefig(OUT + "fig3_input_mf.png"); plt.close()

# Fig 4: output MFs with the aggregated output for Kazakhstan
fig, ax = plt.subplots(figsize=(6.0, 3.0))
for i, k in enumerate(OUT_ORDER):
    p = OUT_MF[k]; ax.plot(Z, trap(Z, *p), lw=1.6, color=C[i])
    ax.text((p[1] + p[2]) / 2 if k not in ("VL", "VH") else (0.05 if k == "VL" else 0.95), 1.05,
            NAMES[k], ha="center", fontsize=8, color=INK)
agg = np.zeros_like(Z)
for kk, ww in R["base"]["act"].items():
    agg = np.maximum(agg, np.minimum(ww, trap(Z, *OUT_MF[kk.split("->")[1]])))
ax.fill_between(Z, agg, color="#86b6ef", alpha=0.6, lw=0, label="Aggregated output, Kazakhstan 2025")
v = R["base"]["CSfinal"]; ax.axvline(v, color=INK, lw=1.2, ls="--")
ax.text(v + 0.012, 0.5, f"centroid\n$CS_{{final}}$ = {v:.3f}", fontsize=8, bbox=dict(fc="white", ec="none", pad=1))
ax.set_xlabel("$CS_{final}$"); ax.set_ylabel("Membership degree"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.15)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8); grid(ax)
fig.tight_layout(); fig.savefig(OUT + "fig4_output_mf.png"); plt.close()

# Fig 5: control surface + rule table
S = np.load(HERE / "surface.npy")
fig, ax = plt.subplots(figsize=(4.6, 3.7))
im = ax.imshow(S, origin="lower", extent=[0, 1, 0, 1], cmap="Blues", vmin=0, vmax=1, aspect="auto")
cs_ = ax.contour(np.linspace(0, 1, 101), np.linspace(0, 1, 101), S, levels=[0.3, 0.5, 0.7, 0.85],
                 colors=INK2, linewidths=0.6)
ax.clabel(cs_, fontsize=8, fmt="%.2f")
pts = {k: (R["Y_all"][k], R["CS_all"][k]) for k in R["Y_all"]}
for k, (y, c) in pts.items():
    ax.plot(y, c, "o", ms=5, mfc="white", mec=INK, mew=1)
    if k in ("KZ-2025", "Uzbekistan", "Estonia", "USA"):
        ax.annotate(k, (y, c), xytext=(4, -9 if k == "KZ-2025" else 4), textcoords="offset points", fontsize=8)
ax.set_xlabel("Threat intensity $Y_1$"); ax.set_ylabel("Baseline level CS")
cb = fig.colorbar(im, ax=ax); cb.set_label("$CS_{final}$"); ax.spines[["top", "right"]].set_visible(True)
fig.tight_layout(); fig.savefig(OUT + "fig5_surface.png"); plt.close()

# Fig 6: D(t) flowchart (redrawn, consistent with Eqs.)
fig, ax = plt.subplots(figsize=(6.3, 3.0)); ax.axis("off"); ax.set_xlim(-0.1, 10.6); ax.set_ylim(0, 4)
boxes = [(0.0, 2.3, "1. Data at time t:\nNCSI, NRI,\nincidents"),
         (2.65, 2.3, "2. Threat intensity\n$Y_1(t)$,\nEqs. (3)-(4)"),
         (5.3, 2.3, "3. Coefficient\n$D(t)=Y_1(t)$\n$-\\,Y_1(t-1)$"),
         (7.95, 2.3, "4. Dynamics class\nfrom |D(t)|,\nTable 3"),
         (2.65, 0.2, "5. Mamdani FIS\n$CS_{final}(t)$,\nEqs. (5)-(9)"),
         (5.3, 0.2, "6. Stability index\nCSI(t) =\n$CS_{final}(t)(1-|D(t)|)$"),
         (7.95, 0.2, "7. Report: $CS_{final}$,\nCSI, linguistic\nlevel, trend sign")]
for x, y, t in boxes:
    ax.add_patch(FancyBboxPatch((x, y), 2.45, 1.4, boxstyle="round,pad=0.03,rounding_size=0.12",
                                fc="#eef4fc", ec="#2a78d6", lw=1))
    ax.text(x + 1.225, y + 0.7, t, ha="center", va="center", fontsize=8, color=INK)
arr = dict(arrowstyle="->", color=INK2, lw=1)
for a, b in [((2.48, 3.0), (2.62, 3.0)), ((5.13, 3.0), (5.27, 3.0)), ((7.78, 3.0), (7.92, 3.0)),
             ((3.87, 2.3), (3.87, 1.63)), ((5.13, 0.9), (5.27, 0.9)), ((7.78, 0.9), (7.92, 0.9)), ((6.52, 2.3), (6.52, 1.63))]:
    ax.annotate("", b, a, arrowprops=arr)
fig.savefig(OUT + "fig6_dt_flow.png", bbox_inches="tight"); plt.close()

# Fig 8: multi-country comparison
obs = list(R["Y_all"]); x = np.arange(len(obs)); w = 0.26
fig, ax = plt.subplots(figsize=(6.3, 3.2))
series = [("CS", [R["CS_all"][o] for o in obs]), ("$Y_1$", [R["Y_all"][o] for o in obs]),
          ("$CS_{final}$ (proposed)", [R["methods"]["Proposed (Fishburn + Mamdani)"]["values"][o] for o in obs])]
for i, (n, v) in enumerate(series):
    ax.bar(x + (i - 1) * w, v, w - 0.03, color=C[i], label=n)
for xi, o in zip(x, obs):
    ax.text(xi + w, R["methods"]["Proposed (Fishburn + Mamdani)"]["values"][o] + 0.02,
            NAMES[R["labels_all"][o]], ha="center", fontsize=8, rotation=90, va="bottom", color=INK2)
ax.set_xticks(x); ax.set_xticklabels(obs, fontsize=8); ax.set_ylim(0, 1.15); ax.set_ylabel("Value")
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8); grid(ax)
fig.tight_layout(); fig.savefig(OUT + "fig8_countries.png"); plt.close()

# Fig 9: sensitivity
fig, axs = plt.subplots(1, 2, figsize=(6.3, 3.0))
for i, (k, lab) in enumerate([("alpha1", r"$\alpha_1$ (C$_1$)"), ("alpha2", r"$\alpha_2$ (C$_2$)"), ("alpha3", r"$\alpha_3$ (C$_3$)"), ("alpha4", r"$\alpha_4$ (C$_4$)")]):
    a = np.array(R["sa_oat"][k]); axs[0].plot(a[:, 0], a[:, 2], "-o", ms=3.5, lw=2, color=C[i], label=lab)
axs[0].axhspan(0.66, 0.78, color="#eef4fc", zorder=0); axs[0].text(0.06, 0.765, "core of 'High'", fontsize=8, color=INK2)
axs[0].set_xlabel("Weight value (others rescaled)"); axs[0].set_ylabel("$CS_{final}$"); axs[0].set_ylim(0.6, 0.85)
axs[0].legend(fontsize=8, loc="lower right"); grid(axs[0])
rng = np.random.default_rng(42); A = rng.dirichlet([1, 1, 1, 1], 10000); comp = np.array(R["base"]["C"])
ys = A @ comp; cs = R["base"]["CS"]
fs = np.array([mamdani(cs, v) for v in ys]); ls = cs * (1 - ys)
bins = np.linspace(0.35, 0.85, 51)
axs[1].hist(ls, bins, color=C[1], alpha=0.85, label="Linear CS(1-$Y_1$)")
axs[1].hist(fs, bins, color=C[0], alpha=0.85, label="Proposed (Mamdani)")
axs[1].set_xlabel("Assessment under random $Y_1$ weights"); axs[1].set_ylabel("Frequency")
axs[1].legend(fontsize=8); grid(axs[1])
fig.tight_layout(); fig.savefig(OUT + "fig9_sensitivity.png"); plt.close()

# Fig 10: (a) real NCSI series for Kazakhstan, (b) stress scenario
d = R["dyn_scen"]
fig, axs = plt.subplots(1, 2, figsize=(6.3, 3.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
ann = R["dyn_annual"]; ax = axs[0]; tt = range(len(ann))
for i, (k, lab) in enumerate([("CS", "CS"), ("Y1", "$Y_1$"), ("CSfinal", "$CS_{final}$"), ("CSI", "CSI")]):
    ax.plot(tt, [r[k] for r in ann], marker="o", ls="-" if k != "CSI" else "--", lw=2, ms=4,
            color=C[i] if k != "CSI" else C[3], label=lab)
ax.plot(tt, [r["C"][3] for r in ann], marker="s", ls=":", lw=1.2, ms=3, color=C[4], label="$C_4$ (incidents)")
for t_, r in zip(tt, ann):
    if r["D"] is not None: ax.text(t_, 0.03, f"{r['D']:+.3f}\n{r['cls']}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(list(tt)); ax.set_xticklabels([str(r["year"]) + ("*" if r["year"] == 2026 else "") for r in ann], fontsize=8)
ax.set_ylim(0, 1.25); ax.set_ylabel("Value"); ax.set_title("(a) Kazakhstan, 2021-2026*", fontsize=8.5)
ax.legend(ncol=3, fontsize=8, loc="upper left", columnspacing=0.8, handlelength=1.5); grid(ax)
ax = axs[1]; t = [r["t"] for r in d]
for i, (k, lab) in enumerate([("Y1", "$Y_1(t)$"), ("CSfinal", "$CS_{final}(t)$"), ("CSI", "CSI(t)"), ("linear", "Linear")]):
    ax.plot(t, [r[k] for r in d], "-o", lw=2, ms=3.5, color=[C[1], C[2], C[3], C[4]][i], label=lab)
for r in d[1:]:
    ax.text(r["t"], 0.02, f"{r['D']:+.2f}\n" + {"moderate": "mod.", "critical": "crit."}.get(r['cls'], r['cls']), ha="center", fontsize=8, color=INK2)
ax.set_xticks(t); ax.set_xticklabels([f"t{i}" for i in t]); ax.set_ylim(0, 1.25)
ax.set_title("(b) Stress scenario (t0 = 2026*)", fontsize=8.5)
ax.legend(ncol=2, fontsize=8, loc="upper left"); grid(ax)
fig.tight_layout(); fig.savefig(OUT + "fig10_dynamic.png"); plt.close()

# Fig 7: normalized indicators, Fishburn weights and contributions (Kazakhstan, Feb 2025)
from data import X as _X
xk = np.array(_X("KZ-2025")[0]); bw = fishburn(12)
short = ["CII", "IR", "Policy", "R&D", "Educ.", "Crisis", "Crime", "Aware.", "Pers.\ndata", "Military", "Coop.", "Enablers"]
fig, ax = plt.subplots(figsize=(6.3, 3.0)); xx = np.arange(12); w = 0.27
for i, (v, lab) in enumerate([(xk, "Normalized value $X_i$"), (bw, r"Fishburn weight $\beta_i$"), (bw * xk, r"Contribution $\beta_i X_i$")]):
    ax.bar(xx + (i - 1) * w, v, w - 0.03, color=C[i], label=lab)
ax.set_xticks(xx); ax.set_xticklabels(short, fontsize=8); ax.set_ylim(0, 1.2); ax.set_ylabel("Value")
ax.set_xlabel("NCSI indicator group (in order of Fishburn rank)")
ax.legend(ncol=3, loc="upper center", fontsize=8); grid(ax)
fig.tight_layout(); fig.savefig(OUT + "fig7_contributions.png"); plt.close()
print("ok")
