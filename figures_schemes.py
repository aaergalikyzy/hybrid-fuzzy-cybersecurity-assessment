from pathlib import Path
HERE = Path(__file__).resolve().parent
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, INK2 = "#0b0b0b", "#52514e"
BLUE, BLUE_F = "#2a78d6", "#eef4fc"
GREEN, GREEN_F = "#1baf7a", "#e9f7f1"
PURP = "#6250d6"
NOTE_F = "#f6f6f4"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "savefig.dpi": 300, "mathtext.fontset": "dejavuserif"})
OUT = str(HERE / "fig") + "/"


def box(ax, x, y, w, h, title, sub=None, fc=BLUE_F, ec=BLUE, ls="-", fs=9, sfs=8.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                fc=fc, ec=ec, lw=1.1, ls=ls))
    if sub is None:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, weight="bold", color=INK)
    else:
        ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center", fontsize=fs, weight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.32, sub, ha="center", va="center", fontsize=sfs, color=INK, linespacing=1.25)


def arrow(ax, a, b, ls="-", color=INK2):
    ax.annotate("", b, a, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, ls=ls, mutation_scale=10))


# ---------------- Fig. 1: overall structure ----------------
steps = [
    ("1. Input indicators", "$X_1,\\dots,X_{12}$: NCSI 3.0 indicator groups",
     "Public data: NCSI country profile\n(capability indicators)"),
    ("2. Normalization", "$X_i$ = fulfilled points / maximum points, $X_i\\in[0,1]$",
     "Common scale for heterogeneous\nindicators"),
    ("3. Fishburn weights", "$\\beta_i = 2(n-i+1)/(n(n+1))$, Eq. (2)",
     "Ordinal ranking of indicators\n(Concept No. 269, Table 4)"),
    ("4. Baseline protection level", "$CS=\\Sigma_{i}\\,\\beta_i X_i$, Eq. (1)",
     "Internal readiness, without\nexternal threats"),
    ("5. Threat intensity", "$Y_1=(C_1+C_2+C_3+C_4)/4$, Eqs. (3)-(4)",
     "NRI, weakness of protective\nmechanisms, KZ-CERT incidents"),
    ("6. Fuzzification", "terms low / medium / high, Table 1",
     "Trapezoidal membership functions\n(strong uniform partition)"),
    ("7. Rule base and Mamdani inference", "9 rules (3 × 3), min activation, max aggregation",
     "Complete, consistent, monotone\nrule base (Table 2)"),
    ("8. Defuzzification", "centroid, Eqs. (8)-(9)",
     "Crisp value of the final\nassessment"),
    ("9. Final cybersecurity state", "$CS_{final}$ and its linguistic level",
     "very low / low / medium /\nhigh / very high"),
    ("10. Dynamics (if $Y_1(t-1)$ exists)", "$D(t)=Y_1(t)-Y_1(t-1)$,  $CSI=CS_{final}(1-|D(t)|)$",
     "Dynamics class (Table 3) and\nstability index, Eqs. (10)-(11)"),
]
H, G = 0.78, 0.30
n = len(steps)
top = n * (H + G)
fig = plt.figure(figsize=(6.3, 7.4)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 10); ax.set_ylim(-0.15, top + 0.05); ax.axis("off")
BX, BW, NX, NW = 0.05, 5.85, 6.6, 3.33
ys = []
for k, (t, s, note) in enumerate(steps):
    y = top - (k + 1) * (H + G) + G
    ys.append(y)
    if k == 8:
        fc, ec, ls = GREEN_F, GREEN, "-"
    elif k == 9:
        fc, ec, ls = BLUE_F, BLUE, "--"
    else:
        fc, ec, ls = BLUE_F, BLUE, "-"
    box(ax, BX, y, BW, H, t, s, fc=fc, ec=ec, ls=ls)
    ax.add_patch(FancyBboxPatch((NX, y + 0.06), NW, H - 0.12, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=NOTE_F, ec="#b9b8b3", lw=0.8))
    ax.text(NX + NW / 2, y + H / 2, note, ha="center", va="center", fontsize=8.2, color=INK, linespacing=1.25)
    x0 = BX + BW + (0.5 if 5 <= k <= 7 else 0.03)
    ax.plot([x0, NX - 0.03], [y + H / 2] * 2, ls=(0, (3, 2)), color="#8a8984", lw=0.9)
    if k > 0:
        arrow(ax, (BX + BW / 2, ys[k - 1]), (BX + BW / 2, y + H + 0.01), ls="--" if k == 9 else "-")
# Mamdani FIS frame (steps 6-8)
y_lo, y_hi = ys[7] - 0.09, ys[5] + H + 0.09
ax.add_patch(FancyBboxPatch((BX - 0.02, y_lo), BW + 0.04, y_hi - y_lo, boxstyle="round,pad=0.03,rounding_size=0.12",
                            fc="none", ec=PURP, lw=1.1, ls=(0, (4, 3))))
ax.text(BX + BW + 0.33, (y_lo + y_hi) / 2, "Mamdani FIS (Fig. 2)", rotation=90, ha="center", va="center",
        fontsize=8.2, color=PURP)
fig.savefig(OUT + "fig1_structure.png", bbox_inches="tight", pad_inches=0.03)
plt.close()

# ---------------- Fig. 2: Mamdani FIS ----------------
fig = plt.figure(figsize=(6.3, 3.2)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-0.15, 10.7); ax.set_ylim(0.1, 5.35); ax.axis("off")
# inputs
box(ax, -0.05, 3.0, 1.95, 1.75, "Input 1", "baseline level $CS$\n\nlow/medium/high", sfs=8)
box(ax, -0.05, 0.65, 1.95, 1.75, "Input 2", "threat intensity $Y_1$\n\nlow/medium/high", sfs=8)
# fuzzification
box(ax, 2.3, 1.55, 1.75, 2.3, "Fuzzification", "crisp inputs\n→ membership\ndegrees\n(Table 1)", sfs=8)
# FIS frame
ax.add_patch(FancyBboxPatch((4.4, 0.25), 2.9, 4.9, boxstyle="round,pad=0.03,rounding_size=0.15",
                            fc="none", ec=PURP, lw=1.1, ls=(0, (4, 3))))
ax.text(5.85, 5.0, "Mamdani inference", ha="center", va="center", fontsize=8.2, color=PURP)
box(ax, 4.6, 3.55, 2.5, 1.2, "Rule base", "9 IF-THEN rules (Table 2)", sfs=8)
box(ax, 4.6, 2.05, 2.5, 1.2, "Rule activation", "$w_r=\\min(\\mu_{A_r},\\mu_{B_r})$", sfs=8.3)
box(ax, 4.6, 0.55, 2.5, 1.2, "Aggregation", "$\\mu_{out}=\\max\\,[\\min(w_r,\\mu_{C_r})]$", sfs=8.3)
# defuzzification / output
box(ax, 7.65, 1.55, 1.35, 2.3, "Defuzzi-\nfication", None, fs=9)
ax.text(8.325, 1.95, "centroid,\nEqs. (8)-(9)", ha="center", va="center", fontsize=8, color=INK)
box(ax, 9.3, 1.55, 1.3, 2.3, "Output", "$CS_{final}$\n\nvery low ...\nvery high", fc=GREEN_F, ec=GREEN, sfs=8)
# arrows
ax.annotate("", (2.27, 3.25), (1.87, 3.85), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1, mutation_scale=10))
ax.annotate("", (2.27, 2.15), (1.87, 1.55), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1, mutation_scale=10))
arrow(ax, (4.07, 2.65), (4.57, 2.65))
arrow(ax, (5.85, 3.55), (5.85, 3.27))
arrow(ax, (5.85, 2.05), (5.85, 1.77))
ax.plot([7.12, 7.4, 7.4], [1.15, 1.15, 2.3], color=INK2, lw=1.1)
arrow(ax, (7.4, 2.3), (7.62, 2.3))
arrow(ax, (9.02, 2.7), (9.27, 2.7))
fig.savefig(OUT + "fig2_mamdani.png", bbox_inches="tight", pad_inches=0.03)
plt.close()
print("ok")
