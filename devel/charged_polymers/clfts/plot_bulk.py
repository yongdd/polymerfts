"""Bulk CL-FTS of the 3:1 salt against one loop on the same grid (references/charged_polymers/brush_clfts_bulk/*.json,
outside git): mean of the 1e4-step intervals
after the first, standard error of the interval means, and the two halves.
usage: python plot_bulk.py out.png"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {0.25: "#eb6834", 0.44: "#1baf7a", 0.66: "#2a78d6"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
rows = []
RUNS = Path(__file__).resolve().parents[3] / "references/charged_polymers/brush_clfts_bulk"
for f in sorted(glob.glob(str(RUNS / "bulk_z3_*.json"))):
    d = json.load(open(f))
    iv = np.array(d["mu_ex_per_interval"])[1:, 0]
    a = round(d["cation_aW_nm"], 2)
    failed = d["status"] != "completed"
    rows.append((a, d["molar"], d["one_loop"][0], iv.mean(), iv.std(ddof=1) / np.sqrt(len(iv)),
                 iv[: len(iv) // 2].mean(), iv[len(iv) // 2:].mean(), failed, d["status"]))
    print(f"a_W {a * 10:4.1f} A {d['molar']:6.3f} M: one loop {d['one_loop'][0]:6.3f}, CL-FTS {iv.mean():7.3f} +- {iv.std(ddof=1) / np.sqrt(len(iv)):.3f}"
          f" (halves {rows[-1][5]:.3f} / {rows[-1][6]:.3f}){'  ' + d['status'] if failed else ''}")
fig, ax = plt.subplots(figsize=(6.2, 4.2))
lims = [-16, 11]
ax.plot(lims, lims, color=MUTED, lw=1)
for a, molar, one, mean, se, h1, h2, failed, _ in rows:
    ax.errorbar(one, mean, se, fmt="s" if failed else "o", ms=7, color=COLORS[a], mfc="white" if failed else COLORS[a],
                mec=COLORS[a], mew=1.6, capsize=3)
    offset = (6, 6) if (a, molar) == (0.44, 0.03) else (6, -10)
    ax.annotate(f"{molar * 1e3:g} mM", (one, mean), textcoords="offset points", xytext=offset, fontsize=7.5, color=MUTED)
for a, color in COLORS.items():
    ax.plot([], [], "o", color=color, label=f"Y$^{{3+}}$ $a_W$ = {a * 10:.1f} Å")
ax.plot([], [], "s", mfc="white", mec=MUTED, label="run diverged before 10$^5$ steps")
ax.set_xlim(0, 11); ax.set_ylim(*lims)
ax.set_xlabel(r"one loop on the grid, $\beta\mu_{\rm ex}$ of Y$^{3+}$")
ax.set_ylabel(r"CL-FTS, $\beta\mu_{\rm ex}$ of Y$^{3+}$")
ax.grid(True, color=GRID, lw=0.6)
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=150)
