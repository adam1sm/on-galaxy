#!/usr/bin/env python3
# Rung 12 merger diagnostics figure. Three stacked panels vs time:
#   (1) COM separation (mass-weighted, from <base>_sep.csv) -- saturates at the
#       tidal-tail bias level after the cores merge.
#   (2) core separation (median + dense-core, from <base>_coresep.csv) -- the
#       tail-robust coalescence proof: -> ~eps when the nuclei merge.
#   (3) relative energy drift |dE|/|E0| (from <base>_energy.csv) -- bounded,
#       not climbing.
import sys
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else base + "_diag.png"


def load(path, cols):
    rows = list(csv.DictReader(open(path)))
    return [[float(r[c]) for r in rows] for c in cols]


fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=False)

t_s, sep = load(base + "_sep.csv", ["time", "sep"])
ax[0].plot(t_s, sep, color="#cc4444")
ax[0].set_ylabel("COM separation")
ax[0].set_title("mass-weighted COM separation (saturates on tidal-tail bias)")
ax[0].axhline(2.0, ls="--", c="gray", lw=0.8)

# core sep is indexed by frame; map frame -> time via the frame stride implied by
# the sep range (frames span the same total time t_end).
fr, msep, csep = load(base + "_coresep.csv", ["frame", "median_sep", "dense_core_sep"])
tend = t_s[-1]
tc = [f / fr[-1] * tend for f in fr]
ax[1].plot(tc, msep, label="median-core sep", color="#3366cc")
ax[1].plot(tc, csep, label="dense-core sep", color="#22aa66")
ax[1].axhline(0.05, ls="--", c="gray", lw=0.8, label="eps=0.05")
ax[1].set_ylabel("core separation")
ax[1].set_title("tail-robust core separation (coalescence proof)")
ax[1].legend()

t_e, drift = load(base + "_energy.csv", ["time", "relDrift"])
ax[2].plot(t_e, [d * 100 for d in drift], color="#8844cc")
ax[2].set_ylabel("|dE|/|E0|  (%)")
ax[2].set_xlabel("time")
ax[2].set_title(f"energy drift (max {max(drift)*100:.2f}%)")

plt.tight_layout()
plt.savefig(out, dpi=110)
print("wrote", out)
