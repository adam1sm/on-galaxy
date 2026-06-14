#!/usr/bin/env python3
"""3D log-log time-vs-N: Direct3D O(N^2), BarnesHut3D O(N log N), FMM3D O(N).
Occupancy-aligned sweep (N=64*8^L) so FMM occupancy is constant per point.
Usage: python plot_scaling3d.py outputs/scaling3d.csv outputs/scaling3d.png
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def slope(N, t, nmin):
    m = np.isfinite(t) & (t > 0) & (N >= nmin)
    if m.sum() < 2: m = np.isfinite(t) & (t > 0)
    return np.polyfit(np.log(N[m]), np.log(t[m]), 1)[0]

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/scaling3d.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/scaling3d.png"
    d = np.genfromtxt(csv_path, delimiter=",", names=True)
    N = np.atleast_1d(d["N"]).astype(float)
    direct, bh, fmm = d["direct_sec"], d["bh_sec"], d["fmm_sec"]
    fig, ax = plt.subplots(figsize=(9, 7))
    # FMM slope from the deepest (most asymptotic) octree points.
    ax.loglog(N, direct, "o-", color="tab:red",   label=f"Direct3D O(N$^2$)  [slope {slope(N,direct,4000):.2f}]")
    ax.loglog(N, bh,     "s-", color="tab:blue",  label=f"Barnes-Hut3D O(N log N)  [slope {slope(N,bh,4000):.2f}]")
    ax.loglog(N, fmm,    "^-", color="tab:green", label=f"FMM3D O(N)  [slope {slope(N,fmm,30000):.2f}, deepest pts]")
    n0 = N[0]
    ax.loglog(N, direct[0]*(N/n0)**2, "--", color="tab:red", alpha=0.35, label="slope 2 ref")
    ax.loglog(N, fmm[1]*(N/N[1]), "--", color="tab:green", alpha=0.35, label="slope 1 ref")
    ax.set_xlabel("N (number of bodies)"); ax.set_ylabel("wall-clock per force evaluation (s)")
    ax.set_title("3D force-evaluation scaling: Direct vs Barnes-Hut vs FMM (uniform cube)")
    ax.grid(alpha=0.3, which="both"); ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); print(f"wrote {out_path}")

if __name__ == "__main__":
    main()
