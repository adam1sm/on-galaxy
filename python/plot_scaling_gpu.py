#!/usr/bin/env python3
"""GPU scaling (Rung 7): GPU-direct O(N^2) vs GPU-FMM O(N), fp32, log-log.
Usage: python plot_scaling_gpu.py outputs/fmm_gpu_scaling.csv outputs/scaling_gpu.png
CSV: N,gpu_direct_sec,fmm_compute_sec,fmm_tree_sec,fmm_total_sec
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def slope(N, t):
    m = np.isfinite(t) & (t > 0)
    return np.polyfit(np.log(N[m]), np.log(t[m]), 1)[0]


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else "outputs/fmm_gpu_scaling.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "outputs/scaling_gpu.png"
    d = np.genfromtxt(csv, delimiter=",", names=True)
    N = np.atleast_1d(d["N"]).astype(float)
    direct = np.atleast_1d(d["gpu_direct_sec"])
    total = np.atleast_1d(d["fmm_total_sec"])
    compute = np.atleast_1d(d["fmm_compute_sec"])
    tree = np.atleast_1d(d["fmm_tree_sec"])

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.loglog(N, direct, "o-", color="tab:red", label=f"GPU-direct O(N$^2$)  [slope {slope(N,direct):.2f}]")
    ax.loglog(N, total, "s-", color="tab:green", label=f"GPU-FMM total  [slope {slope(N,total):.2f}]")
    ax.loglog(N, compute, "^--", color="tab:blue", alpha=0.8, label="GPU-FMM compute only")
    ax.loglog(N, tree, "v:", color="tab:purple", alpha=0.7, label="CPU tree-build")

    faster = np.where(total < direct)[0]
    if faster.size:
        xc = N[faster[0]]
        ax.axvline(xc, color="gray", ls=":", alpha=0.6)
        ax.annotate(f"FMM < direct\nN = {int(xc)}", xy=(xc, total[faster[0]]),
                    xytext=(0.06, 0.80), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="gray"))

    ax.set_xlabel("N (number of bodies)")
    ax.set_ylabel("force-eval time (s)")
    ax.set_title("GPU scaling (RTX 3070, fp32, p=4): direct O(N$^2$) vs FMM O(N)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
