#!/usr/bin/env python3
"""Log-log wall-clock time vs N for each solver -- the hero scaling plot.

Usage:
    python plot_scaling.py outputs/scaling.csv outputs/scaling.png

DirectSolver should trace slope ~2 (O(N^2)); BarnesHutSolver should trace
~N log N (slope a bit above 1). Reference guide lines are drawn for both.
A future FMM line (O(N), slope ~1) drops in if an `fmm_sec` column is present.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fit_slope(N, t, n_min=4000):
    """Asymptotic slope of log t vs log N (least squares on the N>=n_min tail,
    where fixed per-call overhead no longer dominates)."""
    m = np.isfinite(t) & (t > 0) & (N >= n_min)
    if m.sum() < 2:  # fall back to all points if the tail is too short
        m = np.isfinite(t) & (t > 0)
    return np.polyfit(np.log(N[m]), np.log(t[m]), 1)[0]


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/scaling.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/scaling.png"

    d = np.genfromtxt(csv_path, delimiter=",", names=True)
    N = np.atleast_1d(d["N"]).astype(float)
    direct = np.atleast_1d(d["direct_sec"])
    bh = np.atleast_1d(d["bh_sec"])

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.loglog(N, direct, "o-", color="tab:red",
              label=f"Direct O(N$^2$)  [asymptotic slope {fit_slope(N, direct):.2f}]")
    ax.loglog(N, bh, "s-", color="tab:blue",
              label=f"Barnes-Hut O(N log N)  [asymptotic slope {fit_slope(N, bh):.2f}]")

    # Optional FMM line (Rung 2 seam): plotted automatically if the column exists.
    names = d.dtype.names if d.dtype.names else ()
    if "fmm_sec" in names:
        fmm = np.atleast_1d(d["fmm_sec"])
        ax.loglog(N, fmm, "^-", color="tab:green",
                  label=f"FMM O(N)  [asymptotic slope {fit_slope(N, fmm):.2f}]")

    # Reference guide lines anchored at the first N.
    n0, t0d, t0b = N[0], direct[0], bh[0]
    ax.loglog(N, t0d * (N / n0) ** 2, "--", color="tab:red", alpha=0.4,
              label="slope 2 reference")
    ax.loglog(N, t0b * (N / n0) * (np.log(N) / np.log(n0)), "--",
              color="tab:blue", alpha=0.4, label="N log N reference")

    # Mark the crossover N where BH becomes faster than Direct.
    faster = np.where(bh < direct)[0]
    if faster.size:
        xc = N[faster[0]]
        ax.axvline(xc, color="gray", ls=":", alpha=0.6)
        ax.annotate(f"BH < Direct\nN = {int(xc)}",
                    xy=(xc, bh[faster[0]]), xytext=(0.04, 0.78),
                    textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="gray"))

    # Mark the crossover N where FMM becomes faster than BH (the O(N) payoff).
    if "fmm_sec" in names:
        fmm = np.atleast_1d(d["fmm_sec"])
        ffaster = np.where(fmm < bh)[0]
        if ffaster.size:
            xf = N[ffaster[0]]
            ax.axvline(xf, color="tab:green", ls=":", alpha=0.6)
            ax.annotate(f"FMM < BH\nN = {int(xf)}",
                        xy=(xf, fmm[ffaster[0]]), xytext=(0.40, 0.12),
                        textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="tab:green"))

    ax.set_xlabel("N (number of bodies)")
    ax.set_ylabel("wall-clock per force evaluation (s)")
    ax.set_title("Force-evaluation scaling: Direct O(N$^2$) vs Barnes-Hut "
                 "O(N log N) vs FMM O(N)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
