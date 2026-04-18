#!/usr/bin/env python3
"""Plot Barnes-Hut acceleration error vs opening angle theta.

Usage:
    python plot_error_vs_theta.py outputs/bh_error_vs_theta.csv outputs/bh_error_vs_theta.png
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/bh_error_vs_theta.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/bh_error_vs_theta.png"

    d = np.genfromtxt(csv_path, delimiter=",", names=True)
    theta = np.atleast_1d(d["theta"])
    max_rel = np.atleast_1d(d["max_rel_err"])
    rms_rel = np.atleast_1d(d["rms_rel_err"])

    # Floor the theta=0 roundoff point so it is visible on a log axis.
    floor = 1e-16
    max_plot = np.maximum(max_rel, floor)
    rms_plot = np.maximum(rms_rel, floor)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(theta, max_plot, "o-", color="tab:red", label="max relative error")
    ax.semilogy(theta, rms_plot, "s-", color="tab:blue", label="RMS relative error")
    ax.set_xlabel(r"opening angle $\theta$")
    ax.set_ylabel("relative acceleration error")
    ax.set_title("Barnes-Hut error shrinks as theta shrinks\n"
                 r"($\theta=0$ reduces to direct summation: roundoff)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    # Annotate the theta=0 correctness gate.
    i0 = int(np.argmin(theta))
    ax.annotate(f"theta=0 gate\nmax={max_rel[i0]:.1e}",
                xy=(theta[i0], max_plot[i0]),
                xytext=(0.15, 0.25), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="gray"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
