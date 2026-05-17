#!/usr/bin/env python3
"""Overlay Direct vs Barnes-Hut energy traces for the same disk run.

Usage:
    python plot_energy_compare.py <direct_diag.csv> <bh_diag.csv> <out.png>

Confirms that the Barnes-Hut approximation does NOT introduce secular energy
growth: both relative-drift curves should stay in the same bounded band.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    t = d["time"]
    E = d["total"]
    rel = np.abs(E - E[0]) / (abs(E[0]) if E[0] != 0 else 1.0)
    return t, E, rel


def main():
    direct_csv, bh_csv, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    td, Ed, rd = load(direct_csv)
    tb, Eb, rb = load(bh_csv)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(td, Ed, color="tab:red", lw=1.5, label="Direct  total E")
    axes[0].plot(tb, Eb, color="tab:blue", lw=1.2, ls="--", label="Barnes-Hut  total E")
    axes[0].set_ylabel("total energy")
    axes[0].set_title("Direct vs Barnes-Hut (theta=0.5): energy conservation")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    axes[1].plot(td, rd, color="tab:red", lw=1.5,
                 label=f"Direct  (max {rd.max():.2e})")
    axes[1].plot(tb, rb, color="tab:blue", lw=1.2, ls="--",
                 label=f"Barnes-Hut  (max {rb.max():.2e})")
    axes[1].set_ylabel(r"$|E-E_0|/|E_0|$")
    axes[1].set_xlabel("time")
    axes[1].set_title("relative energy drift (bounded, not growing)")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")
    print(f"Direct     max rel drift = {rd.max():.6e}")
    print(f"Barnes-Hut max rel drift = {rb.max():.6e}")


if __name__ == "__main__":
    main()
