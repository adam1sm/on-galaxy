#!/usr/bin/env python3
"""Gate E payoff: energy drift of Direct / BH / FMM(p) on one set of axes.

Usage:
    python plot_energy_vs_p.py <out.png> <label>=<diag.csv> [<label>=<diag.csv> ...]

Drift is normalized by the initial |PE| (NOT |E0|, which sits near the KE/|PE|
cancellation and exaggerates the relative number). All runs share the same IC,
dt, eps and seed, so |PE0| is common across them.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    out_path = sys.argv[1]
    entries = []
    for arg in sys.argv[2:]:
        label, path = arg.rsplit("=", 1)
        entries.append((label, path))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    styles = {
        "Direct": dict(color="black", lw=2.0, ls="-"),
        "BH (theta=0.5)": dict(color="tab:red", lw=1.6, ls="--"),
        "FMM p=4": dict(color="tab:green", lw=1.4, ls="-"),
        "FMM p=8": dict(color="tab:blue", lw=1.4, ls="-"),
        "FMM p=12": dict(color="tab:purple", lw=1.4, ls="-"),
    }

    summary = []
    for label, path in entries:
        d = np.genfromtxt(path, delimiter=",", names=True)
        t = d["time"]
        E = d["total"]
        pe0 = abs(d["potential"][0])
        drift = np.abs(E - E[0]) / pe0
        st = styles.get(label, dict(lw=1.3))
        ax.semilogy(t, np.maximum(drift, 1e-12), label=f"{label}  (max {drift.max():.2e})", **st)
        summary.append((label, drift.max()))

    ax.set_xlabel("time")
    ax.set_ylabel(r"$|E(t)-E_0|\,/\,|PE_0|$")
    ax.set_title("Gate E: energy drift vs solver — FMM converges toward Direct as p grows\n"
                 "(closing the Barnes-Hut drift; eps small enough that far-field pairs are not within eps)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")
    for label, m in summary:
        print(f"  {label:18s} max drift / |PE0| = {m:.6e}")


if __name__ == "__main__":
    main()
