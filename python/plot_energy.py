#!/usr/bin/env python3
"""Plot the energy / momentum diagnostics time series produced by the sim.

Usage:
    python plot_energy.py <prefix>_diagnostics.csv <out.png> [title]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    csv_path = sys.argv[1]
    out_path = sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else "Energy & momentum conservation"

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    t = data["time"]
    E = data["total"]
    KE = data["kinetic"]
    PE = data["potential"]
    Px = data["Px"]
    Py = data["Py"]

    E0 = E[0]
    rel_drift = np.abs(E - E0) / (abs(E0) if E0 != 0 else 1.0)
    mom_mag = np.hypot(Px - Px[0], Py - Py[0])

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(t, E, label="total E", color="black", lw=1.5)
    axes[0].plot(t, KE, label="kinetic", color="tab:red", lw=1, alpha=0.8)
    axes[0].plot(t, PE, label="potential", color="tab:blue", lw=1, alpha=0.8)
    axes[0].set_ylabel("energy")
    axes[0].legend(loc="best")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, rel_drift, color="tab:purple", lw=1.2)
    axes[1].set_ylabel(r"$|E-E_0|/|E_0|$")
    axes[1].set_title(f"relative energy drift (max = {rel_drift.max():.3e})")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, mom_mag, color="tab:green", lw=1.2)
    axes[2].set_ylabel(r"$|P - P_0|$")
    axes[2].set_xlabel("time")
    axes[2].set_title(f"linear-momentum drift (max = {mom_mag.max():.3e})")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")
    print(f"max relative energy drift = {rel_drift.max():.6e}")
    print(f"max momentum drift        = {mom_mag.max():.6e}")


if __name__ == "__main__":
    main()
