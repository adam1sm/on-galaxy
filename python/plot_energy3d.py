#!/usr/bin/env python3
"""Plot 3D diagnostics: energy, plus linear- and angular-momentum drift.

Usage:
    python plot_energy3d.py <prefix>_diagnostics.csv <out.png> [title]

CSV columns: step,time,kinetic,potential,total,Px,Py,Pz,Lx,Ly,Lz
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    csv_path = sys.argv[1]
    out_path = sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else "3D conservation"

    d = np.genfromtxt(csv_path, delimiter=",", names=True)
    t = d["time"]
    E, KE, PE = d["total"], d["kinetic"], d["potential"]
    P = np.vstack([d["Px"], d["Py"], d["Pz"]]).T
    L = np.vstack([d["Lx"], d["Ly"], d["Lz"]]).T

    E0 = E[0]
    rel = np.abs(E - E0) / (abs(E0) if E0 != 0 else 1.0)
    pdrift = np.linalg.norm(P - P[0], axis=1)
    ldrift = np.linalg.norm(L - L[0], axis=1)

    fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    ax[0].plot(t, E, "k", lw=1.5, label="total E")
    ax[0].plot(t, KE, color="tab:red", lw=1, alpha=0.8, label="kinetic")
    ax[0].plot(t, PE, color="tab:blue", lw=1, alpha=0.8, label="potential")
    ax[0].set_ylabel("energy"); ax[0].legend(loc="best"); ax[0].grid(alpha=0.3)
    ax[0].set_title(title)

    ax[1].plot(t, rel, color="tab:purple", lw=1.2)
    ax[1].set_ylabel(r"$|E-E_0|/|E_0|$")
    ax[1].set_title(f"relative energy drift (max = {rel.max():.3e}, bounded)")
    ax[1].grid(alpha=0.3)

    ax[2].plot(t, pdrift, color="tab:green", lw=1.2, label=f"|dP| (max {pdrift.max():.2e})")
    ax[2].plot(t, ldrift, color="tab:orange", lw=1.2, label=f"|dL| vector (max {ldrift.max():.2e})")
    ax[2].set_ylabel("momentum drift"); ax[2].set_xlabel("time")
    ax[2].set_title("linear- & angular-momentum drift")
    ax[2].legend(loc="best"); ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")
    print(f"max rel energy drift = {rel.max():.6e}")
    print(f"max |dP|             = {pdrift.max():.6e}")
    print(f"max |dL| (vector)    = {ldrift.max():.6e}")


if __name__ == "__main__":
    main()
