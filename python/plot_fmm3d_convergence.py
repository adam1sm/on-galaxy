#!/usr/bin/env python3
"""Error-vs-p convergence for the four 3D FMM operators (Rung 4).

Usage: python plot_fmm3d_convergence.py outputs/fmm3d_operator_convergence.csv outputs/fmm3d_convergence.png
CSV columns: operator,quantity,p,rel_err (quantity in {potential, acceleration}).
"""
import sys, csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/fmm3d_operator_convergence.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/fmm3d_convergence.png"
    data = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            data[(row["operator"], row["quantity"])].append((int(row["p"]), float(row["rel_err"])))
    operators = ["P2M", "M2L", "M2M", "L2L"]
    colors = {"P2M":"tab:blue","M2L":"tab:red","M2M":"tab:green","L2L":"tab:purple"}
    markers = {"P2M":"o","M2L":"s","M2M":"^","L2L":"D"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True)
    for ax, quantity in zip(axes, ["potential", "acceleration"]):
        for op in operators:
            pts = sorted(data.get((op, quantity), []))
            if not pts: continue
            p = np.array([x[0] for x in pts]); e = np.maximum([x[1] for x in pts], 1e-16)
            ax.semilogy(p, e, marker=markers[op], color=colors[op], label=op, lw=1.6)
        ax.set_xlabel("expansion order p"); ax.set_title(f"{quantity} relative error (RMS)")
        ax.grid(alpha=0.3, which="both"); ax.legend(title="operator")
    axes[0].set_ylabel("relative error vs DirectSolver3D / direct sum")
    fig.suptitle("3D FMM operator convergence (Rung 4) — spherical-harmonic operators, "
                 "naive O(p^4); M2L over 4 geometries (worst case)", fontsize=12)
    fig.tight_layout(rect=(0,0,1,0.96)); fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")

if __name__ == "__main__":
    main()
