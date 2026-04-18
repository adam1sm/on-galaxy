#!/usr/bin/env python3
"""Error-vs-p convergence for all four FMM operators (Rung 2a).

Usage:
    python plot_fmm_convergence.py outputs/fmm_operator_convergence.csv outputs/fmm_convergence.png

CSV columns: operator,quantity,p,rel_err  (quantity in {potential, acceleration}).
One figure, two panels (potential | acceleration); one line per operator. Clean
geometric (spectral) decay with p confirms each operator is correct in isolation.
"""
import sys
import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/fmm_operator_convergence.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/fmm_convergence.png"

    # data[(operator, quantity)] = list of (p, err)
    data = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            data[(row["operator"], row["quantity"])].append(
                (int(row["p"]), float(row["rel_err"])))

    operators = ["P2M", "M2L", "M2M", "L2L"]
    colors = {"P2M": "tab:blue", "M2L": "tab:red", "M2M": "tab:green", "L2L": "tab:purple"}
    markers = {"P2M": "o", "M2L": "s", "M2M": "^", "L2L": "D"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True)
    for ax, quantity in zip(axes, ["potential", "acceleration"]):
        for op in operators:
            pts = sorted(data.get((op, quantity), []))
            if not pts:
                continue
            p = np.array([x[0] for x in pts])
            e = np.maximum(np.array([x[1] for x in pts]), 1e-16)
            ax.semilogy(p, e, marker=markers[op], color=colors[op], label=op, lw=1.6)
        ax.set_xlabel("expansion order p")
        ax.set_title(f"{quantity} relative error")
        ax.grid(alpha=0.3, which="both")
        ax.legend(title="operator")
    axes[0].set_ylabel("relative error vs DirectSolver / direct sum")

    fig.suptitle("FMM operator convergence (Rung 2a) — error decreases with p "
                 "for every operator (M2L over 4 geometries, worst case)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
