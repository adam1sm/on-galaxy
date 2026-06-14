#!/usr/bin/env python3
"""Rotating 3D scatter animation of a cluster trajectory.

Usage:
    python animate3d.py <prefix>_trajectory.csv <out.gif> [title]

CSV columns: step,time,id,x,y,z,vx,vy,vz (one row per body per frame). The camera
azimuth rotates slowly so the 3D structure reads clearly in a flat gif.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def main():
    csv_path = sys.argv[1]
    out_path = sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else "3D cluster"

    d = np.genfromtxt(csv_path, delimiter=",", names=True)
    steps = np.unique(d["step"]).astype(int)
    n = int(d["id"].max()) + 1
    F = len(steps)

    x = d["x"].reshape(F, n)
    y = d["y"].reshape(F, n)
    z = d["z"].reshape(F, n)
    times = d["time"].reshape(F, n)[:, 0]

    # Symmetric cubic limits from a robust (99th percentile) extent so a few
    # ejected stars don't shrink the cluster to a dot.
    span = np.percentile(np.abs(np.concatenate([x, y, z])), 99) * 1.3
    span = span if span > 0 else 1.0

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    size = max(2.0, 400.0 / n)
    scat = ax.scatter(x[0], y[0], z[0], s=size, c="white", depthshade=True, edgecolors="none")

    def style():
        ax.set_xlim(-span, span); ax.set_ylim(-span, span); ax.set_zlim(-span, span)
        ax.set_box_aspect((1, 1, 1))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_ticklabels([])
            axis.pane.set_facecolor("black")
            axis.pane.set_edgecolor("gray")
        ax.grid(False)

    max_frames = 300
    stride = max(1, F // max_frames)
    frames = list(range(0, F, stride))

    def update(fi):
        ax.cla()
        style()
        ax.scatter(x[fi], y[fi], z[fi], s=size, c="white", depthshade=True, edgecolors="none")
        ax.view_init(elev=20, azim=(fi * 0.6) % 360)  # slow camera rotation
        ax.set_title(f"{title}\nt = {times[fi]:.2f}   N = {n}", color="white")
        return ()

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=25))
    print(f"wrote {out_path}  ({len(frames)} frames, N={n})")


if __name__ == "__main__":
    main()
