#!/usr/bin/env python3
"""Animate a trajectory CSV produced by the sim.

Usage:
    python animate.py <prefix>_trajectory.csv <out.gif|out.mp4> [title]

The CSV has columns: step,time,id,x,y,vx,vy with one row per body per frame.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    csv_path = sys.argv[1]
    out_path = sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else "N-body dynamics"

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    steps = np.unique(data["step"]).astype(int)
    n_bodies = int(data["id"].max()) + 1

    # Reshape into (frame, body) grids. Rows are written grouped by frame.
    x = data["x"].reshape(len(steps), n_bodies)
    y = data["y"].reshape(len(steps), n_bodies)
    times = data["time"].reshape(len(steps), n_bodies)[:, 0]

    # Symmetric fixed limits based on the full extent (with margin).
    span = max(np.abs(x).max(), np.abs(y).max()) * 1.1
    span = span if span > 0 else 1.0

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_aspect("equal")
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.patch.set_facecolor("black")

    # Smaller markers when there are many bodies.
    size = 40 if n_bodies <= 4 else max(2.0, 400.0 / n_bodies)
    scat = ax.scatter(x[0], y[0], s=size, c="white", edgecolors="none")
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, color="white",
                  va="top", ha="left", fontsize=11)
    ax.set_title(title, color="white")

    def update(frame):
        scat.set_offsets(np.column_stack([x[frame], y[frame]]))
        txt.set_text(f"t = {times[frame]:.2f}   N = {n_bodies}")
        return scat, txt

    # Cap to a reasonable number of frames for a quick animation.
    max_frames = 400
    stride = max(1, len(steps) // max_frames)
    frames = range(0, len(steps), stride)

    anim = FuncAnimation(fig, update, frames=frames, blit=True)

    if out_path.endswith(".gif"):
        writer = PillowWriter(fps=30)
    else:
        writer = FFMpegWriter(fps=30, bitrate=2400)
    anim.save(out_path, writer=writer)
    print(f"wrote {out_path}  ({len(list(frames))} frames, N={n_bodies})")


if __name__ == "__main__":
    main()
