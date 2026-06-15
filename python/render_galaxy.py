#!/usr/bin/env python3
"""RUNG 8 offline renderer: density-histogram (additive) galaxy/merger video.

Usage:
  python render_galaxy.py <prefix>_frames.bin <prefix>_species.bin <out.mp4>
        [--extent E] [--res R] [--fps F] [--scale S] [--tilt deg]

Bins projected particle positions into a 2D density image per frame, asinh-
stretches, and maps to colour (two-colour additive when two galaxies are
present; magma ramp for a single disk). Fast enough for ~1e6 points.
"""
import sys, struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


def parse_args(a):
    d = dict(extent=14.0, res=900, fps=30, scale=0.0, tilt=20.0, orbit=0.0, depth_fade=1.5)
    o = {"frames": a[1], "species": a[2], "out": a[3]}
    i = 4
    while i < len(a):
        d[a[i].lstrip("-")] = float(a[i + 1]); i += 2
    o.update(d)
    return o


def main():
    o = parse_args(sys.argv)
    res, extent = int(o["res"]), o["extent"]
    with open(o["frames"], "rb") as f:
        N, nF = struct.unpack("ii", f.read(8))
        data_off = 8
    spec = np.fromfile(o["species"], dtype=np.int8)
    two = spec.max() > 0
    
    tilt = np.radians(o["tilt"])
    orbit_arc = o["orbit"]
    depth_fade = o["depth_fade"]
    
    ce, se = np.cos(tilt), np.sin(tilt)
    Rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
    
    edges = np.linspace(-extent, extent, res + 1)

    def read_frame(k):
        with open(o["frames"], "rb") as f:
            f.seek(data_off + k * N * 3 * 4)
            p = np.frombuffer(f.read(N * 3 * 4), dtype=np.float32).reshape(N, 3)
        return p

    def project(p, k):
        P = p - p.mean(axis=0)  # recenter on the system
        azim = np.radians(orbit_arc * (k / max(1, nF - 1)))
        if azim != 0.0:
            ca, sa = np.cos(azim), np.sin(azim)
            Ry = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
            Q = (P @ Ry.T) @ Rx.T
        else:
            Q = P @ Rx.T
        return Q[:, 0], Q[:, 1], Q[:, 2]

    def density(x, y, depth):
        w = np.exp(depth_fade * np.clip(depth / extent, -2.0, 2.0)) if depth_fade > 0 else None
        h, _, _ = np.histogram2d(x, y, bins=[edges, edges], weights=w)
        return h.T  # row=y

    def frame_rgb(k, scale):
        p = read_frame(k)
        x, y, depth = project(p, k)
        if two:
            mask0, mask1 = (spec == 0), (spec == 1)
            h0 = density(x[mask0], y[mask0], depth[mask0])
            h1 = density(x[mask1], y[mask1], depth[mask1])
            a0 = np.arcsinh(h0 * scale) / norm
            a1 = np.arcsinh(h1 * scale) / norm
            c0 = np.array([0.35, 0.55, 1.0]); c1 = np.array([1.0, 0.55, 0.2])
            rgb = np.clip(a0[..., None] * c0 + a1[..., None] * c1, 0, 1)
            # white-hot cores
            rgb += np.clip((a0 + a1 - 0.7), 0, 1)[..., None] * 0.6
            return np.clip(rgb, 0, 1)
        else:
            h = density(x, y, depth)
            a = np.arcsinh(h * scale) / norm
            cmap = plt.get_cmap("magma")
            return cmap(np.clip(a, 0, 1))[..., :3]

    # auto scale + normalization from a mid frame core density
    k_mid = min(nF - 1, nF // 4)
    pmid = read_frame(k_mid)
    xm, ym, zm = project(pmid, k_mid)
    hm = density(xm, ym, zm)
    scale = o["scale"] if o["scale"] > 0 else 1.0 / max(np.percentile(hm[hm > 0], 99) if (hm > 0).any() else 1.0, 1.0)
    norm = float(np.arcsinh(hm.max() * scale)) or 1.0

    fig = plt.figure(figsize=(8, 8), facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("black"); ax.axis("off")
    im = ax.imshow(frame_rgb(0, scale), origin="lower", extent=[-extent, extent, -extent, extent],
                   interpolation="bilinear")

    def update(k):
        im.set_data(frame_rgb(k, scale))
        return (im,)

    anim = FuncAnimation(fig, update, frames=nF, blit=True)
    if o["out"].endswith(".gif"):
        anim.save(o["out"], writer=PillowWriter(fps=int(o["fps"])))
    else:
        anim.save(o["out"], writer=FFMpegWriter(fps=int(o["fps"]), bitrate=6000))
    print(f"wrote {o['out']}  ({nF} frames, N={N}, two-colour={two})")


if __name__ == "__main__":
    main()
