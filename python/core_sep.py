#!/usr/bin/env python3
# Rung 12 coalescence proof. The driver's <out>_sep.csv tracks the mass-weighted
# COM separation, which SATURATES at the tidal-tail bias level once the cores
# merge (asymmetric ejecta pull each species' COM apart). The tail-robust truth
# is the separation of each species' CORE. Here: median position (robust to tail
# outliers) and a dense-core centroid (closest 20% to the median). When these go
# to ~eps the two nuclei have coalesced into one remnant.
#
# Reads <base>_frames.bin (int32 N, int32 nFrames, then nFrames*N*3 float32) and
# <base>_species.bin (N int8). Prints a per-frame table and writes
# <base>_coresep.csv (frame, median_sep, dense_core_sep).
import sys
import struct
import numpy as np

base = sys.argv[1]
species = np.frombuffer(open(base + "_species.bin", "rb").read(), dtype=np.int8)
N = species.size
i0 = np.where(species == 0)[0]
i1 = np.where(species == 1)[0]

with open(base + "_frames.bin", "rb") as f:
    hN, hF = struct.unpack("<ii", f.read(8))
    assert hN == N, (hN, N)
    out = open(base + "_coresep.csv", "w")
    out.write("frame,median_sep,dense_core_sep\n")
    every = max(1, hF // 30)
    for fr in range(hF):
        pos = np.frombuffer(f.read(N * 3 * 4), dtype=np.float32).reshape(N, 3)

        def med_core(idx):
            p = pos[idx]
            m = np.median(p, axis=0)
            d2 = ((p - m) ** 2).sum(1)
            k = max(1, p.shape[0] // 5)
            core = p[np.argpartition(d2, k - 1)[:k]].mean(0)
            return m, core

        m0, c0 = med_core(i0)
        m1, c1 = med_core(i1)
        msep = float(np.linalg.norm(m0 - m1))
        csep = float(np.linalg.norm(c0 - c1))
        out.write(f"{fr},{msep:.4f},{csep:.4f}\n")
        if fr % every == 0 or fr == hF - 1:
            print(f"frame {fr:4d}  median-sep={msep:7.3f}  dense-core-sep={csep:7.3f}")
    out.close()
print(f"wrote {base}_coresep.csv")
