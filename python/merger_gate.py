#!/usr/bin/env python3
# Rung 12 merger gate: from <out>_sep.csv report the pericenter/apocenter
# sequence (apocenters must DECREASE = orbital decay) and the final separation
# over the last ~10% of the run (must be small/stable for coalescence).
import csv
import sys

path = sys.argv[1]
rows = [(float(r["time"]), float(r["sep"])) for r in csv.DictReader(open(path))]
t = [x[0] for x in rows]
s = [x[1] for x in rows]
n = len(s)
print(f"file={path}  npts={n}  t_end={t[-1]:.2f}")

# local extrema (skip tiny wiggles with a small prominence window)
peris, apos = [], []
for i in range(1, n - 1):
    if s[i] < s[i - 1] and s[i] <= s[i + 1]:
        peris.append((t[i], s[i]))
    elif s[i] > s[i - 1] and s[i] >= s[i + 1]:
        apos.append((t[i], s[i]))

print("PERICENTERS:")
for tt, ss in peris:
    print(f"  t={tt:7.2f}  sep={ss:.3f}")
print("APOCENTERS:")
for tt, ss in apos:
    print(f"  t={tt:7.2f}  sep={ss:.3f}")

if len(apos) >= 2:
    decr = all(apos[i][1] < apos[i - 1][1] for i in range(1, len(apos)))
    print(f"apocenters strictly decreasing (orbital decay): {decr}")

tail = s[int(n * 0.9):]
print(f"LAST 10%: mean={sum(tail)/len(tail):.3f}  min={min(tail):.3f}  max={max(tail):.3f}")
diskRadius = 4.0
print(f"coalescence (final sep < {diskRadius/2:.1f} = half disk radius): "
      f"{max(tail) < diskRadius/2}")
