# galaxy — gravitational N-body simulator (2D FMM + 3D track)

A gravitational N-body simulator built in **rungs**. The **2D track** (complete)
reaches a **Fast Multipole Method** with O(N) force computation, validated against
direct O(N²) and Barnes–Hut O(N log N) baselines. The **3D track** (in progress)
extends to real Newtonian gravity (force ~ 1/r²) and the canonical 3D
spherical-harmonic FMM. The two tracks are fully separate (`src/` vs `src3d/`);
both build.

---

## Rung roadmap

**2D track (`src/`) — complete:**

| Rung | Solver            | Complexity   | Status            |
|------|-------------------|--------------|-------------------|
| 0    | `DirectSolver`    | O(N²)        | ✅ the 2D correctness **oracle** |
| 1    | `BarnesHutSolver` | O(N log N)   | ✅ validated against the oracle |
| 2    | `FMMSolver`       | O(N)         | ✅ uniform 2D FMM (log kernel, complex expansions); gates A–E pass |

**3D track (`src3d/`) — in progress:**

| Rung | Deliverable | Status |
|------|-------------|--------|
| 3    | 3D foundation + `DirectSolver3D` oracle (1/r² Plummer kernel, Vec3, leapfrog) | ✅ |
| 4    | 3D spherical-harmonic FMM operators (P2M/M2M/M2L/L2L), validated in isolation | ✅ (`src3d/FmmExpansions3D.hpp`) |
| 5    | 3D FMM assembly on an octree + BH-3D baseline + scaling | ✅ this rung (`src3d/FMM3D.hpp`, `src3d/BarnesHut3D.hpp`) |
| 6    | GPU / 1M bodies (+ Wigner-rotation M2L, softening-aware far field) | ⛔ next |
| 7    | galaxy render | ⛔ |

The hero deliverables: a **log–log time-vs-N scaling plot** (2D done) and a
**galaxy render** (3D).

---

## HARD CONSTRAINTS — 2D TRACK (`src/`) ONLY

These are deliberate and load-bearing for the 2D FMM. They scope the **2D track
only**; the separate 3D track (`src3d/`, Rung 3+) intentionally uses 3D and the
1/r² kernel. Within `src/`, future sessions must not reintroduce 3D or Euler.

1. **2D only (within `src/`). Never 3D in the 2D track.**
2. **2D logarithmic (Laplace / log) gravity kernel** — *not* the 3D 1/r² kernel.
   Acceleration on body *i*:
   ```
   a_i = G * sum_{j != i} m_j * (z_j - z_i) / (|z_j - z_i|^2 + eps^2)
   ```
   `G` configurable (default 1), `eps` = softening length. The log kernel is chosen
   so the FMM's complex multipole/local expansions line up later.
3. **Integrator: velocity Verlet / leapfrog (symplectic). NEVER forward Euler.**
4. **Positions and velocities are `std::complex<double>`** (`src/Types.hpp`) so FMM
   expansions drop in cleanly. `diff = z_j - z_i` is complex subtraction;
   `|diff|^2 = std::norm(diff)`.

### Self-consistent energy

The pairwise potential is exactly the one whose negative gradient is the coded
force, so the energy check is self-consistent:
```
U_ij = 0.5 * G * m_i * m_j * ln(|z_j - z_i|^2 + eps^2)     (sum over unique pairs i<j)
```
Derivation and the `force = -grad U` check live in `src/Kernel.hpp`.

### Two-body circular orbit (log kernel, NOT Kepler)

For two bodies separated by `D`, balancing centripetal and gravitational
acceleration **for this kernel** gives:
```
omega = sqrt( G (m1 + m2) / (D^2 + eps^2) )
v1 = omega * r1,   v2 = omega * r2,   r1 = D*m2/M,  r2 = D*m1/M
```
This is *not* the 3D Kepler `omega ∝ D^(-3/2)` relation. See
`src/InitialConditions.hpp`.

---

## Architecture / seams

```
src/
  Types.hpp            Complex, Bodies (SoA), Accelerations
  Kernel.hpp           single source of truth for force + potential (log kernel)
  ForceSolver.hpp      abstract interface: computeAccelerations(bodies) -> accels
  DirectSolver.hpp     Rung 0: O(N^2) all-pairs — the ORACLE
  BarnesHutSolver.hpp  Rung 1: O(N log N) quadtree, monopole approximation
  FmmExpansions.hpp    Rung 2a: FMM math — P2M/M2M/M2L/L2L + eval (NO tree/solver)
  FMMSolver.hpp        Rung 2b: uniform FMM solver (tree + passes + lists + P2P)
  Integrator.hpp       velocity Verlet; depends ONLY on ForceSolver
  Diagnostics.hpp      energy (KE+PE) and linear momentum
  InitialConditions.hpp  two-body circular + random disk generators
  Config.hpp           config struct + CLI parser
  main.cpp             driver: build IC -> integrate -> dump CSV + diagnostics
  harness.cpp          Rung 1 validation: force-comparison + timing/scaling
  fmm_tests.cpp        Rung 2a: isolated operator unit tests + error-vs-p table
python/
  plot_energy.py          energy/momentum-vs-time plots
  animate.py              trajectory animation (gif/mp4)
  plot_error_vs_theta.py  BH accuracy vs opening angle
  plot_scaling.py         log-log time-vs-N (hero scaling plot; FMM-ready)
  plot_energy_compare.py  Direct vs BH energy traces on shared axes
  plot_fmm_convergence.py FMM operator error-vs-p (Rung 2a)
```

The integrator and diagnostics depend **only** on the abstract `ForceSolver`, so
Barnes–Hut and FMM drop in without touching the time-stepping or validation code.
`DirectSolver` is the **oracle**: later solvers are validated by diffing their
accelerations against it.

---

## Build & run

Requires a C++17 compiler, CMake ≥ 3.16, Python 3 with `numpy`/`matplotlib`
(and `pillow` for GIF output), and `ffmpeg` (optional, for mp4).

**One command does everything** (build, run both scenarios, make plots + animations):

```bash
# one-time Python env
python3 -m venv .venv && ./.venv/bin/pip install numpy matplotlib pillow

# build + run + artifacts
./run.sh
```

Artifacts land in `outputs/`:
- `twobody_energy.png`, `disk_energy.png` — energy/momentum drift plots
- `twobody.gif`, `disk.gif` — animations
- `*_trajectory.csv`, `*_diagnostics.csv`, `*_log.txt` — raw data & logs

### Build / run manually

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/galaxy --help
./build/galaxy --scenario twobody --dt 0.005 --steps 30000 --out outputs/twobody
./build/galaxy --scenario disk --N 500 --dt 0.002 --steps 8000 --eps 0.05 --out outputs/disk
```

### Config knobs (`./build/galaxy --help`)

`--scenario {twobody,disk}`, `--N`, `--dt`, `--steps`, `--eps`, `--G`, `--seed`,
disk shape (`--disk-radius/-mass/-rotation/-thermal`), two-body
(`--tb-m1/-m2/-sep`), output (`--out`, `--frame-stride`, `--log-stride`). RNG seed
is fixed for reproducibility.

---

## Rung 0 validation

- **Two-body circular orbit** stays a stable closed circle over many periods;
  total energy and linear momentum conserved to a tight tolerance.
- **N-body disk run**: relative energy drift `|dE|/|E0|` stays **bounded**
  (oscillating, not growing) over a long run — see `disk_energy.png`.
- A short animation of each run is produced for eyeballing the dynamics.

Measured drift numbers are printed in the conservation summary at the end of each
run and in `outputs/*_log.txt`.

---

## Rung 1 — Barnes-Hut O(N log N)

`BarnesHutSolver` (`src/BarnesHutSolver.hpp`) is another `ForceSolver` behind the
existing interface — the kernel, integrator, and `Types` are untouched. It builds
an adaptive quadtree each step (root box recomputed to enclose all bodies),
stores per-node total mass + COM, and uses the **monopole** approximation with
opening criterion `node_size / distance_to_COM < theta` (default `theta=0.5`).
Robustness: max-depth cap + per-leaf bucket so coincident bodies can't subdivide
forever; closeness handled by the same `eps`. Self-interaction is skipped.

Select it at run time:
```bash
./build/galaxy --scenario disk --solver bh --theta 0.5 --N 5000 --out outputs/disk_bh
```

### theta = 0 correctness gate

At `theta = 0` the opening criterion (`node_size² < 0`) **never** fires, so the
tree walk always recurses to the leaves and sums every other body directly — i.e.
Barnes-Hut provably reduces to direct summation and must match `DirectSolver` to
roundoff. The harness asserts this:

```
theta=0  max_rel_err = 4.56e-14   (GATE PASS, tol 1e-9)
```

### Validation harness

`build/harness` is the core Rung 1 deliverable:

```bash
./build/harness --mode error   --N 4000   # force-comparison + theta=0 gate
./build/harness --mode scaling --theta 0.5 # log-log timing sweep 100..64000
```

- **Accuracy** (`--mode error`): per-body relative acceleration error of BH vs the
  oracle, reported as MAX and RMS, swept over `theta ∈ {0, 0.1, 0.3, 0.5, 0.8}`.
  Error shrinks monotonically as `theta` shrinks → `outputs/bh_error_vs_theta.png`.
- **Scaling** (`--mode scaling`): wall-clock per single force evaluation (tree
  rebuild included) for both solvers across N. Direct traces slope ≈ 2, BH traces
  ≈ N log N → `outputs/scaling.png`. This is the first version of the hero scaling
  plot; `plot_scaling.py` and the CSV already leave a column seam for an FMM line.
- **Dynamics** (`--solver bh` in the main sim): the disk under BH at `theta=0.5`
  takes a one-time energy offset during the initial violent collapse, then the
  relative drift settles into a **bounded band** (no secular growth). Compared
  against the symplectic Direct run on shared axes →
  `outputs/energy_direct_vs_bh.png`. Lower `theta` (or Rung 2's higher-order
  expansions) tightens this.

---

## Rung 2a — FMM operators (math only, validated in isolation)

Rung 2 is split deliberately: the operators — especially **M2L** — are where FMM
implementations silently break, so they are built and gated individually *before*
any tree/pass/interaction-list assembly. `src/FmmExpansions.hpp` is a standalone
math module (no tree, no solver); `FMMSolver.hpp` stays an unassembled seam.

**Convention (anchored to `Kernel.hpp`/`DirectSolver`).** Treat each body as a
charge `q = mass` at complex `z`, with complex potential `w(z) = Σ q_j log(z−z_j)`.
Then `potential = G·Re(w)` and `acceleration = G·Σ q_j (z_j−z)/|z−z_j|² =
−G·conj(w'(z))`, which is exactly DirectSolver's acceleration at `eps=0`. Every
operator is derived from a direct Taylor/series expansion (see the comments) so
there are no half-remembered binomial indices.

**Operators**: P2M (+ multipole eval), M2M, M2L, L2L (+ local eval), plus
`fieldToAccel` (`a = −G·conj(w')`). Expansion order `p` configurable.

**Isolated gates** (`build/fmm_tests`, fixed seeds): each operator is checked
against the oracle for BOTH potential (vs direct summation) and acceleration (vs
`DirectSolver`, `eps=0`), requiring (a) error decreasing monotonically in `p` and
(b) final error < 1e-5 at `p=12`. M2L is tested over four geometries (axis-aligned,
diagonal, off-quadrant) with the worst case gated. All four operators show clean
spectral (geometric) decay to roundoff → `outputs/fmm_convergence.png`.

```bash
cmake --build build --target fmm_tests && ./build/fmm_tests
```

---

## Rung 2b — assembled uniform FMM, O(N)

`src/FMMSolver.hpp` assembles the Rung 2a operators (reused unchanged) into a
complete **uniform** (non-adaptive) 2D FMM behind the `ForceSolver` interface.
Adaptive trees are a Rung 3 stretch.

- **Uniform quadtree**, depth `L ~ log4(N)` so `4^L ~ N` and mean leaf occupancy
  is O(1) → O(N). (We target occupancy ~4, one level shallower, to balance the
  M2L and P2P constants.) Rebuilt every step; root box recomputed each step.
- **Upward**: P2M at leaves, M2M up to a multipole at every box.
- **Downward**: M2L from each box's interaction list (children of the parent's
  neighbours that aren't the box's own neighbours, ≤27 in 2D) into its local
  expansion, then L2L down to children.
- **Leaves**: L2P (local-expansion derivative = far-field acceleration) + P2P
  direct sum over the leaf and its 8 neighbours. Softening `eps` is applied
  ONLY in P2P (matching DirectSolver); the far field uses the pure kernel.

```bash
./build/galaxy --scenario disk --solver fmm --p 8 --N 5000 --out outputs/disk_fmm
./build/harness --mode fmm        # gates A (depth-0), B (small), C (global err-vs-p)
./build/harness --mode scaling    # hero plot: Direct, BH, FMM
```

### Gates (vs the DirectSolver oracle)

- **A — depth-0 reduction**: at depth 0 the domain is one leaf, so FMM is pure
  P2P and must equal DirectSolver to roundoff. Match: **3.0e-15** (validates P2P
  + plumbing independent of the expansions).
- **B — small system, depth 3, eps=0**: error decreases monotonically with p.
- **C — global error-vs-p, N=8000, eps=0**: RMS force error `7.4e-4 → 1.2e-5 →
  2.7e-7` for p=4,8,12 — clean monotone convergence, well below 1e-6 by p=12.
  (Slightly slower than the isolated operators because interaction-list boxes sit
  only ~2 cells apart, the worst admissible separation.)
- **D — scaling**: on uniform-random points, FMM traces slope ≈ 1.08 (O(N)) vs
  BH ≈ 1.26 and Direct ≈ 1.95. FMM overtakes BH at **N ≈ 8000** and runs ~2×
  faster than BH at N = 256k → `outputs/scaling.png`.
- **E — energy payoff**: identical disk under Direct / BH / FMM(p=4,8,12), drift
  normalized by initial |PE|. FMM drift shrinks with p toward Direct and sits
  far below BH → `outputs/energy_vs_p.png`.

### Softening caveat (important)

The far field uses the **pure** log kernel, so the design assumption is that
well-separated pairs are never within `eps`. This holds only when **`eps ≪ leaf
size`**. If `eps` is comparable to a leaf (e.g. the clustered disk at `eps=0.05`,
leaf ~0.25), the closest well-separated pairs carry a fixed `~(eps/r)²` kernel
mismatch that does **not** shrink with p (FMM force error floors at ~5e-3). The
correctness gates use `eps=0` (no mismatch); the Gate E dynamics use `eps=0.005`
so the assumption holds and p-convergence is visible. Removing this restriction
(softening-aware near/far split, or an adaptive tree) is future work.

---

## 3D track (`src3d/`) — Rung 3: foundation + oracle

A fully separate 3D track that does **not** touch the 2D code. Real Newtonian
gravity with Plummer softening (force ~ 1/r²), built toward the canonical 3D
spherical-harmonic FMM (Rung 4+). The architecture mirrors 2D: an abstract
`ForceSolver3D`, a `DirectSolver3D` oracle, a dimension-agnostic velocity-Verlet
integrator, and named seams for `BarnesHut3D` / `FMM3D` (unimplemented).

```
src3d/
  Vec3.hpp               double 3-vector (dot, cross, norm)
  Types3D.hpp            Bodies3D (SoA), Accelerations3D
  Kernel3D.hpp           1/r^2 Plummer kernel + potential (force = -grad U)
  ForceSolver3D.hpp      abstract interface
  DirectSolver3D.hpp     Rung 3: O(N^2) all-pairs — the 3D ORACLE
  BarnesHut3D.hpp        future seam (octree); FMM3D.hpp future seam (spherical harmonics)
  Integrator3D.hpp       velocity Verlet on Vec3
  Diagnostics3D.hpp      energy, linear momentum, angular-momentum VECTOR
  InitialConditions3D.hpp  Keplerian two-body (tilted) + Plummer sphere
  Config3D.hpp, main3d.cpp
python/
  plot_energy3d.py       energy + linear/angular-momentum drift
  animate3d.py           rotating 3D scatter animation
```

**Kernel** (`Kernel3D.hpp`): `a_i = G Σ_{j≠i} m_j (r_j−r_i)/(|r_j−r_i|²+eps²)^(3/2)`,
with self-consistent potential `U_ij = −G m_i m_j /√(|r_j−r_i|²+eps²)` (the 3/2
exponent is the real 3D law, not the 2D log kernel). The two-body circular orbit
**is Keplerian**: `ω = √(G M/(D²+eps²)^(3/2))`, so `T² = 4π²D³/(GM)` at eps=0.

```bash
cmake --build build --target galaxy3d
./build/galaxy3d --scenario twobody --eps 0 --dt 0.005 --steps 50000 --out outputs/threed_twobody
./build/galaxy3d --scenario plummer --N 1000 --steps 10000 --out outputs/threed_cluster
./.venv/bin/python python/plot_energy3d.py outputs/threed_cluster_diagnostics.csv outputs/threed_cluster_energy.png
./.venv/bin/python python/animate3d.py    outputs/threed_cluster_trajectory.csv  outputs/threed_cluster.gif
```

**Rung 3 validation** (artifacts in `outputs/`):
- **Two-body** (tilted so all three L components are nonzero): closed orbit over
  ~20 periods; energy drift 9.8e-12, linear momentum 0, **angular-momentum vector
  1.1e-14**; Kepler's third law `T²GM/4π²D³ = 1.000000` for D=2,4,8.
- **Plummer sphere** N=1000 to t=20: relative energy drift **1.9e-7** (bounded,
  oscillating), |dP| 9.4e-17, |dL| (vector) 2.9e-16 →
  `outputs/threed_cluster_energy.png`, `outputs/threed_cluster.gif`.

### Rung 4 — 3D spherical-harmonic FMM operators (math only)

`src3d/FmmExpansions3D.hpp` implements the 3D Laplace FMM operators in a **single
convention** (Greengard–Rokhlin solid harmonics): `Y_n^m = √((n−|m|)!/(n+|m|)!)
P_n^|m| e^{imφ}` (no Condon–Shortley phase), regular `R_n^m = r^n Y_n^m` (local)
and singular `S_n^m = Y_n^m/r^{n+1}` (multipole), moments `M_n^m = Σ qᵢ ρᵢⁿ
Y_n^{−m}`. Harmonics use a stable seminormalized Legendre recurrence (no factorial
overflow). The **naive O(p⁴)** translation operators are used (no Wigner/rotation
acceleration — correctness first). It is math-only; the `FMM3D` seam stays
unassembled (Rung 5).

The **acceleration** is the gradient of the expansion, computed with its own
solid-harmonic derivative recurrences (`∂_z, ∂_x±i∂_y` mapping degree `n→n∓1`),
derived for this exact normalization and FD-validated in the basis check.

`fmm3d_tests` runs all gates (fixed seeds), writes `outputs/fmm3d_convergence.png`:
- **Gate 0** basis: harmonics vs closed forms (1.1e-16); gradient recurrences vs
  finite differences (7e-10).
- **P2M / M2L / M2M / L2L**: error-vs-p (p=2..10) for potential AND acceleration,
  multiple geometries; M2L tested over 4 orientations (worst case gated). All show
  clean monotone spectral convergence (see table below).

```bash
cmake --build build --target fmm3d_tests && ./build/fmm3d_tests
```

### Rung 5 — assembled uniform 3D FMM + Barnes-Hut 3D baseline

`src3d/FMM3D.hpp` assembles the Rung 4 operators (reused unchanged) on a
**uniform octree** behind `ForceSolver3D`; `src3d/BarnesHut3D.hpp` is the
O(N log N) baseline (adaptive octree, COM, opening angle). `src3d/harness3d.cpp`
runs the gates and the scaling sweep.

- Depth `L` with `8^L ~ N`; occupancy tuned **high** (~64) because each M2L is
  O(p⁴) and the 3D interaction list holds up to **189** boxes — a deep tree
  (occupancy ~1) would drown in M2L. Still O(N).
- Downward pass: M2L from the 189-box interaction list, then L2L to children.
  Leaves: L2P (gradient of the local expansion = far-field acceleration) + P2P
  over the leaf and its ≤26 neighbours. **Softening eps only in P2P**; far field
  uses the pure kernel.

```bash
./build/galaxy3d --solver fmm3d --p 8 --scenario plummer --N 4000 --out outputs/p_fmm
./build/harness3d --mode fmm        # gates A (depth-0), B (small), C (global) + softening floor
./build/harness3d --mode scaling    # 3D hero plot: Direct3D, BarnesHut3D, FMM3D
```

**Gates (vs DirectSolver3D):**
- **A — depth-0 reduction**: pure P2P equals DirectSolver3D to **1.4e-15**.
- **B — small, depth 2, eps=0**: acceleration rms `6.7e-3 → 1.8e-5` for p=2..8.
- **C — global, N=8000, eps=0**: acceleration rms `6.6e-4 → 4.3e-5 → 6.7e-6 →
  1.4e-6` for p=4,6,8,10 (max rel `1.3e-2 → 7.5e-5`), clean monotone.
- **D — scaling** (uniform cube, occupancy-aligned): Direct **slope 1.97**, BH
  **1.19**, FMM **1.17** (trending to 1). FMM is sub-quadratic and beats Direct
  beyond N≈5000. FMM's constant exceeds BH's (189-list × O(p⁴)), so the FMM<BH
  crossover is **beyond CPU range** — the slope is the claim; absolute speed and
  1M bodies are Rung 6 (GPU + Wigner). → `outputs/scaling3d.png`.
- **E — energy** (Plummer N=2000, depth 3, eps=0.02): drift shrinks toward the
  Direct baseline (1.80e-6) as p grows — `6.0e-6, 3.1e-6, 1.9e-6, 1.80e-6` for
  p=1,2,3,4. → `outputs/energy_vs_p_3d.png`.

**Softening floor (eyes-open for the galaxy rung):** the far field uses the pure
kernel, so force error floors at ~(eps/leaf)² for well-separated pairs. Measured
(N=8000, p=8, leaf=0.5): eps/leaf = 0.10/0.04/0.02/0.01 → rms force error
1.6e-3 / 2.5e-4 / 6.2e-5 / 1.7e-5. Keep **eps ≪ leaf**; a softening-aware far
field is planned (Rung 6).
