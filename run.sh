#!/usr/bin/env bash
# One command to build the simulator + harness, run the Rung 0 scenarios and the
# Rung 1 Barnes-Hut validation, and produce all artifacts (plots + animations)
# under outputs/.
#
# Usage:  ./run.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "No project venv found at .venv. Create it with:"
    echo "  python3 -m venv .venv && ./.venv/bin/pip install numpy matplotlib pillow"
    exit 1
fi

mkdir -p outputs

# ---- Build (single command) ------------------------------------------------
echo "==> Configuring & building (Release)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build --parallel >/dev/null
BIN="$ROOT/build/galaxy"
HARNESS="$ROOT/build/harness"

# ---- Scenario (a): two-body circular orbit, many periods -------------------
# Period for default (m1=m2=1, D=2, eps=0.01) is ~8.886; 30000*0.005 ~ 33.8 T.
echo
echo "==> Running two-body circular orbit (validation)"
"$BIN" --scenario twobody --dt 0.005 --steps 30000 \
       --tb-sep 2 --tb-m1 1 --tb-m2 1 --eps 0.01 \
       --frame-stride 20 --log-stride 3000 \
       --out outputs/twobody | tee outputs/twobody_log.txt

# ---- Scenario (b): random disk, long N-body run ----------------------------
echo
echo "==> Running random disk N-body (validation)"
"$BIN" --scenario disk --N 500 --dt 0.002 --steps 8000 \
       --disk-radius 1 --disk-mass 1 --eps 0.05 --seed 42 \
       --frame-stride 20 --log-stride 1000 \
       --out outputs/disk | tee outputs/disk_log.txt

# ---- Plots -----------------------------------------------------------------
echo
echo "==> Generating energy/momentum plots"
"$PY" python/plot_energy.py outputs/twobody_diagnostics.csv \
      outputs/twobody_energy.png "Two-body circular orbit (log kernel)"
"$PY" python/plot_energy.py outputs/disk_diagnostics.csv \
      outputs/disk_energy.png "Random disk N=500 (log kernel)"

# ---- Animations ------------------------------------------------------------
echo
echo "==> Generating animations"
"$PY" python/animate.py outputs/twobody_trajectory.csv \
      outputs/twobody.gif "Two-body circular orbit"
"$PY" python/animate.py outputs/disk_trajectory.csv \
      outputs/disk.gif "Random disk N=500"

# ===========================================================================
# RUNG 1: Barnes-Hut validation
# ===========================================================================

# ---- Force-comparison harness + theta=0 correctness gate -------------------
echo
echo "==> [Rung 1] Force-comparison vs oracle (theta sweep + theta=0 gate)"
"$HARNESS" --mode error --N 4000 --eps 0.05 --seed 42 | tee outputs/bh_error_log.txt

# ---- Timing / scaling sweep (hero plot: Direct, BH, FMM) -------------------
# Uniform-random points; Direct O(N^2), BH O(N log N), FMM O(N) (p=4). Direct is
# skipped above 64k. This single sweep serves Rung 1 and Rung 2b (Gate D).
echo
echo "==> [Rung 1/2b] Timing / scaling sweep (Direct vs Barnes-Hut vs FMM)"
"$HARNESS" --mode scaling --theta 0.5 --p 4 --eps 0.05 --seed 42 | tee outputs/scaling_log.txt

# ---- Dynamics sanity: identical disk, Direct vs Barnes-Hut (theta=0.5) -----
# Run a LONG window (t=80) so the energy-drift character is visible: a symplectic
# Direct run stays flat, while BH takes a one-time offset during the initial
# violent collapse and then settles into a BOUNDED band (no secular growth).
# Identical seed/N/dt/eps so the two traces are directly comparable.
DISK_ARGS=(--scenario disk --N 500 --dt 0.002 --steps 40000 \
           --disk-radius 1 --disk-mass 1 --eps 0.05 --seed 42 \
           --frame-stride 50 --log-stride 8000)
echo
echo "==> [Rung 1] Disk N-body, Direct (long run, energy reference)"
"$BIN" "${DISK_ARGS[@]}" --solver direct \
       --out outputs/disk_direct_long | tee outputs/disk_direct_long_log.txt
echo
echo "==> [Rung 1] Disk N-body, Barnes-Hut theta=0.5 (long run)"
"$BIN" "${DISK_ARGS[@]}" --solver bh --theta 0.5 \
       --out outputs/disk_bh_long | tee outputs/disk_bh_long_log.txt

# ---- Rung 1 plots ----------------------------------------------------------
echo
echo "==> [Rung 1] Generating plots"
"$PY" python/plot_error_vs_theta.py outputs/bh_error_vs_theta.csv outputs/bh_error_vs_theta.png
"$PY" python/plot_scaling.py outputs/scaling.csv outputs/scaling.png
"$PY" python/plot_energy_compare.py outputs/disk_direct_long_diagnostics.csv \
      outputs/disk_bh_long_diagnostics.csv outputs/energy_direct_vs_bh.png
"$PY" python/animate.py outputs/disk_bh_long_trajectory.csv \
      outputs/disk_bh.gif "Random disk N=500 (Barnes-Hut, theta=0.5)"

# ===========================================================================
# RUNG 2a: FMM operator unit tests (math only -- no tree, no solver wiring)
# ===========================================================================
echo
echo "==> [Rung 2a] FMM operator unit tests (P2M, M2L, M2M, L2L)"
"$ROOT/build/fmm_tests" | tee outputs/fmm_operator_log.txt
echo
echo "==> [Rung 2a] FMM operator convergence plot"
"$PY" python/plot_fmm_convergence.py outputs/fmm_operator_convergence.csv \
      outputs/fmm_convergence.png

# ===========================================================================
# RUNG 2b: assembled uniform FMM behind the ForceSolver interface
# ===========================================================================
# Gates A (depth-0 reduction), B (small system), C (global error-vs-p) -- and a
# disk-accuracy-vs-eps diagnostic. (Gate D scaling is the hero sweep above.)
echo
echo "==> [Rung 2b] FMM solver gates A/B/C (vs DirectSolver oracle)"
"$HARNESS" --mode fmm --seed 42 | tee outputs/fmm_solver_log.txt

# Gate E (payoff): identical disk under Direct / BH / FMM(p=4,8,12), long window.
# eps=0.005 is chosen small relative to the leaf size so the far-field "pure
# kernel" assumption holds (well-separated pairs are never within eps); then the
# FMM drift shrinks toward Direct as p grows, closing the Barnes-Hut drift.
GE_ARGS=(--scenario disk --N 500 --dt 0.002 --steps 40000 \
         --eps 0.005 --seed 42 --frame-stride 50 --log-stride 40000)
echo
echo "==> [Rung 2b] Gate E energy dynamics (Direct, BH, FMM p=4/8/12; eps=0.005)"
"$BIN" "${GE_ARGS[@]}" --solver direct          --out outputs/ge_direct  >/dev/null
"$BIN" "${GE_ARGS[@]}" --solver bh --theta 0.5   --out outputs/ge_bh      >/dev/null
"$BIN" "${GE_ARGS[@]}" --solver fmm --p 4         --out outputs/ge_fmm_p4  >/dev/null
"$BIN" "${GE_ARGS[@]}" --solver fmm --p 8         --out outputs/ge_fmm_p8  >/dev/null
"$BIN" "${GE_ARGS[@]}" --solver fmm --p 12        --out outputs/ge_fmm_p12 >/dev/null
"$PY" python/plot_energy_vs_p.py outputs/energy_vs_p.png \
      "Direct=outputs/ge_direct_diagnostics.csv" \
      "BH (theta=0.5)=outputs/ge_bh_diagnostics.csv" \
      "FMM p=4=outputs/ge_fmm_p4_diagnostics.csv" \
      "FMM p=8=outputs/ge_fmm_p8_diagnostics.csv" \
      "FMM p=12=outputs/ge_fmm_p12_diagnostics.csv"

echo
echo "==> Done. Artifacts in outputs/:"
ls -1 outputs/*.png outputs/*.gif
