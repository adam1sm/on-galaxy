#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "BarnesHutSolver.hpp"
#include "Config.hpp"
#include "Diagnostics.hpp"
#include "DirectSolver.hpp"
#include "FMMSolver.hpp"
#include "ForceSolver.hpp"
#include "InitialConditions.hpp"
#include "Integrator.hpp"
#include "Kernel.hpp"
#include "Types.hpp"

using namespace galaxy;

namespace {

// Build the initial body set for the chosen scenario.
Bodies buildInitialConditions(const Config& cfg, double& period_out) {
    period_out = 0.0;
    if (cfg.scenario == Scenario::TwoBody) {
        TwoBodyConfig tb;
        tb.m1 = cfg.tb_m1;
        tb.m2 = cfg.tb_m2;
        tb.separation = cfg.tb_separation;
        tb.G = cfg.G;
        tb.eps = cfg.eps;
        period_out = twoBodyPeriod(tb);
        return makeTwoBodyCircular(tb);
    }
    DiskConfig dc;
    dc.N = cfg.N;
    dc.radius = cfg.disk_radius;
    dc.total_mass = cfg.disk_mass;
    dc.rotation = cfg.disk_rotation;
    dc.thermal = cfg.disk_thermal;
    dc.seed = cfg.seed;
    return makeRandomDisk(dc);
}

} // namespace

int main(int argc, char** argv) {
    Config cfg;
    if (!parseConfig(argc, argv, cfg)) return 0;

    const KernelParams kparams{cfg.G, cfg.eps};

    double period = 0.0;
    Bodies bodies = buildInitialConditions(cfg, period);
    const std::size_t n = bodies.size();

    // The integrator depends only on the abstract ForceSolver interface; the
    // concrete solver is chosen here and swapped without touching anything else.
    std::unique_ptr<ForceSolver> solver;
    if (cfg.solver == SolverKind::BarnesHut) {
        solver = std::make_unique<BarnesHutSolver>(kparams, cfg.theta,
                                                   cfg.leaf_capacity, cfg.max_depth);
    } else if (cfg.solver == SolverKind::FMM) {
        solver = std::make_unique<FMMSolver>(kparams, cfg.p, cfg.fmm_depth);
    } else {
        solver = std::make_unique<DirectSolver>(kparams);
    }
    VelocityVerlet integrator(*solver);
    integrator.initialize(bodies);

    // Output streams: a trajectory (for animation) and a diagnostics time series.
    const std::string traj_path = cfg.out_prefix + "_trajectory.csv";
    const std::string diag_path = cfg.out_prefix + "_diagnostics.csv";
    std::ofstream traj(traj_path);
    std::ofstream diag(diag_path);
    if (!traj || !diag) {
        std::cerr << "ERROR: cannot open output files under prefix '"
                  << cfg.out_prefix << "'. Does the directory exist?\n";
        return 1;
    }
    traj << "step,time,id,x,y,vx,vy\n";
    diag << "step,time,kinetic,potential,total,Px,Py\n";

    auto writeFrame = [&](long step, double t) {
        for (std::size_t i = 0; i < n; ++i) {
            traj << step << ',' << t << ',' << i << ','
                 << bodies.pos[i].real() << ',' << bodies.pos[i].imag() << ','
                 << bodies.vel[i].real() << ',' << bodies.vel[i].imag() << '\n';
        }
    };
    auto writeDiag = [&](long step, double t, const Diagnostics& d) {
        diag << step << ',' << t << ','
             << d.kinetic << ',' << d.potential << ',' << d.total << ','
             << d.momentum.real() << ',' << d.momentum.imag() << '\n';
    };

    // Report configuration.
    std::cout << "=== galaxy Rung 0 (DirectSolver) ===\n";
    std::cout << "scenario   : " << (cfg.scenario == Scenario::TwoBody ? "twobody" : "disk") << "\n";
    std::cout << "N          : " << n << "\n";
    std::cout << "dt         : " << cfg.dt << "\n";
    std::cout << "steps      : " << cfg.step_count << "\n";
    std::cout << "G, eps     : " << cfg.G << ", " << cfg.eps << "\n";
    if (cfg.scenario == Scenario::TwoBody) {
        std::cout << "period T   : " << period
                  << "  (" << (cfg.dt * cfg.step_count / period) << " periods)\n";
    }
    std::cout << "solver     : " << solver->name();
    if (cfg.solver == SolverKind::BarnesHut) std::cout << "  theta=" << cfg.theta;
    if (cfg.solver == SolverKind::FMM) {
        const int L = cfg.fmm_depth >= 0 ? cfg.fmm_depth : FMMSolver::autoDepth(n);
        std::cout << "  p=" << cfg.p << "  depth=" << L;
    }
    std::cout << "\n";
    std::cout << "outputs    : " << traj_path << ", " << diag_path << "\n";

    // Initial diagnostics define the reference energy E0.
    Diagnostics d0 = computeDiagnostics(bodies, kparams);
    const double E0 = d0.total;
    double max_abs_drift = 0.0;     // max |E - E0|
    double max_rel_drift = 0.0;     // max |E - E0| / |E0|
    double max_mom_drift = 0.0;     // max |P - P0|
    const Complex P0 = d0.momentum;

    std::printf("step %8ld  t %10.4f  E % .8e  KE % .6e  PE % .6e  |P| %.3e\n",
                0L, 0.0, d0.total, d0.kinetic, d0.potential, std::abs(d0.momentum));

    writeFrame(0, 0.0);
    writeDiag(0, 0.0, d0);

    // Main integration loop.
    for (long step = 1; step <= cfg.step_count; ++step) {
        integrator.step(bodies, cfg.dt);
        const double t = step * cfg.dt;

        const bool do_diag = (step % cfg.frame_stride == 0) ||
                             (step % cfg.log_stride == 0) ||
                             (step == cfg.step_count);
        Diagnostics d;
        if (do_diag) {
            d = computeDiagnostics(bodies, kparams);
            const double abs_drift = std::abs(d.total - E0);
            max_abs_drift = std::max(max_abs_drift, abs_drift);
            max_rel_drift = std::max(max_rel_drift,
                                     std::abs(E0) > 0 ? abs_drift / std::abs(E0) : abs_drift);
            max_mom_drift = std::max(max_mom_drift, std::abs(d.momentum - P0));
        }

        if (step % cfg.frame_stride == 0 || step == cfg.step_count) {
            writeFrame(step, t);
            writeDiag(step, t, d);
        }
        if (step % cfg.log_stride == 0 || step == cfg.step_count) {
            std::printf("step %8ld  t %10.4f  E % .8e  KE % .6e  PE % .6e  |P| %.3e\n",
                        step, t, d.total, d.kinetic, d.potential, std::abs(d.momentum));
        }
    }

    // Final summary -- the headline acceptance numbers.
    std::cout << "\n=== conservation summary ===\n";
    std::printf("E0                     : % .10e\n", E0);
    std::printf("max |dE|               : % .6e\n", max_abs_drift);
    std::printf("max |dE|/|E0|          : % .6e   <-- energy drift (bounded)\n", max_rel_drift);
    std::printf("max |dP| (from P0)     : % .6e\n", max_mom_drift);
    std::cout << "wrote: " << traj_path << "\n";
    std::cout << "wrote: " << diag_path << "\n";
    return 0;
}
