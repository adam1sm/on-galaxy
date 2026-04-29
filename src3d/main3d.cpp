#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include "BarnesHut3D.hpp"
#include "Config3D.hpp"
#include "Diagnostics3D.hpp"
#include "DirectSolver3D.hpp"
#include "FMM3D.hpp"
#include "ForceSolver3D.hpp"
#include "InitialConditions3D.hpp"
#include "Integrator3D.hpp"
#include "Kernel3D.hpp"
#include "Types3D.hpp"

using namespace galaxy3d;

namespace {

Bodies3D buildInitialConditions(const Config3D& cfg, double& period_out) {
    period_out = 0.0;
    if (cfg.scenario == Scenario3D::TwoBody) {
        TwoBodyConfig3D tb;
        tb.m1 = cfg.tb_m1; tb.m2 = cfg.tb_m2;
        tb.separation = cfg.tb_separation;
        tb.G = cfg.G; tb.eps = cfg.eps;
        period_out = twoBodyPeriod(tb);
        return makeTwoBodyCircular(tb);
    }
    PlummerConfig pc;
    pc.N = cfg.N; pc.a = cfg.plummer_a; pc.total_mass = cfg.plummer_mass;
    pc.G = cfg.G; pc.seed = cfg.seed;
    return makePlummerSphere(pc);
}

// Bonus: verify Kepler's third law from the derived (eps=0) relation, T^2 ~ D^3.
void printKeplerCheck(const Config3D& cfg) {
    std::cout << "\n=== Kepler's third law check (eps=0): T^2 = 4*pi^2 D^3/(G*M) ===\n";
    std::printf("%10s  %14s  %18s\n", "D", "T", "T^2 G M /(4pi^2 D^3)");
    for (double D : {cfg.tb_separation, 2.0 * cfg.tb_separation, 4.0 * cfg.tb_separation}) {
        TwoBodyConfig3D tb;
        tb.m1 = cfg.tb_m1; tb.m2 = cfg.tb_m2; tb.separation = D; tb.G = cfg.G; tb.eps = 0.0;
        const double M = tb.m1 + tb.m2;
        const double T = twoBodyPeriod(tb);
        const double kepler = T * T * cfg.G * M / (4.0 * M_PI * M_PI * D * D * D);
        std::printf("%10.4f  %14.6f  %18.12f\n", D, T, kepler);
    }
}

} // namespace

int main(int argc, char** argv) {
    Config3D cfg;
    if (!parseConfig3D(argc, argv, cfg)) return 0;

    const KernelParams3D kparams{cfg.G, cfg.eps};

    double period = 0.0;
    Bodies3D bodies = buildInitialConditions(cfg, period);
    const std::size_t n = bodies.size();

    // The integrator depends only on the abstract ForceSolver3D interface; the
    // concrete solver is chosen here and swapped without touching anything else.
    std::unique_ptr<ForceSolver3D> solver;
    if (cfg.solver == Solver3D::BarnesHut)
        solver = std::make_unique<BarnesHut3D>(kparams, cfg.theta);
    else if (cfg.solver == Solver3D::FMM)
        solver = std::make_unique<FMM3D>(kparams, cfg.p, cfg.fmm_depth);
    else
        solver = std::make_unique<DirectSolver3D>(kparams);
    VelocityVerlet3D integrator(*solver);
    integrator.initialize(bodies);

    const std::string traj_path = cfg.out_prefix + "_trajectory.csv";
    const std::string diag_path = cfg.out_prefix + "_diagnostics.csv";
    std::ofstream traj(traj_path), diag(diag_path);
    if (!traj || !diag) {
        std::cerr << "ERROR: cannot open output files under prefix '" << cfg.out_prefix
                  << "'. Does the directory exist?\n";
        return 1;
    }
    // High precision so the (tiny, bounded) conservation drifts survive the CSV
    // round-trip and are visible in the plots.
    traj << std::setprecision(10);
    diag << std::setprecision(15);
    traj << "step,time,id,x,y,z,vx,vy,vz\n";
    diag << "step,time,kinetic,potential,total,Px,Py,Pz,Lx,Ly,Lz\n";

    auto writeFrame = [&](long step, double t) {
        for (std::size_t i = 0; i < n; ++i)
            traj << step << ',' << t << ',' << i << ','
                 << bodies.pos[i].x << ',' << bodies.pos[i].y << ',' << bodies.pos[i].z << ','
                 << bodies.vel[i].x << ',' << bodies.vel[i].y << ',' << bodies.vel[i].z << '\n';
    };
    auto writeDiag = [&](long step, double t, const Diagnostics3D& d) {
        diag << step << ',' << t << ','
             << d.kinetic << ',' << d.potential << ',' << d.total << ','
             << d.momentum.x << ',' << d.momentum.y << ',' << d.momentum.z << ','
             << d.angular.x << ',' << d.angular.y << ',' << d.angular.z << '\n';
    };

    std::cout << "=== galaxy 3D Rung 3 (DirectSolver3D) ===\n";
    std::cout << "scenario   : " << (cfg.scenario == Scenario3D::TwoBody ? "twobody" : "plummer") << "\n";
    std::cout << "N          : " << n << "\n";
    std::cout << "dt         : " << cfg.dt << "\n";
    std::cout << "steps      : " << cfg.step_count << "\n";
    std::cout << "G, eps     : " << cfg.G << ", " << cfg.eps << "\n";
    if (cfg.scenario == Scenario3D::TwoBody)
        std::cout << "period T   : " << period
                  << "  (" << (cfg.dt * cfg.step_count / period) << " periods)\n";
    std::cout << "solver     : " << solver->name();
    if (cfg.solver == Solver3D::BarnesHut) std::cout << "  theta=" << cfg.theta;
    if (cfg.solver == Solver3D::FMM) {
        const int Ld = cfg.fmm_depth >= 0 ? cfg.fmm_depth : FMM3D::autoDepth(n);
        std::cout << "  p=" << cfg.p << "  depth=" << Ld;
    }
    std::cout << "\n";
    std::cout << "outputs    : " << traj_path << ", " << diag_path << "\n";

    const Diagnostics3D d0 = computeDiagnostics(bodies, kparams);
    const double E0 = d0.total;
    const Vec3 P0 = d0.momentum;
    const Vec3 L0 = d0.angular;
    double max_rel_drift = 0.0, max_mom_drift = 0.0, max_ang_drift = 0.0;

    std::printf("step %8ld  t %9.3f  E % .8e  |P| %.3e  L=(% .4e,% .4e,% .4e)\n",
                0L, 0.0, d0.total, galaxy3d::abs(d0.momentum),
                d0.angular.x, d0.angular.y, d0.angular.z);

    writeFrame(0, 0.0);
    writeDiag(0, 0.0, d0);

    for (long step = 1; step <= cfg.step_count; ++step) {
        integrator.step(bodies, cfg.dt);
        const double t = step * cfg.dt;

        const bool do_diag = (step % cfg.frame_stride == 0) ||
                             (step % cfg.log_stride == 0) || (step == cfg.step_count);
        Diagnostics3D d;
        if (do_diag) {
            d = computeDiagnostics(bodies, kparams);
            const double abs_drift = std::abs(d.total - E0);
            max_rel_drift = std::max(max_rel_drift,
                                     std::abs(E0) > 0 ? abs_drift / std::abs(E0) : abs_drift);
            max_mom_drift = std::max(max_mom_drift, galaxy3d::abs(d.momentum - P0));
            max_ang_drift = std::max(max_ang_drift, galaxy3d::abs(d.angular - L0));
        }
        if (step % cfg.frame_stride == 0 || step == cfg.step_count) { writeFrame(step, t); writeDiag(step, t, d); }
        if (step % cfg.log_stride == 0 || step == cfg.step_count)
            std::printf("step %8ld  t %9.3f  E % .8e  |P| %.3e  L=(% .4e,% .4e,% .4e)\n",
                        step, t, d.total, galaxy3d::abs(d.momentum),
                        d.angular.x, d.angular.y, d.angular.z);
    }

    std::cout << "\n=== conservation summary ===\n";
    std::printf("E0                     : % .10e\n", E0);
    std::printf("max |dE|/|E0|          : % .6e   <-- energy drift (bounded)\n", max_rel_drift);
    std::printf("max |dP| (from P0)     : % .6e\n", max_mom_drift);
    std::printf("max |dL| (vector)      : % .6e   (L0 = %.4e,%.4e,%.4e)\n",
                max_ang_drift, L0.x, L0.y, L0.z);

    if (cfg.scenario == Scenario3D::TwoBody) printKeplerCheck(cfg);

    std::cout << "wrote: " << traj_path << "\n";
    std::cout << "wrote: " << diag_path << "\n";
    return 0;
}
