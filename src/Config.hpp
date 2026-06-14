#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

namespace galaxy {

// Initial-condition selector.
enum class Scenario { TwoBody, Disk };

// Force-solver selector.
enum class SolverKind { Direct, BarnesHut, FMM };

// Full run configuration. Populated from command-line flags (see parseConfig).
struct Config {
    Scenario scenario = Scenario::Disk;
    SolverKind solver = SolverKind::Direct;

    // Barnes-Hut parameters (ignored by the direct solver)
    double theta = 0.5;
    int leaf_capacity = 1;
    int max_depth = 64;

    // FMM parameters (ignored by other solvers)
    int p = 8;             // expansion order
    int fmm_depth = -1;    // uniform tree depth; -1 = auto (4^L ~ N)

    // Kernel / physics
    double G = 1.0;
    double eps = 0.01;

    // Integration
    double dt = 0.002;
    long step_count = 5000;

    // Disk initial condition
    std::size_t N = 500;
    double disk_radius = 1.0;
    double disk_mass = 1.0;
    double disk_rotation = 0.6;
    double disk_thermal = 0.05;
    unsigned seed = 42;

    // Two-body initial condition
    double tb_m1 = 1.0;
    double tb_m2 = 1.0;
    double tb_separation = 2.0;

    // Output
    std::string out_prefix = "outputs/run"; // -> <prefix>_trajectory.csv etc.
    long frame_stride = 10;  // write a trajectory frame every N steps
    long log_stride = 100;   // log diagnostics to stdout every N steps
};

inline void printUsage(const char* prog) {
    std::cout <<
        "Usage: " << prog << " [options]\n"
        "  --solver <direct|bh|fmm>    force solver (default: direct)\n"
        "  --theta <float>             Barnes-Hut opening angle (default: 0.5)\n"
        "  --leaf-capacity <int>       BH bodies per leaf bucket (default: 1)\n"
        "  --max-depth <int>           BH max tree depth (default: 64)\n"
        "  --p <int>                   FMM expansion order (default: 8)\n"
        "  --fmm-depth <int>           FMM uniform tree depth (default: auto, 4^L~N)\n"
        "  --scenario <twobody|disk>   initial condition (default: disk)\n"
        "  --N <int>                   body count for disk (default: 500)\n"
        "  --dt <float>                timestep (default: 0.002)\n"
        "  --steps <int>               number of steps (default: 5000)\n"
        "  --eps <float>               softening length (default: 0.01)\n"
        "  --G <float>                 gravitational constant (default: 1)\n"
        "  --seed <int>                RNG seed for disk (default: 42)\n"
        "  --disk-radius <float>       disk radius (default: 1)\n"
        "  --disk-mass <float>         total disk mass (default: 1)\n"
        "  --disk-rotation <float>     rigid rotation scale (default: 0.6)\n"
        "  --disk-thermal <float>      thermal velocity stddev (default: 0.05)\n"
        "  --tb-m1 <float>             two-body mass 1 (default: 1)\n"
        "  --tb-m2 <float>             two-body mass 2 (default: 1)\n"
        "  --tb-sep <float>            two-body separation (default: 2)\n"
        "  --out <prefix>              output path prefix (default: outputs/run)\n"
        "  --frame-stride <int>        steps between trajectory frames (default: 10)\n"
        "  --log-stride <int>          steps between stdout diagnostics (default: 100)\n"
        "  --help                      show this message\n";
}

// Minimal, dependency-free flag parser. Returns false if the program should
// exit (after --help or on error).
inline bool parseConfig(int argc, char** argv, Config& cfg) {
    auto need = [&](int& i) -> std::string {
        if (i + 1 >= argc) {
            std::cerr << "missing value for " << argv[i] << "\n";
            std::exit(2);
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { printUsage(argv[0]); return false; }
        else if (a == "--solver") {
            std::string s = need(i);
            if (s == "direct") cfg.solver = SolverKind::Direct;
            else if (s == "bh") cfg.solver = SolverKind::BarnesHut;
            else if (s == "fmm") cfg.solver = SolverKind::FMM;
            else { std::cerr << "unknown solver: " << s << "\n"; std::exit(2); }
        }
        else if (a == "--theta") cfg.theta = std::stod(need(i));
        else if (a == "--leaf-capacity") cfg.leaf_capacity = std::stoi(need(i));
        else if (a == "--max-depth") cfg.max_depth = std::stoi(need(i));
        else if (a == "--p") cfg.p = std::stoi(need(i));
        else if (a == "--fmm-depth") cfg.fmm_depth = std::stoi(need(i));
        else if (a == "--scenario") {
            std::string s = need(i);
            if (s == "twobody") cfg.scenario = Scenario::TwoBody;
            else if (s == "disk") cfg.scenario = Scenario::Disk;
            else { std::cerr << "unknown scenario: " << s << "\n"; std::exit(2); }
        }
        else if (a == "--N") cfg.N = std::stoul(need(i));
        else if (a == "--dt") cfg.dt = std::stod(need(i));
        else if (a == "--steps") cfg.step_count = std::stol(need(i));
        else if (a == "--eps") cfg.eps = std::stod(need(i));
        else if (a == "--G") cfg.G = std::stod(need(i));
        else if (a == "--seed") cfg.seed = static_cast<unsigned>(std::stoul(need(i)));
        else if (a == "--disk-radius") cfg.disk_radius = std::stod(need(i));
        else if (a == "--disk-mass") cfg.disk_mass = std::stod(need(i));
        else if (a == "--disk-rotation") cfg.disk_rotation = std::stod(need(i));
        else if (a == "--disk-thermal") cfg.disk_thermal = std::stod(need(i));
        else if (a == "--tb-m1") cfg.tb_m1 = std::stod(need(i));
        else if (a == "--tb-m2") cfg.tb_m2 = std::stod(need(i));
        else if (a == "--tb-sep") cfg.tb_separation = std::stod(need(i));
        else if (a == "--out") cfg.out_prefix = need(i);
        else if (a == "--frame-stride") cfg.frame_stride = std::stol(need(i));
        else if (a == "--log-stride") cfg.log_stride = std::stol(need(i));
        else { std::cerr << "unknown option: " << a << "\n"; printUsage(argv[0]); std::exit(2); }
    }
    return true;
}

} // namespace galaxy
