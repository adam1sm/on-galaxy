#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

namespace galaxy3d {

enum class Scenario3D { TwoBody, Plummer };
enum class Solver3D { Direct, BarnesHut, FMM };

// Full 3D run configuration, populated from command-line flags.
struct Config3D {
    Scenario3D scenario = Scenario3D::Plummer;
    Solver3D solver = Solver3D::Direct;

    // Solver parameters
    double theta = 0.5;   // Barnes-Hut opening angle
    int p = 8;            // FMM expansion order
    int fmm_depth = -1;   // FMM octree depth; -1 = auto (8^L ~ N)

    // Kernel / physics
    double G = 1.0;
    double eps = 0.05;

    // Integration
    double dt = 0.002;
    long step_count = 4000;

    // Plummer cluster
    std::size_t N = 1000;
    double plummer_a = 1.0;
    double plummer_mass = 1.0;
    unsigned seed = 7;

    // Two-body
    double tb_m1 = 1.0;
    double tb_m2 = 1.0;
    double tb_separation = 2.0;

    // Output
    std::string out_prefix = "outputs/run3d";
    long frame_stride = 10;
    long log_stride = 100;
};

inline void printUsage3D(const char* prog) {
    std::cout <<
        "Usage: " << prog << " [options]\n"
        "  --solver <direct3d|bh3d|fmm3d> force solver (default: direct3d)\n"
        "  --theta <float>               BH opening angle (default: 0.5)\n"
        "  --p <int>                     FMM expansion order (default: 8)\n"
        "  --fmm-depth <int>             FMM octree depth (default: auto, 8^L~N)\n"
        "  --scenario <twobody|plummer>  initial condition (default: plummer)\n"
        "  --N <int>                     body count for plummer (default: 1000)\n"
        "  --dt <float>                  timestep (default: 0.002)\n"
        "  --steps <int>                 number of steps (default: 4000)\n"
        "  --eps <float>                 Plummer softening (default: 0.05)\n"
        "  --G <float>                   gravitational constant (default: 1)\n"
        "  --seed <int>                  RNG seed (default: 7)\n"
        "  --plummer-a <float>           Plummer scale radius (default: 1)\n"
        "  --plummer-mass <float>        total cluster mass (default: 1)\n"
        "  --tb-m1 <float>               two-body mass 1 (default: 1)\n"
        "  --tb-m2 <float>               two-body mass 2 (default: 1)\n"
        "  --tb-sep <float>              two-body separation (default: 2)\n"
        "  --out <prefix>                output path prefix (default: outputs/run3d)\n"
        "  --frame-stride <int>          steps between trajectory frames (default: 10)\n"
        "  --log-stride <int>            steps between stdout diagnostics (default: 100)\n"
        "  --help                        show this message\n";
}

inline bool parseConfig3D(int argc, char** argv, Config3D& cfg) {
    auto need = [&](int& i) -> std::string {
        if (i + 1 >= argc) { std::cerr << "missing value for " << argv[i] << "\n"; std::exit(2); }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { printUsage3D(argv[0]); return false; }
        else if (a == "--solver") {
            std::string s = need(i);
            if (s == "direct3d") cfg.solver = Solver3D::Direct;
            else if (s == "bh3d") cfg.solver = Solver3D::BarnesHut;
            else if (s == "fmm3d") cfg.solver = Solver3D::FMM;
            else { std::cerr << "unknown solver: " << s << "\n"; std::exit(2); }
        }
        else if (a == "--theta") cfg.theta = std::stod(need(i));
        else if (a == "--p") cfg.p = std::stoi(need(i));
        else if (a == "--fmm-depth") cfg.fmm_depth = std::stoi(need(i));
        else if (a == "--scenario") {
            std::string s = need(i);
            if (s == "twobody") cfg.scenario = Scenario3D::TwoBody;
            else if (s == "plummer") cfg.scenario = Scenario3D::Plummer;
            else { std::cerr << "unknown scenario: " << s << "\n"; std::exit(2); }
        }
        else if (a == "--N") cfg.N = std::stoul(need(i));
        else if (a == "--dt") cfg.dt = std::stod(need(i));
        else if (a == "--steps") cfg.step_count = std::stol(need(i));
        else if (a == "--eps") cfg.eps = std::stod(need(i));
        else if (a == "--G") cfg.G = std::stod(need(i));
        else if (a == "--seed") cfg.seed = static_cast<unsigned>(std::stoul(need(i)));
        else if (a == "--plummer-a") cfg.plummer_a = std::stod(need(i));
        else if (a == "--plummer-mass") cfg.plummer_mass = std::stod(need(i));
        else if (a == "--tb-m1") cfg.tb_m1 = std::stod(need(i));
        else if (a == "--tb-m2") cfg.tb_m2 = std::stod(need(i));
        else if (a == "--tb-sep") cfg.tb_separation = std::stod(need(i));
        else if (a == "--out") cfg.out_prefix = need(i);
        else if (a == "--frame-stride") cfg.frame_stride = std::stol(need(i));
        else if (a == "--log-stride") cfg.log_stride = std::stol(need(i));
        else { std::cerr << "unknown option: " << a << "\n"; printUsage3D(argv[0]); std::exit(2); }
    }
    return true;
}

} // namespace galaxy3d
