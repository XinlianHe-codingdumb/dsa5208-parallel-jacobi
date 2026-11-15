#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

struct Options {
    int nx = 512;
    int ny = 512;
    int steps = 500;
    int threads = 1;
    std::string mode = "serial";
    bool validate = false;
};

int positive_int(const char* value, const char* name) {
    const std::string text(value);
    std::size_t end = 0;
    const long parsed = std::stol(text, &end);
    if (end != text.size() || parsed < 1 || parsed > 100000000) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
    return static_cast<int>(parsed);
}

Options parse_options(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--validate") {
            opt.validate = true;
        } else if (arg == "--help") {
            std::cout << "Usage: jacobi [--nx N] [--ny N] [--steps N] "
                         "[--mode serial|omp] [--threads N] [--validate]\n";
            std::exit(0);
        } else {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
            const char* value = argv[++i];
            if (arg == "--nx") opt.nx = positive_int(value, "nx");
            else if (arg == "--ny") opt.ny = positive_int(value, "ny");
            else if (arg == "--steps") opt.steps = positive_int(value, "steps");
            else if (arg == "--threads") opt.threads = positive_int(value, "threads");
            else if (arg == "--mode") opt.mode = value;
            else throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if (opt.nx < 3 || opt.ny < 3) throw std::invalid_argument("nx and ny must be at least 3");
    if (opt.mode != "serial" && opt.mode != "omp") {
        throw std::invalid_argument("mode must be serial or omp");
    }
#ifndef _OPENMP
    if (opt.mode == "omp" || opt.validate) {
        throw std::runtime_error("OpenMP support is required; compile with -fopenmp");
    }
#endif
    return opt;
}

std::vector<double> initial_grid(int nx, int ny) {
    std::vector<double> grid(static_cast<std::size_t>(nx) * ny, 0.0);
    // Boundary cells remain zero throughout the simulation.
    for (int y = 1; y < ny - 1; ++y) {
        for (int x = 1; x < nx - 1; ++x) {
            if (x >= 2 * nx / 5 && x < 3 * nx / 5 &&
                y >= 2 * ny / 5 && y < 3 * ny / 5) {
                grid[static_cast<std::size_t>(y) * nx + x] = 1.0;
            }
        }
    }
    return grid;
}

std::vector<double> solve(int nx, int ny, int steps, bool parallel) {
    std::vector<double> current = initial_grid(nx, ny);
    std::vector<double> next(current.size(), 0.0);
    for (int step = 0; step < steps; ++step) {
        if (parallel) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
            for (int y = 1; y < ny - 1; ++y) {
                for (int x = 1; x < nx - 1; ++x) {
                    const std::size_t k = static_cast<std::size_t>(y) * nx + x;
                    next[k] = 0.25 * (current[k - nx] + current[k + nx] +
                                      current[k - 1] + current[k + 1]);
                }
            }
#endif
        } else {
            for (int y = 1; y < ny - 1; ++y) {
                for (int x = 1; x < nx - 1; ++x) {
                    const std::size_t k = static_cast<std::size_t>(y) * nx + x;
                    next[k] = 0.25 * (current[k - nx] + current[k + nx] +
                                      current[k - 1] + current[k + 1]);
                }
            }
        }
        current.swap(next);
    }
    return current;
}

int main(int argc, char** argv) {
    try {
        const Options opt = parse_options(argc, argv);
#ifdef _OPENMP
        omp_set_dynamic(0);
        omp_set_num_threads(opt.threads);
#endif
        if (opt.validate) {
            const auto reference = solve(opt.nx, opt.ny, opt.steps, false);
            const auto candidate = solve(opt.nx, opt.ny, opt.steps, true);
            double max_error = 0.0;
            for (std::size_t i = 0; i < reference.size(); ++i) {
                max_error = std::max(max_error, std::abs(reference[i] - candidate[i]));
            }
            const auto small = solve(5, 5, 1, false);
            double small_error = 0.0;
            for (int y = 0; y < 5; ++y) {
                for (int x = 0; x < 5; ++x) {
                    const double expected = ((x == 2 && (y == 1 || y == 3)) ||
                                             (y == 2 && (x == 1 || x == 3))) ? 0.25 : 0.0;
                    small_error = std::max(small_error,
                                           std::abs(small[static_cast<std::size_t>(y) * 5 + x] - expected));
                }
            }
            std::cout << std::setprecision(17)
                      << "max_abs_error=" << max_error << '\n'
                      << "small_case_error=" << small_error << '\n';
            return max_error <= 1e-12 && small_error <= 1e-12 ? 0 : 1;
        }

        const auto start = std::chrono::steady_clock::now();
        const auto grid = solve(opt.nx, opt.ny, opt.steps, opt.mode == "omp");
        const auto end = std::chrono::steady_clock::now();
        double checksum = 0.0;
        for (double value : grid) checksum += value;
        const double seconds = std::chrono::duration<double>(end - start).count();
        std::cout << "mode,nx,ny,steps,threads,seconds,checksum\n"
                  << opt.mode << ',' << opt.nx << ',' << opt.ny << ',' << opt.steps
                  << ',' << (opt.mode == "omp" ? opt.threads : 1) << ','
                  << std::setprecision(17) << seconds << ',' << checksum << '\n';
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 2;
    }
}
