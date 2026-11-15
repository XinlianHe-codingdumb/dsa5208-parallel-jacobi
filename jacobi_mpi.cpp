#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif

struct Options {
    int nx = 512, ny = 512, steps = 500, threads = 1;
    std::string mode = "mpi";
    bool validate = false;
};

int positive_int(const char* value) {
    const std::string text(value);
    std::size_t end = 0;
    const long number = std::stol(text, &end);
    if (end != text.size() || number < 1 || number > 100000000)
        throw std::invalid_argument("expected a positive integer");
    return static_cast<int>(number);
}

Options parse(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--validate") opt.validate = true;
        else if (arg == "--help") {
            std::cout << "Usage: jacobi_mpi --mode mpi|hybrid --nx N --ny N "
                         "--steps N --threads N [--validate]\n";
            std::exit(0);
        } else {
            if (++i >= argc) throw std::invalid_argument("missing argument for " + arg);
            if (arg == "--nx") opt.nx = positive_int(argv[i]);
            else if (arg == "--ny") opt.ny = positive_int(argv[i]);
            else if (arg == "--steps") opt.steps = positive_int(argv[i]);
            else if (arg == "--threads") opt.threads = positive_int(argv[i]);
            else if (arg == "--mode") opt.mode = argv[i];
            else throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if (opt.nx < 3 || opt.ny < 3 || static_cast<long long>(opt.nx) * opt.ny > INT_MAX)
        throw std::invalid_argument("grid must be at least 3x3 and have at most INT_MAX cells");
    if (opt.mode != "mpi" && opt.mode != "hybrid")
        throw std::invalid_argument("mode must be mpi or hybrid");
#ifndef _OPENMP
    if (opt.mode == "hybrid") throw std::runtime_error("compile with -fopenmp");
#endif
    return opt;
}

int rows_for(int rank, int ranks, int interior_rows) {
    return interior_rows / ranks + (rank < interior_rows % ranks ? 1 : 0);
}

int first_row(int rank, int ranks, int interior_rows) {
    return 1 + rank * (interior_rows / ranks) +
           std::min(rank, interior_rows % ranks);
}

std::vector<double> serial_reference(int nx, int ny, int steps) {
    std::vector<double> a(static_cast<std::size_t>(nx) * ny, 0.0), b(a.size(), 0.0);
    for (int y = 1; y < ny - 1; ++y)
        for (int x = 1; x < nx - 1; ++x)
            if (x >= 2 * nx / 5 && x < 3 * nx / 5 &&
                y >= 2 * ny / 5 && y < 3 * ny / 5)
                a[static_cast<std::size_t>(y) * nx + x] = 1.0;
    for (int step = 0; step < steps; ++step) {
        for (int y = 1; y < ny - 1; ++y)
            for (int x = 1; x < nx - 1; ++x) {
                const std::size_t k = static_cast<std::size_t>(y) * nx + x;
                b[k] = 0.25 * (a[k - nx] + a[k + nx] + a[k - 1] + a[k + 1]);
            }
        a.swap(b);
    }
    return a;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try {
        if (provided < MPI_THREAD_FUNNELED)
            throw std::runtime_error("MPI_THREAD_FUNNELED is unavailable");
        const Options opt = parse(argc, argv);
        const int interior = opt.ny - 2;
        if (ranks > interior) throw std::invalid_argument("too many ranks for grid rows");
#ifdef _OPENMP
        omp_set_dynamic(0);
        omp_set_num_threads(opt.mode == "hybrid" ? opt.threads : 1);
#endif
        const int rows = rows_for(rank, ranks, interior);
        const int start_y = first_row(rank, ranks, interior);
        const int nx = opt.nx;
        const int above = rank == 0 ? MPI_PROC_NULL : rank - 1;
        const int below = rank == ranks - 1 ? MPI_PROC_NULL : rank + 1;
        MPI_Barrier(MPI_COMM_WORLD);
        const double start_time = MPI_Wtime();
        std::vector<double> a(static_cast<std::size_t>(rows + 2) * nx, 0.0);
        std::vector<double> b(a.size(), 0.0);
        double comm_seconds_local = 0.0;
        double compute_seconds_local = 0.0;
        for (int local_y = 1; local_y <= rows; ++local_y) {
            const int global_y = start_y + local_y - 1;
            for (int x = 1; x < nx - 1; ++x)
                if (x >= 2 * nx / 5 && x < 3 * nx / 5 &&
                    global_y >= 2 * opt.ny / 5 && global_y < 3 * opt.ny / 5)
                    a[static_cast<std::size_t>(local_y) * nx + x] = 1.0;
        }
        for (int step = 0; step < opt.steps; ++step) {
            const double comm_start = MPI_Wtime();
            MPI_Request requests[4];
            MPI_Irecv(&a[0], nx, MPI_DOUBLE, above, 11, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(&a[static_cast<std::size_t>(rows + 1) * nx], nx, MPI_DOUBLE,
                      below, 10, MPI_COMM_WORLD, &requests[1]);
            MPI_Isend(&a[nx], nx, MPI_DOUBLE, above, 10, MPI_COMM_WORLD, &requests[2]);
            MPI_Isend(&a[static_cast<std::size_t>(rows) * nx], nx, MPI_DOUBLE,
                      below, 11, MPI_COMM_WORLD, &requests[3]);
            MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
            comm_seconds_local += MPI_Wtime() - comm_start;
            const double compute_start = MPI_Wtime();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(opt.mode == "hybrid")
#endif
            for (int local_y = 1; local_y <= rows; ++local_y)
                for (int x = 1; x < nx - 1; ++x) {
                    const std::size_t k = static_cast<std::size_t>(local_y) * nx + x;
                    b[k] = 0.25 * (a[k - nx] + a[k + nx] + a[k - 1] + a[k + 1]);
                }
            a.swap(b);
            compute_seconds_local += MPI_Wtime() - compute_start;
        }
        const double local_seconds = MPI_Wtime() - start_time;
        double seconds = 0.0;
        MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        double comm_seconds = 0.0, compute_seconds = 0.0;
        MPI_Reduce(&comm_seconds_local, &comm_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&compute_seconds_local, &compute_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        double local_sum = 0.0, checksum = 0.0;
        for (int y = 1; y <= rows; ++y)
            for (int x = 0; x < nx; ++x)
                local_sum += a[static_cast<std::size_t>(y) * nx + x];
        MPI_Reduce(&local_sum, &checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (opt.validate) {
            std::vector<int> counts, displacements;
            std::vector<double> gathered;
            if (rank == 0) {
                counts.resize(ranks);
                displacements.resize(ranks);
                gathered.assign(static_cast<std::size_t>(nx) * opt.ny, 0.0);
                for (int r = 0; r < ranks; ++r) {
                    counts[r] = nx * rows_for(r, ranks, interior);
                    displacements[r] = nx * first_row(r, ranks, interior);
                }
            }
            MPI_Gatherv(&a[nx], rows * nx, MPI_DOUBLE,
                        rank == 0 ? &gathered[0] : NULL,
                        rank == 0 ? &counts[0] : NULL,
                        rank == 0 ? &displacements[0] : NULL,
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
            int passed = 1;
            if (rank == 0) {
                const std::vector<double> reference =
                    serial_reference(nx, opt.ny, opt.steps);
                double error = 0.0;
                for (std::size_t i = 0; i < reference.size(); ++i)
                    error = std::max(error, std::abs(reference[i] - gathered[i]));
                std::cout << std::setprecision(17) << "max_abs_error=" << error << '\n';
                passed = error <= 1e-12 ? 1 : 0;
            }
            MPI_Bcast(&passed, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Finalize();
            return passed ? 0 : 1;
        }
        if (rank == 0)
            std::cout << "mode,nx,ny,steps,ranks,threads,seconds,checksum,comm_seconds_max,compute_seconds_max\n"
                      << opt.mode << ',' << nx << ',' << opt.ny << ',' << opt.steps
                      << ',' << ranks << ',' << (opt.mode == "hybrid" ? opt.threads : 1)
                      << ',' << std::setprecision(17) << seconds << ',' << checksum
                      << ',' << comm_seconds << ',' << compute_seconds << '\n';
        MPI_Finalize();
        return 0;
    } catch (const std::exception& e) {
        if (rank == 0) std::cerr << "error: " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }
}
