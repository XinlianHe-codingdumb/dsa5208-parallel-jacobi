# DSA5208 Project 3: Parallel 2D Jacobi diffusion

C++11 implementations of a five-point Jacobi update using serial execution,
OpenMP, MPI, and MPI + OpenMP. The grid has fixed zero boundaries and a central
hot square initialized to one. Each iteration replaces an interior cell with
the mean of its four neighbours. Two arrays separate the old and new states.
MPI ranks exchange neighbouring halo rows at every iteration.

## 1. Download and requirements

```sh
git clone https://github.com/XinlianHe-codingdumb/dsa5208-parallel-jacobi.git
cd dsa5208-parallel-jacobi
```

- Compilation and execution: Linux, a C++11 compiler, OpenMP, and an MPI
  installation providing `mpicxx` and `mpirun`.
- Supplied cluster jobs: an NUS Atlas8 account and PBS access.
- CSV/plot regeneration only: Python 3 and Matplotlib 3.6 or newer. MPI and
  Atlas access are not needed to analyze the supplied log.

Run compute-intensive commands on an allocated compute node, rather than a
cluster login node. Request enough cores for ranks x threads per rank.

## 2. Compile

Use the same MPI installation for compilation and execution. On the Atlas8
environment used for the experiment, initialize modules with:

```sh
source /etc/profile.d/rec_modules.sh
module load openmpi/2.1.1
```

On another Linux system, load its available compiler/MPI environment instead.
Compile from the repository directory:

```sh
g++ -O3 -std=c++11 -fopenmp jacobi.cpp -o jacobi
mpicxx -O3 -std=c++11 -fopenmp jacobi_mpi.cpp -o jacobi_mpi
```

| File | Implementations |
| --- | --- |
| `jacobi.cpp` | Serial and OpenMP |
| `jacobi_mpi.cpp` | MPI and MPI + OpenMP |

## 3. Check correctness

Run within a compute allocation with at least 12 cores:

```sh
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=FALSE
./jacobi --nx 64 --ny 64 --steps 20 --threads 4 --validate
mpirun --bind-to none -np 4 ./jacobi_mpi --mode mpi --nx 64 --ny 64 --steps 20 --validate
mpirun --bind-to none -np 4 ./jacobi_mpi --mode hybrid --nx 64 --ny 64 --steps 20 --threads 3 --validate
```

`--bind-to none` is an OpenMPI option; adjust launcher options for a different
MPI implementation.

`jacobi` compares OpenMP with serial cell by cell and checks an independent
5 x 5, one-step example. Expected successful output:

```text
max_abs_error=0
small_case_error=0
```

For MPI and hybrid, rank zero gathers the final grid and compares it cell by
cell with a serial reference. Each command is expected to print:

```text
max_abs_error=0
```

The pass threshold is maximum absolute error <= `1e-12`. Each validation
command returns exit status 0 on success; a nonzero status indicates a failed
check or execution error. Validation returns before printing benchmark CSV.

## 4. Run each implementation

These examples use the production workload: 2048 x 2048 cells and 1000 steps.
Execute them within an allocation with at least 12 cores.

```sh
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=FALSE

# Serial: one core
./jacobi --mode serial --nx 2048 --ny 2048 --steps 1000

# OpenMP: 12 threads
./jacobi --mode omp --nx 2048 --ny 2048 --steps 1000 --threads 12

# MPI: 12 ranks, one thread per rank
mpirun --bind-to none -np 12 ./jacobi_mpi --mode mpi --nx 2048 --ny 2048 --steps 1000

# Hybrid: four ranks, three threads per rank (12 cores total)
mpirun --bind-to none -np 4 ./jacobi_mpi --mode hybrid --nx 2048 --ny 2048 --steps 1000 --threads 3
```

Both programs accept `--nx`, `--ny`, `--steps`, `--threads`, `--mode`,
`--validate`, and `--help`. Modes are `serial|omp` for `jacobi` and `mpi|hybrid`
for `jacobi_mpi`. Dimensions must be at least 3; ranks must not exceed
`ny - 2`. Hybrid `--threads` means threads per rank; pure MPI uses one.

Benchmark commands print a CSV header and one result row with seconds and a
checksum. MPI/hybrid output also includes ranks and maximum per-rank
communication and computation times. Total timings include allocation,
initialization, and iterations, but exclude launch, MPI initialization,
checksum calculation, and output. Checksums provide a consistency check;
cell-by-cell validation above is the correctness test.

## 5. Reproduce the Atlas8 experiment

Submit from the repository directory on Atlas8:

```sh
qsub atlas_validate.pbs
qsub atlas_mpi_validate.pbs
qsub atlas_strong_scaling.pbs
```

| Script | Purpose |
| --- | --- |
| `atlas_validate.pbs` | Compile and validate serial/OpenMP; run the small example |
| `atlas_mpi_validate.pbs` | Compile and validate MPI/hybrid; run an MPI timing example |
| `atlas_strong_scaling.pbs` | Compile both programs and run the production benchmark |

Jobs are independent; inspect validation output before interpreting benchmark
results. `qsub` returns a job ID; use `qstat JOB_ID` to monitor it. PBS writes
combined output/error logs to the submission directory. Successful validation
scripts print `validation_passed` or `mpi_validation_passed`.

The production job requests one node, 12 CPU cores, 4 GB memory and a one-hour
walltime. It runs three repetitions of 14 configurations (42 runs total):

- Serial: one core.
- OpenMP: 1, 2, 4, 6 and 12 threads.
- MPI: 1, 2, 4, 6 and 12 ranks.
- Hybrid: 2 x 6, 4 x 3 and 6 x 2 ranks x threads.

The completed production log should contain 42 result rows,
`strong_scaling_complete`, and the Atlas resource summary with `Exit Status: 0`.

Scripts reflect Atlas8 on 8 November 2026. They request 12 cores, the minimum
observed in the `parallel` queue at that time. MPI scripts load
`openmpi/2.1.1`. Queue names, module paths and resource settings may need
adjustment on another system.

The production script uses `OMP_PROC_BIND=FALSE` and `mpirun --bind-to none`
because binding caused severe waiting in the original environment. Validation
scripts retain their original `OMP_PROC_BIND=TRUE` setting. The MPI scripts'
`OMPI_MCA_btl=self,vader` setting is specific to their single-node OpenMPI
setup; these jobs do not establish multi-node scaling.

## 6. Regenerate the reported CSV and figures

Analyze the supplied completed log, `dsa5208_strong.o1036561`, without running
the simulation again:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install 'matplotlib>=3.6'
python3 analyze_strong.py
```

The noninteractive plotting backend needs no graphical desktop. The script
checks completion markers, 42 runs, three repetitions per configuration,
workload dimensions, and checksum agreement within `1e-6`. If an assertion
fails, inspect the input log before interpreting results.

Files are written beside the script, replacing existing copies:

| Output | Contents |
| --- | --- |
| `strong_scaling_summary.csv` | Median, minimum and maximum times, speedup and efficiency |
| `strong_scaling.png` | Three-panel scaling figure |
| `strong_scaling_report.png` | Figure layout used in the report |

For the supplied log, console output begins `serial median: 3.1983 s`.
The 12-rank MPI median is about 0.5321 s, speedup 6.01 and efficiency 0.50.
Speedup uses the serial median as baseline; efficiency divides speedup by
ranks x threads. Plot appearance may vary slightly across Matplotlib versions.

### Analyze a new PBS run

The parser reads the fixed filename `dsa5208_strong.o1036561`; it does not
automatically discover new jobs. Preserve the supplied evidence by analyzing
a new log in a separate directory. Replace `/path/to/NEW_PBS_LOG` with the
completed production log from your job:

```sh
mkdir -p rerun_analysis
cp analyze_strong.py rerun_analysis/
cp /path/to/NEW_PBS_LOG rerun_analysis/dsa5208_strong.o1036561
python3 rerun_analysis/analyze_strong.py
```

The fixed filename in this separate directory is only the parser's input
name; the log still contains your actual job ID. Outputs go into
`rerun_analysis/`. Keep the same workload and 42-run configuration matrix for
the checks to apply. Use the full Atlas log, including its completion marker
and successful PBS exit status.

New runs reproduce the procedure, not identical timings: placement,
scheduling, memory bandwidth and system load affect performance. The report
uses medians of three repetitions and discusses the observed timing spread.

## Included evidence and scope

The raw production log, summary CSV and report figure are included. Separate
validation-job logs are not included; use the source and PBS scripts to rerun
validation. The PDF report is submitted separately and describes the method,
validation, design choices, results, limitations and AI use. Input is
synthetic; no real-data conclusions are claimed.
