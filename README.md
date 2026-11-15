# DSA5208 Project 3: Parallel 2D Jacobi diffusion

This archive contains two C++11 programs:

- `jacobi.cpp`: serial and OpenMP variants.
- `jacobi_mpi.cpp`: MPI and MPI + OpenMP variants.

The grid has fixed zero boundaries and a central hot square. Each step replaces
an interior cell with the mean of its four neighbours. Options are `--nx N`,
`--ny N`, `--steps N`, `--threads N`, and `--mode`. The MPI executable additionally
accepts `--validate` to gather the result and compare it with a serial reference.
The non-MPI executable's `--validate` checks OpenMP against serial and a known
5 by 5 result. A nonzero exit status means validation failed.

## Build and run

Compile and run on an allocated compute node with GCC, MPI and OpenMP available:

```sh
g++ -O3 -std=c++11 -fopenmp jacobi.cpp -o jacobi
mpicxx -O3 -std=c++11 -fopenmp jacobi_mpi.cpp -o jacobi_mpi
./jacobi --nx 64 --ny 64 --steps 20 --threads 4 --validate
mpirun -np 4 ./jacobi_mpi --mode mpi --nx 64 --ny 64 --steps 20 --validate
mpirun -np 4 ./jacobi_mpi --mode hybrid --nx 64 --ny 64 --steps 20 --threads 3 --validate
```

On NUS Atlas8, submit the included PBS scripts from the directory containing
the C++ files:

```sh
qsub atlas_validate.pbs
qsub atlas_mpi_validate.pbs
qsub atlas_strong_scaling.pbs
```

The scripts request 12 cores, the minimum seen in the Atlas8 `parallel` queue
on 8 November 2026. The MPI scripts load `openmpi/2.1.1`. The production
benchmark sets `OMP_PROC_BIND=FALSE`; the opposite setting caused severe MPI
waiting on this system. These scripts may need adjustment if queue or module
settings change.

## Results and analysis

`dsa5208_strong.o1036561` is the raw Atlas8 production job log, including
three runs for each configuration and the PBS exit status. The run used a
2048 x 2048 grid, 1000 steps and one compute node. `strong_scaling_summary.csv`
contains median times and derived speedup and efficiency. The Python script
`analyze_strong.py` checks row counts and checksums, then regenerates the CSV
and figures; it requires `matplotlib`.

The PDF report documents the method, validation, design choices, results,
limitations and AI use. Synthetic input was used; no real-data conclusions are
claimed.
