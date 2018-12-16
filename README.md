# All-Pair-Shortest-Path GPU Implementation
____
**Author:** Yair Schiff\
**Project:** Fall 2018 CSCI-GA 3033-004: GPUs\
**Instructor:** Prof. Zahran
____

##Overview
This project compares an efficient GPU implementation of the Floyd-Warshall (FW)
algorithm for solving the all-pair-shortest-path (APSP) problem with that of a
CPU implementation and other GPU implementations.


##Requirements
1) CUDA compiler - `module load cuda-9.1`
2) OpenACC compiler - `module load module load pgi`


##How to use
### Compilation
To compile the CUDA version of the parallel FW algorithm use the `Makefile` provided in the `src` directory.

```bash
make ARCH=<sm architecture>
```

This will compile the FW.cu file and produce an executable `FW` binary for the GPU architecture provided in the argument `ARCH`. To remove this file you can run
```bash
make clean
```

To compile the OpenACC version of the code use one of the following two commands (once the compiler has been loaded)

(For GPU)
```bash
pgcc -o FW_acc -acc -Minfo FW_openacc.c
```

(For Multi-Core)
```bash
pgcc -o FW_multi -acc -ta=multicore -Minfo FW_openacc.c
```

The GPU binary file, `FW_acc`, and that from the multi-core compilations, `FW_multi`, will have the same usage as the CUDA code, as described below.


###Usage
To run the algorithm on a CPU, GPU, or Multi-Core processor use the following command line from within the `/src` directory where the code was compiled (shown for the CUDA binary file):
```bash
./FW <path to input file> <N> <machine> <verbose>
```
The first argument `<path to input file>` points to the input matrix. You can use `../inputs/20181101.as-rel.txt` to use the CAIDA AS Relationship data from the project report. Note, when using other inputs, the code assumes that the data is in the following format: `i | j | weight\n`, which represents a weighted directed edge from vertex i to vertex j (both of which are integers).

The second argument, indicates how much of the input matrix to use. For example, `N=1000` will use the first 1,000 vertices of the graph.

The third argument should be either `0` (sequential implementation on CPU) or `1` (parallel implementation on GPU or multi-core).

The final argument should be `0`, but when set to `1`, will print an `N`x`N` table of the original input matrix.

Once the algorithm has finished running it will output the time taken and will save a file titled: `solution_path_<N>.txt` to the `/outputs` directory which contains the path and length (if a path exists) between all `(i, j)` vertex pairs.
