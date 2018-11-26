/*
 * Author: Yair Schiff
 * Project: Fall 2018 CSCI.GA 3033-004: GPUs
 * Instructor: Prof. Zahran
 *
 * Project Description: This project explores the efficient implementation of the Floyd-Warshall (FW) algorithm, a
 * solution for the All-Pairs-Shortest-Path (APSP) and Transitive Closure problems
 *
 * Program Description: This program implements sequential (CPU) and parallel (GPU) versions of the algorithm.
 */

#include "FW_helper.h"

/*****************************************************************/
// Forward declarations
void FW_sequential(int *adj_matrix, int *go_to, unsigned int N);
void FW_parallel(int *adj_matrix, int *go_to, unsigned int N);
/*****************************************************************/


int main(int argc, char *argv[]) {
    // Check that correct number of command line arguments given
    if (argc != 4) {
        fprintf(stderr, "usage: FW_seq <input> <N>\n");
        fprintf(stderr, "input = file containing adjacency matrix for the graph\n");
        fprintf(stderr, "N = number for vertices from input graph to use\n");
        fprintf(stderr, "who = 0: sequential code on CPU, 1: GPU execution\n");
        exit(1);
    }

    // Parse command line arguments
    const char *input_file_name = argv[1]; // input file
    unsigned int N;  // Number of vertices to use
    N = convert(argv[2]);
    if (N > MAX_GRAPH) {
        fprintf(stderr, "Max graph size allowed %u x %u. Defaulting to this size.", MAX_GRAPH, MAX_GRAPH);
        N = MAX_GRAPH;
    }
    int type_of_device = 0; // CPU or GPU
    type_of_device = atoi(argv[3]);

    // Allocate memory for NxN adjacency matrix
    int *adj_matrix;
    adj_matrix = calloc( N * N, sizeof(int));
    if (adj_matrix == NULL) {
        fprintf(stderr, "malloc for adjacency matrix of size %u x %u failed.", N, N);
        exit(1);
    }

    // Allocate memory for NxN go_to matrix:
    int *go_to;
    go_to = (int *) malloc(sizeof(int) * N * N);
    if (go_to == NULL) {
        fprintf(stderr, "malloc for go_to matrix of size %u x %u failed.", N, N);
        exit(1);
    }

    // Read input and populate edges
    read_input(input_file_name, adj_matrix, N);

    // Pre-process adjacency matrix and next index matrix
    preprocess_graph(adj_matrix, go_to, N);
    print_adj(adj_matrix, N);

    // Declare variables for tracking time
    double time_taken;
    clock_t clock_start, clock_end;

    // Dispatch FW to either sequential or parallel version based on flag passed in
    if (!type_of_device) { // The CPU sequential version
        clock_start = clock();
        FW_sequential(adj_matrix, go_to, N);
        clock_end = clock();
        time_taken = ((double) clock_end - clock_start) / CLOCKS_PER_SEC;
        printf("Time taken to run FW algorithm sequentially: %lf seconds\n", time_taken);
    }
    else { // The GPU version
        clock_start = clock();
        FW_parallel(adj_matrix, go_to, N);
        clock_end = clock();
        time_taken = ((double) clock_end - clock_start) / CLOCKS_PER_SEC;
        printf("Time taken to run FW algorithm in parallel: %lf seconds\n", time_taken);
    }

    // Print solution path between every pair of vertices
    print_path(adj_matrix, go_to, N);

    free(adj_matrix);
    free(go_to)
    return 0;
}

/*******************************************************************************************************************
 * Floyd-Warshall algorithm to solve APSP problem sequentially
 *******************************************************************************************************************/
void FW_sequential(int *adj_matrix, int *go_to, unsigned int N) {
    unsigned int i, j, k;
    for (k = 0; k < N; k++) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                if (adj_matrix[index(i, j, N)] > (adj_matrix[index(i, k, N)] + adj_matrix[index(k, j, N)])) {
                    adj_matrix[index(i, j, N)] = adj_matrix[index(i, k, N)] + adj_matrix[index(k, j, N)];
                    go_to[index(i, j, N)] = (int) k;
                }
            }
        }
    }
}

/*******************************************************************************************************************
 * Kernel for running inner double for-loops of FW in parallel
 *******************************************************************************************************************/
__global__
void FW_kernel(int *adj_matrix, int *go_to, unsigned int N, int k) {
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) { // Boundary check
        if (adj_matrix[index(i, j, N)] > (adj_matrix[index(i, k, N)] + adj_matrix[index(k, j, N)])) {
            adj_matrix[index(i, j, N)] = adj_matrix[index(i, k, N)] + adj_matrix[index(k, j, N)];
            go_to[index(i, j, N)] = k;
        }
    }
}

/*******************************************************************************************************************
 * Floyd-Warshall algorithm to solve APSP problem on GPU
 *******************************************************************************************************************/
void FW_parallel(int *adj_matrix, int *go_to, unsigned int N) {
    // Allocate memory on GPU for NxN adjacency and next index matrices
    int num_bytes = sizeof(int) * N * N;
    int * adj_matrix_d, go_to_d;
    cudaError_t err = cudaMalloc((void **) &adj_matrix_d, num_bytes);
    CUDA_ERROR_CHECK(err);
    err = cudaMemcpy(adj_matrix_d, adj_matrix, num_bytes, cudaMemcpyHostToDevice);
    CUDA_ERROR_CHECK(err);
    err = cudaMalloc((void **) &go_to_d, num_bytes);
    CUDA_ERROR_CHECK(err);
    err = cudaMemcpy(go_to_d, go_to, num_bytes, cudaMemcpyHostToDevice);
    CUDA_ERROR_CHECK(err);

    // Get warp size from device properties and set it as block size
    err = cudaGetDeviceProperties(&dev_prop, 0);
    CUDA_ERROR_CHECK(err);
    int warp_size = dev_prop.warpSize;
    int dim_helper = ceil(prime_end/((double) warp_size));
    dim3 dimGrid(dim_helper, dim_helper);
    dim3 dimBlock(warp_size, warp_size);

    // Run FW triple-loop by launching a new kernel for each k
    unsigned int k;
    for (k = 0; k < N; k++) {
        FW_kernel<<dimGrid, dimBlock>>(adj_matrix_d, go_to_d, N, (int) k);
        err = cudaGetLastError();
        CUDA_ERROR_CHECK(err);
    }

    // Copy solution back to host
    err = cudaMemcpy(adj_matrix, adj_matrix_d, num_bytes, cudaMemcpyDeviceToHost);
    CUDA_ERROR_CHECK(err);
    err = cudaFree(adj_matrix_d);
    CUDA_ERROR_CHECK(err);
    err = cudaMemcpy(go_to, go_to_d, num_bytes, cudaMemcpyDeviceToHost);
    CUDA_ERROR_CHECK(err);
    err = cudaFree(go_to_d);
    CUDA_ERROR_CHECK(err);
}
