/*
 * Author: Yair Schiff
 * Project: Fall 2018 CSCI.GA 3033-004: GPUs
 * Instructor: Prof. Zahran
 *
 * Project Description: This project explores the efficient implementation of the Floyd-Warshall (FW) algorithm, a
 * solution for the All-Pairs-Shortest-Path (APSP) and Transitive Closure problems. This project will compare sequential
 * (CPU) and parallel (GPU) versions of the algorithm.
 */

#include <ctype.h>
#include <cuda.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/*****************************************************************
 * Macros
*****************************************************************/
#define MAX_GRAPH 397020 // max graph size
#define MAX_BUF 1000 // integer size of buffer for file reading
#define index(i, j, N)  ((i)*(N)) + (j) // To index element (i,j) of a 2D array stored as 1D
// Macro for error checking cuda API calls
#define CUDA_ERROR_CHECK(err) {\
    if (err != cudaSuccess) {\
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);\
        exit(1);\
    }\
}

/*****************************************************************
 * Forward declarations
*****************************************************************/
unsigned int convert(char *st);
void read_input(const char *fn, int *adj_matrix, unsigned int N);
void preprocess_graph(int *adj_matrix, int *go_to, unsigned int N);
void print_adj(int *adj_matrix, unsigned int N);
void print_path(int *adj_matrix, int *go_to, unsigned int N);
void print_path_recursive(int *go_to, unsigned int i, unsigned int j, unsigned int N);
void FW_sequential(int *adj_matrix, int *go_to, unsigned int N);
void FW_parallel(int *adj_matrix, int *go_to, unsigned int N);
/*****************************************************************/

/*****************************************************************
 * main method
*****************************************************************/
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
    adj_matrix = (int *) calloc( N * N, sizeof(int));
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
    free(go_to);
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
    int *adj_matrix_d;
    int *go_to_d;
    cudaError_t err = cudaMalloc((void **) &adj_matrix_d, num_bytes);
    CUDA_ERROR_CHECK(err);
    err = cudaMemcpy(adj_matrix_d, adj_matrix, num_bytes, cudaMemcpyHostToDevice);
    CUDA_ERROR_CHECK(err);
    err = cudaMalloc((void **) &go_to_d, num_bytes);
    CUDA_ERROR_CHECK(err);
    err = cudaMemcpy(go_to_d, go_to, num_bytes, cudaMemcpyHostToDevice);
    CUDA_ERROR_CHECK(err);

    // Get warp size from device properties and set it as block size
    cudaDeviceProp dev_prop;
    err = cudaGetDeviceProperties(&dev_prop, 0);
    CUDA_ERROR_CHECK(err);
    int warp_size = dev_prop.warpSize;
    int dim_helper = ceil(N/((double) warp_size));
    dim3 dimGrid(dim_helper, dim_helper);
    dim3 dimBlock(warp_size, warp_size);

    // Run FW triple-loop by launching a new kernel for each k
    unsigned int k;
    for (k = 0; k < N; k++) {
        FW_kernel<<<dimGrid, dimBlock>>>(adj_matrix_d, go_to_d, N, (int) k);
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

/*******************************************************************************************************************
 * Read input graph file and populate adjacency matrix
 *******************************************************************************************************************/
void read_input(const char *fn, int *adj_matrix, unsigned int N) {
    const char *fileName = fn;
    FILE *input = fopen(fileName, "r");
    if (input == NULL) {
        fprintf(stderr, "Error while opening the file.\n");
        exit(1);
    }

    char buffer[MAX_BUF];

    // Read file
    int line = 0;
    while (1) {
        line++;
        fgets(buffer, MAX_BUF, input); // get next line
        // Skip lines starting with '#' and empty lines
        if (buffer[0] == '#' || buffer[0] == '\n' || buffer[0] == ' ') continue;
        int i; // row
        int j; // column
        int rel; // relationship (take absolute value, below)
        int rc = sscanf(buffer, "%d|%d|%d",&i, &j, &rel);
        if (rc != 3) {
            fprintf(stderr, "Input file not well formatted (Line %d). "
                            "Expected format of graph lines: <v1>|<v2>|<edge>.\n", line);
            exit(1);
        }
        if (i <= N && j <= N) adj_matrix[index(i-1, j-1, N)] = abs(rel);
        if (feof(input)) break;
    }

    // Close file
    fclose(input);
}

/*******************************************************************************************************************
 * Convert command line input to integer
 * Code taken from https://stackoverflow.com/questions/34206446/how-to-convert-string-into-unsigned-int-c
 *******************************************************************************************************************/
unsigned int convert(char *st) {
    char *x;
    for (x = st ; *x ; x++) {
        if (!isdigit(*x))
            return 0L;
    }
    return (strtoul(st, 0L, 10));
}

/*******************************************************************************************************************
 * Pre-process adjacency matrix and next index matrix:
 * Fill non-edges with int_max/2 in adjacency matrix and -1 in next index on path matrix
 *******************************************************************************************************************/
void preprocess_graph(int *adj_matrix, int *go_to, unsigned int N) {
    unsigned int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (adj_matrix[index(i, j, N)] >= 1) {
                go_to[index(i, j, N)] = j;
            }
            else {
                adj_matrix[index(i, j, N)] = INT_MAX / 2;
                go_to[index(i, j, N)] = -1;
            }
        }
    }
}

/*******************************************************************************************************************
 * Print adjacency matrix read in from file
 *******************************************************************************************************************/
void print_adj(int *adj_matrix, unsigned int N) {
    unsigned int i, j;
    printf("Original adjacency matrix:\n");
    printf("    |");
    for (i = 0; i < N; i++) printf(" %2d |", i+1);
    printf("\n----|----|----|----|----|----|----|----|----|----|----|\n");
    for (i = 0; i < N; i++) {
        printf(" %2d |", i+1);
        for (j = 0; j < N; j++) {
            if (adj_matrix[index(i, j, N)] != INT_MAX/2) printf(" %2d |", adj_matrix[index(i, j, N)]);
            else printf("  - |");
        }
        printf("\n----|----|----|----|----|----|----|----|----|----|----|\n");
    }
}

/*******************************************************************************************************************
 * Print path between all vertex pairs i,j
 *******************************************************************************************************************/
void print_path(int *adj_matrix, int *go_to, unsigned int N) {
    unsigned int i, j;
    printf("\nAPSP solution:\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (go_to[index(i, j, N)] == -1) {
                printf("No path exists between %u and %u.\n", i+1, j+1);
            }
            else {
                printf("Path from %u to %u (length: %d): %u", i+1, j+1, adj_matrix[index(i, j, N)], i+1);
                print_path_recursive(go_to, i, j, N);
                printf("\n");
            }
        }
    }
}

/*******************************************************************************************************************
 * Recursive method for printing path
 *******************************************************************************************************************/
void print_path_recursive(int *go_to, unsigned int i, unsigned int j, unsigned int N) {
    unsigned int next = go_to[index(i, j, N)];
    if (next == j) {
        printf("->%u", next+1);
        return;
    }
    else {
        print_path_recursive(go_to, i, next, N);
        print_path_recursive(go_to, next, j, N);
    }
}