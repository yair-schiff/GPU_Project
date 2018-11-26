/*
 * Author: Yair Schiff
 * Project: Fall 2018 CSCI.GA 3033-004: GPUs
 * Instructor: Prof. Zahran
 *
 * Project Description: This project explores the efficient implementation of the Floyd-Warshall (FW) algorithm, a
 * solution for the All-Pairs-Shortest-Path (APSP) and Transitive Closure problems
 *
 * File Description: This is the header file to be used in implementing FW algorithm sequentially and in parallel
 */

#ifndef FW_H
#define FW_H

#include <ctype.h>
#include <cuda.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

// Macros
#define MAX_GRAPH 397020 // max graph size
#define MAX_BUF 1000 // integer size of buffer for file reading
#define index(i, j, N)  ((i)*(N)) + (j) // To index element (i,j) of a 2D array stored as 1D
// For error checking cuda API calls
#define CUDA_ERROR_CHECK(err) {\
    if (err != cudaSuccess) {\
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);\
        exit(1);\
    }\
}

// Function declarations
unsigned int convert(char *st);
void read_input(const char *fn, int *adj_matrix, unsigned int N);
void preprocess_graph(int *adj_matrix, int *go_to, unsigned int N);
void print_adj(int *adj_matrix, unsigned int N);
void print_path(int *adj_matrix, int *go_to, unsigned int N);
void print_path_recursive(int *go_to, unsigned int i, unsigned int j, unsigned int N);


#endif //FW_H
