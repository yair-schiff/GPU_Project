/*
 * Author: Yair Schiff
 * Project: Fall 2018 CSCI.GA 3033-004: GPUs
 * Instructor: Prof. Zahran
 *
 * Project Description: This project explores the efficient implementation of the Floyd-Warshall (FW) algorithm, a
 * solution for the All-Pairs-Shortest-Path (APSP) and Transitive Closure problems
 *
 * File Description: This file implements the functions defined in FW_helper.h
 */

#include "FW_helper.h"

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

