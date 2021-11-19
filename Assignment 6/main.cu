/**
 * @file main.c
 * @author Cristiano Caon, Nicolas Ferradas
 * @brief Project 6. Parallelize code using GPU cards with CUDA framework
 * @version 0.1
 * @date 2021-11-18
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

__global__ void findMinValues(int N, int *edge)
{
  int i, j, k;
  // int result[N][N];
  // int vBuf[N], hBuf[N];

  for (k = 0; k < N; k++)
  {
    printf("%d\n", edge[k]);
    // for (i = 0; i < N; i++)
    //   vBuf[i] = edge[i][k];

    // for (j = 0; j < N; j++)
    //   hBuf[j] = edge[k][j];

    // for (i = 0; i < N; i++)
    //   for (j = 0; j < N; j++)
    //     edge[i][j] = MIN(edge[i][j], (vBuf[i] + hBuf[j]));
  }
}

void main(int argc, char **argv)
{
  int *edge, *cudaEdge, size, i, j, N = 6;

  size = pow(N, 2) * sizeof(int);
  edge = (int *)malloc(size);

  for (i = 0; i < pow(N, 2); i++)
  {
    if (i % N == 0)
      edge[i] = 0;

    edge[i] = 2 * i;
  }

  cudaMalloc((void **)&cudaEdge, size);
  cudaMemcpy(cudaEdge, edge, size, cudaMemcpyHostToDevice);

  findMinValues<<<1, 1>>>(N, edge);

  free(edge);
  cudaFree(cudaEdge);
} /****************** End of function main() ********************/
