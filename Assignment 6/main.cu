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

#define N 6
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

__global__ void findMinValues(int *edge)
{
  int i, j, k;
  int *vBuf, *hBuf;

  // for (k = 0; k < N; k++)
  // {
  // for (i = 0; i < N; i++)
  //   vBuf[i] = edge[i][k];

  // for (j = 0; j < N; j++)
  //   hBuf[j] = edge[k][j];

  // for (i = 0; i < N; i++)
  //   for (j = 0; j < N; j++)
  //     edge[i][j] = MIN(edge[i][j], (vBuf[i] + hBuf[j]));
  // }
}

int main(int argc, char **argv)
{
  int *edge, *cudaEdge, size, i;

  size = sizeof(int) * pow(N, 2);
  edge = (int *)malloc(size);

  for (i = 0; i < N; i++)
    edge[i] = i % N == 0 ? 0 : i;

  cudaMalloc((void **)&cudaEdge, size);
  cudaMemcpy(cudaEdge, edge, size, cudaMemcpyHostToDevice);

  findMinValues<<<1, 1>>>(cudaEdge);

  cudaFree(cudaEdge);
  free(edge);

  return 1;
} /****************** End of function main() ********************/
