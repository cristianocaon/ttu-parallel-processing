/*********************************************************************
 Input:  n and edge[n][n], where n is the number of vertices of a graph
         edge[i][j] is the length of the edge from vertex i to vertex j
 Output: distance[n], the distance from the SOURCE vertex to vertex i.
 *********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define ROWS 6
#define COLS 6

int min(int a, int b)
{
  if (a < b)
  {
    return a;
  }
  else
  {
    return b;
  }
}

void dijkstra(int SOURCE, int n, int edge[ROWS][COLS], int *distance)
{
  int i, j, count, tmp, least, leastPos, *found;
  int pid, np, gap, mtag;
  MPI_Status status;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  found = (int *)calloc(n, sizeof(int));

  for (i = 0; i < n; i++)
  {
    found[i] = 0;
    distance[i] = edge[SOURCE][i];
  }
  found[SOURCE] = 1;
  count = 1;

  while (count < n)
  {
    least = 2147483647;
    for (i = pid * n / np; i < (pid + 1) * n / np - 1; i++)
    {
      tmp = distance[i];
      if ((!found[i]) && (tmp < least))
      {
        least = tmp;
        leastPos = i;
      }
    }

    mtag = 1;
    int receivedWeights[2] = {100, 100};
    int weights[2] = {least, leastPos};

    if (pid == 0) // First process/chunk
    {
      MPI_Send(weights, 2, MPI_INT, pid + 1, mtag, MPI_COMM_WORLD);
    }
    else if (pid > 0 && pid < np - 1) // Processes in between
    {
      /* Receive least and leastPos from previous process */
      MPI_Recv(&receivedWeights, 2, MPI_INT, pid - 1, mtag, MPI_COMM_WORLD, &status);
      /* Recalculate least and leastPos before sending */
      if (receivedWeights[0] < weights[0])
      {
        weights[0] = receivedWeights[0];
        weights[1] = receivedWeights[1];
      }
      /* Send least and leastPos to next process */
      MPI_Send(weights, 2, MPI_INT, pid + 1, mtag, MPI_COMM_WORLD);
    }
    else // Last process/chunk
    {
      MPI_Recv(&receivedWeights, 2, MPI_INT, pid - 1, mtag, MPI_COMM_WORLD, &status);
      /* Recalculate least and leastPos before sending */
      if (receivedWeights[0] < weights[0])
      {
        weights[0] = receivedWeights[0];
        weights[1] = receivedWeights[1];
      }
    }

    mtag = 2;
    if (pid == np - 1)
    {
      for (i = 0; i < np - 1; i++)
      {
        MPI_Send(weights, 2, MPI_INT, i, mtag, MPI_COMM_WORLD);
      }
    }
    else
    {
      MPI_Recv(&weights, 2, MPI_INT, np - 1, mtag, MPI_COMM_WORLD, &status);
    }

    found[weights[1]] = 1;
    for (i = pid * n / np; i < (pid + 1) * n / np - 1; i++)
    {
      if (!(found[i]))
      {
        distance[i] = min(distance[i], weights[0] + edge[weights[1]][i]);
      }
    }
    count += 1;
  }
  free(found);
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int SOURCE = 0;
  int n = 6;

  int edge[ROWS][COLS] = {
      {0, 2, 1, 3, 100, 100},
      {2, 0, 100, 1, 4, 100},
      {1, 100, 0, 1, 100, 100},
      {3, 1, 1, 0, 2, 1},
      {100, 4, 100, 2, 0, 4},
      {100, 100, 100, 1, 4, 0},
  };

  int distance[6] = {100, 100, 100, 100, 100, 100};

  /* Call Dijkstra Function with MPI Parallelization */
  dijkstra(SOURCE, n, edge, distance);

  MPI_Finalize();

  return 1;
} /****************** End of function main() ********************/
