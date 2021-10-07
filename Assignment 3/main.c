#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

void dijkstra(int SOURCE, int n, int **edge, int *distance)
{
  int i, j, count, tmp, least, leastPos, *found;

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
    least = 9876543210;
    for (i = 0; i < n; i++)
    {
      tmp = distance[i];
      if ((!found[i]) && (tmp < least))
      {
        least = tmp;
        leastPos = i;
      }
    }
  }

  found[leastPos] = 1;
  count++;
  for (i = 0; i < n; i++)
  {
    if (!(found[i]))
    {
      distance[i] = min(distance[i], least + edge[leastPos][i]);
    }
  }
  free(found);
}

int main(int argc, char **argv)
{
  int pid, np, gap, mtag;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  /* Call Dijkstra Function with MPI Parallelization */

  MPI_Finalize();

  return 1;
} /****************** End of function main() ********************/
