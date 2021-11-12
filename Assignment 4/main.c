/*********************************************************************
  Input: edge, N x N matrix with 0 on diagonal,
            positive values other places
  Output: result, N x N matrix
 *********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define N 6

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

void subMatrices(int edge[N][N])
{
  int i, j, k, pid, np, mtag, pidComm1, pidComm2;
  int result[N][N];

  MPI_Status status;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  int receivedWeights[2] = {100, 100};

  for (k = 0; k < N; k++)
  {
    for (i = (pid / (int)sqrt((double)np)) * (N / np) * sqrt(np);
         i < (pid / (int)sqrt((double)np)) * (N / np) * sqrt(np) + (N / np * sqrt(np));
         i++)
    {
      for (j = pid % (int)sqrt((double)np) * (N / (int)sqrt((double)np));
           j < (pid + 1) % (int)sqrt((double)np) * (N / (int)sqrt((double)np));
           j++)
      {
        pidComm1 = (i * (int)sqrt((double)np) + k) / (N / np * (int)sqrt((double)np));
        pidComm2 = (k * (int)sqrt((double)np) + j) / (N / np * (int)sqrt((double)np));
        // if pids are not same as the current, we receive the data
        if (pid != pidComm1 && pid != pidComm2)
        {
          mtag = 0;
          MPI_Recv(&receivedWeights[0], 1, MPI_INT, pidComm1, mtag, MPI_COMM_WORLD, &status);
          mtag = 1;
          MPI_Recv(&receivedWeights[1], 1, MPI_INT, pidComm2, mtag, MPI_COMM_WORLD, &status);
          result[i][j] = min(edge[i][j], receivedWeights[0] + receivedWeights[1]);
        }
        else // if they are the same, we send our data to all the processes that need it
        {
          /* 
          *  This part sends the local data to all the other processes that need it.
          *  We were unable to figure out the looping condition for it,
          *  but this is the main idea that we would use to solve this problem.
          */

          // for (/* loop through all the processes that need our data */)
          // {
          //   mtag = 0; // update mtag with iteration
          //   MPI_Send(edge[i][j], 1, MPI_INT, /* current process to send it to */, mtag, MPI_COMM_WORLD);
          // }
          // result[i][j] = min(edge[i][j], edge[i][k] + edge[k][j]);
        }
      }
    }

    for (i = (pid / (int)sqrt((double)np)) * (N / np) * sqrt(np);
         i < (pid / (int)sqrt((double)np)) * (N / np) * sqrt(np) + (N / np * sqrt(np));
         i++)
    {
      for (j = pid % (int)sqrt((double)np) * (N / (int)sqrt((double)np));
           j < (pid + 1) % (int)sqrt((double)np) * (N / (int)sqrt((double)np));
           j++)
      {
        edge[i][j] = result[i][j];
      }
    }
  }
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int edge[N][N] = {
      {0, 2, 1, 3, 100, 100},
      {2, 0, 100, 1, 4, 100},
      {1, 100, 0, 1, 100, 100},
      {3, 1, 1, 0, 2, 1},
      {100, 4, 100, 2, 0, 4},
      {100, 100, 100, 1, 4, 0},
  };

  subMatrices(edge);

  MPI_Finalize();

  return 1;
} /****************** End of function main() ********************/
