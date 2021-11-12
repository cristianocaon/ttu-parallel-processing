/**
 * @file main.c
 * 
 * @author Cristiano Caon, Nicolas Ferradas
 * 
 * @brief Project 5. Parallelize the force calculation for n particles so tha 
 * the computation loads on all processes are balanced, and communication cost
 * has a complexity of O(n log_2 p) for n particles and p processors.
 * 
 * @version 0.1
 * 
 * @date 2021-11-12
 * 
 */

#include <math.h>
#include <mpi.h>
#include <stdio.h>

#define c1 1.23456
#define c2 6.54321
#define N 8

double sgn(double x)
/**
 * Calculates the sine value of input.
 */
{
  return x < 0.0 ? -1.0 : 1.0;
}

int addFunction(int n)
/**
 * Sums all integers from 1 -> n.
 */
{
  int sum = 0;
  for (int i = n; i > 0; i--)
    sum += i;
  return sum;
}

int findIndexI(int num_iter)
/**
 * Finds the 'i' index based on number of iterations.
 */
{
  for (int i = 1; i < N; i++)
    if (addFunction(i) > num_iter)
      return i;

  return 0;
}

int findIndexJ(int num_iter, int i)
/**
 * Finds the 'j' index based on number of iterations and current 'i' index.
 */
{
  int num = addFunction(i);
  int diff = num % num_iter;
  int j = i - diff;
  return j;
}

void calc_forces(int num, double *particles)
/**
 * Input: num, particles[num].Note that particles[i] != particles[j] for different i, j.
 * Output: f[num]. 
 */
{
  int i, j, k, pid, np, mtag, gap, numIterations = 0;
  int basePID, basePairPID, matchingPID;
  double diff, tmp, results[num], receivedResults[num];

  MPI_Status status;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  for (i = 0; i < num; i++)
  {
    results[i] = 0.0;
    receivedResults[i] = 0.0;
    numIterations += i + 1;
  }
  numIterations /= np; // number of iterations per process

  // Start indexes for both i and j for current process
  int startI = findIndexI(numIterations * pid);
  int startJ = findIndexJ(numIterations, startI);

  // End indexes for both i and j for current process
  int endI = findIndexI(numIterations * (pid + 1));
  int endJ = findIndexJ(numIterations, endI) - 1;

  // Calculating local results for each process
  // printf("hellos");
  for (i = startI; i <= endI; i++)
    for (j = startJ; j < i; j++)
      // Validate that the loop is within the range it is supposed to be
      if ((i == endI && j < endJ) || i < endI)
      {
        diff = particles[i] - particles[j];
        tmp = 1.0 / diff;
        tmp = c1 * pow(tmp, 3) - c2 * pow(tmp, 2) * sgn(tmp);
        results[i] += tmp;
        results[j] -= tmp;
      }
      else
      {
        break;
      }

  // Communicating local results to other processes and updating based on received results
  mtag = 1;
  for (gap = np; gap > 1; gap /= 2)
  {
    basePID = pid % gap;
    basePairPID = gap - 1 - basePID;
    matchingPID = basePairPID + gap * (pid / gap);

    MPI_Send(results, N, MPI_INT, matchingPID, mtag, MPI_COMM_WORLD);

    MPI_Recv(receivedResults, N, MPI_INT, matchingPID, mtag, MPI_COMM_WORLD, &status);

    // Updating the results with received results from other processes
    for (k = 0; k < sizeof(receivedResults) / sizeof(double); k++)
      results[k] += receivedResults[k];
    mtag++;
  }
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  double particles[N] = {4, 2, 1, 3, 9, 8, 10, 12};

  calc_forces(N, particles);

  MPI_Finalize();

  return 1;
} /****************** End of function main() ********************/