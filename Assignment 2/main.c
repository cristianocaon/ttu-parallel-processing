#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int pid, np, gap, mtag;
    int basePID, basePairPID, matchingPID;
    int data[1] = {1};
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    mtag = 1;
    for (gap = np; gap > 1; gap /= 2)
    {
        basePID = pid % gap;
        basePairPID = gap - 1 - basePID;
        matchingPID = basePairPID + gap * (pid / gap);
        MPI_Send(data, 1, MPI_INT, matchingPID, mtag, MPI_COMM_WORLD);
        printf("%i sent to %i\n", pid, matchingPID);
        MPI_Recv(&data[0], 1, MPI_INT, matchingPID, mtag, MPI_COMM_WORLD, &status);
        printf("%i received from %i\n", pid, matchingPID);
        mtag++;
    }

    MPI_Finalize();

    return 1;
} /****************** End of function main() ********************/
