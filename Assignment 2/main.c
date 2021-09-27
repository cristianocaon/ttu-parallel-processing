#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int pid, np;
    double t0, t1;
    MPI_Status status;
    MPI_Request req_s;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    MPI_Finalize();

    return 1;
} /****************** End of function main() ********************/
