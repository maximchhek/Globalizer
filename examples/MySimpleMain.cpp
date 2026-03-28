
#include "Globalizer.h"
#include <cmath>
#include <mpi.h>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Вычисление функций задачи
double ArchipelagTask(const double* y, int fNumber)
{
    double x1 = y[0];
    double x2 = y[1];
    double res = 0.0;

    switch (fNumber)
    {
    case 0:
        // g(x) = 0.5 - sin(2*x1)*cos(2*x2) <= 0
        res = 0.5 - sin(2.0 * x1) * cos(2.0 * x2);
        break;

    case 1:
    {
        double term1 = -20.0 * exp(-0.2 * sqrt(0.5 * (x1 * x1 + x2 * x2)));
        double term2 = -exp(0.5 * (cos(2.0 * M_PI * x1) + cos(2.0 * M_PI * x2)));
        res = term1 + term2 + exp(1.0) + 20.0;
    }
    break;
    }

    return res;
}

int main(int argc, char* argv[])
{

    GlobalizerInitialization(argc, argv, true);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank != 0) {
        static std::ofstream devNull("NUL");
        std::cout.rdbuf(devNull.rdbuf());
    }

    if (rank == 0) {
        printf("Master process started. Total workers: %d\n", size);
    }
    else {
        printf("Worker %d is alive.\n", rank);
    }

    parameters.Dimension = 2;
    parameters.IsPlot = true;
    parameters.CalcsType = 0;
    parameters.FigureType = 1;

    std::vector<double> low = { -5.0, -5.0 };
    std::vector<double> up = { 5.0, 5.0 };

    IProblem* problem = new ProblemFromFunctionPointers(
        parameters.Dimension,
        low,
        up,
        ArchipelagTask,
        2
    );

    problem->Initialize();

    Solver solver(problem);

    if (rank != 0) {
        parameters.IsPlot = false;
    }

    try {
        if (solver.Solve() != SYSTEM_OK)
            throw EXCEPTION("Solver error!");

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Optimization finished successfully.\n");
        }
    }
    catch (const std::exception& e) {
        if (rank == 0) printf("Error: %s\n", e.what());
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // 5. Закрываем MPI
    MPI_Finalize();

    return 0;
}