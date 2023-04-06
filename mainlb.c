#include <stdio.h>
#include "mpi.h"
#include <stdlib.h>

void matrix_print(double **matrix, int N){
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            printf("%lg ", matrix[i][j]);
        }
        printf("\n");
    }
}

extern double func  (double t, double x);
extern double fi    (double x);
extern double ksi   (double t);

extern double t_max;
extern double x_max;
extern double t_step;
extern double x_step;

double triple_dots(double value, int m, int k){
    return ((func(k * t_step, m * x_step) - (value) / (2 * x_step)) * t_step + 0.5 * (value));
}

double cross(double value, double value_down, int m, int k){
    return (func(k * t_step, m * x_step) - (value) / (2 * x_step)) * 2 * t_step + value_down;
}

double corner(double value_2, double value_1, int m, int k){
    return (func(k * t_step, m * x_step) - (value_2 - value_1) / (x_step)) * t_step + value_2;
}

int main( int argc, char **argv ){
    int size, rank, K = t_max/t_step, M = x_max/x_step, ibeg, complete = 0;;
    double **matrix;
    double res[M];
    matrix = (double**)calloc(K, sizeof(double*));
    for(int p = 0; p < K; ++p){
        matrix[p] = (double*)calloc(M, sizeof(double));
    }
    for(int k = 0; k < K; k++){
        matrix[k][0] = ksi(k * t_step);
    }
    for(int m = 0; m < M; ++m){
        matrix[0][m] = fi(m * x_step);
    }

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 0){
        for(int m = 1; m < M - 1; ++m){
            matrix[1][m] = triple_dots(matrix[0][m+1] - matrix[0][m-1], m, 0);
        }
        matrix[1][M-1] = corner(matrix[0][M-1], matrix[0][M-2], M - 1, 0);
    }

    ibeg = rank + 2;
    if(rank > 0){
        while(ibeg <= (K - 1)){
            if(ibeg != (K - 1)) {
            MPI_Recv(&matrix[ibeg - 1][1], 1, MPI_DOUBLE, rank - 1, 5, MPI_COMM_WORLD, &status);
            MPI_Recv(&matrix[ibeg - 2][1], 1, MPI_DOUBLE, rank - 1, 4, MPI_COMM_WORLD, &status);
            for(int m = 1; m < M - 1; ++m){
                    MPI_Recv(&matrix[ibeg - 1][m + 1], 1, MPI_DOUBLE, rank - 1, 5, MPI_COMM_WORLD, &status);
                    MPI_Recv(&matrix[ibeg - 2][m + 1], 1, MPI_DOUBLE, rank - 1, 4, MPI_COMM_WORLD, &status);
                    matrix[ibeg][m] = cross(matrix[ibeg - 1][m+1] - matrix[ibeg - 1][m-1], matrix[ibeg - 2][m], m, ibeg - 1);
                    MPI_Send(&matrix[ibeg][m], 1, MPI_DOUBLE, (rank + 1) % size, 5, MPI_COMM_WORLD);
                    MPI_Send(&matrix[ibeg - 1][m], 1, MPI_DOUBLE, (rank + 1) % size, 4, MPI_COMM_WORLD);
                }
            matrix[ibeg][M-1] = corner(matrix[ibeg - 1][M - 1], matrix[ibeg - 1][M - 1], M - 1, ibeg);  
            MPI_Send(&matrix[ibeg][M - 1], 1, MPI_DOUBLE, (rank + 1) % size, 5, MPI_COMM_WORLD);
            MPI_Send(&matrix[ibeg - 1][M - 1], 1, MPI_DOUBLE, (rank + 1) % size, 4, MPI_COMM_WORLD);
            ibeg = ibeg + size;
            } else {
                double res[M];
                MPI_Recv(&matrix[ibeg - 1][1], 1, MPI_DOUBLE, rank - 1, 5, MPI_COMM_WORLD, &status);
                MPI_Recv(&matrix[ibeg - 2][1], 1, MPI_DOUBLE, rank - 1, 4, MPI_COMM_WORLD, &status);
                for(int m = 1; m < M - 1; ++m){
                        MPI_Recv(&matrix[ibeg - 1][m + 1], 1, MPI_DOUBLE, rank - 1, 5, MPI_COMM_WORLD, &status);
                        MPI_Recv(&matrix[ibeg - 2][m + 1], 1, MPI_DOUBLE, rank - 1, 4, MPI_COMM_WORLD, &status);
                        matrix[ibeg][m] = cross(matrix[ibeg - 1][m+1] - matrix[ibeg - 1][m-1], matrix[ibeg - 2][m], m, ibeg - 1);
                        res[m] = matrix[ibeg][m];
                    }
                matrix[ibeg][M-1] = corner(matrix[ibeg - 1][M - 1], matrix[ibeg - 1][M - 1], M - 1, ibeg);
                res[M - 1] = matrix[ibeg][M-1];
                res[0] = matrix[ibeg][0];
                ibeg = ibeg + size;
                MPI_Send(&res, M, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
            }
            //matrix_print(matrix, 10);
            //printf("\n rank = %d, ibeg = %d\n\n", rank, ibeg);

        }
    }
    if(rank == 0){
        for(int m = 1; m < M - 1; ++m){
                matrix[ibeg][m] = cross(matrix[ibeg - 1][m+1] - matrix[ibeg - 1][m-1], matrix[ibeg - 2][m], m, ibeg - 1);
                MPI_Send(&matrix[ibeg][m], 1, MPI_DOUBLE, 1, 5, MPI_COMM_WORLD);
                MPI_Send(&matrix[ibeg - 1][m], 1, MPI_DOUBLE, 1, 4, MPI_COMM_WORLD);
            }
        matrix[ibeg][M-1] = corner(matrix[ibeg - 1][M - 1], matrix[ibeg - 1][M - 1], M - 1, 0);  
        MPI_Send(&matrix[ibeg][M - 1], 1, MPI_DOUBLE, 1, 5, MPI_COMM_WORLD);
        MPI_Send(&matrix[ibeg - 1][M - 1], 1, MPI_DOUBLE, 1, 4, MPI_COMM_WORLD);
        ibeg = ibeg + size;

        while(ibeg <= (K - 1)){
            if(ibeg != (K - 1)) {
                MPI_Recv(&matrix[ibeg - 1][1], 1, MPI_DOUBLE, size - 1, 5, MPI_COMM_WORLD, &status);
                MPI_Recv(&matrix[ibeg - 2][1], 1, MPI_DOUBLE, size - 1, 4, MPI_COMM_WORLD, &status);
                for(int m = 1; m < M - 1; ++m){
                        MPI_Recv(&matrix[ibeg - 1][m + 1], 1, MPI_DOUBLE, size - 1, 5, MPI_COMM_WORLD, &status);
                        MPI_Recv(&matrix[ibeg - 2][m + 1], 1, MPI_DOUBLE, size - 1, 4, MPI_COMM_WORLD, &status);
                        matrix[ibeg][m] = cross(matrix[ibeg - 1][m+1] - matrix[ibeg - 1][m-1], matrix[ibeg - 2][m], m, ibeg - 1);
                        MPI_Send(&matrix[ibeg][m], 1, MPI_DOUBLE, 1 , 5, MPI_COMM_WORLD);
                        MPI_Send(&matrix[ibeg - 1][m], 1, MPI_DOUBLE, 1 , 4, MPI_COMM_WORLD);
                    }
                matrix[ibeg][M-1] = corner(matrix[ibeg - 1][M - 1], matrix[ibeg - 1][M - 1], M - 1, ibeg);  
                MPI_Send(&matrix[ibeg][M - 1], 1, MPI_DOUBLE, 1 , 5, MPI_COMM_WORLD);    
                MPI_Send(&matrix[ibeg - 1][M - 1], 1, MPI_DOUBLE, 1, 4, MPI_COMM_WORLD);
                ibeg = ibeg + size;
            } else {
                MPI_Recv(&matrix[ibeg - 1][1], 1, MPI_DOUBLE, size - 1, 5, MPI_COMM_WORLD, &status);
                MPI_Recv(&matrix[ibeg - 2][1], 1, MPI_DOUBLE, size - 1, 4, MPI_COMM_WORLD, &status);
                for(int m = 1; m < M - 1; ++m){
                        MPI_Recv(&matrix[ibeg - 1][m + 1], 1, MPI_DOUBLE, size - 1, 5, MPI_COMM_WORLD, &status);
                        MPI_Recv(&matrix[ibeg - 2][m + 1], 1, MPI_DOUBLE, size - 1, 4, MPI_COMM_WORLD, &status);
                        matrix[ibeg][m] = cross(matrix[ibeg - 1][m+1] - matrix[ibeg - 1][m-1], matrix[ibeg - 2][m], m, ibeg - 1);
                        res[m] = matrix[ibeg][m];                   
                    }
                matrix[ibeg][M-1] = corner(matrix[ibeg - 1][M - 1], matrix[ibeg - 1][M - 1], M - 1, ibeg); 
                res[M - 1] = matrix[ibeg][M-1];
                res[0] = matrix[ibeg][0];
                ibeg = ibeg + size;
                complete = 1;
            } 
            //matrix_print(matrix, 10);
            //printf("\n rank = %d, ibeg = %d\n\n", rank, ibeg);

        }

        if(complete != 1){
            MPI_Recv(&res, M, MPI_DOUBLE, MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, &status); 
            for(int y = 0; y < M; ++y){
                printf("%g  ", res[y]);
            }
        }   
    }



    MPI_Finalize();

    //printf("OK rank = %d\n", rank);
    for(int p = 0; p < K; ++p){
        free(matrix[p]);
    }
    free(matrix);
    return 0;
}