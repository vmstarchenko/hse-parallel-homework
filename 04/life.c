#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int DEBUG = 0;
#define printf                                                                 \
    if (DEBUG)                                                                 \
    printf

#define ALIVE 'X'
#define DEAD '.'
#define master 0

int toindex(int row, int col, int N);
void printgrid(char* grid, FILE* f, int N);
char* read_data(char* filename, int N, int size, int width, int rank);
void write_data(char* filename, int N, int size, int width, int rank,
                char* grid);
void recalc_row(int i, int N, char* grid, char* buf);

int main(int argc, char* argv[]) {
    // parse args
    if (argc != 5) {
        fprintf(stderr, "Usage: %s N input_file iterations output_file\n",
                argv[0]);
        return 1;
    }

    int N = atoi(argv[1]); // grid size
    int iterations = atoi(argv[3]);

    // init mpi
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Status status;

    int me = rank;

    int width = (N / size);
    if (N % size)
        ++width;

    int real_width = width;
    if (rank == size - 1)
        real_width = N - width * (size - 1);

    printf("r%d rw%d\n", rank, real_width);

    int down = (me + 1) % size;
    int up = (me + size - 1) % size;

    char* grid = read_data(argv[2], N, size, width, rank);
    char* buf = (char*)calloc((width + 2) * N, sizeof(*buf));

    for (int iter = 0; iter < iterations; ++iter) {
        // fill first and last

        MPI_Request send_request_up, recv_request_up, send_request_down,
            recv_request_down;
        /* Send up */
        MPI_Isend(grid + 1 * N, N, MPI_CHAR, (rank + size - 1) % size, 1,
                  MPI_COMM_WORLD, &send_request_up);
        MPI_Irecv(grid + (real_width + 1) * N, N, MPI_CHAR, (rank + 1) % size,
                  1, MPI_COMM_WORLD, &recv_request_up);

        /* Send down */
        printf("r%d sr%d %c\n", rank, (rank + 1) % size,
               (grid + real_width * N)[0]);
        MPI_Isend(grid + real_width * N, N, MPI_CHAR, (rank + 1) % size, 0,
                  MPI_COMM_WORLD, &send_request_down);
        MPI_Irecv(grid, N, MPI_CHAR, (rank + size - 1) % size, 0,
                  MPI_COMM_WORLD, &recv_request_down);

        if (DEBUG) {
            MPI_Barrier(MPI_COMM_WORLD);
            for (int i = 0; i < size; i++) {
                if (i == rank) {
                    printf("kek r%d [%11.11s]\n", rank, grid);
                    fflush(stdout);
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        // exchange first and last

        // recalc field
        for (int i = 2; i < real_width; ++i) { // row
            recalc_row(i, N, grid, buf);
        }

        MPI_Wait(&recv_request_up, &status);
        MPI_Wait(&send_request_up, &status);
        MPI_Wait(&recv_request_down, &status);
        MPI_Wait(&send_request_down, &status);

        recalc_row(1, N, grid, buf);
        if (real_width > 1) {
          recalc_row(real_width, N, grid, buf);
        }

        char* tmp = grid;
        grid = buf;
        buf = tmp;
    }

    write_data(argv[4], N, size, width, rank, grid);

    free(grid);
    free(buf);
    MPI_Finalize();
    return 0;
}

void write_data(char* filename, int N, int size, int width, int rank,
                char* grid) {

    char* common_grid;
    if (rank == 0) {
        common_grid = (char*)calloc(width * N * size, sizeof(*common_grid));
    }

    /* MPI_Scatter(common_grid, width * N, MPI_CHAR, grid + N, width * N,
     * MPI_CHAR, */
    /*             0, MPI_COMM_WORLD); */
    MPI_Gather(grid + N, width * N, MPI_CHAR, common_grid, width * N, MPI_CHAR,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE* output = fopen(filename, "w");
        printf("r%d write [%100s]\n", rank, common_grid);
        printgrid(common_grid, output, N);
        fclose(output);
        free(common_grid);
    }
}

char* read_data(char* filename, int N, int size, int width, int rank) {
    char* grid = (char*)calloc((width + 2) * N, sizeof(*grid));

    // read data
    char* common_grid;
    if (rank == master) {
        common_grid = (char*)calloc(width * N * size, sizeof(*common_grid));

        FILE* input = fopen(filename, "r");
        for (int i = 0; i < N; ++i) {
            fscanf(input, "%s", common_grid + i * N);
        }
        fclose(input);

        printf("r%d read [%*.*s]\n", rank, N * N, N * N, common_grid);
    }

    MPI_Scatter(common_grid, width * N, MPI_CHAR, grid + N, width * N, MPI_CHAR,
                master, MPI_COMM_WORLD);

    if (DEBUG) {
        MPI_Barrier(MPI_COMM_WORLD);
        for (int i = 0; i < size; i++) {
            if (i == rank) {
                printf("kek r%d [%*.*s]\n", rank, width * N, width * N,
                       grid + N);
                fflush(stdout);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == master) {
        free(common_grid);
    }

    return grid;
}

void printgrid(char* grid, FILE* f, int N) {
    char* buf = (char*)malloc(N * N * sizeof(char));
    for (int i = 0; i < N; ++i) {
        strncpy(buf, grid + i * N, N);
        buf[N] = 0;
        fprintf(f, "%s\n", buf);
    }
    free(buf);
}

int toindex(int row, int col, int N) {
    if (row < 0) {
        row = row + N;
    } else if (row >= N) {
        row = row - N;
    }
    if (col < 0) {
        col = col + N;
    } else if (col >= N) {
        col = col - N;
    }
    return row * N + col;
}

void recalc_row(int i, int N, char* grid, char* buf) {
    for (int j = 0; j < N; ++j) { // column
        int alive_count = 0;      // count alive
        for (int di = -1; di <= 1; ++di) {
            for (int dj = -1; dj <= 1; ++dj) {
                if ((di != 0 || dj != 0) &&
                    grid[toindex(i + di, j + dj, N)] == ALIVE) {
                    ++alive_count;
                }
            }
        }

        int current = i * N + j;
        if (alive_count == 3 || (alive_count == 2 && grid[current] == ALIVE)) {
            buf[current] = ALIVE;
        } else {
            buf[current] = DEAD;
        }
    }
}
