#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>

#define INITIAL_CAPACITY 1024

/**
 * Reads the input file on rank 0 and distributes the live cells to the relevant processes.
 */
void read_input_file(int **live_cells, int *num_live_cells, const char *input_file_name, int rank, int size, int rows_per_proc, int remainder) {
    if (rank == 0) {
        FILE *input_file = fopen(input_file_name, "r");
        if (!input_file) {
            perror("Input file cannot be opened");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        char line[256];
        int x, y;
        int **proc_live_cells_flat = (int **)malloc(size * sizeof(int *));
        int *proc_num_cells = (int *)calloc(size, sizeof(int));
        int *proc_capacities = (int *)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            proc_capacities[i] = INITIAL_CAPACITY;
            proc_live_cells_flat[i] = (int *)malloc(proc_capacities[i] * sizeof(int));
            proc_num_cells[i] = 0;
        }

        while (fgets(line, sizeof(line), input_file)) {
            if (sscanf(line, "%d,%d", &x, &y) != 2) {
                continue;
            }

            for (int p = 0; p < size; p++) {
                int proc_rows = rows_per_proc + (p < remainder ? 1 : 0);
                int offset = p * rows_per_proc + ((p < remainder) ? p : remainder);
                if (x >= offset && x < offset + proc_rows) {
                    int index = proc_num_cells[p];
                    if (index + 2 > proc_capacities[p]) {
                        proc_capacities[p] *= 2;
                        proc_live_cells_flat[p] = (int *)realloc(proc_live_cells_flat[p], proc_capacities[p] * sizeof(int));
                    }
                    proc_live_cells_flat[p][index++] = x;
                    proc_live_cells_flat[p][index++] = y;
                    proc_num_cells[p] = index;
                    break;
                }
            }
        }
        fclose(input_file);

        // Send live cells to each process
        for (int p = 1; p < size; p++) {
            int num_cells_p = proc_num_cells[p] / 2; 
            MPI_Send(&num_cells_p, 1, MPI_INT, p, 0, MPI_COMM_WORLD);

            if (num_cells_p > 0) {
                MPI_Send(proc_live_cells_flat[p], num_cells_p * 2, MPI_INT, p, 0, MPI_COMM_WORLD);
            }
            free(proc_live_cells_flat[p]);
        }

        // For rank 0, initialize live_cells with its own data
        *num_live_cells = proc_num_cells[0] / 2;
        if (*num_live_cells > 0) {
            *live_cells = (int *)malloc(proc_num_cells[0] * sizeof(int));
            memcpy(*live_cells, proc_live_cells_flat[0], proc_num_cells[0] * sizeof(int));
        } else {
            *live_cells = NULL;
        }
        free(proc_live_cells_flat[0]);

        free(proc_live_cells_flat);
        free(proc_num_cells);
        free(proc_capacities);
    } else {
        // Receive live cells from rank 0
        int num_cells_p;
        MPI_Recv(&num_cells_p, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (num_cells_p > 0) {
            int total_ints = num_cells_p * 2;
            *live_cells = (int *)malloc(total_ints * sizeof(int));
            MPI_Recv(*live_cells, total_ints, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            *num_live_cells = num_cells_p;
        } else {
            *live_cells = NULL;
            *num_live_cells = 0;
        }
    }
}

/**
 * Performs the Game of Life computation for a single generation.
 */
void compute(int *life, int *previous_life, int local_rows, int Y_limit, int cols, int rank, int size) {
    MPI_Request request[4];
    int top = rank - 1;
    int bottom = rank + 1;

    memcpy(previous_life, life, (local_rows + 2) * cols * sizeof(int));

    // Non-blocking communication with neighboring processes
    if (rank > 0) {
        MPI_Isend(&previous_life[1 * cols], cols, MPI_INT, top, 1, MPI_COMM_WORLD, &request[0]);
        MPI_Irecv(&previous_life[0 * cols], cols, MPI_INT, top, 2, MPI_COMM_WORLD, &request[1]);
    } else {
        memset(&previous_life[0 * cols], 0, cols * sizeof(int));
        request[0] = MPI_REQUEST_NULL;
        request[1] = MPI_REQUEST_NULL;
    }
    if (rank < size - 1) {
        MPI_Isend(&previous_life[local_rows * cols], cols, MPI_INT, bottom, 2, MPI_COMM_WORLD, &request[2]);
        MPI_Irecv(&previous_life[(local_rows + 1) * cols], cols, MPI_INT, bottom, 1, MPI_COMM_WORLD, &request[3]);
    } else {
        memset(&previous_life[(local_rows + 1) * cols], 0, cols * sizeof(int));
        request[2] = MPI_REQUEST_NULL;
        request[3] = MPI_REQUEST_NULL;
    }

    // Compute inner cells (excluding boundary rows)
    for (int i = 2; i < local_rows; i++) {
        for (int j = 1; j <= Y_limit; j++) {
            int idx = i * cols + j;
            int neighbors = previous_life[(i - 1) * cols + (j - 1)] + previous_life[(i - 1) * cols + j] + previous_life[(i - 1) * cols + (j + 1)] +
                            previous_life[i * cols + (j - 1)] + previous_life[i * cols + (j + 1)] +
                            previous_life[(i + 1) * cols + (j - 1)] + previous_life[(i + 1) * cols + j] + previous_life[(i + 1) * cols + (j + 1)];

            if (previous_life[idx] == 0) {
                life[idx] = (neighbors == 3) ? 1 : 0;
            } else {
                life[idx] = (neighbors == 2 || neighbors == 3) ? 1 : 0;
            }
        }
    }

    MPI_Waitall(4, request, MPI_STATUSES_IGNORE);

    // Compute boundary cells (first and last rows)
    for (int boundary = 1; boundary <= local_rows; boundary += local_rows - 1) {
        int i = boundary;
        for (int j = 1; j <= Y_limit; j++) {
            int idx = i * cols + j;
            int neighbors = previous_life[(i - 1) * cols + (j - 1)] + previous_life[(i - 1) * cols + j] + previous_life[(i - 1) * cols + (j + 1)] +
                            previous_life[i * cols + (j - 1)] + previous_life[i * cols + (j + 1)] +
                            previous_life[(i + 1) * cols + (j - 1)] + previous_life[(i + 1) * cols + j] + previous_life[(i + 1) * cols + (j + 1)];

            if (previous_life[idx] == 0) {
                life[idx] = (neighbors == 3) ? 1 : 0;
            } else {
                life[idx] = (neighbors == 2 || neighbors == 3) ? 1 : 0;
            }
        }
    }
}

/**
 * Gathers the live cells from all processes to rank 0 and writes them to the output file.
 */
void write_output(int *life, int local_rows, int Y_limit, int cols, int offset, const char *input_name, int num_of_generations, int rank, int size) {
    int *local_live_cells = NULL;
    int local_capacity = INITIAL_CAPACITY;
    int local_count = 0;

    local_live_cells = (int *)malloc(local_capacity * sizeof(int));

    // Collect local live cells.
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 1; j <= Y_limit; j++) {
            int idx = i * cols + j;
            if (life[idx] == 1) {
                if (local_count + 2 > local_capacity) {
                    local_capacity *= 2;
                    local_live_cells = (int *)realloc(local_live_cells, local_capacity * sizeof(int));
                }
                local_live_cells[local_count++] = offset + i - 1;
                local_live_cells[local_count++] = j - 1; 
            }
        }
    }

    // Gather live cells counts from all processes.
    int *counts = NULL;
    if (rank == 0) {
        counts = (int *)malloc(size * sizeof(int));
    }
    int local_count_ints = local_count;
    MPI_Gather(&local_count_ints, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute displacements and total_count on rank 0
    int *displs = NULL;
    int total_count = 0;
    if (rank == 0) {
        displs = (int *)malloc(size * sizeof(int));
        displs[0] = 0;
        total_count += counts[0];
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + counts[i - 1];
            total_count += counts[i];
        }
    }

    int *all_live_cells = NULL;
    if (rank == 0 && total_count > 0) {
        all_live_cells = (int *)malloc(total_count * sizeof(int));
    }

    MPI_Gatherv(local_live_cells, local_count_ints, MPI_INT,
                all_live_cells, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    // Rank 0 writes the output file.
    if (rank == 0) {
        char output_file_name[256];
        strncpy(output_file_name, input_name, strlen(input_name) - 5);
        output_file_name[strlen(input_name) - 5] = '\0';
        snprintf(output_file_name + strlen(output_file_name), sizeof(output_file_name) - strlen(output_file_name), ".%d.csv", num_of_generations);

        FILE *output_file = fopen(output_file_name, "w");
        if (!output_file) {
            perror("Output file cannot be opened");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < total_count; i += 2) {
            int x = all_live_cells[i];
            int y = all_live_cells[i + 1];
            fprintf(output_file, "%d,%d\n", x, y);
        }
        fclose(output_file);
    }

    free(local_live_cells);
    if (rank == 0) {
        free(counts);
        free(displs);
        free(all_live_cells);
    }
}

/**
 * The main function to execute the Game of Life simulations using MPI.
 */
int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);  
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 5) {
        if (rank == 0)
            fprintf(stderr, "Expected arguments: ./life <input_file> <num_of_generations> <X_limit> <Y_limit>\n");
        MPI_Finalize();
        return 1;
    }

    const char *input_file_name = argv[1];
    int num_of_generations = atoi(argv[2]);
    int X_limit = atoi(argv[3]);
    int Y_limit = atoi(argv[4]);

    int rows_per_proc = X_limit / size;
    int remainder = X_limit % size;
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    int offset = rank * rows_per_proc + ((rank < remainder) ? rank : remainder);

    int *live_cells = NULL;
    int num_live_cells = 0;
    read_input_file(&live_cells, &num_live_cells, input_file_name, rank, size, rows_per_proc, remainder);

    int rows = local_rows + 2;  // +2 for padding (ghost rows)
    int cols = Y_limit + 2;     // +2 for padding (ghost columns)
    int *life = (int *)malloc(rows * cols * sizeof(int));
    int *previous_life = (int *)malloc(rows * cols * sizeof(int));

    memset(life, 0, rows * cols * sizeof(int));
    memset(previous_life, 0, rows * cols * sizeof(int));

    for (int i = 0; i < num_live_cells * 2; i += 2) {
        int x = live_cells[i];
        int y = live_cells[i + 1];
        int idx_i = x - offset + 1;  // +1 for padding
        int idx_j = y + 1;           // +1 for padding
        if (idx_i >= 1 && idx_i <= local_rows && idx_j >= 1 && idx_j <= Y_limit) {
            life[idx_i * cols + idx_j] = 1;
        }
    }

    if (live_cells != NULL) {
        free(live_cells);
    }

    double start = MPI_Wtime();
    for (int numg = 0; numg < num_of_generations; numg++) {
        compute(life, previous_life, local_rows, Y_limit, cols, rank, size);
    }
    double end = MPI_Wtime();
    double local_time = end - start;

    double min_time, max_time, avg_time;
    MPI_Reduce(&local_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    avg_time /= size;

    // if (rank == 0) {
    //    printf("TIME: Min: %f s Avg: %f s Max: %f s\n", min_time, avg_time, max_time);
    // }

    write_output(life, local_rows, Y_limit, cols, offset, input_file_name, num_of_generations, rank, size);

    free(life);
    free(previous_life);

    MPI_Finalize();
    return 0;
}
