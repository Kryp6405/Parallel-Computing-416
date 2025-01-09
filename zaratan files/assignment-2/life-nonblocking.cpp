#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
using namespace std;
#include <vector>
#include <mpi.h>
#include <omp.h>

/*
 * Reads the input file line by line and stores it in a 1D array.
 */
void read_input_file(int *life, int X_limit, int Y_limit, string const &input_file_name) {

    // Open the input file for reading.
    ifstream input_file;
    input_file.open(input_file_name);
    if (!input_file.is_open())
        perror("Input file cannot be opened");
    
    int x, y;
    string line;

    while (getline(input_file, line)) {
        size_t comma = line.find(',');
        x = stoi(line.substr(0, comma));
        y = stoi(line.substr(comma + 1));
        
        // Formula for 1D decomp
        life[x * Y_limit + y] = 1;
    }
    input_file.close();
}

/* 
 * Writes out the final state of the 1D array to a csv file. 
 */
void write_output(int *result_array, int X_limit, int Y_limit, string const &input_name, int num_of_generations) {
    
    // Open the output file for writing.
    ofstream output_file;
    string input_file_name = input_name.substr(0, input_name.length() - 5);
    output_file.open(input_file_name + "." + to_string(num_of_generations) + ".csv");
    if (!output_file.is_open())
        perror("Output file cannot be opened");

    for (int i = 0; i < X_limit; i++) {
        for (int j = 0; j < Y_limit; j++) {
            
            // Mapping 1D array onto csv output file
            if (result_array[i * Y_limit + j] == 1) {
                output_file << i << "," << j << "\n";
            }
        }
    }
    output_file.close();
}

/* 
 * Calculates the state of the cell based on neighbor count.
 */
int apply_rules(int cell_value, int neighbors) {
    // Alive
    if (cell_value == 1) {
        if (neighbors == 2 || neighbors == 3) {
            return 1;
        } else {
            return 0;
        }
    }
    // Dead 
    else {
        if (neighbors == 3) {
            return 1;
        } else {
            return 0;
        }
    }
}

/*
 * Processes the life array for the specified number of iterations.
 */
void compute(int *life, int *previous_life, int local_rows, int Y_limit, int rank, int size, int num_of_generations) {
    MPI_Request requests[4];
    int total_rows = local_rows + 2;

    for (int gen = 0; gen < num_of_generations; gen++) {
        memcpy(previous_life, life, total_rows * Y_limit * sizeof(int));

        // Non-blocking communication for before and after process
        if (rank > 0) {
            MPI_Isend(&previous_life[1 * Y_limit], Y_limit, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(&previous_life[0], Y_limit, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, &requests[1]);
        } else {
            memset(&previous_life[0 * Y_limit], 0, Y_limit * sizeof(int));
            requests[0] = MPI_REQUEST_NULL;
            requests[1] = MPI_REQUEST_NULL;
        }
        if (rank < size - 1) {
            MPI_Isend(&previous_life[local_rows * Y_limit], Y_limit, MPI_INT, rank + 1, 1, MPI_COMM_WORLD, &requests[2]);
            MPI_Irecv(&previous_life[(local_rows + 1) * Y_limit], Y_limit, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &requests[3]);
        } else {
            memset(&previous_life[(local_rows + 1) * Y_limit], 0, Y_limit * sizeof(int));
            requests[2] = MPI_REQUEST_NULL;
            requests[3] = MPI_REQUEST_NULL;
        }

        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

        // Compute all data
	#pragma omp parallel for private(index, neighbors)
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 0; j < Y_limit; j++) {
                int index = i * Y_limit + j;
                int neighbors = 0;

                // Top row neighbors
                if (j > 0) {
                    neighbors += previous_life[(i - 1) * Y_limit + (j - 1)];
                }
                neighbors += previous_life[(i - 1) * Y_limit + j];
                if (j < Y_limit - 1) {
                    neighbors += previous_life[(i - 1) * Y_limit + (j + 1)];
                }

                // Same row neighbors
                if (j > 0) {
                    neighbors += previous_life[i * Y_limit + (j - 1)];
                }
                if (j < Y_limit - 1) {
                    neighbors += previous_life[i * Y_limit + (j + 1)];
                }

                // Bottom row neighbors
                if (j > 0) {
                    neighbors += previous_life[(i + 1) * Y_limit + (j - 1)];
                }
                neighbors += previous_life[(i + 1) * Y_limit + j];
                if (j < Y_limit - 1) {
                    neighbors += previous_life[(i + 1) * Y_limit + (j + 1)];
                }

                // Rules
                life[index] = apply_rules(previous_life[index], neighbors);
            }
        }
    }
}

/**
  * The main function to execute "Game of Life" simulations.
  */
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    if (argc != 5) {
        perror("Expected arguments: ./life <input_file> <num_of_generations> <X_limit> <Y_limit>");

        // Stop parallel environment if error encountered
        MPI_Finalize();
    }

    int rank, size, local_rows;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    string input_file_name = argv[1];
    int num_of_generations = stoi(argv[2]);
    int X_limit = stoi(argv[3]);
    int Y_limit = stoi(argv[4]);

    int row_distribution = X_limit / size;
    int remainder = X_limit % size;
    if (rank < remainder) {
        local_rows = row_distribution + 1;
    } else {
        local_rows = row_distribution;
    }

    int *rows_per_process = new int[size];
    int *chunks = new int[size];
    int *offsets = new int[size];
    int temp_offset = 0;

    // Use chunks and offsets
    for (int i = 0; i < size; i++) {
        if (i < remainder) {
            rows_per_process[i] = row_distribution + 1;
        } else {
            rows_per_process[i] = row_distribution;
        }
        chunks[i] = rows_per_process[i] * Y_limit;
        offsets[i] = temp_offset;
        temp_offset += chunks[i];
    }

    int total_rows = local_rows + 2;  // Including ghost rows

    // Creating life and previous_life 1D arrays
    int *life = new int[total_rows * Y_limit]();
    int *previous_life = new int[total_rows * Y_limit ]();

    // If first process, distribute workload
    if (rank == 0) {
        int *all_grid = new int[X_limit * Y_limit]();
        read_input_file(all_grid, X_limit, Y_limit, input_file_name);
        memcpy(&life[1 * Y_limit], &all_grid[offsets[0]], chunks[0] * sizeof(int));
        // Send data to each process
        for (int process = 1; process < size; process++) {
            MPI_Send(&all_grid[offsets[process]], chunks[process], MPI_INT, process, 0, MPI_COMM_WORLD);
        }
        delete[] all_grid;
    } else {
        // Receive data from parent
        MPI_Recv(&life[1 * Y_limit], chunks[rank], MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Start timing calculation
    double start_time = MPI_Wtime();
    compute(life, previous_life, local_rows, Y_limit, rank, size, num_of_generations);
    double end_time = MPI_Wtime();

    double diff_time = end_time - start_time;
    double min_time, max_time, avg_time;

    MPI_Reduce(&diff_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&diff_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&diff_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    avg_time /= size;

    if (rank == 0) {
       printf("TIME: Min: %f s Avg: %f s Max: %f s\n", min_time, avg_time, max_time);
    }
    
    // Gather all data and write
    if (rank == 0) {
        int *all_grid = new int[X_limit * Y_limit]();
        memcpy(&all_grid[offsets[rank]], &life[1 * Y_limit], chunks[rank] * sizeof(int));
        for (int process = 1; process < size; process++) {
            MPI_Recv(&all_grid[offsets[process]], chunks[process], MPI_INT, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        write_output(all_grid, X_limit, Y_limit, input_file_name, num_of_generations);
        delete[] all_grid;
    } else {
        MPI_Send(&life[1 * Y_limit], chunks[rank], MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    // Free allocated memory
    delete[] life;
    delete[] previous_life;
    delete[] rows_per_process;
    delete[] chunks;
    delete[] offsets;

    MPI_Finalize();
    return 0;
}
