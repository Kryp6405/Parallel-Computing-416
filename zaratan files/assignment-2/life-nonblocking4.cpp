#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstring>   // For memset
#include <algorithm> // For min function

using namespace std;

/**
 * Reads the input file on rank 0 and distributes the live cells to the relevant processes.
 */
void read_input_file(vector<pair<int, int>> &live_cells, const string &input_file_name, int rank, int size, int rows_per_proc, int remainder) {
    if (rank == 0) {
        // Open the input file for reading.
        ifstream input_file(input_file_name);
        if (!input_file.is_open()) {
            perror("Input file cannot be opened");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        string line, val;
        int x, y;
        vector<vector<int>> proc_live_cells_flat(size); // Flattened x and y coordinates
        while (getline(input_file, line)) {
            stringstream ss(line);

            // Read x coordinate.
            getline(ss, val, ',');
            x = stoi(val);

            // Read y coordinate.
            getline(ss, val);
            y = stoi(val);

            // Determine which process the cell belongs to
            for (int p = 0; p < size; p++) {
                int proc_rows = rows_per_proc + (p < remainder ? 1 : 0);
                int offset = p * rows_per_proc + min(p, remainder);
                if (x >= offset && x < offset + proc_rows) {
                    proc_live_cells_flat[p].push_back(x);
                    proc_live_cells_flat[p].push_back(y);
                    break;
                }
            }
        }
        input_file.close();

        // Send live cells to each process
        for (int p = 0; p < size; p++) {
            if (p == 0) continue;

            int num_cells_p = proc_live_cells_flat[p].size() / 2;
            MPI_Send(&num_cells_p, 1, MPI_INT, p, 0, MPI_COMM_WORLD);

            if (num_cells_p > 0) {
                MPI_Send(proc_live_cells_flat[p].data(), num_cells_p * 2, MPI_INT, p, 0, MPI_COMM_WORLD);
            }
        }

        // For rank 0, initialize live_cells with its own data
        live_cells.clear();
        for (size_t i = 0; i < proc_live_cells_flat[0].size(); i += 2) {
            x = proc_live_cells_flat[0][i];
            y = proc_live_cells_flat[0][i + 1];
            live_cells.push_back(make_pair(x, y));
        }
    } else {
        // Receive live cells from rank 0
        int num_cells_p;
        MPI_Recv(&num_cells_p, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        vector<int> proc_live_cells_flat(num_cells_p * 2);
        if (num_cells_p > 0) {
            MPI_Recv(proc_live_cells_flat.data(), num_cells_p * 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Initialize live_cells with received data
        live_cells.clear();
        for (size_t i = 0; i < proc_live_cells_flat.size(); i += 2) {
            int x = proc_live_cells_flat[i];
            int y = proc_live_cells_flat[i + 1];
            live_cells.push_back(make_pair(x, y));
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

    // Copy life to previous_life
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

    // Wait for communication to complete
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
void write_output(int *life, int local_rows, int Y_limit, int cols, int offset, const string &input_name, int num_of_generations, int rank, int size) {
    vector<int> local_live_cells;

    // Collect local live cells.
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 1; j <= Y_limit; j++) {
            int idx = i * cols + j;
            if (life[idx] == 1) {
                local_live_cells.push_back(offset + i - 1); // x coordinate
                local_live_cells.push_back(j - 1);          // y coordinate
            }
        }
    }

    // Gather live cells counts from all processes.
    int local_count = local_live_cells.size(); // Number of integers to send
    vector<int> counts(size);
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute displacements and total_count on rank 0
    vector<int> displs(size);
    int total_count = 0;
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 0; i < size; i++) {
            if (i > 0)
                displs[i] = displs[i - 1] + counts[i - 1];
            total_count += counts[i];
        }
    }

    vector<int> all_live_cells(total_count);
    MPI_Gatherv(local_live_cells.data(), local_count, MPI_INT,
                all_live_cells.data(), counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    // Rank 0 writes the output file.
    if (rank == 0) {
        // Open the output file for writing.
        ofstream output_file;
        string input_file_name = input_name.substr(0, input_name.length() - 5); // Remove ".csv"
        output_file.open(input_file_name + "." + to_string(num_of_generations) + ".csv");
        if (!output_file.is_open()) {
            perror("Output file cannot be opened");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Output each live cell on a new line.
        for (size_t i = 0; i < all_live_cells.size(); i += 2) {
            int x = all_live_cells[i];
            int y = all_live_cells[i + 1];
            output_file << x << "," << y << "\n";
        }
        output_file.close();
    }
}

/**
 * The main function to execute the Game of Life simulations using MPI.
 */
int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);                // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get current process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get total number of processes

    // Process command line arguments
    if (argc != 5) {
        if (rank == 0)
            cerr << "Expected arguments: ./life <input_file> <num_of_generations> <X_limit> <Y_limit>" << endl;
        MPI_Finalize();
        return 1;
    }

    string input_file_name = argv[1];
    int num_of_generations = stoi(argv[2]);
    int X_limit = stoi(argv[3]);
    int Y_limit = stoi(argv[4]);

    // Determine the number of rows each process will handle
    int rows_per_proc = X_limit / size;
    int remainder = X_limit % size;
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

    // Calculate the offset for each process
    int offset = rank * rows_per_proc + min(rank, remainder);

    // Read input file and initialize live cells
    vector<pair<int, int>> live_cells;
    read_input_file(live_cells, input_file_name, rank, size, rows_per_proc, remainder);

    // Allocate memory for life and previous_life arrays as 1D arrays
    int rows = local_rows + 2;  // +2 for padding (ghost rows)
    int cols = Y_limit + 2;     // +2 for padding (ghost columns)
    int *life = new int[rows * cols];
    int *previous_life = new int[rows * cols];

    // Initialize arrays to zero
    memset(life, 0, rows * cols * sizeof(int));
    memset(previous_life, 0, rows * cols * sizeof(int));

    // Initialize the life matrix with live cells
    for (const auto &cell : live_cells) {
        int x = cell.first;
        int y = cell.second;
        int i = x - offset + 1;  // +1 for padding
        int j = y + 1;           // +1 for padding
        if (i >= 1 && i <= local_rows && j >= 1 && j <= Y_limit) {
            life[i * cols + j] = 1;
        }
    }

    // Time the computation
    double start = MPI_Wtime();
    for (int numg = 0; numg < num_of_generations; numg++) {
        compute(life, previous_life, local_rows, Y_limit, cols, rank, size);
    }
    double end = MPI_Wtime();
    double local_time = end - start;

    // Get min, max, avg times
    double min_time, max_time, avg_time;
    MPI_Reduce(&local_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    avg_time /= size;

    if (rank == 0) {
        cout << "TIME: Min: " << min_time << " s Avg: " << avg_time << " s Max: " << max_time << " s\n";
    }

    // Write out the final state to the output file.
    write_output(life, local_rows, Y_limit, cols, offset, input_file_name, num_of_generations, rank, size);

    // Clean up memory
    delete[] life;
    delete[] previous_life;

    MPI_Finalize();
    return 0;
}

