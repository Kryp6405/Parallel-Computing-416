#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <mpi.h>
using namespace std;

/*
 * Reads the input file and broadcasts the live cell positions to all processes.
 */
void read_input_file(vector<pair<int, int>> &live_cells, string const &input_file_name, int rank) {
    if (rank == 0) {
        // Open the input file for reading.
        ifstream input_file(input_file_name);
        if (!input_file.is_open()) {
            perror("Input file cannot be opened");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        string line, val;
        int x, y;
        while (getline(input_file, line)) {
            stringstream ss(line);

            // Read x coordinate.
            getline(ss, val, ',');
            x = stoi(val);

            // Read y coordinate.
            getline(ss, val);
            y = stoi(val);

            // Store live cell coordinates.
            live_cells.push_back(make_pair(x, y));
        }
        input_file.close();
    }

    // Broadcast the number of live cells.
    int num_live_cells = live_cells.size();
    MPI_Bcast(&num_live_cells, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast the live cells.
    if (rank != 0) {
        live_cells.resize(num_live_cells);
    }
    MPI_Bcast(live_cells.data(), num_live_cells * sizeof(pair<int, int>), MPI_BYTE, 0, MPI_COMM_WORLD);
}

/* 
 * Gathers the live cells from all processes to rank 0 and writes to the output file.
 */
void write_output(int **life, int local_rows, int Y_limit, int offset, string const &input_name, int num_of_generations, int rank, int size) {
    vector<int> local_live_cells;

    // Collect local live cells.
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 1; j <= Y_limit; j++) {
            if (life[i][j] == 1) {
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
        string input_file_name = input_name.substr(0, input_name.length() - 5);
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
    } else{ printf("not in rank 0\n"); }
}

/*
 * Processes the life array for the specified number of iterations.
 */
void compute(int **life, int **previous_life, int local_rows, int Y_limit, int rank, int size) {
    int neighbors = 0;
    MPI_Request request[4];
    int top = rank - 1;
    int bottom = rank + 1;

    // Start of generations loop (assuming it's called in a loop)
    // For each generation:
    // 1. Copy life to previous_life
    // 2. Exchange boundary rows
    // 3. Compute inner cells
    // 4. Wait for communication to complete
    // 5. Compute boundary cells

    // Copy life to previous_life
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 1; j <= Y_limit; j++) {
            previous_life[i][j] = life[i][j];
        }
    }

    // Non-blocking communication with neighboring processes
    int tag1 = 1, tag2 = 2;
    if (rank > 0) {
        // Send top row to the process above
        MPI_Isend(previous_life[1], Y_limit + 2, MPI_INT, top, tag1, MPI_COMM_WORLD, &request[0]);
        // Receive from the process above
        MPI_Irecv(previous_life[0], Y_limit + 2, MPI_INT, top, tag2, MPI_COMM_WORLD, &request[1]);
    } else {
        // Set the padding row to zero for top edge
        for (int j = 0; j <= Y_limit + 1; j++) {
            previous_life[0][j] = 0;
        }
        request[0] = MPI_REQUEST_NULL;
        request[1] = MPI_REQUEST_NULL;
    }
    if (rank < size - 1) {
        // Send bottom row to the process below
        MPI_Isend(previous_life[local_rows], Y_limit + 2, MPI_INT, bottom, tag2, MPI_COMM_WORLD, &request[2]);
        // Receive from the process below
        MPI_Irecv(previous_life[local_rows + 1], Y_limit + 2, MPI_INT, bottom, tag1, MPI_COMM_WORLD, &request[3]);
    } else {
        // Set the padding row to zero for bottom edge
        for (int j = 0; j <= Y_limit + 1; j++) {
            previous_life[local_rows + 1][j] = 0;
        }
        request[2] = MPI_REQUEST_NULL;
        request[3] = MPI_REQUEST_NULL;
    }

    // Compute inner cells (excluding boundary rows)
    for (int i = 2; i < local_rows; i++) {
        for (int j = 1; j <= Y_limit; j++) {
            neighbors = previous_life[i - 1][j - 1] + previous_life[i - 1][j] + previous_life[i - 1][j + 1] +
                        previous_life[i][j - 1] + previous_life[i][j + 1] +
                        previous_life[i + 1][j - 1] + previous_life[i + 1][j] + previous_life[i + 1][j + 1];

            if (previous_life[i][j] == 0) {
                // A cell is born only when an unoccupied cell has 3 neighbors.
                if (neighbors == 3)
                    life[i][j] = 1;
                else
                    life[i][j] = 0;
            } else {
                // An occupied cell survives only if it has either 2 or 3 neighbors.
                if (neighbors == 2 || neighbors == 3)
                    life[i][j] = 1;
                else
                    life[i][j] = 0;
            }
        }
    }

    // Wait for non-blocking communication to complete
    MPI_Waitall(4, request, MPI_STATUSES_IGNORE);

    // Compute boundary cells
    int i = 1; // First row
    for (int j = 1; j <= Y_limit; j++) {
        neighbors = previous_life[i - 1][j - 1] + previous_life[i - 1][j] + previous_life[i - 1][j + 1] +
                    previous_life[i][j - 1] + previous_life[i][j + 1] +
                    previous_life[i + 1][j - 1] + previous_life[i + 1][j] + previous_life[i + 1][j + 1];

        if (previous_life[i][j] == 0) {
            if (neighbors == 3)
                life[i][j] = 1;
            else
                life[i][j] = 0;
        } else {
            if (neighbors == 2 || neighbors == 3)
                life[i][j] = 1;
            else
                life[i][j] = 0;
        }
    }
    i = local_rows; // Last row
    for (int j = 1; j <= Y_limit; j++) {
        neighbors = previous_life[i - 1][j - 1] + previous_life[i - 1][j] + previous_life[i - 1][j + 1] +
                    previous_life[i][j - 1] + previous_life[i][j + 1] +
                    previous_life[i + 1][j - 1] + previous_life[i + 1][j] + previous_life[i + 1][j + 1];

        if (previous_life[i][j] == 0) {
            if (neighbors == 3)
                life[i][j] = 1;
            else
                life[i][j] = 0;
        } else {
            if (neighbors == 2 || neighbors == 3)
                life[i][j] = 1;
            else
                life[i][j] = 0;
        }
    }
}

/**
 * The main function to execute "Game of Life" simulations on a 2D board using MPI.
 */
int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);                // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get current process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get total number of processes

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
    int local_rows = rows_per_proc;
    if (rank < remainder)
        local_rows++;

    // Calculate the offset for each process
    int offset = rank * rows_per_proc + min(rank, remainder);

    // Allocate memory for life and previous_life arrays
    int **life = new int *[local_rows + 2];
    int **previous_life = new int *[local_rows + 2];
    for (int i = 0; i < local_rows + 2; i++) {
        life[i] = new int[Y_limit + 2];
        previous_life[i] = new int[Y_limit + 2];
        for (int j = 0; j < Y_limit + 2; j++) {
            life[i][j] = 0;
            previous_life[i][j] = 0;
        }
    }

    // Read input file and initialize live cells
    vector<pair<int, int>> live_cells;
    read_input_file(live_cells, input_file_name, rank);

    // Initialize the life matrix with live cells
    for (const auto &cell : live_cells) {
        int x = cell.first;
        int y = cell.second;
        if (x >= offset && x < offset + local_rows && y >= 0 && y < Y_limit) {
            life[x - offset + 1][y + 1] = 1;  // +1 for padding
        }
    }

    double start = MPI_Wtime();
    for (int numg = 0; numg < num_of_generations; numg++) {
        compute(life, previous_life, local_rows, Y_limit, rank, size);
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
    write_output(life, local_rows, Y_limit, offset, input_file_name, num_of_generations, rank, size);

    // Clean up memory
    for (int i = 0; i < local_rows + 2; i++) {
        delete[] life[i];
        delete[] previous_life[i];
    }
    delete[] life;
    delete[] previous_life;

    MPI_Finalize();
    return 0;
}

