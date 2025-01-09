all: problem1 problem1_parallel problem2 problem2_parallel problem3 problem3_parallel problem4 problem4_parallel

CC = g++
CFLAGS = -O2 -fopenmp

problem1: problem1.cpp
	$(CC) $(CFLAGS) -o problem1 $<

problem1_parallel: problem1_parallel.cpp
	$(CC) $(CFLAGS) -o problem1_parallel $<

problem2: problem2.cpp
	$(CC) $(CFLAGS) -o problem2 $<

problem2_parallel: problem2_parallel.cpp
	$(CC) $(CFLAGS) -o problem2_parallel $<

problem3: problem3.cpp
	$(CC) $(CFLAGS) -o problem3 $<

problem3_parallel: problem3_parallel.cpp
	$(CC) $(CFLAGS) -o problem3_parallel $<

problem4: problem4.cpp
	$(CC) $(CFLAGS) -o problem4 $<

problem4_parallel: problem4_parallel.cpp
	$(CC) $(CFLAGS) -o problem4_parallel $<

EXECUTABLES := problem1 problem1_parallel problem2 problem2_parallel problem3 problem3_parallel problem4 problem4_parallel

clean:
	rm -f $(EXECUTABLES) *.out
