CXX = g++
CXXFLAGS = -std=c++11 -O2

MPICXX = mpic++

all: life life-nonblocking

life: serial.cpp
	$(CXX) $(CXXFLAGS) -o life $<

life-nonblocking: life-nonblocking5.cpp
	module load openmpi/gcc && $(MPICXX) $(CXXFLAGS) -o life-nonblocking $<

clean:
	rm -f life life-nonblocking *.out *.csv
