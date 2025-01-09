CXXFLAGS = -std=c++11 -O2
MPICXX = mpic++

all: life-nonblocking
life-nonblocking: life-nonblocking.cpp
	$(MPICXX) $(CXXFLAGS) -o life-nonblocking $<

clean:
	rm -f life-nonblocking *.out *.csv
