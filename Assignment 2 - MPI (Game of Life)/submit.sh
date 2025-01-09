#!/bin/bash

# Allocate 16 cores on a single node for 5 minutes
#SBATCH -N 1
#SBATCH --ntasks=128
#SBATCH -t 00:05:00
#SBATCH -A cmsc416-class


# This is to suppress the warning about not finding a GPU resource
export OMPI_MCA_mpi_cuda_support=0

# Load OpenMPI
module load openmpi/gcc
make clean
make all

# Run the executable
mpirun -np 4 ./life-nonblocking life.1.512x512.data 500 512 512 > life-nonblocking4.out
mpirun -np 8 ./life-nonblocking life.1.512x512.data 500 512 512 > life-nonblocking8.out
mpirun -np 16 ./life-nonblocking life.1.512x512.data 500 512 512 > life-nonblocking16.out
mpirun -np 32 ./life-nonblocking life.1.512x512.data 500 512 512 > life-nonblocking32.out
mpirun -np 64 ./life-nonblocking life.1.512x512.data 500 512 512 > life-nonblocking64.out
mpirun -np 128 ./life-nonblocking life.1.512x512.data 500 512 512 > life-nonblocking128.out

