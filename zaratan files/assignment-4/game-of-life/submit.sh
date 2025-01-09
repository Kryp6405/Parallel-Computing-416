#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_1g.5gb
#SBATCH -A cmsc416-class
#SBATCH -o cuda-game-of-life-%A.out
#SBATCH -J cuda-game-of-life
#SBATCH -t 00:00:15

# load cuda libraries
module load cuda

# 512 2b problem
./life life.512x512.data 100 512 512
./game-of-life life.512x512.data 100 512 512 32 32 gpu.life.512x512.100.csv
diff life.512x512.100.csv gpu.life.512x512.100.csv
