import os

N = 0
M = 2

for i in range(N, M):
    os.system(f'sbatch main.sh {i+1}')

