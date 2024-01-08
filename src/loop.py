import os

N = 0
M = 200

for i in range(N, M):
    os.system(f'sbatch main.sh {1} {i+1}')
