import numpy as np
from pipeline import Pipeline
from pyoperators import *
import sys

seed = int(sys.argv[1])
it = int(sys.argv[2])

### MPI common arguments
comm = MPI.COMM_WORLD

if __name__ == "__main__":
    
    pipeline = Pipeline(comm, seed, it)
    
    pipeline.main()