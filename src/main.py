import numpy as np
from pipeline import Pipeline
from pyoperators import *

### MPI common arguments
comm = MPI.COMM_WORLD

if __name__ == "__main__":
    
    pipeline = Pipeline(comm)
    
    pipeline.main()