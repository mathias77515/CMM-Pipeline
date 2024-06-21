import yaml

class PresetMain:


    """
    
    Instance to initialize the Components Map-Making. It contains the main functions to initialize all the different files.
    
    Arguments : 
    ===========
        - comm    : MPI common communicator (define by MPI.COMM_WORLD).
        - seed    : Int number for CMB realizations.
        - it      : Int number for noise realizations.
        - verbose : bool, Display message or not.
    
    """

    def __init__(self, comm, verbose = True):
        
        ### MPI common arguments
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        ### Open parameters file
        if verbose:
            self._print_message('    => Reading parameters file')
        with open('params.yml', "r") as stream:
            self.params = yaml.safe_load(stream)

    def _print_message(self, message):

        """
        Method that display a `message` only for the first rank because of MPI multiprocessing.
        """
        
        if self.rank == 0: 
            print(message)