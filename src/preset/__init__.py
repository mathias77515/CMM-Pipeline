from .preset_tools import *
from .preset_qubic import *
from .preset_foregrounds import *
from .preset_gain import *
from .preset_acquisition import *
from .preset_external_data import *
from .preset_sky import *
from .preset_mixingmatrix import *

class PresetInitialisation:
    """
    
    Instance to initialize the Components Map-Making reading all the preset files.
    
    Arguments : 
    ===========
        - comm    : MPI common communicator (define by MPI.COMM_WORLD).
        - seed    : Int number for CMB realizations.
        - it      : Int number for noise realizations.
        - verbose : bool, Display message or not.
    
    """
    def __init__(self, comm, seed, seed_noise):
        """
        Initialize the class with MPI communication and optional verbosity.

        Args:
            comm: MPI communicator object.
            verbose (bool): If True, print detailed initialization messages. Default is True.
        """
        ### MPI common arguments
        self.comm = comm
        self.seed = seed
        self.seed_noise = seed_noise

        self.tools = None
        self.qubic = None
        self.fg = None
        self.gain = None
        self.acquisition = None
        self.external = None
        self.sky = None
        self.mixingmatrix = None

    def initialize(self):
        self.tools = PresetTools(self.comm)
        self.tools._print_message('========= Initialization =========')
        self.tools._print_message('    => Checking simulation parameters')
        self.tools._check_for_errors()
        self.tools._print_message('    => No error detected !')
        self.tools._print_message('    => Getting job ID')
        self.job_id = os.environ.get('SLURM_JOB_ID')
        self.tools._print_message('    => Creating folders')
        if self.tools.rank == 0:
            if self.tools.params['save_iter'] != 0:
                self.tools.params['foldername'] = f"{self.tools.params['Foregrounds']['Dust']['type']}_{self.tools.params['Foregrounds']['Dust']['model_d']}_{self.tools.params['QUBIC']['instrument']}_" + self.tools.params['foldername']
                self.tools.create_folder_if_not_exists(self.tools.params['foldername'])
            if self.tools.params['Plots']['maps'] == True or self.tools.params['Plots']['conv_beta'] == True:
                self.tools.create_folder_if_not_exists(f'jobs/{self.job_id}/I')
                self.tools.create_folder_if_not_exists(f'jobs/{self.job_id}/Q')
                self.tools.create_folder_if_not_exists(f'jobs/{self.job_id}/U')
                self.tools.create_folder_if_not_exists(f'jobs/{self.job_id}/allcomps')

        self.tools._print_message('========= External Data =========')
        self.external = PresetExternal(self.tools)

        self.tools._print_message('========= QUBIC =========')
        self.qubic = PresetQubic(self.tools, self.external)

        self.tools._print_message('========= Foregrounds =========')
        self.fg = PresetFG(self.tools, self.qubic, self.seed)

        self.tools._print_message('========= Sky =========')
        self.sky = PresetSky(self.tools, self.qubic)

        self.tools._print_message('========= GAIN =========')
        self.gain = PresetGain(self.tools, self.qubic)

        self.tools._print_message('========= Mixing Matrix =========')
        self.mixingmatrix = PresetMixingMatrix(self.tools, self.qubic, self.fg)

        self.tools._print_message('========= Acquisition =========')
        self.acquisition = PresetAcquisition(self.seed_noise, self.tools, self.external, self.qubic, self.sky, self.fg, self.mixingmatrix, self.gain)

        self.tools.display_simulation_configuration() 

        return self