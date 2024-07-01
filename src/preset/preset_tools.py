import yaml
import os

class PresetTools:
    """
    
    Instance to initialize the Components Map-Making. It contains tool functions to initialize all the different files.

    """

    def __init__(self, comm):
        """
        Initialize the class with MPI communication and parameters from a YAML file.

        Args:
            comm: MPI communicator object.
        """
        
        ### MPI common arguments
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        ### Open parameters file
        self._print_message('========= Initialization =========')
        self._print_message('    => Reading parameters file')
        with open('params.yml', "r") as stream:
            self.params = yaml.safe_load(stream)
            
    def _print_message(self, message):
        """
        Display a `message` only for the first rank in an MPI multiprocessing environment.

        Parameters:
        message (str): The message to be displayed.

        Returns:
        None
        """
        if self.rank == 0: 
            print(message)

    def create_folder_if_not_exists(self, folder_name):
        """
        Creates a folder with the specified name if it does not already exist.

        Args:
            folder_name (str): The name of the folder to create.

        Returns:
            None
        """
        # Check if the folder exists
        if not os.path.exists(folder_name):
            try:
                # Create the folder if it doesn't exist
                os.makedirs(folder_name)
                print(f"The folder '{folder_name}' has been created.")
            except OSError as e:
                print(f"Error creating the folder '{folder_name}': {e}")
        else:
            pass
        
    def _check_for_errors(self):
        """
        Checks for various parameter errors in the 'params.yml' file.
        
        Raises:
            TypeError: If any of the parameter checks fail.
        """
        with open('params.yml', "r") as stream:
            params = yaml.safe_load(stream)

        # Check if the instrument is either 'DB' or 'UWB'
        if params['QUBIC']['instrument'] not in ['DB', 'UWB']:
            raise TypeError('You must choose DB or UWB instrument')

        # Check if bin_mixing_matrix is even
        if params['Foregrounds']['bin_mixing_matrix'] % 2 != 0:
            raise TypeError('The argument bin_mixing_matrix should be even')

        # Check if nsub_in is even
        if params['QUBIC']['nsub_in'] % 2 != 0:
            raise TypeError('The argument nsub_in should be even')

        # Check if nsub_out is even
        if params['QUBIC']['nsub_out'] % 2 != 0:
            raise TypeError('The argument nsub_out should be even')

        # Check if blind_method is one of the allowed methods
        if params['Foregrounds']['blind_method'] not in ['alternate', 'minimize', 'PCG']:
            raise TypeError('You must choose alternate, minimize or PCG method')

        # Check if nsub_out is greater than or equal to bin_mixing_matrix
        if params['QUBIC']['nsub_out'] < params['Foregrounds']['bin_mixing_matrix']:
            raise TypeError('nsub_out should be higher than bin_mixing_matrix')

        # Check if bin_mixing_matrix is a multiple of nsub_out when either Dust or Synchrotron type is 'blind'
        if params['Foregrounds']['Dust']['type'] == 'blind' or params['Foregrounds']['Synchrotron']['type'] == 'blind':
            if params['QUBIC']['nsub_out'] % params['Foregrounds']['bin_mixing_matrix'] != 0:
                raise TypeError('bin_mixing_matrix should be a multiple of nsub_out')
            
    def display_simulation_configuration(self):
        """
        Display the simulation configuration details.
        
        This method prints out the configuration settings for the simulation, including
        details about the sky input and output, QUBIC instrument settings, and MPI tasks.
        The configuration is only displayed if the rank of the process is 0.
        """
        if self.rank == 0:
            print('******************** Configuration ********************\n')
            print('    - QUBIC :')
            print(f"        Npointing : {self.params['QUBIC']['npointings']}")
            print(f"        Nsub : {self.params['QUBIC']['nsub_in']}")
            print(f"        Ndet : {self.params['QUBIC']['NOISE']['ndet']}")
            print(f"        Npho150 : {self.params['QUBIC']['NOISE']['npho150']}")
            print(f"        Npho220 : {self.params['QUBIC']['NOISE']['npho220']}")
            print(f"        RA : {self.params['SKY']['RA_center']}")
            print(f"        DEC : {self.params['SKY']['DEC_center']}")
            if self.params['QUBIC']['instrument'] == 'DB':
                print(f"        Type : Dual Bands")
            else:
                print(f"        Type : Ultra Wide Band")
            print('    - Sky In :')
            print(f"        CMB : {self.params['CMB']['cmb']}")
            print(f"        Dust : {self.params['Foregrounds']['Dust']['Dust_in']} - {self.params['Foregrounds']['Dust']['model_d']}")
            print(f"        Synchrotron : {self.params['Foregrounds']['Synchrotron']['Synchrotron_in']} - {self.params['Foregrounds']['Synchrotron']['model_s']}")
            print(f"        CO : {self.params['Foregrounds']['CO']['CO_in']}\n")
            print('    - Sky Out :')
            print(f"        CMB : {self.params['CMB']['cmb']}")
            print(f"        Dust : {self.params['Foregrounds']['Dust']['Dust_out']} - {self.params['Foregrounds']['Dust']['model_d']}")
            print(f"        Synchrotron : {self.params['Foregrounds']['Synchrotron']['Synchrotron_out']} - {self.params['Foregrounds']['Synchrotron']['model_s']}")
            print(f"        CO : {self.params['Foregrounds']['CO']['CO_out']}\n")
            
            print(f"        MPI Tasks : {self.size}")
