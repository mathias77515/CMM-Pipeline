import numpy as np

import qubic

from preset.preset_main import *

class PresetQubic:
    """
    
    Instance to initialize the Components Map-Making. It reads the `params.yml` file to define QUBIC instrument.
    
    Arguments : 
    ===========
        - comm    : MPI common communicator (define by MPI.COMM_WORLD).
        - seed    : Int number for CMB realizations.
        - it      : Int number for noise realizations.
        - verbose : bool, Display message or not.
    
    """
    def __init__(self, comm, verbose=True):
        """
        Initialize the class with MPI communication and optional verbosity.

        Args:
            comm: MPI communicator object.
            verbose (bool): If True, print detailed initialization messages. Default is True.
        """

        self.preset = PresetMain(comm, verbose=verbose)

        self.comm = comm
        self.size = self.comm.Get_size()
        self.verbose = verbose

        ### QUBIC dictionary
        if self.verbose:
            self.preset._print_message('    => Reading QUBIC dictionary')
        self.dict = self._get_dict()

        ### QUBIC parameters
        self.instrument = self.preset.params['QUBIC']['instrument']
        self.fit_gain = self.preset.params['QUBIC']['GAIN']['fit_gain']
        self.convolution_out = self.preset.params['QUBIC']['convolution_out']
        self.nsub_out = self.preset.params['QUBIC']['nsub_out']
        self.dtheta = self.preset.params['QUBIC']['dtheta']
        
    def _get_ultrawideband_config(self):
        """
        Method to define Ultra Wide Band configuration.

        This method calculates the average frequency (average_frequency) and the normalized 
        frequency deviation (2*difference_frequency/average_frequency) for the Ultra Wide Band.

        Returns:
            tuple: A tuple containing:
                - average_frequency (float): The average frequency.
                - normalized_deviation (float): The normalized frequency deviation.
        """
        maximum_frequency = 247.5
        minimum_frequency = 131.25
        average_frequency = np.mean(np.array([maximum_frequency, minimum_frequency]))
        difference_frequency = maximum_frequency - average_frequency

        return average_frequency, 2*difference_frequency/average_frequency

    def _get_dict(self):
        """
        Method to define and modify the QUBIC dictionary.

        This method retrieves the ultrawideband configuration and constructs a dictionary
        with various parameters required for the QUBIC pipeline. It then reads a default
        dictionary from a file and updates it with the constructed parameters.

        Returns:
            qubic.qubicdict.qubicDict: The modified QUBIC dictionary.
        """

        # Retrieve ultrawideband configuration
        average_frequency, difference_frequency_nu_over_nu = self._get_ultrawideband_config()

        # Construct the arguments dictionary with required parameters
        args = {
            'npointings': self.preset.params['QUBIC']['npointings'],
            'nf_recon': 1,
            'nf_sub': self.preset.params['QUBIC']['nsub_in'],
            'nside': self.preset.params['SKY']['nside'],
            'MultiBand': True,
            'period': 1,
            'RA_center': self.preset.params['SKY']['RA_center'],
            'DEC_center': self.preset.params['SKY']['DEC_center'],
            'filter_nu': average_frequency * 1e9,
            'noiseless': False,
            'comm': self.comm,
            'kind': 'IQU',
            'config': 'FI',
            'verbose': False,
            'dtheta': self.preset.params['QUBIC']['dtheta'],
            'nprocs_sampling': 1,
            'nprocs_instrument': self.size,
            'photon_noise': True,
            'nhwp_angles': 3,
            'effective_duration': 3,
            'filter_relative_bandwidth': difference_frequency_nu_over_nu,
            'type_instrument': 'wide',
            'TemperatureAtmosphere150': None,
            'TemperatureAtmosphere220': None,
            'EmissivityAtmosphere150': None,
            'EmissivityAtmosphere220': None,
            'detector_nep': float(self.preset.params['QUBIC']['NOISE']['detector_nep']),
            'synthbeam_kmax': self.preset.params['QUBIC']['SYNTHBEAM']['synthbeam_kmax'],
            'synthbeam_fraction': self.preset.params['QUBIC']['SYNTHBEAM']['synthbeam_fraction']
        }

        # Get the default dictionary
        dictfilename = 'dicts/pipeline_demo.dict'
        d = qubic.qubicdict.qubicDict()
        d.read_from_file(dictfilename)

        # Update the default dictionary with the constructed parameters
        for i in args.keys():
            d[str(i)] = args[i]

        return d