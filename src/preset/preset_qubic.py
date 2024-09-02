import numpy as np

import qubic
from acquisition.Qacquisition import JointAcquisitionComponentsMapMaking

import fgb.component_model as c

class PresetQubic:
    """
    
    Instance to initialize the Components Map-Making. It defines QUBIC operator.

    Self variables :    - dict : dictionnary
                        - joint_in : class
                        - joint_out : class
    
    """
    def __init__(self, preset_tools, preset_external):
        """
        
        Initialize the class with preset tools and external.

        Args:
            preset_tools: Class containing tools and simulation parameters.
            preset_external: Class containing external frequencies.
        """
        ### Import preset tools
        self.preset_tools = preset_tools

        ### Define QUBIC parameters variable
        self.params_qubic = self.preset_tools.params['QUBIC']

        ### MPI common arguments
        self.comm = self.preset_tools.comm
        self.size = self.comm.Get_size()

        ### QUBIC dictionary
        self.preset_tools._print_message('    => Reading QUBIC dictionary')
        self.dict = self._get_dict()

        ### Define model for reconstruction
        components_fgb_in, _ = self._get_components_fgb(key='in')
        components_fgb_out, _ = self._get_components_fgb(key='out')

        if self.preset_tools.params['Foregrounds']['CO']['CO_in']:
            nu_co = self.preset_tools.params['Foregrounds']['CO']['nu0_co']
        else:
            nu_co = None

        ### Joint acquisition for QUBIC operator
        self.preset_tools._print_message('    => Building QUBIC operator')
        self.joint_in = JointAcquisitionComponentsMapMaking(self.dict, 
                                                        self.params_qubic['instrument'], 
                                                        components_fgb_in, 
                                                        self.params_qubic['nsub_in'],
                                                        preset_external.external_nus,
                                                        preset_external.params_external['nintegr_planck'],
                                                        nu_co=nu_co)

        if self.params_qubic['nsub_in'] == self.params_qubic['nsub_out']:
            self.joint_out = JointAcquisitionComponentsMapMaking(self.dict, 
                                                        self.params_qubic['instrument'], 
                                                        components_fgb_out, 
                                                        self.params_qubic['nsub_out'],
                                                        preset_external.external_nus,
                                                        preset_external.params_external['nintegr_planck'],
                                                        nu_co=nu_co,
                                                        H=self.joint_in.qubic.H)
        else:
            self.joint_out = JointAcquisitionComponentsMapMaking(self.dict, 
                                                        self.params_qubic['instrument'], 
                                                        components_fgb_out, 
                                                        self.params_qubic['nsub_out'],
                                                        preset_external.external_nus,
                                                        preset_external.params_external['nintegr_planck'],
                                                        nu_co=nu_co,
                                                        H=None)

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

        ### Retrieve ultrawideband configuration
        average_frequency, difference_frequency_nu_over_nu = self._get_ultrawideband_config()

        ### Construct the arguments dictionary with required parameters
        args = {
            'npointings': self.params_qubic['npointings'],
            'nf_recon': 1,
            'nf_sub': self.params_qubic['nsub_in'],
            'nside': self.preset_tools.params['SKY']['nside'],
            'MultiBand': True,
            'period': 1,
            'RA_center': self.preset_tools.params['SKY']['RA_center'],
            'DEC_center': self.preset_tools.params['SKY']['DEC_center'],
            'filter_nu': average_frequency * 1e9,
            'noiseless': False,
            'comm': self.comm,
            'kind': 'IQU',
            'config': 'FI',
            'verbose': False,
            'dtheta': self.params_qubic['dtheta'],
            'nprocs_sampling': 1,
            'nprocs_instrument': self.size,
            'photon_noise': True,
            'nhwp_angles': 3,
            'effective_duration': 3,
            'filter_relative_bandwidth': 0.25,#difference_frequency_nu_over_nu,
            #'type_instrument': 'wide',
            'TemperatureAtmosphere150': None,
            'TemperatureAtmosphere220': None,
            'EmissivityAtmosphere150': None,
            'EmissivityAtmosphere220': None,
            'detector_nep': float(self.params_qubic['NOISE']['detector_nep']),
            'synthbeam_kmax': self.params_qubic['SYNTHBEAM']['synthbeam_kmax'],
            'synthbeam_fraction': self.params_qubic['SYNTHBEAM']['synthbeam_fraction']
        }

        ### Get the default dictionary
        dictfilename = 'dicts/pipeline_demo.dict'
        d = qubic.qubicdict.qubicDict()
        d.read_from_file(dictfilename)

        ### Update the default dictionary with the constructed parameters
        for i in args.keys():
            d[str(i)] = args[i]

        return d
    def _get_components_fgb(self, key):
        """
        Method to define sky model taken from FGBuster code. Note that we add `COLine` instance to define monochromatic description.

        Parameters:
        key (str): The key to identify specific components in the preset parameters.

        Returns:
        tuple: A tuple containing two lists:
            - components (list): List of component instances.
            - components_name (list): List of component names corresponding to the instances.
        """

        components = []
        components_name = []

        if self.preset_tools.params['CMB']['cmb']:
            components += [c.CMB()]
            components_name += ['CMB']
            
        if self.preset_tools.params['Foregrounds']['Dust'][f'Dust_{key}']:
            components += [c.Dust(nu0=self.preset_tools.params['Foregrounds']['Dust']['nu0_d'], temp=20, beta_d=None)]
            components_name += ['Dust']

        if self.preset_tools.params['Foregrounds']['Synchrotron'][f'Synchrotron_{key}']:
            components += [c.Synchrotron(nu0=self.preset_tools.params['Foregrounds']['Synchrotron']['nu0_s'])]
            components_name += ['Synchrotron']

        if self.preset_tools.params['Foregrounds']['CO'][f'CO_{key}']:
            components += [c.COLine(nu=self.preset_tools.params['Foregrounds']['CO']['nu0_co'], active=False)]
            components_name += ['CO']
        
        return components, components_name
