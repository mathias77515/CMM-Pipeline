import fgb.component_model as c

from preset.preset_main import *

class PresetSky:
    """
    
    Instance to initialize the Components Map-Making. It reads the `params.yml` file to define the Sky.
    
    Arguments : 
    ===========
        - comm    : MPI common communicator (define by MPI.COMM_WORLD).
        - seed    : Int number for CMB realizations.
        - it      : Int number for noise realizations.
        - verbose : bool, Display message or not.
    
    """
    def __init__(self, comm, seed, verbose=True):
        """
        Initialize the class with MPI communication and optional verbosity.

        Args:
            comm: MPI communicator object.
            verbose (bool): If True, print detailed initialization messages. Default is True.
        """

        self.preset = PresetMain(comm, verbose=verbose)

        ### Define seed for CMB generation and noise
        self.preset.params['CMB']['seed'] = seed

        ### Skyconfig
        self.skyconfig_in = self._get_sky_config(key='in')
        self.skyconfig_out = self._get_sky_config(key='out')

        ### Define model for reconstruction
        if verbose:
            self.preset._print_message('    => Creating model')
            
        self.comps_in, self.comps_name_in = self._get_components_fgb(key='in', method=self.preset.params['Foregrounds']['Dust']['type'])
        self.comps_out, self.comps_name_out = self._get_components_fgb(key='out', method=self.preset.params['Foregrounds']['Dust']['type'])
        

    def _get_sky_config(self, key):
        """
        Method to define the sky model used by PySM to generate a fake sky.

        Args:
            key (str): The key used to access specific parameters in the preset configuration.

        Returns:
            dict: A dictionary containing the sky model configuration with keys such as 'cmb', 'Dust', 'Synchrotron', and 'coline'.

        Example:

            sky = {'cmb': 42, 'Dust': 'd0'}
        """
        
        sky = {}
        if self.preset.params['CMB']['cmb']:
            sky['cmb'] = self.preset.params['CMB']['seed']
        
        if self.preset.params['Foregrounds']['Dust'][f'Dust_{key}']:
            sky['Dust'] = self.preset.params['Foregrounds']['Dust']['model_d']
        
        if self.preset.params['Foregrounds']['Synchrotron'][f'Synchrotron_{key}']:
            sky['Synchrotron'] = self.preset.params['Foregrounds']['Synchrotron']['model_s']
        
        if self.preset.params['Foregrounds']['CO'][f'CO_{key}']:
            sky['coline'] = 'co2'
        
        return sky
    
    def _get_components_fgb(self, key, method='blind'):
        """
        Method to define sky model taken from FGBuster code. Note that we add `COLine` instance to define monochromatic description.

        Parameters:
        key (str): The key to identify specific components in the preset parameters.
        method (str): The method to use for defining components. Default is 'blind'.

        Returns:
        tuple: A tuple containing two lists:
            - comps (list): List of component instances.
            - comps_name (list): List of component names corresponding to the instances.
        """

        comps = []
        comps_name = []

        if method == 'blind':
            beta_d = None
        else:
            beta_d = None

        if self.preset.params['CMB']['cmb']:
            comps += [c.CMB()]
            comps_name += ['CMB']
            
        if self.preset.params['Foregrounds']['Dust'][f'Dust_{key}']:
            comps += [c.Dust(nu0=self.preset.params['Foregrounds']['Dust']['nu0_d'], temp=20, beta_d=beta_d)]
            comps_name += ['Dust']

        if self.preset.params['Foregrounds']['Synchrotron'][f'Synchrotron_{key}']:
            comps += [c.Synchrotron(nu0=self.preset.params['Foregrounds']['Synchrotron']['nu0_s'])]
            comps_name += ['Synchrotron']

        if self.preset.params['Foregrounds']['CO'][f'CO_{key}']:
            comps += [c.COLine(nu=self.preset.params['Foregrounds']['CO']['nu0_co'], active=False)]
            comps_name += ['CO']
        
        return comps, comps_name
