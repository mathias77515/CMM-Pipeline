import os
import numpy as np
import healpy as hp
import pysm3
import pysm3.units as u
from pysm3 import utils

from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

class PresetFG:
    """
    
    Instance to initialize the Components Map-Making. It defines the Foregrounds.

    Self variables :    - params_cmb: dict
                        - params_foregrounds: dict
                        - seed: int
                        - skyconfig: dict
                        - components_model: list (Ncomp)
                        - components_name: list (Ncomp)
                        - components: ndarray (Ncomp, Npix, Nstokes)
                        - components_convolved: ndarray (Ncomp, Npix, Nstokes)
                        - components_iter: ndarray (Ncomp, Npix, Nstokes)
                        - nu_co: int / bool
    
    """
    def __init__(self, preset_tools, preset_qubic, seed):
        """
        Initialize the class with preset tools & qubic and a seed value.

        Args:
            preset_tools: Class containing tools and simulation parameters.
            preset_qubic: Class containing qubic operator and variables.
            seed: Seed value for CMB generation and noise.
        """
        ### Import preset QUBIC & tools
        self.preset_tools = preset_tools
        self.preset_qubic = preset_qubic

        ### Define variable for Foregrounds parameters
        self.params_cmb = self.preset_tools.params['CMB']
        self.params_foregrounds = self.preset_tools.params['Foregrounds']

        ### Define seed for CMB generation and noise
        self.seed = seed

        ### Skyconfig
        self.preset_tools._print_message('    => Creating sky configuration')
        self.skyconfig_in = self._get_sky_config(key='in')
        self.skyconfig_out = self._get_sky_config(key='out')

        ### Define model for reconstruction
        self.preset_tools._print_message('    => Creating model')
        self.components_model_in, self.components_name_in = self.preset_qubic._get_components_fgb(key='in')
        self.components_model_out, self.components_name_out = self.preset_qubic._get_components_fgb(key='out')

        ### Compute true components
        self.preset_tools._print_message('    => Creating components')
        self.components_in, self.components_convolved_in, _ = self._get_components(self.skyconfig_in)
        self.components_out, self.components_convolved_out, self.components_iter = self._get_components(self.skyconfig_out)

        ### Monochromatic emission
        if self.preset_tools.params['Foregrounds']['CO']['CO_in']:
            self.nu_co = self.preset_tools.params['Foregrounds']['CO']['nu0_co']
        else:
            self.nu_co = None

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
        if self.params_cmb['cmb']:
            sky['CMB'] = self.seed
        
        if self.preset_tools.params['Foregrounds']['Dust'][f'Dust_{key}']:
            sky['Dust'] = self.preset_tools.params['Foregrounds']['Dust']['model_d']
        
        if self.preset_tools.params['Foregrounds']['Synchrotron'][f'Synchrotron_{key}']:
            sky['Synchrotron'] = self.preset_tools.params['Foregrounds']['Synchrotron']['model_s']
        
        if self.preset_tools.params['Foregrounds']['CO'][f'CO_{key}']:
            sky['coline'] = 'co2'
        
        return sky
    def give_cl_cmb(self, r=0, Alens=1.):
        """
        Generates the CMB power spectrum with optional lensing and tensor contributions.

        Parameters:
        r (float): Tensor-to-scalar ratio. Default is 0.
        Alens (float): Lensing amplitude. Default is 1.

        Returns:
        numpy.ndarray: The modified CMB power spectrum.
        """
        # Read the lensed scalar power spectrum from the FITS file
        power_spectrum = hp.read_cl(os.getcwd() + '/data/' + 'Cls_Planck2018_lensed_scalar.fits')[:,:4000]
        
        # Adjust the lensing amplitude if Alens is not the default value
        if Alens != 1.:
            power_spectrum[2] *= Alens
        
        # Add tensor contributions if r is not zero
        if r:
            power_spectrum += r * hp.read_cl(os.getcwd() + '/data/' + 'Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits')[:,:4000]
        
        return power_spectrum
    def polarized_I(self, m, nside, polarization_fraction=0):
        """
        Calculate the polarized intensity map.

        Parameters:
        m (array-like): Input map to be polarized.
        nside (int): The nside parameter of the HEALPix map.
        polarization_fraction (float, optional): Fraction of polarization. Default is 0.

        Returns:
        numpy.ndarray: Array containing the polarized intensity map with cosine and sine components.
        """
        
        # Read and downgrade the polarization angle map to the desired nside resolution
        polangle = hp.ud_grade(hp.read_map(os.getcwd() + '/data/' + 'psimap_dust90_512.fits'), nside)
        
        # Read and downgrade the depolarization map to the desired nside resolution
        depolmap = hp.ud_grade(hp.read_map(os.getcwd() + '/data/' + 'gmap_dust90_512.fits'), nside)
        
        # Calculate the cosine of twice the polarization angle
        cospolangle = np.cos(2.0 * polangle)
        
        # Calculate the sine of twice the polarization angle
        sinpolangle = np.sin(2.0 * polangle)
        
        # Calculate the polarized intensity map by scaling the input map with the depolarization map and polarization fraction
        P_map = polarization_fraction * depolmap * hp.ud_grade(m, nside)
        
        # Return the polarized intensity map with cosine and sine components
        return P_map * np.array([cospolangle, sinpolangle])
    def _get_components(self, skyconfig):

        """
        Read configuration dictionary which contains every component and the model.

        Example: d = {'cmb': 42, 'Dust': 'd0', 'Synchrotron': 's0'}

        The CMB is randomly generated from a specific seed. Astrophysical foregrounds come from PySM 3.

        Parameters:
        skyconfig (dict): Dictionary containing the configuration for each component.

        Returns:
        tuple: A tuple containing:
            - components (ndarray): Array of shape (len(skyconfig), 12 * nside^2, 3) with the generated components.
            - components_convolved (ndarray): Array of shape (len(skyconfig), 12 * nside^2, 3) with the convolved components.
            - components_iter (ndarray): Array of shape (len(skyconfig), 12 * nside^2, 3) with the components for iteration.
        """
        
        ### Initialization
        components = np.zeros((len(skyconfig), 12*self.preset_tools.params['SKY']['nside']**2, 3))
        components_convolved = np.zeros((len(skyconfig), 12*self.preset_tools.params['SKY']['nside']**2, 3))
        
        ### Compute convolution operator if needed
        if self.preset_qubic.params_qubic['convolution_in'] or self.preset_qubic.params_qubic['convolution_out']:
            C = HealpixConvolutionGaussianOperator(fwhm=self.preset_qubic.joint_in.qubic.allfwhm[-1], lmax=3*self.preset_tools.params['SKY']['nside'])
        else:
            C = HealpixConvolutionGaussianOperator(fwhm=0)
            
        ### Compute CMB power spectrum according Planck data
        mycls = self.give_cl_cmb(r=self.params_cmb['r'], 
                            Alens=self.params_cmb['Alens'])
        
        ### Build components list
        for icomp, comp_name in enumerate(skyconfig.keys()):
            # CMB case
            if comp_name == 'CMB':
                np.random.seed(skyconfig[comp_name])
                cmb = hp.synfast(mycls, self.preset_tools.params['SKY']['nside'], verbose=False, new=True).T
                components[icomp] = cmb.copy()
                components_convolved[icomp] = C(cmb).copy()

            # Dust case
            elif comp_name == 'Dust':
                sky_dust=pysm3.Sky(nside=self.preset_tools.params['SKY']['nside'], 
                              preset_strings=[self.preset_tools.params['Foregrounds']['Dust']['model_d']], 
                              output_unit="uK_CMB")

                sky_dust.components[0].mbb_temperature = 20*sky_dust.components[0].mbb_temperature.unit
                map_Dust = np.array(sky_dust.get_emission(self.preset_tools.params['Foregrounds']['Dust']['nu0_d'] * u.GHz, None).T * \
                                  utils.bandpass_unit_conversion(self.preset_tools.params['Foregrounds']['Dust']['nu0_d']*u.GHz, None, u.uK_CMB)) * self.preset_tools.params['Foregrounds']['Dust']['amplification_d'] / 2
                components[icomp] = map_Dust.copy()
                components_convolved[icomp] = C(map_Dust).copy()

            # Synchrotron case   
            elif comp_name == 'Synchrotron':
                sky_sync = pysm3.Sky(nside=self.preset_tools.params['SKY']['nside'], 
                                preset_strings=[self.preset_tools.params['Foregrounds']['Synchrotron']['model_s']], 
                                output_unit="uK_CMB")
                
                map_sync = np.array(sky_sync.get_emission(self.preset_tools.params['Foregrounds']['Synchrotron']['nu0_s'] * u.GHz, None).T * \
                                utils.bandpass_unit_conversion(self.preset_tools.params['Foregrounds']['Synchrotron']['nu0_s'] * u.GHz, None, u.uK_CMB)) * self.preset_tools.params['Foregrounds']['Synchrotron']['amplification_s']
                components[icomp] = map_sync.copy() 
                components_convolved[icomp] = C(map_sync).copy()

            # CO emission case
            elif comp_name == 'coline':
                map_co = hp.ud_grade(hp.read_map('data/CO_line.fits') * 10, self.preset_tools.params['SKY']['nside'])
                map_co_polarised = self.polarized_I(map_co, self.preset_tools.params['SKY']['nside'], polarization_fraction=self.preset_tools.params['Foregrounds']['CO']['polarization_fraction'])
                sky_co = np.zeros((12*self.preset_tools.params['SKY']['nside']**2, 3))
                sky_co[:, 0] = map_co.copy()
                sky_co[:, 1:] = map_co_polarised.T.copy()
                components[icomp] = sky_co.copy()
                components_convolved[icomp] = C(sky_co).copy()

            else:
                raise TypeError('Choose right foreground model (d0, s0, ...)')
        
        # if self.preset_tools.params['Foregrounds']['Dust']['nside_beta_out'] != 0:
        #     components = components.T.copy()
        components_iter = components.copy() 

        return components, components_convolved, components_iter
    
    
