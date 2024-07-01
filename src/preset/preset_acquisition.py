import numpy as np

from acquisition.Qacquisition import *
from simtools.noise_timeline import *

from pyoperators import DiagonalOperator

class PresetAcquisition:
    """
    
    Instance to initialize the Components Map-Making. It defines the acquisition of data.
    
    
    """
    def __init__(self, seed_noise, preset_tools, preset_external, preset_qubic, preset_sky, preset_fg, preset_mixing_matrix, preset_gain):
        """
        Initialize the class with the given presets and seed noise.

        Args:
            seed_noise: Seed for noise generation.
            preset_tools: Object containing tools and parameters.
            preset_qubic: preset_qubic: Object containing qubic operator.
            preset_sky: Object containing sky-related operations.
            preset_fg: Object containing foreground-related operations.
            preset_mixing_matrix: Object initializing mixing-matrix.
            preset_gain: Object initializing detector gain.
        """
        ### Import preset Gain, Mixing Matrix, Foregrounds, Sky, QUBIC & tools
        self.preset_tools = preset_tools
        self.preset_external = preset_external
        self.preset_qubic = preset_qubic
        self.preset_sky = preset_sky
        self.preset_fg = preset_fg
        self.preset_mixingmatrix = preset_mixing_matrix
        self.preset_gain = preset_gain

        ### Set noise seed
        self.seed_noise = seed_noise
        
        ### Define tolerance of the rms variations
        self.rms_tolerance = self.preset_tools.params['PCG']['tol_rms']
        self.ites_rms_tolerance = self.preset_tools.params['PCG']['ites_to_converge']
        self.rms_plot = np.zeros((1, 2))

        ### Inverse noise-covariance matrix
        self.preset_tools._print_message('    => Building inverse noise covariance matrix')
        self.invN = self.preset_qubic.joint_out.get_invntt_operator(mask=self.preset_sky.mask)

        ### Preconditioning
        self.preset_tools._print_message('    => Creating preconditioner')
        self._get_preconditioner()
        print(self.M.shapein)

        ### Get convolution
        self.preset_tools._print_message('    => Getting convolution')
        self._get_convolution()

        ### Get observed data
        self.preset_tools._print_message('    => Getting observational data')
        self._get_tod()

        ### Compute initial guess for PCG
        self.preset_tools._print_message('    => Initializing starting point')
        self._get_x0()

    def _get_approx_hth(self):

        # Approximation of H.T H
        approx_hth = np.empty((self.preset_qubic.params_qubic['nsub_in'],) + self.preset_qubic.joint_out.qubic.H[0].shapein) # has shape (self.preset_qubic.params_qubic['nsub_in'], npixel, 3)
        vector = np.ones(self.preset_qubic.joint_out.qubic.H[0].shapein)
        for index in range(self.preset_qubic.params_qubic['nsub_in']):
            approx_hth[index] = self.preset_qubic.joint_out.qubic.H[index].T * self.preset_qubic.joint_out.qubic.invn220 * self.preset_qubic.joint_out.qubic.H[index](vector)

        return approx_hth

    def _get_preconditioner(self):

        approx_hth = self._get_approx_hth()
        
        preconditioner = np.ones((len(self.preset_fg.components_model_out), approx_hth.shape[1], approx_hth.shape[2]))

        # np.array(A_list) has shape (self.preset_qubic.params_qubic['nsub_in'], n_comp, npixel, 3)
        # We sum over the frequencies, take the inverse, and only keep the information on the patch.
        
        for icomp in range(len(self.preset_fg.components_model_out)):
            self.preset_tools._print_message(f'Optimized preconditioner moved to component {icomp}')
            Asub = self.preset_mixingmatrix.Amm_in[:self.preset_qubic.params_qubic['nsub_in'], icomp].copy()
            for stk in range(3):
                preconditioner[icomp, self.preset_sky.seenpix, stk] = (approx_hth[:, :, stk].T @ Asub**2)[self.preset_sky.seenpix]

        if self.preset_qubic.params_qubic['preconditionner']:
            if self.preset_tools.params['PLANCK']['fixI']:
                self.M = DiagonalOperator(preconditioner[:, :, 1:])
            elif self.preset_tools.params['PLANCK']['fix_pixels_outside_patch']:
                self.M = DiagonalOperator(preconditioner[:, self.preset_sky.seenpix, :])
            else:
                self.M = DiagonalOperator(preconditioner)
        else:
            self.M = None 

    def _get_convolution(self):
        """
        Method to define all angular resolutions of the instrument at each frequency.
        
        This method sets the Full Width at Half Maximum (FWHM) for Time-Ordered Data (TOD) and map-making processes.
        `self.fwhm_tod` represents the real angular resolution, and `self.fwhm_mapmaking` represents the beams used for reconstruction.
        
        The method checks the `convolution_in` and `convolution_out` parameters to determine the appropriate FWHM values.
        It also calculates the reconstructed FWHM based on these parameters.
        
        Attributes:
            self.fwhm_tod (array): FWHM for TOD making.
            self.fwhm_mapmaking (array): FWHM for map-making.
            self.fwhm_reconstructed (float): Reconstructed FWHM.
        """
        
        self.fwhm_tod = self.preset_qubic.joint_in.qubic.allfwhm * 0
        self.fwhm_mapmaking = self.preset_qubic.joint_in.qubic.allfwhm * 0
        
        if self.preset_qubic.params_qubic['convolution_in']:
            self.fwhm_tod = self.preset_qubic.joint_in.qubic.allfwhm
        
        if self.preset_qubic.params_qubic['convolution_out']:
            self.fwhm_mapmaking = np.sqrt(self.preset_qubic.joint_in.qubic.allfwhm**2 - np.min(self.preset_qubic.joint_in.qubic.allfwhm)**2)
        
        if self.preset_qubic.params_qubic['convolution_in'] and self.preset_qubic.params_qubic['convolution_out']:
            self.fwhm_reconstructed = np.min(self.preset_qubic.joint_in.qubic.allfwhm)
        elif self.preset_qubic.params_qubic['convolution_in'] and not self.preset_qubic.params_qubic['convolution_out']:
            self.fwhm_reconstructed = np.mean(self.preset_qubic.joint_in.qubic.allfwhm)
        elif not self.preset_qubic.params_qubic['convolution_in'] and not self.preset_qubic.params_qubic['convolution_out']:
            self.fwhm_reconstructed = 0
        
        self.preset_tools._print_message(f'FWHM for TOD making : {self.fwhm_tod}')
        self.preset_tools._print_message(f'FWHM for reconstruction : {self.fwhm_mapmaking}')
        self.preset_tools._print_message(f'Reconstructed FWHM : {self.fwhm_reconstructed}')

    def _get_noise(self):
        """
        Method to define QUBIC noise, can generate Dual band or Wide Band noise by following:

        - Dual Band: n = [Ndet + Npho_150, Ndet + Npho_220]
        - Wide Band: n = [Ndet + Npho_150 + Npho_220]

        Depending on the instrument type specified in the preset_qubic parameters, this method will
        instantiate either QubicWideBandNoise or QubicDualBandNoise and return the total noise.

        Returns:
            numpy.ndarray: The total noise array, flattened.
        """

        if self.preset_qubic.params_qubic['instrument'] == 'UWB':
            noise = QubicWideBandNoise(self.preset_qubic.dict, 
                                    self.preset_qubic.params_qubic['npointings'], 
                                    detector_nep=self.preset_qubic.params_qubic['NOISE']['detector_nep'],
                                    duration=np.mean([self.preset_qubic.params_qubic['NOISE']['duration_150'], self.preset_qubic.params_qubic['NOISE']['duration_220']]))
        else:
            noise = QubicDualBandNoise(self.preset_qubic.dict, 
                                    self.preset_qubic.params_qubic['npointings'], 
                                    detector_nep=self.preset_qubic.params_qubic['NOISE']['detector_nep'],
                                    duration=[self.preset_qubic.params_qubic['NOISE']['duration_150'], self.preset_qubic.params_qubic['NOISE']['duration_220']])

        return noise.total_noise(self.preset_qubic.params_qubic['NOISE']['ndet'], 
                                self.preset_qubic.params_qubic['NOISE']['npho150'], 
                                self.preset_qubic.params_qubic['NOISE']['npho220'],
                                seed_noise=self.seed_noise).ravel()
    
    def _get_tod(self):
        """
        Generate fake observational data from QUBIC and external experiments.

        This method simulates observational data, including astrophysical foregrounds contamination using `self.beta` and systematics using `self.g`.
        The data generation follows the formula: d = H . A . c + n.

        Note:
            - The data uses MPI communication to utilize multiple cores.
            - Full data are stored in `self.TOD_qubic_BAND_ALL`, where `self.TOD_qubic` is a subset of all the data.
            - Multiprocessing is achieved by dividing the number of detectors per process.

        Attributes:
            self.H (Operator): The operator combining various systematics and foregrounds.
            self.TOD_qubic (ndarray): The simulated observational data for QUBIC.
            self.nsampling_x_ndetectors (int): The number of samples in `self.TOD_qubic`.
            self.TOD_external (ndarray): The simulated observational data for external experiments.
            self.TOD_obs (ndarray): The combined observational data from QUBIC and external experiments.
        """
        ### Build joint acquisition operator
        self.H = self.preset_qubic.joint_in.get_operator(beta=self.preset_mixingmatrix.beta_in, Amm=self.preset_mixingmatrix.Amm_in, gain=self.preset_gain.gain_in, fwhm=self.fwhm_tod)
        
        ### Create seed
        if self.preset_tools.rank == 0:
            np.random.seed(None)
            seed_pl = np.random.randint(10000000)
        else:
            seed_pl = None
        seed_pl = self.preset_tools.comm.bcast(seed_pl, root=0)
        
        ### Build noise variables
        noise_external = self.preset_qubic.joint_in.external.get_noise(seed=seed_pl) * self.preset_tools.params['PLANCK']['level_noise_planck']
        noise_qubic = self._get_noise()

        ### Create QUBIC TOD
        self.TOD_qubic = (self.H.operands[0])(self.preset_fg.components_in[:, :, :]) + noise_qubic
        self.nsampling_x_ndetectors = self.TOD_qubic.shape[0]

        ### Create external TOD
        self.TOD_external = (self.H.operands[1])(self.preset_fg.components_in[:, :, :]) + noise_external
        
        ### Reconvolve Planck data toward QUBIC angular resolution
        if self.preset_qubic.params_qubic['convolution_in'] or self.preset_qubic.params_qubic['convolution_out']:
            _r = ReshapeOperator(self.TOD_external.shape, (len(self.preset_external.external_nus), 12*self.preset_sky.params_sky['nside']**2, 3))
            maps_external = _r(self.TOD_external)
            C = HealpixConvolutionGaussianOperator(fwhm=self.preset_qubic.joint_in.qubic.allfwhm[-1], lmax=3*self.preset_sky.params_sky['nside'])
            for i in range(maps_external.shape[0]):
                maps_external[i] = C(maps_external[i])
            
            self.TOD_external = _r.T(maps_external)

        self.TOD_obs = np.r_[self.TOD_qubic, self.TOD_external]
        
    def _spectral_index_modifiedblackbody(self, nside):
        """
        Method to define input spectral indices if the d1 model is used for thermal Dust description.

        Parameters:
        nside (int): The nside parameter defines the resolution of the HEALPix map.

        Returns:
        numpy.ndarray: An array containing the spectral indices for the thermal Dust model.
        """
        sky = pysm3.Sky(nside=nside, preset_strings=['d1'])
        
        return np.array(sky.components[0].mbb_index)
    
    def _spectral_index_powerlaw(self, nside):
        """
        Define input spectral indices if the s1 model is used for Synchrotron description.

        Parameters:
        nside (int): The nside parameter for the Sky object.

        Returns:
        np.array: Array of spectral indices for the Synchrotron component.
        """
        sky = pysm3.Sky(nside=nside, preset_strings=['s1'])
        return np.array(sky.components[0].pl_index)

    def _get_x0(self):
        """
        Define starting point of the convergence.

        The argument 'set_comp_to_0' multiplies the pixel values by a given factor. You can decide 
        to convolve the map by a beam with an FWHM in radians.

        This method initializes the beta_iter and Amm_iter attributes based on the foreground model parameters.
        It also applies convolution and noise to the components based on the preset parameters.

        Raises:
        TypeError: If an unrecognized component name is encountered.
        """
        ### Create seed
        if self.preset_tools.rank == 0:
            seed = np.random.randint(100000000)
        else:
            seed = None
        seed = self.preset_tools.comm.bcast(seed, root=0)
        np.random.seed(seed)

        ### Build Mixing Matrix for d0 & d6 models
        self.beta_iter = None
        if self.preset_fg.params_foregrounds['Dust']['model_d'] in ['d0', 'd6']:
            self.beta_iter = np.array([])
            self.Amm_iter = self.preset_mixingmatrix._get_Amm(
                self.preset_fg.components_model_in, 
                self.preset_fg.components_name_in, 
                self.preset_mixingmatrix.nus_eff_in, 
                beta_d=self.preset_fg.params_foregrounds['Dust']['beta_d_init'][0], 
                beta_s=self.preset_fg.params_foregrounds['Synchrotron']['beta_s_init'][0],
                init=True
            )
            if self.preset_fg.params_foregrounds['Dust']['Dust_out']:
                self.beta_iter = np.append(
                    self.beta_iter, 
                    np.random.normal(
                        self.preset_fg.params_foregrounds['Dust']['beta_d_init'][0], 
                        self.preset_fg.params_foregrounds['Dust']['beta_d_init'][1], 
                        1
                    )
                )
            if self.preset_fg.params_foregrounds['Synchrotron']['Synchrotron_out']:
                self.beta_iter = np.append(
                    self.beta_iter, 
                    np.random.normal(
                        self.preset_fg.params_foregrounds['Synchrotron']['beta_s_init'][0], 
                        self.preset_fg.params_foregrounds['Synchrotron']['beta_s_init'][1], 
                        1
                    )
                )
        
        else:
            self.Amm_iter = None
            self.beta_iter = np.zeros(
                (12 * self.preset_fg.params_foregrounds['Dust']['nside_beta_out']**2, 
                len(self.preset_fg.components_model_out) - 1)
            )
            for iname, name in enumerate(self.preset_fg.components_name_out):
                if name == 'CMB':
                    pass
                elif name == 'Dust':
                    self.beta_iter[:, iname - 1] = self._spectral_index_modifiedblackbody(
                        self.preset_fg.params_foregrounds['Dust']['nside_beta_out']
                    )
                elif name == 'Synchrotron':
                    self.beta_iter[:, iname - 1] = self._spectral_index_powerlaw(
                        self.preset_fg.params_foregrounds['Dust']['nside_beta_out']
                    )

        if self.preset_fg.params_foregrounds['Dust']['nside_beta_out'] == 0:
            # Build beta map for spatially varying spectral index
            self.allbeta = np.array([self.beta_iter])
            C1 = HealpixConvolutionGaussianOperator(
                fwhm=self.fwhm_reconstructed, 
                lmax=3 * self.preset_tools.params['SKY']['nside']
            )
            C2 = HealpixConvolutionGaussianOperator(
                fwhm=self.preset_tools.params['INITIAL']['fwhm0'], 
                lmax=3 * self.preset_tools.params['SKY']['nside']
            )
            # Constant spectral index -> maps have shape (Ncomp, Npix, Nstk)
            for i in range(len(self.preset_fg.components_model_out)):
                if self.preset_fg.components_name_out[i] == 'CMB':
                    self.preset_fg.components_iter[i] = C2(C1(self.preset_fg.components_iter[i]))
                    self.preset_fg.components_iter[i, self.preset_sky.seenpix_qubic, 1:] *= self.preset_tools.params['INITIAL']['qubic_patch_cmb']
                    self.preset_fg.components_iter[i, self.preset_sky.seenpix_qubic, 1:] += np.random.normal(
                        0, 
                        self.preset_tools.params['INITIAL']['sig_map_noise'], 
                        self.preset_fg.components_iter[i, self.preset_sky.seenpix_qubic, 1:].shape
                    )
                elif self.preset_fg.components_name_out[i] == 'Dust':
                    self.preset_fg.components_iter[i] = C2(C1(self.preset_fg.components_iter[i]))
                    self.preset_fg.components_iter[i, self.preset_sky.seenpix_qubic, 1:] *= self.preset_tools.params['INITIAL']['qubic_patch_dust']
                    self.preset_fg.components_iter[i, self.preset_sky.seenpix_qubic, 1:] += np.random.normal(
                        0, 
                        self.preset_tools.params['INITIAL']['sig_map_noise'], 
                        self.preset_fg.components_iter[i, self.preset_sky.seenpix_qubic, 1:].shape
                    )
                elif self.preset_fg.components_name_out[i] == 'Synchrotron':
                    self.preset_fg.components_iter[i] = C2(C1(self.preset_fg.components_iter[i]))
                    self.preset_fg.components_iter[i, self.preset_sky.seenpix_qubic, 1:] *= self.preset_tools.params['INITIAL']['qubic_patch_sync']
                    self.preset_fg.components_iter[i, self.preset_sky.seenpix_qubic, 1:] += np.random.normal(
                        0, 
                        self.preset_tools.params['INITIAL']['sig_map_noise'], 
                        self.preset_fg.components_iter[i, self.preset_sky.seenpix_qubic, 1:].shape
                    )
                elif self.preset_fg.components_name_out[i] == 'CO':
                    self.preset_fg.components_iter[i] = C2(C1(self.preset_fg.components_iter[i]))
                    self.preset_fg.components_iter[i, self.preset_sky.seenpix_qubic, 1:] *= self.preset_tools.params['INITIAL']['qubic_patch_co']
                    self.preset_fg.components_iter[i, self.preset_sky.seenpix_qubic, 1:] += np.random.normal(
                        0, 
                        self.preset_tools.params['INITIAL']['sig_map_noise'], 
                        self.preset_fg.components_iter[i, self.preset_sky.seenpix_qubic, 1:].shape
                    )
                else:
                    raise TypeError(f'{self.preset_fg.components_name_out[i]} not recognized')
        else:
            self.allbeta = np.array([self.beta_iter])
            C1 = HealpixConvolutionGaussianOperator(
                fwhm=self.fwhm_reconstructed, 
                lmax=3 * self.preset_tools.params['SKY']['nside']
            )
            C2 = HealpixConvolutionGaussianOperator(
                fwhm=self.preset_tools.params['INITIAL']['fwhm0'], 
                lmax=3 * self.preset_tools.params['SKY']['nside']
            )
            # Varying spectral indices -> maps have shape (Nstk, Npix, Ncomp)
            for i in range(len(self.preset_fg.components_model_out)):
                if self.preset_fg.components_name_out[i] == 'CMB':
                    self.preset_fg.components_iter[:, :, i] = C2(C1(self.preset_fg.components_iter[:, :, i].T)).T
                    self.preset_fg.components_iter[1:, self.seenpix, i] *= self.preset_tools.params['INITIAL']['qubic_patch_cmb']
                elif self.preset_fg.components_name_out[i] == 'Dust':
                    self.preset_fg.components_iter[:, :, i] = C2(C1(self.preset_fg.components_iter[:, :, i])).T + np.random.normal(
                        0, 
                        self.preset_tools.params['INITIAL']['sig_map_noise'], 
                        self.preset_fg.components_iter[:, :, i].T.shape
                    ).T
                    self.preset_fg.components_iter[1:, self.seenpix, i] *= self.preset_tools.params['INITIAL']['qubic_patch_Dust']
                elif self.preset_fg.components_name_out[i] == 'Synchrotron':
                    self.preset_fg.components_iter[:, :, i] = C2(C1(self.preset_fg.components_iter[:, :, i])).T + np.random.normal(
                        0, 
                        self.preset_tools.params['INITIAL']['sig_map_noise'], 
                        self.preset_fg.components_iter[:, :, i].T.shape
                    ).T
                    self.preset_fg.components_iter[1:, self.seenpix, i] *= self.preset_tools.params['INITIAL']['qubic_patch_sync']
                elif self.preset_fg.components_name_out[i] == 'CO':
                    self.preset_fg.components_iter[:, :, i] = C2(C1(self.preset_fg.components_iter[:, :, i])).T + np.random.normal(
                        0, 
                        self.preset_tools.params['INITIAL']['sig_map_noise'], 
                        self.preset_fg.components_iter[:, :, i].T.shape
                    ).T
                    self.preset_fg.components_iter[1:, self.seenpix, i] *= self.preset_tools.params['INITIAL']['qubic_patch_co']
                else:
                    raise TypeError(f'{self.preset_fg.components_name_out[i]} not recognized')
