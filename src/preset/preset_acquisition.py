import numpy as np

from acquisition.Qacquisition import *
from simtools.noise_timeline import *
import matplotlib.pyplot as plt
import healpy as hp

from pyoperators import DiagonalOperator

class PresetAcquisition:
    """
    
    Instance to initialize the Components Map-Making. It defines the acquisition of data.
    
    Self variables :    - seed_noise: int
                        - params_foregrounds: dict
                        - seed_noise: int
                        - rms_tolerance: float
                        - ites_rms_tolerance: int
                        - invN: BlockDiagonalOperator (Ndet*Nsamples + Npix*Nplanck*Nstokes) ---> (Ndet*Nsamples + Npix*Nplanck*Nstokes)
                            -> .operands[0] = QUBIC / .operands[1] = Planck
                            -> .operands[0].operands[0] = ReshapeOperator (Ndet, Nsamples) ---> (Ndet*Nsamples) / .operands[0].operands[1] = ReshapeOperator (Ndet*Nsamples) ---> (Ndet, Nsamples)
                            -> .operands[0].operands[0/1].operands[0] = 150 GHz focal plane / .operands[0].operands[0/1].operands[1] = 220 GHz focal plane
                        - M: DiagonalOperator (Ncomp, Npix, Nstokes) ---> (Ncomp, Npix, Nstokes)
                        - fwhm_reconstructed: float
                        - fwhm_mapmaking: ndarray (Nsub)
                        - H: BlockColumnOperator (Ncomp, Npix, Nstokes) ---> (Ndet*Nsamples + Npix*Nplanck*Nstokes)
                            -> .operands[0] = QUBIC / .operands[1] = Planck
                        - TOD_qubic: ndarray (Ndet*Nsamples)
                        - TOD_external: ndarray (Npix*Nplanck*Nstokes)
                        - TOD_obs: ndarray (Ndet*Nsamples + Npix*Nplanck*Nstokes)
                        - beta_iter: ndarray / if d1 (iter, 12*nside_beta**2, Ncomp-1) / if not (Ncomp-1)
                        - allbeta: ndarray / if d1 (iter, 12*nside_beta**2, Ncomp-1) / if not (iter, Ncomp-1)
                        - Amm_iter: ndarray (Nsub + Nplanck*Nintegr, Ncomp-1)
    
    """
    def __init__(self, seed_noise, preset_tools, preset_external, preset_qubic, preset_sky, preset_fg, preset_mixing_matrix, preset_gain):
        """
        Initialize the class with the given presets and seed noise.

        Args:
            seed_noise: Seed for noise generation.
            preset_tools: Class containing tools and simulation parameters.
            preset_qubic: Class containing qubic operator and variables.
            preset_sky: Class containing sky varaibles.
            preset_fg: Class containing foreground variables.
            preset_mixing_matrix: Class containing mixing-matrix variables.
            preset_gain: Class containing detector gain variables.
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
        #stop
        ### Preconditioning
        #if self.preset_qubic.params_qubic['preconditionner']:
            #self.preset_tools._print_message('    => Creating preconditioner')
            #self._get_preconditioner()
        #else:
        #    self.M = None 

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
        """
        Calculates the approximation of H.T H.

        Returns:
            numpy.ndarray: The approximation of H.T H with shape (nsub_in, npixel, 3).
        """

        # Approximation of H.T H
        approx_hth = np.empty((self.preset_qubic.params_qubic['nsub_in'],) + self.preset_qubic.joint_out.qubic.H[0].shapein)
        vector = np.ones(self.preset_qubic.joint_out.qubic.H[0].shapein)
        for index in range(self.preset_qubic.params_qubic['nsub_in']):
            approx_hth[index] = self.preset_qubic.joint_out.qubic.H[index].T * self.preset_qubic.joint_out.qubic.invn220 * self.preset_qubic.joint_out.qubic.H[index](vector)
        
        invN_ext = self.preset_qubic.joint_out.external.get_invntt_operator(mask=self.preset_sky.mask)
        
        _r = ReshapeOperator((len(self.preset_qubic.joint_out.external.nus), approx_hth.shape[1], approx_hth.shape[2]), invN_ext.shapein)
        approx_hth_ext = invN_ext(np.ones(invN_ext.shapein))
        
        return approx_hth, _r.T(approx_hth_ext)
    def _get_relative_weight(self, A_qubic, A_ext):
        
        approx_hth, approx_hth_ext = self._get_approx_hth()
        
        AtHtnHA_qubic = approx_hth[:, :, 0].T @ A_qubic[..., 0]**2
        AtHtnHA_qubic /= AtHtnHA_qubic.max()
        AtHtnHA_ext = approx_hth_ext[:, :, 0].T @ A_ext[..., 0]**2 / AtHtnHA_qubic.max()
        
        #plt.figure()
        
        #hp.mollview(AtHtnHA_qubic, cmap='jet', sub=(1, 3, 1))
        #hp.mollview(AtHtnHA_ext, cmap='jet', sub=(1, 3, 2))
        #hp.mollview(AtHtnHA_qubic + AtHtnHA_ext, cmap='jet', sub=(1, 3, 3))
        
        #plt.show()
        #print(AtHtnHA_qubic)
        #print(AtHtnHA_ext)
        #stop
        
        
        
        
        #preconditioner[icomp, :, istk] = (approx_hth_ext[:, :, 0].T @ A_ext[..., icomp]**2)
        
    def _get_preconditioner(self, A_qubic, A_ext):
        """
        Calculates and returns the preconditioner matrix for the optimization process.

        Returns:
            np.ndarray: The preconditioner matrix.
        """

        # Calculate the approximate H^T * H matrix
        #approx_hth = self._get_approx_hth()
        approx_hth, approx_hth_ext = self._get_approx_hth()
        
        # Create a preconditioner matrix with dimensions (number of components, number of pixels, 3)
        preconditioner = np.ones((len(self.preset_fg.components_model_out), approx_hth.shape[1], approx_hth.shape[2]))

        if self.preset_external.params_external['weight_planck'] > 0 :
            for icomp in range(len(self.preset_fg.components_model_out)):
                self.preset_tools._print_message(f'Optimized preconditioner moved to component {icomp}')
            
                for istk in range(3):
                    
                    preconditioner[icomp, :, istk] = 1/(approx_hth_ext[:, :, 0].T @ A_ext[..., icomp]**2)
        #plt.figure()
        _min, _max = np.min(preconditioner[0, :, 0]), np.max(preconditioner[0, :, 0])
        #hp.mollview(preconditioner[0, :, 0], sub=(1, 3, 1),min=_min, max=_max)
        
        # We sum over the frequencies, take the inverse, and only keep the information on the patch.
        for icomp in range(len(self.preset_fg.components_model_out)):
            self.preset_tools._print_message(f'Optimized preconditioner moved to component {icomp}')
            
            for istk in range(3):
                precond_qubic = 1/(approx_hth[:, :, 0].T @ A_qubic[..., icomp]**2)

                preconditioner[icomp, self.preset_sky.seenpix, istk] += precond_qubic[self.preset_sky.seenpix]
                #print(precond_qubic == np.inf)
                precond_qubic[precond_qubic == np.inf] = 0
        #print(precond_qubic.shape, precond_qubic)
        
        #hp.mollview(precond_qubic, sub=(1, 3, 2),min=_min, max=_max)
        #hp.mollview(preconditioner[0, :, 0], sub=(1, 3, 3), min=_min, max=_max)
        #plt.show()
        #stop
        if self.preset_tools.params['PCG']['fixI']:
            M = DiagonalOperator(preconditioner[:, :, 1:])
        elif self.preset_tools.params['PCG']['fix_pixels_outside_patch']:
            M = DiagonalOperator(preconditioner[:, self.preset_sky.seenpix, :])
        else:
            M = DiagonalOperator(preconditioner)

        return M  
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
        
        # Initialize FWHM arrays to 0
        self.fwhm_tod = self.preset_qubic.joint_in.qubic.allfwhm * 0
        self.fwhm_mapmaking = self.preset_qubic.joint_in.qubic.allfwhm * 0
        
        # Check if convolution_in is True
        if self.preset_qubic.params_qubic['convolution_in']:
            self.fwhm_tod = self.preset_qubic.joint_in.qubic.allfwhm
        
        # Check if convolution_out is True
        if self.preset_qubic.params_qubic['convolution_out']:
            self.fwhm_mapmaking = np.sqrt(self.preset_qubic.joint_in.qubic.allfwhm**2 - np.min(self.preset_qubic.joint_in.qubic.allfwhm)**2)
        
        # Calculate the reconstructed FWHM based on convolution parameters
        if self.preset_qubic.params_qubic['convolution_in'] and self.preset_qubic.params_qubic['convolution_out']:
            self.fwhm_reconstructed = np.min(self.preset_qubic.joint_in.qubic.allfwhm)
        elif self.preset_qubic.params_qubic['convolution_in'] and not self.preset_qubic.params_qubic['convolution_out']:
            self.fwhm_reconstructed = np.mean(self.preset_qubic.joint_in.qubic.allfwhm)
        elif not self.preset_qubic.params_qubic['convolution_in'] and not self.preset_qubic.params_qubic['convolution_out']:
            self.fwhm_reconstructed = 0
        
        # Print the FWHM values
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
        self.H = self.preset_qubic.joint_in.get_operator(A=self.preset_mixingmatrix.Amm_in, gain=self.preset_gain.gain_in, fwhm=self.fwhm_tod)

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
        self.TOD_qubic = (self.H.operands[0])(self.preset_fg.components_in) + noise_qubic
        self.nsampling_x_ndetectors = self.TOD_qubic.shape[0]

        ### Create external TOD
        self.TOD_external = (self.H.operands[1])(self.preset_fg.components_in[:, :, :]) + noise_external
        
        _r = ReshapeOperator(self.TOD_external.shape, (len(self.preset_external.external_nus), 12*self.preset_sky.params_sky['nside']**2, 3))
        maps_external = _r(self.TOD_external)
        
        ### Reconvolve Planck data toward QUBIC angular resolution
        if self.preset_qubic.params_qubic['convolution_in'] or self.preset_qubic.params_qubic['convolution_out']:
            
            C = HealpixConvolutionGaussianOperator(fwhm=self.preset_qubic.joint_in.qubic.allfwhm[-1], lmax=3*self.preset_sky.params_sky['nside'])
            for i in range(maps_external.shape[0]):
                maps_external[i] = C(maps_external[i])
            
        if self.preset_tools.params['PCG']['fix_pixels_outside_patch']:
            print('Removing pixels outside patch')
            maps_external[:, ~self.preset_sky.seenpix, :] = 0
        self.TOD_external = _r.T(maps_external)
        
        #self.seenpix_external = np.tile(self.preset_sky.seenpix_qubic, (maps_external.shape[0], 3, 1)).reshape(maps_external.shape)
        
        ### Planck dataset with 0 outside QUBIC patch (Planck is assumed on the full sky)
        self.TOD_external_zero_outside_patch = _r.T(maps_external)
        
        ### Observed TOD (Planck is assumed on the full sky)
        self.TOD_obs = np.r_[self.TOD_qubic, self.TOD_external]
        self.TOD_obs_zero_outside = np.r_[self.TOD_qubic, self.TOD_external_zero_outside_patch]    
        
        #plt.figure()
        #plt.plot(self.TOD_external)
        #hp.mollview(maps_external[0, :, 0])
        #plt.plot(self.TOD_external_zero_outside_patch)
        #plt.show()
        #stop
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

        self.beta_iter, self.Amm_iter = self.preset_mixingmatrix._get_beta_iter()
        
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
        istk=0
        for i in range(len(self.preset_fg.components_model_out)):
            if self.preset_fg.components_name_out[i] == 'CMB':
                self.preset_fg.components_iter[i] = C2(C1(self.preset_fg.components_iter[i]))
                for istk in range(3):
                    if istk == 0:
                        key = 'I'
                    else:
                        key = 'P'
                    self.preset_fg.components_iter[i, self.preset_sky.seenpix_qubic, istk] *= self.preset_tools.params['INITIAL'][f'qubic_patch_{key}_cmb']
                    self.preset_fg.components_iter[i, self.preset_sky.seenpix_qubic, istk] += np.random.normal(
                        0, 
                        self.preset_tools.params['INITIAL']['sig_map_noise'], 
                        self.preset_fg.components_iter[i, self.preset_sky.seenpix_qubic, istk].shape
                        )   
            elif self.preset_fg.components_name_out[i] == 'Dust':
                self.preset_fg.components_iter[i] = C2(C1(self.preset_fg.components_iter[i]))
                for istk in range(3):
                    if istk == 0:
                        key = 'I'
                    else:
                        key = 'P'
                    self.preset_fg.components_iter[i, self.preset_sky.seenpix_qubic, istk] *= self.preset_tools.params['INITIAL'][f'qubic_patch_{key}_dust']
                    self.preset_fg.components_iter[i, self.preset_sky.seenpix_qubic, istk] += np.random.normal(
                        0, 
                        self.preset_tools.params['INITIAL']['sig_map_noise'], 
                        self.preset_fg.components_iter[i, self.preset_sky.seenpix_qubic, istk].shape
                        )
            elif self.preset_fg.components_name_out[i] == 'Synchrotron':
                self.preset_fg.components_iter[i] = C2(C1(self.preset_fg.components_iter[i]))
                for istk in range(3):
                    if istk == 0:
                        key = 'I'
                    else:
                        key = 'P'
                    
                    self.preset_fg.components_iter[i, self.preset_sky.seenpix_qubic, istk] *= self.preset_tools.params['INITIAL'][f'qubic_patch_{key}_sync']
                    self.preset_fg.components_iter[i, self.preset_sky.seenpix_qubic, istk] += np.random.normal(
                        0, 
                        self.preset_tools.params['INITIAL']['sig_map_noise'], 
                        self.preset_fg.components_iter[i, self.preset_sky.seenpix_qubic, istk].shape
                        )
            elif self.preset_fg.components_name_out[i] == 'CO':
                self.preset_fg.components_iter[i] = C2(C1(self.preset_fg.components_iter[i]))
                self.preset_fg.components_iter[i, self.preset_sky.seenpix_qubic, :] *= self.preset_tools.params['INITIAL']['qubic_patch_co']
                self.preset_fg.components_iter[i, self.preset_sky.seenpix_qubic, :] += np.random.normal(
                    0, 
                    self.preset_tools.params['INITIAL']['sig_map_noise'], 
                    self.preset_fg.components_iter[i, self.preset_sky.seenpix_qubic, :].shape
                    )
            else:
                raise TypeError(f'{self.preset_fg.components_name_out[i]} not recognized')

        # else:
        #     self.allbeta = np.array([self.beta_iter])
        #     C1 = HealpixConvolutionGaussianOperator(fwhm=self.fwhm_reconstructed, lmax=3*self.preset_tools.params['SKY']['nside'])
        #     C2 = HealpixConvolutionGaussianOperator(fwhm=self.preset_tools.params['INITIAL']['fwhm0'], lmax=3*self.preset_tools.params['SKY']['nside'])
        #     ### Varying spectral indices -> maps have shape (Nstk, Npix, Ncomp)
        #     for i in range(len(self.preset_fg.components_model_out)):
        #         if self.preset_fg.components_name_out[i] == 'CMB':
        #             #print(self.preset_fg.components_iter.shape)
        #             self.preset_fg.components_iter[:, :, i] = C2(C1(self.preset_fg.components_iter[:, :, i].T)).T
        #             self.preset_fg.components_iter[1:, self.preset_sky.seenpix, i] *= self.preset_tools.params['INITIAL']['qubic_patch_cmb']
                    
        #         elif self.preset_fg.components_name_out[i] == 'Dust':
        #             self.preset_fg.components_iter[:, :, i] = C2(C1(self.preset_fg.components_iter[:, :, i].T)).T + np.random.normal(0, self.preset_tools.params['INITIAL']['sig_map_noise'], self.preset_fg.components_iter[:, :, i].T.shape).T 
        #             self.preset_fg.components_iter[1:, self.preset_sky.seenpix, i] *= self.preset_tools.params['INITIAL']['qubic_patch_dust']
                    
        #         elif self.preset_fg.components_name_out[i] == 'Synchrotron':
        #             self.preset_fg.components_iter[:, :, i] = C2(C1(self.preset_fg.components_iter[:, :, i].T)).T + np.random.normal(0, self.preset_tools.params['INITIAL']['sig_map_noise'], self.preset_fg.components_iter[:, :, i].T.shape).T
        #             self.preset_fg.components_iter[1:, self.preset_sky.seenpix, i] *= self.preset_tools.params['INITIAL']['qubic_patch_sync']
                    
        #         elif self.preset_fg.components_name_out[i] == 'CO':
        #             self.preset_fg.components_iter[:, :, i] = C2(C1(self.preset_fg.components_iter[:, :, i].T)).T + np.random.normal(0, self.preset_tools.params['INITIAL']['sig_map_noise'], self.preset_fg.components_iter[:, :, i].T.shape).T
        #             self.preset_fg.components_iter[1:, self.preset_sky.seenpix, i] *= self.preset_tools.params['INITIAL']['qubic_patch_co']
        #         else:
        #             raise TypeError(f'{self.preset_fg.components_name_out[i]} not recognize')  
