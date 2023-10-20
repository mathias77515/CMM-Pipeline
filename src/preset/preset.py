import numpy as np
import yaml
import qubic
import pickle

import fgb.mixing_matrix as mm
import fgb.component_model as c

from acquisition.systematics import *

from simtools.mpi_tools import *
from simtools.noise_timeline import *
from simtools.foldertools import *

import healpy as hp
import matplotlib.pyplot as plt
from functools import partial
from pyoperators import *
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
import os
import sys
from scipy.optimize import minimize
from solver.cg import *

class PresetSims:


    """
    
    Instance to initialize the Components Map-Making. It reads the `params.yml` file to define QUBIC acquisitions.
    
    Arguments : 
    ===========
        - comm    : MPI common communicator (define by MPI.COMM_WORLD).
        - seed    : Int number for CMB realizations.
        - it      : Int number for noise realizations.
        - verbose : bool, Display message or not.
    
    """

    def __init__(self, comm, seed, it, verbose=True):
        
        self.verbose = verbose

        ### MPI common arguments
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        if self.verbose:
            self._print_message('========= Initialization =========')

        ### Open parameters file
        if self.verbose:
            self._print_message('    => Reading parameters file')
        with open('params.yml', "r") as stream:
            self.params = yaml.safe_load(stream)
            
        ### Define seed for CMB generation and noise
        self.params['CMB']['seed'] = seed
        self.params['CMB']['iter'] = it
        

        ### Get job id for plots
        self.job_id = os.environ.get('SLURM_JOB_ID')

        ### Create folder for saving data and figures
        if self.rank == 0:
            if self.params['save'] != 0:
                print(self.params['CMB']['seed'])
                create_folder_if_not_exists(self.params['foldername']+f"_seed{str(self.params['CMB']['seed'])}")
            if self.params['Plots']['maps'] == True or self.params['Plots']['conv_beta'] == True:
                create_folder_if_not_exists(f'figures_{self.job_id}/I')
                create_folder_if_not_exists(f'figures_{self.job_id}/Q')
                create_folder_if_not_exists(f'figures_{self.job_id}/U')
        
        
        ### QUBIC dictionary
        if self.verbose:
            self._print_message('    => Reading QUBIC dictionary')
        self.dict = self._get_dict()

        ### Skyconfig
        self.skyconfig = self._get_sky_config()

        ### Define model for reconstruction
        if self.verbose:
            self._print_message('    => Creating model')
            
        self._get_components_fgb()

        ### Center of the QUBIC patch
        self.center = qubic.equ2gal(self.dict['RA_center'], self.dict['DEC_center'])

        ### External frequencies
        self.external_nus = self._get_external_nus()
        
        ### Joint acquisition
        if self.params['Foregrounds']['CO']:
            self.nu_co = self.params['Foregrounds']['nu0_co']
        else:
            self.nu_co = None
        
        if self.verbose:
            self._print_message('    => Creating acquisition')
            
        ### Joint acquisition for QUBIC operator
        self.joint = JointAcquisitionComponentsMapMaking(self.dict, 
                                                         self.params['MapMaking']['qubic']['type'], 
                                                         self.comps, 
                                                         self.params['MapMaking']['qubic']['nsub'],
                                                         self.external_nus,
                                                         self.params['MapMaking']['planck']['nintegr'],
                                                         nu_co=self.nu_co)
        
        
        ### Compute coverage map
        self.coverage = self.joint.qubic.coverage
        self.seenpix_qubic = self.coverage/self.coverage.max() > 0
        self.seenpix = self.coverage/self.coverage.max() > self.params['MapMaking']['planck']['thr']
        self.seenpix_plot = self.coverage/self.coverage.max() > self.params['Plots']['thr_plot']
        if self.params['Foregrounds']['nside_fit'] != 0:
            self.seenpix_beta = hp.ud_grade(self.seenpix, self.params['Foregrounds']['nside_fit'])
        
        ### Compute true components
        if self.verbose:
            self._print_message('    => Creating components')
        self._get_components()

        ### Get input spectral index
        if self.verbose:
            self._print_message('    => Reading spectral indices')
        self._get_beta_input()
        
        ### Mask for weight Planck data
        self.mask = np.ones(12*self.params['MapMaking']['qubic']['nside']**2)
        self.mask[self.seenpix] = self.params['MapMaking']['planck']['kappa']
        
        self.mask_beta = np.ones(12*self.params['MapMaking']['qubic']['nside']**2)
        #self.mask_beta = np.zeros(12*self.params['MapMaking']['qubic']['nside']**2)
        #self.mask_beta[self.seenpix_qubic] = 1
        C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['planck']['fwhm_kappa'])
        self.mask = C(self.mask)
        self.mask_beta = C(self.mask_beta)
        
        ### Inverse noise-covariance matrix
        self.invN = self.joint.get_invntt_operator(mask=self.mask)
        self.invN_beta = self.joint.get_invntt_operator(mask=self.mask_beta)
       
        ### Preconditionning
        self.M = get_preconditioner(np.ones(12*self.params['MapMaking']['qubic']['nside']**2))
        self._get_convolution()
        
        ### Get observed data
        if self.verbose:
            self._print_message('    => Getting observational data')
        self._get_tod()

        ### Compute initial guess for PCG
        if self.verbose:
            self._print_message('    => Initialize starting point')
        self._get_x0()    
    def _get_noise(self):

        """
        
        Method to define QUBIC noise, can generate Dual band or Wide Band noise by following :

            - Dual Band : n = [Ndet + Npho_150, Ndet + Npho_220]
            - Wide Band : n = [Ndet + Npho_150 + Npho_220]

        """

        if self.params['MapMaking']['qubic']['type'] == 'wide':
            noise = QubicWideBandNoise(self.dict, 
                                       self.params['MapMaking']['qubic']['npointings'], 
                                       detector_nep=self.params['MapMaking']['qubic']['detector_nep'])
        else:
            noise = QubicDualBandNoise(self.dict, 
                                       self.params['MapMaking']['qubic']['npointings'], 
                                       detector_nep=self.params['MapMaking']['qubic']['detector_nep'])

        return noise.total_noise(self.params['MapMaking']['qubic']['ndet'], 
                                 self.params['MapMaking']['qubic']['npho150'], 
                                 self.params['MapMaking']['qubic']['npho220']).ravel()
    def _get_tod(self):

        """
        
        Method to define fake observational data from QUBIC. It includes astrophysical foregrounds contamination using `self.beta` and systematics using `self.g`.
        We generate also fake observational data from external experiments. We generate data in the following way : d = H . A . c + n

        Be aware that the data used the MPI communication to use several cores. Full data are stored in `self.TOD_Q_BAND_ALL` where `self.TOD_Q` is a part
        of all the data. The multiprocessing is done by divide the number of detector per process.
        
        """

        self._get_input_gain()
        self.H = self.joint.get_operator(beta=self.beta, gain=self.g, fwhm=self.fwhm)
        
        self.array_of_operators = self.joint.qubic.operator
        self.array_of_operators150 = self.array_of_operators[:int(self.params['MapMaking']['qubic']['nsub']/2)]
        self.array_of_operators220 = self.array_of_operators[int(self.params['MapMaking']['qubic']['nsub']/2):self.params['MapMaking']['qubic']['nsub']]

        seed_pl = 42
        ne = self.joint.external.get_noise(seed=seed_pl) * self.params['MapMaking']['planck']['level_planck_noise']
        nq = self._get_noise()
        #self.components *= 0
        self.TOD_Q = self.H.operands[0](self.components) + nq
        self.TOD_E = self.H.operands[1](self.components) + ne
        
        ### Reconvolve Planck dataç toward QUBIC angular resolution
        if self.params['MapMaking']['qubic']['convolution']:
            _r = ReshapeOperator(self.TOD_E.shape, (len(self.external_nus), 12*self.params['MapMaking']['qubic']['nside']**2, 3))
            maps_e = _r(self.TOD_E)
            C = HealpixConvolutionGaussianOperator(fwhm=np.min(self.joint.qubic.allfwhm))
            for i in range(maps_e.shape[0]):
                maps_e[i] = C(maps_e[i])
            
            self.TOD_E = _r.T(maps_e)

        self.TOD_obs = np.r_[self.TOD_Q, self.TOD_E] 

        if self.params['MapMaking']['qubic']['type'] == 'wide':
            R2det_i = ReshapeOperator(self.joint.qubic.ndets*self.joint.qubic.nsamples, (self.joint.qubic.ndets, self.joint.qubic.nsamples))
            self.TOD_Q_ALL = self.comm.allreduce(self.TOD_Q, op=MPI.SUM)#join_data(self.comm, R2det_i(self.TOD_Q))
        else:
            R2det_i = ReshapeOperator(2*self.joint.qubic.ndets*self.joint.qubic.nsamples, (2*self.joint.qubic.ndets, self.joint.qubic.nsamples))
            self.TOD_Q_150 = R2det_i(self.TOD_Q)[:self.joint.qubic.ndets]
            self.TOD_Q_220 = R2det_i(self.TOD_Q)[self.joint.qubic.ndets:2*self.joint.qubic.ndets]
            self.TOD_Q_150_ALL = self.comm.allreduce(self.TOD_Q_150, op=MPI.SUM)
            self.TOD_Q_220_ALL = self.comm.allreduce(self.TOD_Q_220, op=MPI.SUM)
    def _spectral_index(self):

        """
        
        Method to define input spectral indices if the d1 model is used for thermal dust description.
        
        """

        sky = pysm3.Sky(nside=self.params['Foregrounds']['nside_pix'], preset_strings=['d1'])
        return np.array(sky.components[0].mbb_index)
    def _get_beta_input(self):

        """
        
        Method to define the input spectral indices. If the model is d0, the input is 1.54, if not the model assumes varying spectral indices across the sky
        by calling the previous method. In this case, the shape of beta is (Nbeta, Ncomp).
        
        """
        if self.params['Foregrounds']['Dust']:
            if self.params['Foregrounds']['model_d'] == 'd0':
                if self.params['Foregrounds']['nside_fit'] == 0:
                    self.beta = np.array([1.54])
                else:
                    self.beta = np.array([[1.54]*(12*self.params['Foregrounds']['nside_pix']**2)]).T
                    
            else:
                self.beta = np.array([self._spectral_index()]).T
        else:
            self.beta = np.array([])
    def _get_components(self):

        """
        
        Read configuration dictionary which contains every components adn the model. 
        
        Example : d = {'cmb':42, 'dust':'d0', 'synchrotron':'s0'}

        The CMB is randomly generated fram specific seed. Astrophysical foregrounds come from PySM 3.
        
        """

        self.components = np.zeros((len(self.skyconfig), 12*self.params['MapMaking']['qubic']['nside']**2, 3))
        mycls = give_cl_cmb(r=self.params['CMB']['r'], 
                            Alens=self.params['CMB']['Alens'])

        for k, kconf in enumerate(self.skyconfig.keys()):
            if kconf == 'cmb':

                np.random.seed(self.skyconfig[kconf])
                cmb = hp.synfast(mycls, self.params['MapMaking']['qubic']['nside'], verbose=False, new=True).T
                self.components[k] = cmb.copy()
            
            elif kconf == 'dust':

                sky=pysm3.Sky(nside=self.params['MapMaking']['qubic']['nside'], 
                              preset_strings=[self.params['Foregrounds']['model_d']], 
                              output_unit="uK_CMB")
                
                sky.components[0].mbb_temperature = 20*sky.components[0].mbb_temperature.unit
                self.components[k] = np.array(sky.get_emission(self.params['Foregrounds']['nu0_d'] * u.GHz, None).T * \
                                  utils.bandpass_unit_conversion(self.params['Foregrounds']['nu0_d']*u.GHz, None, u.uK_CMB))
                    

            elif kconf == 'synchrotron':

                sky = pysm3.Sky(nside=self.params['MapMaking']['qubic']['nside'], 
                                preset_strings=[self.params['Foregrounds']['model_s']], 
                                output_unit="uK_CMB")
                self.components[k] = np.array(sky.get_emission(self.params['Foregrounds']['nu0_s'] * u.GHz, None).T * \
                                utils.bandpass_unit_conversion(self.params['Foregrounds']['nu0_s']*u.GHz, None, u.uK_CMB))
                
            elif kconf == 'coline':
                
                m = hp.ud_grade(hp.read_map('data/CO_line.fits') * 10, self.params['MapMaking']['qubic']['nside'])
                mP = polarized_I(m, self.params['MapMaking']['qubic']['nside'])
                myco = np.zeros((12*self.params['MapMaking']['qubic']['nside']**2, 3))
                myco[:, 0] = m.copy()
                myco[:, 1:] = mP.T.copy()
                self.components[k] = myco.copy()
            else:
                raise TypeError('Choose right foreground model (d0, s0, ...)')
        if self.params['Foregrounds']['nside_fit'] == 0:
        #if self.params['Foregrounds']['model_d'] == 'd0':
            self.components_iter = self.components.copy()
        else:
            self.components = self.components.T.copy()
            self.components_iter = self.components.copy()
    def _get_ultrawideband_config(self):
        

        """
        
        Method to simply define Ultra Wide Band configuration.
        
        """
        nu_up = 247.5
        nu_down = 131.25
        nu_ave = np.mean(np.array([nu_up, nu_down]))
        delta = nu_up - nu_ave
    
        return nu_ave, 2*delta/nu_ave
    def _get_dict(self):
    
        """

        Method to define and modify the QUBIC dictionary.
        
        """

        nu_ave, delta_nu_over_nu = self._get_ultrawideband_config()

        args = {'npointings':self.params['MapMaking']['qubic']['npointings'], 
                'nf_recon':1, 
                'nf_sub':self.params['MapMaking']['qubic']['nsub'], 
                'nside':self.params['MapMaking']['qubic']['nside'], 
                'MultiBand':True, 
                'period':1, 
                'RA_center':self.params['MapMaking']['sky']['RA_center'], 
                'DEC_center':self.params['MapMaking']['sky']['DEC_center'],
                'filter_nu':nu_ave*1e9, 
                'noiseless':False, 
                'comm':self.comm, 
                'dtheta':self.params['MapMaking']['qubic']['dtheta'],
                'nprocs_sampling':1, 
                'nprocs_instrument':self.size,
                'photon_noise':True, 
                'nhwp_angles':self.params['MapMaking']['qubic']['nhwp_angles'], 
                'effective_duration':3, 
                'filter_relative_bandwidth':delta_nu_over_nu, 
                'type_instrument':'wide', 
                'TemperatureAtmosphere150':None, 
                'TemperatureAtmosphere220':None,
                'EmissivityAtmosphere150':None, 
                'EmissivityAtmosphere220':None, 
                'detector_nep':float(self.params['MapMaking']['qubic']['detector_nep']), 
                'synthbeam_kmax':self.params['MapMaking']['qubic']['synthbeam_kmax']}

        ### Get the default dictionary
        dictfilename = 'dicts/pipeline_demo.dict'
        d = qubic.qubicdict.qubicDict()
        d.read_from_file(dictfilename)
        for i in args.keys():
        
            d[str(i)] = args[i]

    
        return d
    def _get_sky_config(self):
        
        """
        
        Method to define sky model used by PySM to generate fake sky. It create dictionary like :

                sky = {'cmb':42, 'dust':'d0'}
        
        """
        
        sky = {}
        for ii, i in enumerate(self.params.keys()):

            if i == 'CMB':
                if self.params['CMB']['cmb']:
                    sky['cmb'] = self.params['CMB']['seed']
            else:
                for jj, j in enumerate(self.params['Foregrounds']):
                    #print(j, self.params['Foregrounds'][j])
                    if j == 'Dust':
                        if self.params['Foregrounds'][j]:
                            sky['dust'] = self.params['Foregrounds']['model_d']
                    elif j == 'Synchrotron':
                        if self.params['Foregrounds'][j]:
                            sky['synchrotron'] = self.params['Foregrounds']['model_s']
                    elif j == 'CO':
                        if self.params['Foregrounds'][j]:
                            sky['coline'] = 'co2'
        return sky
    def _get_components_fgb(self):

        """
        
        Method to define sky model taken form FGBuster code. Note that we add `COLine` instance to define monochromatic description.

        """

        self.comps = []
        self.comps_name = []

        if self.params['CMB']['cmb']:
            self.comps += [c.CMB()]
            self.comps_name += ['CMB']
            
        if self.params['Foregrounds']['Dust']:
            self.comps += [c.Dust(nu0=self.params['Foregrounds']['nu0_d'], temp=self.params['Foregrounds']['temp'])]
            self.comps_name += ['Dust']

        if self.params['Foregrounds']['Synchrotron']:
            self.comps += [c.Synchrotron(nu0=self.params['Foregrounds']['nu0_s'], beta_pl=-3)]
            self.comps_name += ['Synchrotron']

        if self.params['Foregrounds']['CO']:
            self.comps += [c.COLine(nu=self.params['Foregrounds']['nu0_co'], active=False)]
            self.comps_name += ['CO']
    def _get_external_nus(self):


        """
        
        Method to create python array of external frequencies by reading the `params.yml` file.

        """

        allnus = [30, 44, 70, 100, 143, 217, 353]
        external = []
        for inu, nu in enumerate(allnus):
            if self.params['MapMaking']['planck'][f'{nu:.0f}GHz']:
                external += [nu]

        return external
    def _get_convolution(self):

        """
        
        Method to define all agular resolutions of the instrument at each frequencies. `self.fwhm` are the real angular resolution and `self.fwhm_recon` are the 
        beams used for the reconstruction. 
        
        """
        
        if self.params['MapMaking']['qubic']['convolution']:
            self.fwhm_recon = np.sqrt(self.joint.qubic.allfwhm**2 - np.min(self.joint.qubic.allfwhm)**2)
            self.fwhm = self.joint.qubic.allfwhm
        else:
            self.fwhm_recon = None
            self.fwhm = None
    def _get_input_gain(self):

        """
        
        Method to define gain detector of QUBIC focal plane. It is a random generation following normal law. Note that `self.g` contains gain for the i-th process
        that contains few detectors, all the gain are stored in `self.G`.
        
        """
        
        np.random.seed(None)
        if self.params['MapMaking']['qubic']['type'] == 'wide':
            self.g = np.random.random(self.joint.qubic.ndets) * self.params['MapMaking']['qubic']['sig_gain'] + 1
            self.g /= self.g[0]
        else:
            self.g = np.random.random((self.joint.qubic.ndets, 2)) * self.params['MapMaking']['qubic']['sig_gain'] + 1
            self.g /= self.g[0]

        self.G = join_data(self.comm, self.g)
        self.g_iter = np.ones(self.g.shape)
        self.allg = np.array([self.g_iter])
    def _get_x0(self):

        """
        
        Method to define starting point of the convergence. The argument 'set_comp_to_0' multiply the pixels values by a given factor. You can decide 
        to convolve also the map by a beam with an fwhm in radians.
        
        """
        #if self.params['Foregrounds']['nside_fit'] == 0:
        
        #    self.beta_iter = np.random.normal(self.params['MapMaking']['initial']['mean_beta_x0'],
        #                                  self.params['MapMaking']['initial']['sig_beta_x0'],
        #                                  self.beta.shape)
            
        if self.rank == 0:
            seed = np.random.randint(100000000)
        else:
            seed = None
        
        seed = self.comm.bcast(seed, root=0)
        np.random.seed(seed)
        
        if self.params['Foregrounds']['nside_fit'] == 0:
            self.beta_iter = np.random.normal(self.params['MapMaking']['initial']['mean_beta_x0'], 
                                              self.params['MapMaking']['initial']['sig_beta_x0'], 
                                              self.beta.shape)
        else:
            self.beta_iter = self.beta.copy()
            _index_seenpix_beta = np.where(self.seenpix_beta == 1)[0]
            #self.beta_iter[_index_seenpix_beta, 0] += np.random.normal(0, 
            #                                                       self.params['MapMaking']['initial']['sig_beta_x0'], 
            #                                                       _index_seenpix_beta.shape)
            self.beta_iter[_index_seenpix_beta, 0] = np.random.normal(self.params['MapMaking']['initial']['mean_beta_x0'], 
                                                                      self.params['MapMaking']['initial']['sig_beta_x0'], 
                                                                      _index_seenpix_beta.shape)
        
        

        if self.params['Foregrounds']['nside_fit'] == 0:
            self.allbeta = np.array([self.beta_iter])
            ### Constant spectral index -> maps have shape (Ncomp, Npix, Nstk)
            for i in range(len(self.comps)):
                if self.comps_name[i] == 'CMB':
                    C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['initial']['fwhm_x0'])
                    self.components_iter[i] = C(self.components_iter[i] + np.random.normal(0, self.params['MapMaking']['initial']['sig_map'], self.components_iter[i].shape)) * self.params['MapMaking']['initial']['set_cmb_to_0']
                    self.components_iter[i, self.seenpix, :] *= self.params['MapMaking']['initial']['qubic_patch_cmb']

                elif self.comps_name[i] == 'Dust':
                    C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['initial']['fwhm_x0'])
                    self.components_iter[i] = C(self.components_iter[i] + np.random.normal(0, self.params['MapMaking']['initial']['sig_map'], self.components_iter[i].shape)) * self.params['MapMaking']['initial']['set_dust_to_0']
                    self.components_iter[i, self.seenpix, :] *= self.params['MapMaking']['initial']['qubic_patch_dust']

                elif self.comps_name[i] == 'Synchrotron':
                    C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['initial']['fwhm_x0'])
                    self.components_iter[i] = C(self.components_iter[i] + np.random.normal(0, self.params['MapMaking']['initial']['sig_map'], self.components_iter[i].shape)) * self.params['MapMaking']['initial']['set_sync_to_0']
                    self.components_iter[i, self.seenpix, :] *= self.params['MapMaking']['initial']['qubic_patch_sync']

                elif self.comps_name[i] == 'CO':
                    C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['initial']['fwhm_x0'])
                    self.components_iter[i] = C(self.components_iter[i] + np.random.normal(0, self.params['MapMaking']['initial']['sig_map'], self.components_iter[i].shape)) * self.params['MapMaking']['initial']['set_co_to_0']
                    self.components_iter[i, self.seenpix, :] *= self.params['MapMaking']['initial']['qubic_patch_co']
                else:
                    raise TypeError(f'{self.comps_name[i]} not recognize')
        else:
            self.allbeta = np.array([self.beta_iter[np.where(self.seenpix_beta == 1)[0]]])
            ### Varying spectral indices -> maps have shape (Nstk, Npix, Ncomp)
            for i in range(len(self.comps)):
                if self.comps_name[i] == 'CMB':
                    C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['initial']['fwhm_x0'])
                    self.components_iter[:, :, i] = C(self.components_iter[:, :, i].T + np.random.normal(0, self.params['MapMaking']['initial']['sig_map'], self.components_iter[:, :, i].T.shape)).T * self.params['MapMaking']['initial']['set_cmb_to_0']
                    self.components_iter[:, self.seenpix, i] *= self.params['MapMaking']['initial']['qubic_patch_cmb']
                    
                elif self.comps_name[i] == 'Dust':
                    C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['initial']['fwhm_x0'])
                    self.components_iter[:, :, i] = C(self.components_iter[:, :, i].T + np.random.normal(0, self.params['MapMaking']['initial']['sig_map'], self.components_iter[:, :, i].T.shape)).T * self.params['MapMaking']['initial']['set_dust_to_0']
                    self.components_iter[:, self.seenpix, i] *= self.params['MapMaking']['initial']['qubic_patch_dust']
                    
                elif self.comps_name[i] == 'Synchrotron':
                    C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['initial']['fwhm_x0'])
                    self.components_iter[:, :, i] = C(self.components_iter[:, :, i].T + np.random.normal(0, self.params['MapMaking']['initial']['sig_map'], self.components_iter[:, :, i].T.shape)).T * self.params['MapMaking']['initial']['set_sync_to_0']
                    self.components_iter[:, self.seenpix, i] *= self.params['MapMaking']['initial']['qubic_patch_sync']
                    
                elif self.comps_name[i] == 'CO':
                    C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['initial']['fwhm_x0'])
                    self.components_iter[:, :, i] = C(self.components_iter[:, :, i].T + np.random.normal(0, self.params['MapMaking']['initial']['sig_map'], self.components_iter[:, :, i].T.shape)).T * self.params['MapMaking']['initial']['set_co_to_0']
                    self.components_iter[:, self.seenpix, i] *= self.params['MapMaking']['initial']['qubic_patch_co']
                else:
                    raise TypeError(f'{self.comps_name[i]} not recognize')            
    def _print_message(self, message):

        """
        
        Method that display a `message` only for the first rank because of MPI multiprocessing.
        
        """
        
        if self.rank == 0:
            print(message)

