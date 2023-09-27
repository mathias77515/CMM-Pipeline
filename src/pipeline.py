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
        C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['planck']['fwhm_kappa'])
        self.mask = C(self.mask)
        
        ### Inverse noise-covariance matrix
        self.invN = self.joint.get_invntt_operator(mask=self.mask)
       
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
            self.TOD_Q_150_ALL = join_data(self.comm, self.TOD_Q_150)
            self.TOD_Q_220_ALL = join_data(self.comm, self.TOD_Q_220)
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
                if self.verbose:
                    print('Define constant spectral index')
                self.beta = np.array([1.54])
            else:
                if self.verbose:
                    print('Define varying spectral index')
                #raise TypeError('Not yet implemented')
                self.beta = np.array([self._spectral_index()]).T
        else:
            self.beta = np.array([])
            
        #if self.params['Foregrounds']['Synchrotron']:
        #    self.beta = np.append(self.beta, -3)
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

        if self.params['Foregrounds']['model_d'] == 'd0':
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
            self.comps_name += ['DUST']

        if self.params['Foregrounds']['Synchrotron']:
            self.comps += [c.Synchrotron(nu0=self.params['Foregrounds']['nu0_s'], beta_pl=-3)]
            self.comps_name += ['SYNCHROTRON']

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
        
        self.beta_iter = np.random.normal(self.params['MapMaking']['initial']['mean_beta_x0'],
                                          self.params['MapMaking']['initial']['sig_beta_x0'],
                                          self.beta.shape)
        
        if self.params['Foregrounds']['model_d'] == 'd0':
            
            ### Constant spectral index -> maps have shape (Ncomp, Npix, Nstk)
            for i in range(len(self.comps)):
                if self.comps_name[i] == 'CMB':
                    C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['initial']['fwhm_x0'])
                    self.components_iter[i] = C(self.components_iter[i] + np.random.normal(0, self.params['MapMaking']['initial']['sig_map'], self.components_iter[i].shape)) * self.params['MapMaking']['initial']['set_cmb_to_0']
                    self.components_iter[i, self.seenpix, :] *= self.params['MapMaking']['initial']['qubic_patch_cmb']

                elif self.comps_name[i] == 'DUST':
                    C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['initial']['fwhm_x0'])
                    self.components_iter[i] = C(self.components_iter[i] + np.random.normal(0, self.params['MapMaking']['initial']['sig_map'], self.components_iter[i].shape)) * self.params['MapMaking']['initial']['set_dust_to_0']
                    self.components_iter[i, self.seenpix, :] *= self.params['MapMaking']['initial']['qubic_patch_dust']

                elif self.comps_name[i] == 'SYNCHROTRON':
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
            
            ### Varying spectral indices -> maps have shape (Nstk, Npix, Ncomp)
            for i in range(len(self.comps)):
                if self.comps_name[i] == 'CMB':
                    C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['initial']['fwhm_x0'])
                    self.components_iter[:, :, i] = C(self.components_iter[:, :, i].T + np.random.normal(0, self.params['MapMaking']['initial']['sig_map'], self.components_iter[:, :, i].shape)).T * self.params['MapMaking']['initial']['set_cmb_to_0']

                elif self.comps_name[i] == 'DUST':
                    C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['initial']['fwhm_x0'])
                    self.components_iter[:, :, i] = C(self.components_iter[:, :, i].T + np.random.normal(0, self.params['MapMaking']['initial']['sig_map'], self.components_iter[:, :, i].shape)).T * self.params['MapMaking']['initial']['set_dust_to_0']

                elif self.comps_name[i] == 'SYNCHROTRON':
                    C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['initial']['fwhm_x0'])
                    self.components_iter[:, :, i] = C(self.components_iter[:, :, i].T + np.random.normal(0, self.params['MapMaking']['initial']['sig_map'], self.components_iter[:, :, i].shape)).T * self.params['MapMaking']['initial']['set_sync_to_0']

                elif self.comps_name[i] == 'CO':
                    C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['initial']['fwhm_x0'])
                    self.components_iter[:, :, i] = C(self.components_iter[:, :, i].T + np.random.normal(0, self.params['MapMaking']['initial']['sig_map'], self.components_iter[:, :, i].shape)).T * self.params['MapMaking']['initial']['set_co_to_0']
                else:
                    raise TypeError(f'{self.comps_name[i]} not recognize')
    def _print_message(self, message):

        """
        
        Method that display a `message` only for the first rank because of MPI multiprocessing.
        
        """
        
        if self.rank == 0:
            print(message)

class Chi2(PresetSims):

    """
    
    Instance that define Chi^2 function for many configurations. The instance initialize first the PresetSims instance and knows every parameters.

    Arguments : 
    ===========
        - comm    : MPI common communicator (define by MPI.COMM_WORLD).
        - seed    : Int number for CMB realizations.
        - it      : Int number for noise realizations.

    """
    
    def __init__(self, comm, seed, it):

        PresetSims.__init__(self, comm, seed, it)

    def chi2_tot(self, x, solution):

        """
    
        Method to define chi^2 function for all experience :

            chi^2 = chi^2_QUBIC + chi^2_external

        """
        #print(x)
        xi2_external = self.chi2_external(x, solution)
        if self.params['MapMaking']['qubic']['type'] == 'wide':
            self.chi2_Q = self.wide(x, solution)
        elif self.params['MapMaking']['qubic']['type'] == 'two':
            xi2_150 = self.two150(x, solution)
            xi2_220 = self.two220(x, solution)
        
            self.chi2_Q = xi2_150 + xi2_220

        return xi2_external + self.chi2_Q
    def chi2_tot_varying(self, x, patch_id, allbeta, solution):

        """
    
        Method to define chi^2 function for all experience :

            chi^2 = chi^2_QUBIC + chi^2_external

        """
        xi2_external = self.chi2_external_varying(x, patch_id, allbeta, solution)
        if self.params['MapMaking']['qubic']['type'] == 'wide':
            xi2_w = self.wide_varying(x, patch_id, allbeta, solution)
            #print(xi2_w, xi2_external)
        return xi2_w + xi2_external
    def chi2_external_varying(self, x, patch_id, allbeta, solution):

        """
    
        Define chi^2 function for external data with shape :

            chi^2 = (TOD_true - sum_nsub(H * A * c))^2 

        """
        allbeta[patch_id, 0] = x

        Hexternal = self.joint.external.get_operator(beta=allbeta, convolution=False, comm=self.joint.qubic.mpidist, nu_co=None)
        Hexternal2 = self.joint.external.get_operator(beta=allbeta*10, convolution=False, comm=self.joint.qubic.mpidist, nu_co=None)
        
        tod_s_i = Hexternal(solution[-1])
        diff = (self.TOD_E/self.TOD_E.max()) - (tod_s_i/tod_s_i.max())
        return np.sum(diff**2)
    def chi2_external(self, x, solution):

        """
    
        Define chi^2 function for external data with shape :

            chi^2 = (TOD_true - sum_nsub(H * A * c))^2 

        """

        tod_s_i = self.TOD_E.copy() * 0

        Hexternal = self.joint.external.get_operator(beta=x, convolution=False)

        tod_s_i = Hexternal(solution[-1])
        
        #diff = self.TOD_E/self.TOD_E.max() - tod_s_i/tod_s_i.max()
        tod_sim_norm = self.invN.operands[1](tod_s_i.ravel()**2)
        tod_obs_norm = self.invN.operands[1](self.TOD_E.ravel()**2)
        tod_sim_norm = self.comm.allreduce(tod_sim_norm, op=MPI.SUM)
        tod_obs_norm = self.comm.allreduce(tod_obs_norm, op=MPI.SUM)
        diff = tod_obs_norm - tod_sim_norm
        #print(np.sum(diff**2))
        self.chi2_P = np.sum(diff**2)
        return self.chi2_P
    def wide_varying(self, x, patch_id, allbeta, solution):

        """
    
        Define chi^2 function for Wide Band TOD with shape :

            chi^2 = (TOD_true - sum_nsub(H * A * c))^2 

        """

        allbeta[patch_id, 0] = x
    
        tod_s_i = self.TOD_Q_ALL.copy() * 0
        G = DiagonalOperator(self.g_iter, broadcast='rightward', shapein=(self.joint.qubic.ndets, self.joint.qubic.nsamples))
        
        k=0
        for ii, i in enumerate(self.array_of_operators[:self.params['MapMaking']['qubic']['nsub']]):
        
            A = get_mixing_operator(allbeta, nus=np.array([self.joint.qubic.allnus[k]]), comp=self.comps, nside=self.params['MapMaking']['qubic']['nside'], active=False)
            Hi = G * i.copy()
            Hi.operands[-1] = A

            tod_s_i += Hi(solution[ii]).ravel()
            k+=1
    
        if self.nu_co is not None:
            A = Acq.get_mixing_operator(x, nus=np.array([self.nu_co]), comp=self.comps, nside=self.params['MapMaking']['qubic']['nside'], active=True)
            Hi = array_of_operators[-1].copy()
            Hi.operands[-1] = A

            tod_s_i += Hi(solution[-1]).ravel()


        if self.comm is not None:
            s = self.comm.allreduce(tod_s_i, op=MPI.SUM)
        else:
            s = tod_s_i.copy()
    
        diff = ((self.TOD_Q_ALL.ravel()) - (s.ravel()))
    
        return np.sum(diff**2)
    def wide(self, x, solution):

        """
    
        Define chi^2 function for Wide Band TOD with shape :

            chi^2 = (TOD_true - sum_nsub(H * A * c))^2 

        """

        tod_s_i = self.TOD_Q * 0

        G = DiagonalOperator(self.g_iter, broadcast='rightward', shapein=(self.joint.qubic.ndets, self.joint.qubic.nsamples))

        k=0
        for ii, i in enumerate(self.array_of_operators[:self.params['MapMaking']['qubic']['nsub']]):
        
            A = get_mixing_operator(x, nus=np.array([self.joint.qubic.allnus[k]]), comp=self.comps, nside=self.params['MapMaking']['qubic']['nside'], active=False)
            Hi = G * i.copy()
            Hi.operands[-1] = A
            
            tod_s_i += Hi(solution[ii]).ravel()
            k+=1

        if self.nu_co is not None:
            A = get_mixing_operator(x, nus=np.array([self.nu_co]), comp=self.comps, nside=self.params['MapMaking']['qubic']['nside'], active=True)
            Hi = self.array_of_operators[-1].copy()
            Hi.operands[-1] = A

            tod_s_i += Hi(solution[-1])
        

        tod_sim_norm = self.invN.operands[0](tod_s_i.ravel()**2)
        tod_obs_norm = self.invN.operands[0](self.TOD_Q.ravel()**2)
        tod_sim_norm = self.comm.allreduce(tod_sim_norm, op=MPI.SUM)
        tod_obs_norm = self.comm.allreduce(tod_obs_norm, op=MPI.SUM)
        
        diff = tod_obs_norm - tod_sim_norm
        
        self.chi2_Q = np.sum(diff**2)
        
        return self.chi2_Q
    def two150(self, x, solution):
        
        tod_s_i = self.TOD_Q_150.ravel() * 0

        G = DiagonalOperator(self.g_iter[:, 0], broadcast='rightward', shapein=(self.joint.qubic.ndets, self.joint.qubic.nsamples))
        k=0
        for ii, i in enumerate(self.array_of_operators150):
        
            A = get_mixing_operator(x, nus=np.array([self.joint.qubic.allnus[k]]), comp=self.comps, nside=self.params['MapMaking']['qubic']['nside'], active=False)
            Hi = G * i.copy()
            Hi.operands[-1] = A
            
            tod_s_i += Hi(solution[ii]).ravel()
            k+=1
        _r = ReshapeOperator((self.joint.qubic.ndets, self.joint.qubic.nsamples), (self.joint.qubic.ndets*self.joint.qubic.nsamples))

        invn = CompositionOperator([_r, self.invN.operands[0].operands[1].operands[0], _r.T])
        
        tod_sim_norm = invn(tod_s_i.ravel()**2)
        tod_obs_norm = invn(self.TOD_Q_150.ravel()**2)
        tod_sim_norm = self.comm.allreduce(tod_sim_norm, op=MPI.SUM)
        tod_obs_norm = self.comm.allreduce(tod_obs_norm, op=MPI.SUM)
        diff = tod_obs_norm - tod_sim_norm
        
        self.chi2_Q_150 = np.sum(diff**2)
        
        return self.chi2_Q_150
    def two220(self, x, solution):
        
        tod_s_i = self.TOD_Q_220.ravel() * 0

        G = DiagonalOperator(self.g_iter[:, 1], broadcast='rightward', shapein=(self.joint.qubic.ndets, self.joint.qubic.nsamples))
        k=0
        for ii, i in enumerate(self.array_of_operators220[:self.params['MapMaking']['qubic']['nsub']]):
            
            mynus = np.array([self.joint.qubic.allnus[k+int(self.params['MapMaking']['qubic']['nsub']/2)]])
            A = get_mixing_operator(x, nus=mynus, comp=self.comps, nside=self.params['MapMaking']['qubic']['nside'], active=False)
            Hi = G * i.copy()
            Hi.operands[-1] = A
            
            tod_s_i += Hi(solution[ii+int(self.params['MapMaking']['qubic']['nsub']/2)]).ravel()
            k+=1

        if self.nu_co is not None:
            A = get_mixing_operator(x, nus=np.array([self.nu_co]), comp=self.comps, nside=self.params['MapMaking']['qubic']['nside'], active=True)
            Hi = self.array_of_operators[-1].copy()
            Hi.operands[-1] = A

            tod_s_i += Hi(solution[-1])


        _r = ReshapeOperator((self.joint.qubic.ndets, self.joint.qubic.nsamples), (self.joint.qubic.ndets*self.joint.qubic.nsamples))

        invn = CompositionOperator([_r, self.invN.operands[0].operands[1].operands[1], _r.T])
        
        tod_sim_norm = invn(tod_s_i.ravel()**2)
        tod_obs_norm = invn(self.TOD_Q_220.ravel()**2)
        tod_sim_norm = self.comm.allreduce(tod_sim_norm, op=MPI.SUM)
        tod_obs_norm = self.comm.allreduce(tod_obs_norm, op=MPI.SUM)
        diff = tod_obs_norm - tod_sim_norm
        
        self.chi2_Q_220 = np.sum(diff**2)
        
        return self.chi2_Q_220

class Plots:

    """
    
    Instance to produce plots on the convergence. 
    
    Arguments : 
    ===========
        - jobid : Int number for saving figures.
        - dogif : Bool to produce GIF.
    
    """
    
    def __init__(self, jobid, dogif=False):
        
        self.job_id = jobid
        self.dogif = dogif
    def plot_beta_iteration(self, beta, figsize=(8, 6), truth=None):

        """
        
        Method to plot beta as function of iteration. beta can have shape (niter) of (niter, nbeta)
        
        """
        
        if self.params['Plots']['conv_beta']:
            niter = beta.shape[0]
            alliter = np.arange(1, niter+1, 1)
            plt.figure(figsize=figsize)

            if np.ndim(beta) == 1:
                plt.plot(alliter[1:]-1, beta[1:])
                if truth is not None:
                    plt.axhline(truth, ls='--', color='red')
            else:
                for i in range(niter):
                    plt.plot(alliter, beta[:, i], '-k', alpha=0.3)
                    if truth is not None:
                        plt.axhline(truth[i], ls='--', color='red')

            plt.savefig(f'figures_{self.job_id}/beta_iter{self._steps+1}.png')

            if self._steps > 0:
            
                os.remove(f'figures_{self.job_id}/beta_iter{self._steps}.png')

            plt.close()
    def display_maps(self, seenpix, ngif=0, figsize=(14, 8), nsig=6):
        
        """
        
        Method to display maps at given iteration.
        
        Arguments:
        ----------
            - seenpix : array containing the id of seen pixels.
            - ngif    : Int number to create GIF with ngif PNG image.
            - figsize : Tuple to control size of plots.
            - nsig    : Int number to compute errorbars.
        
        """
        
        if self.params['Plots']['maps']:
            stk = ['I', 'Q', 'U']
            C = HealpixConvolutionGaussianOperator(fwhm=self.params['Plots']['fake_conv'])
            for istk, s in enumerate(stk):
                plt.figure(figsize=figsize)

                k=0
                for icomp in range(len(self.comps)):
                    
                    map_in = C(self.components[icomp, :, istk]).copy()
                    map_out = C(self.components_iter[icomp, :, istk]).copy()
                    map_in[~seenpix] = hp.UNSEEN
                    map_out[~seenpix] = hp.UNSEEN
                    r = map_in - map_out
                    r[~seenpix] = hp.UNSEEN
                    
                    sig = np.std(self.components[icomp, seenpix, istk])
                    
                    hp.gnomview(map_in, rot=self.center, reso=13, notext=True, title='',
                        cmap='jet', sub=(len(self.comps), 3, k+1), min=-2*sig, max=2*sig)
                    hp.gnomview(map_out, rot=self.center, reso=13, notext=True, title='',
                        cmap='jet', sub=(len(self.comps), 3, k+2), min=-2*sig, max=2*sig)
                    
                    hp.gnomview(r, rot=self.center, reso=13, notext=True, title=f"{np.std(r[seenpix]):.3e}",
                        cmap='jet', sub=(len(self.comps), 3, k+3), min=-1*sig, max=1*sig)

                    k+=3

                plt.tight_layout()
                plt.savefig(f'figures_{self.job_id}/{s}/maps_iter{self._steps+1}.png')

                plt.close()
        if self.dogif:
            if ngif%10 == 0:
                do_gif(f'figures_{self.job_id}/I/', self._steps+1)
                do_gif(f'figures_{self.job_id}/Q/', self._steps+1)
                do_gif(f'figures_{self.job_id}/U/', self._steps+1)
    def plot_gain_iteration(self, gain, alpha, figsize=(8, 6)):
        
        """
        
        Method to plot convergence of reconstructed gains.
        
        Arguments :
        -----------
            - gain    : Array containing gain number (1 per detectors). It has the shape (Niteration, Ndet, 2) for Two Bands design and (Niteration, Ndet) for Wide Band design
            - alpha   : Transparency for curves.
            - figsize : Tuple to control size of plots.
            
        """
        
        
        if self.params['Plots']['conv_gain']:
            
            plt.figure(figsize=figsize)

            
            
            niter = gain.shape[0]
            ndet = gain.shape[1]
            alliter = np.arange(1, niter+1, 1)

            if self.params['MapMaking']['qubic']['type'] == 'two':
                color = ['--r', '--b']
                for j in range(2):
                    plt.plot(alliter-1, np.mean(gain, axis=1)[:, j], color[j], alpha=1)
                    for i in range(ndet):
                        plt.plot(alliter-1, gain[:, i, j], color[j], alpha=alpha)
                        
            elif self.params['MapMaking']['qubic']['type'] == 'wide':
                color = ['--g']
                plt.plot(alliter-1, np.mean(gain, axis=1), color[0], alpha=1)
                for i in range(ndet):
                    plt.plot(alliter-1, gain[:, i], color[0], alpha=alpha)
                        
            plt.yscale('log')
            plt.ylabel(r'|$g_{reconstructed} - g_{input}$|', fontsize=12)
            plt.xlabel('Iterations', fontsize=12)
            plt.savefig(f'figures_{self.job_id}/gain_iter{self._steps+1}.png')

            if self._steps > 0:
            
                os.remove(f'figures_{self.job_id}/gain_iter{self._steps}.png')

            plt.close()
                       
class Pipeline(Chi2, Plots):


    """
    
    Main instance to create End-2-End pipeline for components reconstruction.
    
    Arguments :
    -----------
        - comm    : MPI common communicator (define by MPI.COMM_WORLD).
        - seed    : Int number for CMB realizations.
        - it      : Int number for noise realizations.
        
    """
    
    def __init__(self, comm, seed, it):
        
        ### Compute previous instance by heritage
        Chi2.__init__(self, comm, seed, it)
        Plots.__init__(self, self.job_id, self.params['Plots']['gif'])
    def main(self):
        
        """
        
        Method to run the pipeline by following :
        
            1) Initialize simulation using `PresetSims` instance reading `params.yml`.
            
            2) Solve map-making equation knowing spectral index and gains.
            
            3) Fit spectral index knowing components and gains.
            
            4) Fit gains knowing components and sepctral index.
            
            5) Repeat 2), 3) and 4) until convergence.
        
        """
        
        self._info = True
        self._steps = 0
        
        while self._info:


            self._display_iter()
            
            ### Update self.components_iter^{k} -> self.components_iter^{k+1}
            self._update_components()
            
            ### Update self.beta_iter^{k} -> self.beta_iter^{k+1}
            if self.params['Foregrounds']['fit_spectral_index']:
                self._update_spectral_index()
            
            ### Update self.g_iter^{k} -> self.g_iter^{k+1}
            if self.params['MapMaking']['qubic']['fit_gain']:
                self._update_gain()
            
            if self.rank == 0:
                
                ### Display maps
                self.display_maps(self.seenpix_plot, ngif=self._steps+1)
                
                ### Display convergence of beta
                self.plot_beta_iteration(self.beta, truth=None)

                ### Display convergence of beta
                self.plot_gain_iteration(abs(self.allg - self.g), alpha=0.03)

            ### Save data inside pickle file
            self._save_data()

            ### Stop the loop when self._steps > k
            self._stop_condition()
    def _compute_maps_convolved(self):
        
        """
        
        Method to compute convolved maps for each FWHM of QUBIC.
        
        """
        
        ### We make the convolution before beta estimation to speed up the code, we avoid to make all the convolution at each iteration
        ### Constant spectral index
        if self.params['Foregrounds']['model_d'] == 'd0':
            components_for_beta = np.zeros((self.params['MapMaking']['qubic']['nsub'], len(self.comps), 12*self.params['MapMaking']['qubic']['nside']**2, 3))
            for i in range(self.params['MapMaking']['qubic']['nsub']):

                for jcomp in range(len(self.comps)):
                    if self.params['MapMaking']['qubic']['convolution']:
                        C = HealpixConvolutionGaussianOperator(fwhm = self.fwhm_recon[i])
                    else:
                        C = HealpixConvolutionGaussianOperator(fwhm = 0)
                    components_for_beta[i, jcomp] = C(self.components_iter[jcomp])
        else:
            components_for_beta = np.zeros((self.params['MapMaking']['qubic']['nsub'], 3, 12*self.params['MapMaking']['qubic']['nside']**2, len(self.comps)))
            for i in range(self.params['MapMaking']['qubic']['nsub']):
                for jcomp in range(len(self.comps)):
                    if self.params['MapMaking']['qubic']['convolution']:
                        C = HealpixConvolutionGaussianOperator(fwhm = self.fwhm_recon[i])
                    else:
                        C = HealpixConvolutionGaussianOperator(fwhm = 0)

                    components_for_beta[i, :, :, jcomp] = C(self.components_iter[:, :, jcomp].T).T
        return components_for_beta
    def _callback(self, x):
        
        """
        
        Method to make callback function readable by `scipy.optimize.minimize`.
        
        """
        
        if self.rank == 0:
            print(f"{self.nfev:4d}   {x[0]:3.6f}   {self.chi2_Q:3.6e}   {self.chi2_P:3.6e}")
            self.nfev += 1
    def _update_spectral_index(self):
        
        """
        
        Method that perform step 3) of the pipeline for 2 possible designs : Two Bands and Wide Band
        
        """
        
        self.H_i = self.joint.get_operator(self.beta_iter, gain=self.g_iter, fwhm=self.fwhm_recon, nu_co=self.nu_co)
        
        if self.params['Foregrounds']['model_d'] == 'd0':
            chi2 = partial(self.chi2_tot, solution=self._compute_maps_convolved())
            
            ### Callback function
            if self.rank == 0:
                self.nfev = 0
                print('{0:4s}     {1:9s} {2:9s}    {3:9s}'.format('Iter', 'beta', 'logL QUBIC', 'logL Planck'))
            
            self.beta_iter = minimize(chi2, x0=np.array([1.54]), method='L-BFGS-B', tol=1e-6, options={}, callback=self._callback).x
            self.beta = np.append(self.beta, self.beta_iter)
            #print(self.beta_iter)
            self.comm.Barrier()

        else:
            
            
            _index_seenpix_beta = np.where(self.seenpix_beta == 1)[0]

            for i_index, index in enumerate(_index_seenpix_beta):
                chi2 = partial(self.chi2_tot_varying, patch_id=index, allbeta=self.beta_iter, solution=components_conv)
                
                if self.rank == 0:
                    print(f'Fitting pixel {index}')
                self.beta_iter[index, 0] = minimize(chi2, x0=np.array([1.54]), method='Nelder-Mead', tol=1e-5).x
            
            print(self.beta[_index_seenpix_beta, 0])
            print(self.beta_iter[_index_seenpix_beta, 0])
            self.comm.Barrier()
        print()
        print()
    def _save_data(self):
        
        """
        
        Method that save data for each iterations. It saves components, gains, spectral index, coverage, seen pixels.
        
        """
        
        if self.rank == 0:
            if self.params['save'] != 0:
                if (self._steps+1) % self.params['save'] == 0:
                    with open(self.params['foldername'] + f"_seed{str(self.params['CMB']['seed'])}" + '/' + self.params['filename']+f"_{self._steps+1}_{str(self.params['CMB']['iter'])}.pkl", 'wb') as handle:
                        pickle.dump({'components':self.components, 
                                 'components_i':self.components_iter,
                                 'beta':self.beta,
                                 'g':self.g,
                                 'gi':self.g_iter,
                                 'center':self.center,
                                 'coverage':self.coverage,
                                 'seenpix':self.seenpix}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def _update_components(self):
        
        """
        
        Method that solve the map-making equation ( H.T * invN * H ) * components = H.T * invN * TOD using OpenMP / MPI solver. 
        
        """
        
        self.H_i = self.joint.get_operator(self.beta_iter, gain=self.g_iter, fwhm=self.fwhm_recon, nu_co=self.nu_co)

        self.A = self.H_i.T * self.invN * self.H_i
        self.b = self.H_i.T * self.invN * self.TOD_obs

        self._call_pcg()
    def _call_pcg(self):

        """
        
        Method that call the PCG in PyOperators.
        
        """
        
        self.components_iter = pcg(self.A, 
                                   self.b, 
                                   M=self.M, 
                                   tol=self.params['MapMaking']['pcg']['tol'], 
                                   x0=self.components_iter, 
                                   maxiter=self.params['MapMaking']['pcg']['maxiter'], 
                                   disp=True,
                                   create_gif=False,
                                   center=self.center, 
                                   reso=self.params['MapMaking']['qubic']['dtheta'], 
                                   seenpix=self.seenpix, 
                                   truth=self.components)['x']['x']  
    def _stop_condition(self):
        
        """
        
        Method that stop the convergence if there are more than k steps.
        
        """
        
        if self._steps >= self.params['MapMaking']['pcg']['k']-1:
            self._info = False
            
        self._steps += 1
    def _display_iter(self):
        
        """
        
        Method that display the number of a specific iteration k.
        
        """
        
        if self.rank == 0:
            print('========== Iter {}/{} =========='.format(self._steps+1, self.params['MapMaking']['pcg']['k']))
    def _update_gain(self):
        
        """
        
        Method that compute gains of each detectors using semi-analytical method g_i = TOD_obs_i / TOD_sim_i
        
        """
        
        self.H_i = self.joint.get_operator(self.beta_iter, gain=np.ones(self.g_iter.shape), fwhm=self.fwhm_recon, nu_co=self.nu_co)
        
        if self.params['MapMaking']['qubic']['type'] == 'wide':
            R2det_i = ReshapeOperator(self.joint.qubic.ndets*self.joint.qubic.nsamples, (self.joint.qubic.ndets, self.joint.qubic.nsamples))
            #print(R2det_i.shapein, R2det_i.shapeout)
            TOD_Q_ALL_i = R2det_i(self.H_i.operands[0](self.components_iter))
        
            self.g_iter = self._give_me_intercal(TOD_Q_ALL_i, R2det_i(self.TOD_Q))
            self.g_iter /= self.g_iter[0]
            self.allg = np.concatenate((self.allg, np.array([self.g_iter])), axis=0)
            
        elif self.params['MapMaking']['qubic']['type'] == 'two':
            
            R2det_i = ReshapeOperator(2*self.joint.qubic.ndets*self.joint.qubic.nsamples, (2*self.joint.qubic.ndets, self.joint.qubic.nsamples))
            TODi_Q_150 = R2det_i(self.H_i.operands[0](self.components_iter))[:self.joint.qubic.ndets]
            TODi_Q_220 = R2det_i(self.H_i.operands[0](self.components_iter))[self.joint.qubic.ndets:2*self.joint.qubic.ndets]
            
            g150 = self._give_me_intercal(TODi_Q_150, self.TOD_Q_150)
            g220 = self._give_me_intercal(TODi_Q_220, self.TOD_Q_220)
            g150 /= g150[0]
            g220 /= g220[0]
            
            self.g_iter = np.array([g150, g220]).T
            self.allg = np.concatenate((self.allg, np.array([self.g_iter])), axis=0)
    def _give_me_intercal(self, D, d):
        
        """
        
        Semi-analytical method for gains estimation.

        """
        
        return 1/np.sum(D[:]**2, axis=1) * np.sum(D[:] * d[:], axis=1)

