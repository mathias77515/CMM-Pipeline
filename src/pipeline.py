import os
from functools import partial

import numpy as np
import healpy as hp
from scipy.optimize import minimize, fmin_l_bfgs_b

import pickle
import gc
from acquisition.Qacquisition import *

from simtools.foldertools import *
from simtools.mpi_tools import *
from simtools.noise_timeline import *
from simtools.analysis import *
 
from pyoperators import MPI
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

import preset

import fgb.mixing_matrix as mm
from solver.cg import (mypcg)
from plots.plots import *
from costfunc.chi2 import Chi2Parametric, Chi2Parametric_alt, Chi2Blind, Chi2DualBand, Chi2UltraWideBand       
                
class Pipeline:
    """
    Main instance to create End-2-End pipeline for components reconstruction.
    
    Arguments :
    -----------
        - comm    : MPI common communicator (define by MPI.COMM_WORLD).
        - seed    : Int number for CMB realizations.
        - it      : Int number for noise realizations.
    """
    
    def __init__(self, comm, seed, seed_noise=None):

        ### Creating noise seed
        if seed_noise == -1:
            if comm.Get_rank() == 0:
                seed_noise = np.random.randint(100000000)
            else:
                seed_noise = None
        seed_noise = comm.bcast(seed_noise, root=0)

        ### Initialization
        self.preset = preset.PresetInitialisation(comm, seed, seed_noise).initialize()
        self.plots = Plots(self.preset, dogif=True)
        if self.preset.fg.params_foregrounds['Dust']['type'] == 'blind' or self.preset.fg.params_foregrounds['Synchrotron']['type'] == 'blind':
           self.chi2 = Chi2Blind(self.preset)
        else:
           pass

        self.fsub = int(self.preset.qubic.joint_out.qubic.nsub / self.preset.fg.params_foregrounds['bin_mixing_matrix'])

        ### Create variables for stopping condition
        self._rms_noise_qubic_patch_per_ite = np.empty((self.preset.tools.params['PCG']['ites_to_converge'],len(self.preset.fg.components_name_out)))
        self._rms_noise_qubic_patch_per_ite[:] = np.nan
        
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
        #print(self.preset.acquisition.beta_iter)
        while self._info:
            ### Display iteration
            self.preset.tools._display_iter(self._steps)
            
            ### Update self.fg.components_iter^{k} -> self.fg.components_iter^{k+1}
            self._update_components()
            
            ### Update self.preset.acquisition.beta_iter^{k} -> self.preset.acquisition.beta_iter^{k+1}
            if self.preset.fg.params_foregrounds['fit_spectral_index']:
                self._update_spectral_index()
                
            ### Update self.gain.gain_iter^{k} -> self.gain.gain_iter^{k+1}
            if self.preset.qubic.params_qubic['GAIN']['fit_gain']:
                self._update_gain()
            
            ### Wait for all processes and save data inside pickle file
            self.preset.tools.comm.Barrier()
            self._save_data(self._steps)
            
            ### Compute the rms of the noise per iteration to later analyze its convergence in _stop_condition
            #self._compute_maxrms_array()

            ### Stop the loop when self._steps > k
            self._stop_condition()

    def _fisher(self, ell, Nl):
        """
        Fisher to compute an estimation of sigma(r) for a given noise power spectrum.
        
        Parameters:
        ell (array-like): Array of multipole values.
        Nl (array-like): Array of noise power spectrum values.
        
        Returns:
        float: The computed value of sigma(r).
        """
        
        Dl = np.interp(ell, np.arange(1, 4001, 1), self.preset.fg.give_cl_cmb(r=1, Alens=0)[2])
        s = np.sum((ell + 0.5) * self.preset.sky.namaster.fsky * self.preset.tools.params['SPECTRUM']['dl'] * (Dl / (Nl))**2)
        
        return s**(-1/2)
    def _fisher_compute_sigma_r(self):
        """
        Computes the value of sigma(r) using the Fisher matrix.

        Returns:
            float: The value of sigma(r).
        """
        # Apply Gaussian beam convolution
        C = HealpixConvolutionGaussianOperator(fwhm=self.preset.acquisition.fwhm_reconstructed)
        map_to_namaster = C(self.preset.fg.components_iter[0] - self.preset.fg.components_out[0])
 
        # Set unobserved pixels to zero
        map_to_namaster[~self.preset.sky.seenpix, :] = 0
        
        # Compute power spectra using NaMaster
        leff, cls, _ = self.preset.sky.namaster.get_spectra(map_to_namaster.T, beam_correction=np.rad2deg(self.preset.acquisition.fwhm_reconstructed), pixwin_correction=False, verbose=False)
        
        # Compute BB power spectrum and convert to delta ell
        dl_BB = cls[:, 2] / self.preset.sky.cl2dl
        
        # Compute sigma(r) using Fisher matrix
        sigma_r = self._fisher(leff, dl_BB)
        
        # Print the value of sigma(r)
        self.preset.tools._print_message(f'sigma(r) = {sigma_r:.6f}')
    def _call_pcg(self, max_iterations):
        """
        Method that calls the PCG in PyOperators.
        
        Args:
            max_iterations (int): Maximum number of iterations for the PCG algorithm.
        """

        ### Initialize PCG starting point
        #if self.preset.tools.params['PCG']['fix_pixels_outside_patch']:
        initial_maps = self.preset.fg.components_iter[:, self.preset.sky.seenpix, :].copy()
        #elif self.preset.tools.params['PCG']['fixI']:
        #    initial_maps = self.preset.fg.components_iter[:, :, 1:].copy()
        #else:
        #initial_maps = self.preset.fg.components_iter.copy()
        
        ### Run PCG
        self.preset.acquisition.M = self.preset.acquisition._get_preconditioner(A_qubic=self.preset.acquisition.Amm_iter[:self.preset.qubic.params_qubic['nsub_out']],
                                                  A_ext=self.preset.mixingmatrix.Amm_in[self.preset.qubic.params_qubic['nsub_out']:],
                                                  precond=self.preset.qubic.params_qubic['preconditionner'])
        
        #if self._steps > 0:
        #    self.preset.acquisition.M = None
        if self._steps == 0:
            maxiter = self.preset.tools.params['PCG']['n_init_iter_pcg']
        else:
            maxiter = max_iterations
        result = mypcg(self.preset.A, 
                        self.preset.b, 
                        M=self.preset.acquisition.M, 
                        tol=self.preset.tools.params['PCG']['tol_pcg'], 
                        x0=initial_maps, 
                        maxiter=maxiter, 
                        disp=True,
                        create_gif=True,
                        center=self.preset.sky.center, 
                        reso=self.preset.tools.params['PCG']['reso_plot'], 
                        seenpix=self.preset.sky.seenpix, 
                        seenpix_plot=self.preset.sky.seenpix_015, 
                        truth=self.preset.fg.components_out,
                        reuse_initial_state=False,
                        jobid=self.preset.job_id,
                        iter_init=self._steps*self.preset.tools.params['PCG']['n_iter_pcg'])['x']['x'] 
        
        ### Update components
        #if self.preset.tools.params['PCG']['fix_pixels_outside_patch']:
        self.preset.fg.components_iter[:, self.preset.sky.seenpix, :] = result.copy()
        #elif self.preset.tools.params['PCG']['fixI']:
        #    self.preset.fg.components_iter[:, :, 1:] = result
        #else:
        #    self.preset.fg.components_iter = result.copy()
    
        ### Method to compute an approximation of sigma(r) using Fisher matrix at the end of the PCG
        #self._fisher_compute_sigma_r()
        
        ### Plot if asked
        if self.preset.tools.rank == 0:
            do_gif(f'jobs/{self.preset.job_id}/allcomps/', 'iter_', output='animation.gif')
            self.plots.display_maps(self.preset.sky.seenpix_015, ki=self._steps)
            #self.plots._display_allcomponents(self.preset.sky.seenpix, ki=self._steps)
            #self.plots._display_allresiduals(self.preset.fg.components_iter[:, self.preset.sky.seenpix, :], self.preset.sky.seenpix, ki=self._steps)  
            self.plots.plot_rms_iteration(self.preset.acquisition.rms_plot, ki=self._steps) 
    def _update_components(self):
        """
        Method that solves the map-making equation ( H.T * invN * H ) * components = H.T * invN * TOD using OpenMP / MPI solver. 
        
        This method updates the components of the map by solving the map-making equation using an OpenMP / MPI solver. The equation is of the form ( H.T * invN * H ) * components = H.T * invN * TOD, where H_i is the operator obtained from the preset, U is a reshaped operator, and x_planck and xI are intermediate variables used in the calculations.
        
        Parameters:
        - self: The instance of the class.
        
        Returns:
        - None
        """
        
        H_i = self.preset.qubic.joint_out.get_operator(A=self.preset.acquisition.Amm_iter, gain=self.preset.gain.gain_iter, fwhm=self.preset.acquisition.fwhm_mapmaking, nu_co=self.preset.fg.nu_co)

        U = (
            ReshapeOperator((len(self.preset.fg.components_name_out) * sum(self.preset.sky.seenpix) * 3), (len(self.preset.fg.components_name_out), sum(self.preset.sky.seenpix), 3)) *
            PackOperator(np.broadcast_to(self.preset.sky.seenpix[None, :, None], (len(self.preset.fg.components_name_out), self.preset.sky.seenpix.size, 3)).copy())
            ).T
     
        ### Update components when pixels outside the patch are fixed (assumed to be 0)
        self.preset.A = U.T * H_i.T * self.preset.acquisition.invN * H_i * U

        if self.preset.qubic.params_qubic['convolution_out']:
            x_planck = self.preset.fg.components_convolved_out * (1 - self.preset.sky.seenpix[None, :, None])
        else:
            x_planck = self.preset.fg.components_out * (1 - self.preset.sky.seenpix[None, :, None])
        self.preset.b = U.T (  H_i.T * self.preset.acquisition.invN * (self.preset.acquisition.TOD_obs - H_i(x_planck)))

        # TO BE REMOVE
        ### Update components when intensity maps are fixed
        #elif self.preset.tools.params['PCG']['fixI']:
        #    mask = np.ones((len(self.preset.fg.components_name_out), 12*self.preset.sky.params_sky['nside']**2, 3))
        #    mask[:, :, 0] = 0
        #    P = (
        #        ReshapeOperator(PackOperator(mask).shapeout, (len(self.preset.fg.components_name_out), 12*self.preset.sky.params_sky['nside']**2, 2)) * 
        #        PackOperator(mask)
        #        ).T
            
        #    xI = self.preset.fg.components_convolved_out * (1 - mask)
        #    self.preset.A = P.T * H_i.T * self.preset.acquisition.invN * H_i * P
        #    self.preset.b = P.T (H_i.T * self.preset.acquisition.invN * (self.preset.acquisition.TOD_obs - H_i(xI)))
            
        ### Update components
        #else:
        #self.preset.A = H_i.T * self.preset.acquisition.invN * H_i
        #self.preset.b = H_i.T * self.preset.acquisition.invN * self.preset.acquisition.TOD_obs
        
        ### Run PCG
        self._call_pcg(self.preset.tools.params['PCG']['n_iter_pcg'])
    def _get_tod_comp(self):
        """
        Method that produces Time-Ordered Data (TOD) using the component maps computed at the current iteration.

        This method initializes a zero-filled numpy array `tod_comp` with dimensions based on the number of components,
        the number of sub-components (multiplied by 2), and the product of the number of detectors and samples.
        It then iterates over each component and sub-component to compute the TOD by applying a convolution operator
        (if specified) and a mapping operator to the component maps.

        Returns:
            np.ndarray (Ncomp, nsub, Npix): A numpy array containing the computed TOD for each component and sub-component.
        """

        tod_comp = np.zeros((len(self.preset.fg.components_name_out), self.preset.qubic.joint_out.qubic.nsub, self.preset.qubic.joint_out.qubic.ndets*self.preset.qubic.joint_out.qubic.nsamples))
        
        for i in range(len(self.preset.fg.components_name_out)):
            for j in range(self.preset.qubic.joint_out.qubic.nsub):
                if self.preset.qubic.params_qubic['convolution_out']:
                    C = HealpixConvolutionGaussianOperator(fwhm = self.preset.acquisition.fwhm_mapmaking[j], lmax=3*self.preset.sky.params_sky['nside'])
                else:
                    C = HealpixConvolutionGaussianOperator(fwhm = 0, lmax=3*self.preset.sky.params_sky['nside'])
                tod_comp[i, j] = self.preset.qubic.joint_out.qubic.H[j](C(self.preset.fg.components_iter[i])).ravel()
        
        return tod_comp  
    def _callback(self, x):
        """
        Method to make callback function readable by `scipy.optimize.minimize`.

        This method is intended to be used as a callback function during the optimization
        process. It is called by the optimizer at each iteration.

        Parameters:
        x : array_like
            The current parameter values at the current iteration of the optimization.

        The method performs the following actions:
        1. Synchronizes all processes using a barrier to ensure that all processes reach this point before proceeding.
        2. If the current process is the root process (rank 0), it performs the following:
            a. Every 5 iterations (when `self.nfev` is a multiple of 5), it prints the current iteration number and the parameter values rounded to 5 decimal places.
        3. Increments the iteration counter `self.nfev` by 1.
        """
        
        self.preset.tools.comm.Barrier()
        if self.preset.tools.rank == 0:
            if (self.nfev%1) == 0:
                print(f"Iter = {self.nfev:4d}   x = {[np.round(x[i], 5) for i in range(len(x))]}   qubic log(L) = {np.round(self.chi2.Lqubic, 5)}  planck log(L) = {np.round(self.chi2.Lplanck, 5)}")
            self.nfev += 1
    def _get_tod_comp_superpixel(self, index):
        if self.preset.tools.rank == 0:
            print('Computing contribution of each super-pixel')
        _index = np.zeros(12*self.preset.fg.params_foregrounds['Dust']['nside_beta_out']**2)
        _index[index] = index.copy()
        _index_nside = hp.ud_grade(_index, self.preset.qubic.joint_out.external.nside)
        tod_comp = np.zeros((len(index), self.preset.qubic.joint_out.qubic.nsub, len(self.preset.fg.components_name_out), self.preset.qubic.joint_out.qubic.ndets*self.preset.qubic.joint_out.qubic.nsamples))
        
        maps_conv = self.preset.fg.components_iter.copy()

        for j in range(self.preset.qubic.params_qubic['nsub_out']):
            for icomp in range(len(self.preset.fg.components_name_out)):
                if self.preset.qubic.params_qubic['convolution_out']:
                    C = HealpixConvolutionGaussianOperator(fwhm=self.preset.acquisition.fwhm_mapmaking[j], lmax=3*self.preset.sky.params_sky['nside'])
                else:
                    C = HealpixConvolutionGaussianOperator(fwhm=0, lmax=3*self.preset.sky.params_sky['nside'])
                maps_conv[icomp] = C(self.preset.fg.components_iter[icomp, :, :]).copy()
                for ii, i in enumerate(index):
                    maps_conv_i = maps_conv.copy()
                    _i = _index_nside == i
                    for stk in range(3):
                        maps_conv_i[:, :, stk] *= _i
                    tod_comp[ii, j, icomp] = self.preset.qubic.joint_out.qubic.H[j](maps_conv_i[icomp]).ravel()
        return tod_comp
    def _get_constrains(self):
        """
        Generate constraints readable by `scipy.optimize.minimize`.

        Return:
        constraints : list
            A list of constraint dictionaries for optimize.minimize.
        """

        constraints = []
        n = (self.preset.fg.params_foregrounds['bin_mixing_matrix']-1)*(len(self.preset.fg.components_name_out)-1)

        ### Dust only : constraint ==> SED is increasing
        if self.preset.fg.params_foregrounds['Dust']['Dust_out'] and not self.preset.fg.params_foregrounds['Synchrotron']['Synchrotron_out']:
            for i in range(n):
                constraints.append(
                                    {'type': 'ineq', 'fun': lambda x, i=i: x[i+1] - x[i]}
                                  )
        
        ### Synchrotron only : constraint ==> SED is decreasing
        elif not self.preset.fg.params_foregrounds['Dust']['Dust_out'] and self.preset.fg.params_foregrounds['Synchrotron']['Synchrotron_out']:
            for i in range(n):
                constraints.append(
                                    {'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[i+1]}
                                  )
        
        ### No component : constraint ==> None
        elif not self.preset.fg.params_foregrounds['Dust']['Dust_out'] and not self.preset.fg.params_foregrounds['Synchrotron']['Synchrotron_out']:
            return None

        ### Dust & Synchrotron : constraint ==> SED is increasing for one component and decrasing for the other one
        elif self.preset.fg.params_foregrounds['Dust']['Dust_out'] and self.preset.fg.params_foregrounds['Synchrotron']['Synchrotron_out']:
            for i in range(n): 
                # Dust
                if i % 2 == 0:
                    constraints.append(
                                        {'type': 'ineq', 'fun': lambda x, i=i: x[i+2] - x[i]}
                                      )
                # Sync
                else:
                    constraints.append(
                                        {'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[i+2]}
                                      )
        
        return constraints 
    def _update_mixing_matrix(self, beta, previous_mixingmatrix, icomp):
        """
        Method to update the mixing matrix using the current fitted value of the beta parameter and the parametric model associated.
        Only use when hybrid parametric-blind fit is selected !

        Arguments:
        - beta: int
        - previous_mixingmatrix: ndarray (nsub_in + Nintegr * Nplanck, Ncomp)
        - icomp: int

        Return:
        - updated_mixingmatrix: ndarray (nsub_in + Nintegr * Nplanck, Ncomp)
        """
        
        ### Build mixing matrix according to the choosen model and the beta parameter
        mixingmatrix = mm.MixingMatrix(*self.preset.fg.components_out)
        model_mixingmatrix = mixingmatrix.eval(self.preset.qubic.joint_out.qubic.allnus, *beta)

        ### Update the mixing matrix according to the one computed using the beta parameter
        updated_mixingmatrix = previous_mixingmatrix
        for ii in range(self.preset.fg.params_foregrounds['bin_mixing_matrix']):
            updated_mixingmatrix[ii*self.fsub: (ii + 1)*self.fsub, icomp] = model_mixingmatrix[ii*self.fsub: (ii + 1)*self.fsub, icomp]

        return updated_mixingmatrix
    def _update_spectral_index(self):
        """
        
        Method that perform step 3) of the pipeline for 2 possible designs : Two Bands and Ultra Wide Band
        
        """
        method_0 = self.preset.fg.params_foregrounds[self.preset.fg.components_name_out[1]]['type']
        if len(self.preset.fg.components_name_out) > 1:
            cpt = 2
            while cpt < len(self.preset.fg.components_name_out):
                if self.preset.fg.components_name_out[cpt] != 'CO' and self.preset.fg.params_foregrounds[self.preset.fg.components_name_out[cpt]]['type'] != method_0:
                    method = 'parametric_blind'
                cpt+=1
        try :
            method == 'parametric_blind'
        except :
            method = method_0

        tod_comp = self._get_tod_comp()
        self.nfev = 0
        self.preset.mixingmatrix._index_seenpix_beta = 0

        if method == 'parametric':
            if self.preset.fg.params_foregrounds['Dust']['nside_beta_out'] == 0:

                previous_beta = self.preset.acquisition.beta_iter.copy()
                
                if self.preset.qubic.params_qubic['instrument'] == 'DB':
                    self.chi2 = Chi2DualBand(self.preset, tod_comp, parametric=True)
                elif self.preset.qubic.params_qubic['instrument'] == 'UWB':
                    self.chi2 = Chi2UltraWideBand(self.preset, tod_comp, parametric=True)

                self.preset.acquisition.beta_iter = minimize(self.chi2, x0=self.preset.acquisition.beta_iter, method='BFGS', callback=self._callback, tol=1e-10).x

                self.preset.acquisition.Amm_iter = self.chi2._get_mixingmatrix(nus=self.preset.qubic.joint_out.allnus, x=self.preset.acquisition.beta_iter)
                #print(Ai.shape, Ai)
                #for inu in range(self.preset.qubic.joint_out.qubic.nsub):
                #    for icomp in range(1, len(self.preset.fg.components_name_out)):
                #        self.preset.acquisition.Amm_iter[inu, icomp] = Ai[inu, icomp]
                
                del tod_comp
                gc.collect()
                
                if self.preset.tools.rank == 0:
                
                    print(f'Iteration k     : {previous_beta}')
                    print(f'Iteration k + 1 : {self.preset.acquisition.beta_iter.copy()}')
                    print(f'Truth           : {self.preset.mixingmatrix.beta_in.copy()}')
                    print(f'Residuals       : {self.preset.mixingmatrix.beta_in - self.preset.acquisition.beta_iter}')
                    
                    #if len(self.preset.fg.components_name_out) > 2:
                    #    self.plots.plot_beta_iteration(self.preset.acquisition.allbeta[:, 0], truth=self.preset.mixingmatrix.beta_in[0], ki=self._steps)
                    #else:
                    #    self.plots.plot_beta_iteration(self.preset.acquisition.allbeta, truth=self.preset.mixingmatrix.beta_in, ki=self._steps)
            
                self.preset.tools.comm.Barrier()

                self.preset.acquisition.allbeta = np.concatenate((self.preset.acquisition.allbeta, np.array([self.preset.acquisition.beta_iter])), axis=0) 
            
            else:
            
                index_num = hp.ud_grade(self.preset.sky.seenpix_qubic, self.preset.fg.params_foregrounds['Dust']['nside_beta_out'])    #
                self.preset.mixingmatrix._index_seenpix_beta = np.where(index_num == True)[0]

                ### Simulated TOD for each components, nsub, npix with shape (npix, nsub, ncomp, nsnd)
                tod_comp = self._get_tod_comp_superpixel(self.preset.mixingmatrix._index_seenpix_beta)
                
                ### Store fixed beta (those denoted with hp.UNSEEN are variable)
                beta_fixed = self.preset.acquisition.beta_iter.copy()
                beta_fixed[:, self.preset.mixingmatrix._index_seenpix_beta] = hp.UNSEEN
                chi2 = Chi2DualBand(self.preset, tod_comp, parametric=True, full_beta_map=beta_fixed)
                #chi2 = Chi2Parametric(self.preset, tod_comp, self.preset.acquisition.beta_iter, seenpix_wrap=None)

                previous_beta = self.preset.acquisition.beta_iter[:, self.preset.mixingmatrix._index_seenpix_beta].copy()
                self.nfev = 0
                
                beta_i = fmin_l_bfgs_b(chi2, x0=previous_beta, callback=self._callback, approx_grad=True, epsilon=1e-6, maxls=20, maxiter=20)[0]

                self.preset.acquisition.beta_iter[chi2.seenpix_beta] = beta_i
                
                del tod_comp
                gc.collect()
                
                self.preset.acquisition.allbeta = np.concatenate((self.preset.acquisition.allbeta, np.array([self.preset.acquisition.beta_iter])), axis=0)
                
                if self.preset.tools.rank == 0:
                
                    print(f'Iteration k     : {previous_beta}')
                    print(f'Iteration k + 1 : {self.preset.acquisition.beta_iter[:, self.preset.mixingmatrix._index_seenpix_beta].copy()}')
                    print(f'Truth           : {self.preset.mixingmatrix.beta_in[:, self.preset.mixingmatrix._index_seenpix_beta].copy()}')
                    print(f'Residuals       : {self.preset.mixingmatrix.beta_in[:, self.preset.mixingmatrix._index_seenpix_beta] - self.preset.acquisition.beta_iter[:, self.preset.mixingmatrix._index_seenpix_beta]}')
                    self.plots.plot_beta_iteration(self.preset.acquisition.allbeta[:, :, self.preset.mixingmatrix._index_seenpix_beta], 
                                                   truth=self.preset.mixingmatrix.beta_in[:, self.preset.mixingmatrix._index_seenpix_beta], 
                                                   ki=self._steps)  
        elif method == 'blind':
            previous_step = self.preset.acquisition.Amm_iter[:self.preset.qubic.joint_out.qubic.nsub, 1:].copy()
            if self._steps == 0:
                self.allAmm_iter = np.array([self.preset.acquisition.Amm_iter]) 

            
            if self.preset.fg.params_foregrounds['blind_method'] == 'minimize' :
                
                if self.preset.qubic.params_qubic['instrument'] == 'DB':
                    self.chi2 = Chi2DualBand(self.preset, tod_comp, parametric=False)
                elif self.preset.qubic.params_qubic['instrument'] == 'UWB':
                    self.chi2 = Chi2UltraWideBand(self.preset, tod_comp, parametric=False)
                
                
                
                x0 = []
                bnds = []
                for inu in range(self.preset.fg.params_foregrounds['bin_mixing_matrix']):
                    for icomp in range(1, len(self.preset.fg.components_name_out)):
                        x0 += [np.mean(self.preset.acquisition.Amm_iter[inu*self.fsub:(inu+1)*self.fsub, icomp])]
                        bnds += [(0, None)]
                
                Ai = minimize(self.chi2, 
                                   x0=x0, 
                                   bounds=bnds, 
                                   method='L-BFGS-B',
                                   #constraints=self._get_constrains(),
                                   callback=self._callback,
                                   tol=1e-12).x
                Ai = self.chi2._fill_A(Ai)#Ai.reshape((self.preset.qubic.joint_out.qubic.nsub, len(self.preset.fg.components_name_out)-1))
                
                for inu in range(self.preset.qubic.joint_out.qubic.nsub):
                    for icomp in range(1, len(self.preset.fg.components_name_out)):
                        self.preset.acquisition.Amm_iter[inu, icomp] = Ai[inu, icomp]

                
                '''
                ### Function to minimize
                fun = partial(self.chi2._qu, tod_comp=tod_comp)
                
                ### Starting point
                x0 = []
                bnds = []
                for ii in range(self.preset.fg.params_foregrounds['bin_mixing_matrix']):
                    for i in range(1, len(self.preset.fg.components_name_out)):
                        x0 += [np.mean(self.preset.acquisition.Amm_iter[ii*self.fsub:(ii+1)*self.fsub, i])]
                        bnds += [(0, None)]
                if self._steps == 0:
                    x0 = np.array(x0) * 1 + 0
                ### Constraints on frequency evolution
                constraints = self._get_constrains()
                
                Ai = minimize(fun, x0=x0, 
                            #constraints=constraints, 
                            callback=self._callback, 
                            bounds=bnds, 
                            method='L-BFGS-B', 
                            tol=1e-10).x
                
                k=0
                for ii in range(self.preset.fg.params_foregrounds['bin_mixing_matrix']):
                    for i in range(1, len(self.preset.fg.components_name_out)):
                        self.preset.acquisition.Amm_iter[ii*self.fsub:(ii+1)*self.fsub, i] = Ai[k]
                        k+=1
                '''
                

            elif self.preset.fg.params_foregrounds['blind_method'] == 'PCG' :
                tod_comp_binned = np.zeros((tod_comp.shape[0], self.preset.fg.params_foregrounds['bin_mixing_matrix'], tod_comp.shape[-1]))
                for k in range(len(self.preset.fg.components_name_out)):
                    for i in range(self.preset.fg.params_foregrounds['bin_mixing_matrix']):
                        tod_comp_binned[k, i] = np.sum(tod_comp[k, i*self.fsub:(i+1)*self.fsub], axis=0)
            
                tod_cmb150 = self.preset.tools.comm.allreduce(np.sum(tod_comp[0, :int(tod_comp.shape[1]/2)], axis=0), op=MPI.SUM)
                tod_cmb220 = self.preset.tools.comm.allreduce(np.sum(tod_comp[0, int(tod_comp.shape[1]/2):int(tod_comp.shape[1])], axis=0), op=MPI.SUM)
            
                tod_in_150 = self.preset.tools.comm.allreduce(self.preset.TOD_Q[:int(self.preset.TOD_Q.shape[0]/2)], op=MPI.SUM)
                tod_in_220 = self.preset.tools.comm.allreduce(self.preset.TOD_Q[int(self.preset.TOD_Q.shape[0]/2):int(self.preset.TOD_Q.shape[0])], op=MPI.SUM)
            
                tod_without_cmb = np.r_[tod_in_150 - tod_cmb150, tod_in_220 - tod_cmb220]
                tod_without_cmb_reshaped = np.sum(tod_without_cmb.reshape((2, int(self.preset.nsnd/2))), axis=0)

                dnu = self.preset.tools.comm.allreduce(tod_comp_binned[1:], op=MPI.SUM)
                dnu = dnu.reshape((dnu.shape[0]*dnu.shape[1], dnu.shape[2]))
  
                A = dnu @ dnu.T
                b = dnu @ tod_without_cmb_reshaped
            
                s = mypcg(A, b, disp=False, tol=1e-20, maxiter=10000)['x']
            
                k=0
                for i in range(1, len(self.preset.fg.components_name_out)):
                    for ii in range(self.preset.fg.params_foregrounds['bin_mixing_matrix']):
                        self.preset.acquisition.Amm_iter[ii*self.fsub:(ii+1)*self.fsub, i] = s['x'][k]#Ai[k]
                        k+=1
                
            elif self.preset.fg.params_foregrounds['blind_method'] == 'alternate' :
                for i in range(len(self.preset.fg.components_name_out)):
                    if self.preset.fg.components_name_out[i] != 'CMB':
                        print('I am fitting ', self.preset.fg.components_name_out[i])
                        fun = partial(self.chi2._qu_alt, tod_comp=tod_comp, A=self.preset.acquisition.Amm_iter, icomp=i)
                
                        ### Starting point
                        x0 = []
                        bnds = []
                        for ii in range(self.preset.fg.params_foregrounds['bin_mixing_matrix']):
                            x0 += [np.mean(self.preset.acquisition.Amm_iter[ii*self.fsub:(ii+1)*self.fsub, i])]
                            bnds += [(0, None)]
                        if self._steps == 0:
                            x0 = np.array(x0) * 1 + 0

                        Ai = minimize(fun, x0=x0,
                                callback=self._callback, 
                                bounds=bnds, 
                                method='SLSQP', 
                                tol=1e-10).x
  
                        for ii in range(self.preset.fg.params_foregrounds['bin_mixing_matrix']):
                            self.preset.acquisition.Amm_iter[ii*self.fsub:(ii+1)*self.fsub, i] = Ai[ii]
                
            else:
                raise TypeError(f"{self.preset.fg.params_foregrounds['blind_method']} is not yet implemented..")           

            self.allAmm_iter = np.concatenate((self.allAmm_iter, np.array([self.preset.acquisition.Amm_iter])), axis=0)
            
            if self.preset.tools.rank == 0:
                print(f'Iteration k     : {previous_step.ravel()}')
                print(f'Iteration k + 1 : {self.preset.acquisition.Amm_iter[:self.preset.qubic.joint_out.qubic.nsub, 1:].ravel()}')
                print(f'Truth           : {self.preset.mixingmatrix.Amm_in[:self.preset.qubic.joint_out.qubic.nsub, 1:].ravel()}')
                print(f'Residuals       : {self.preset.mixingmatrix.Amm_in[:self.preset.qubic.joint_out.qubic.nsub, 1:].ravel() - self.preset.acquisition.Amm_iter[:self.preset.qubic.joint_out.qubic.nsub, 1:].ravel()}')
                self.plots.plot_sed(self.preset.qubic.joint_out.qubic.allnus, 
                                    self.allAmm_iter[:, :self.preset.qubic.joint_out.qubic.nsub, 1:], 
                                    ki=self._steps, truth=self.preset.mixingmatrix.Amm_in[:self.preset.qubic.joint_out.qubic.nsub, 1:])

            del tod_comp
            gc.collect()
        elif method == 'parametric_blind':
            previous_step = self.preset.acquisition.Amm_iter[:self.preset.qubic.joint_out.qubic.nsub*2, 1:].copy()
            if self._steps == 0:
                self.allAmm_iter = np.array([self.preset.acquisition.Amm_iter]) 
            for i in range(len(self.preset.fg.components_name_out)):
                if self.preset.fg.components_name_out[i] != 'CMB':
                    if self.preset.fg.params_foregrounds[self.preset.fg.components_name_out[i]]['type'] == 'parametric':
                        print('I am fitting ', self.preset.fg.components_name_out[i], i)

                        #if self._steps==0:
                        #    self.preset.acquisition.beta_iter = self.preset.acquisition.beta_iter
                        previous_beta = self.preset.acquisition.beta_iter.copy()
                            
                        chi2 = Chi2Parametric_alt(self.preset, tod_comp, self.preset.acquisition.Amm_iter, i, seenpix_wrap=None)
                    
                        self.preset.acquisition.beta_iter[i-1] = np.array([fmin_l_bfgs_b(chi2, 
                                                                        x0 = self.preset.acquisition.beta_iter[i-1], callback=self._callback, approx_grad=True, epsilon=1e-6)[0]])

                        self.preset.acquisition.Amm_iter = self._update_mixing_matrix(self.preset.acquisition.beta_iter, self.preset.acquisition.Amm_iter, i)
                    
                    else:
                        print('I am fitting ', self.preset.fg.components_name_out[i], i)

                        fun = partial(self.chi2._qu_alt, tod_comp=tod_comp, A=self.preset.acquisition.Amm_iter, icomp=i)
                
                        ### Starting point
                        x0 = []
                        bnds = []
                        for ii in range(self.preset.fg.params_foregrounds['bin_mixing_matrix']):
                            for j in range(1, len(self.preset.fg.components_name_out)):
                                x0 += [np.mean(self.preset.acquisition.Amm_iter[ii*self.fsub:(ii+1)*self.fsub, j])]
                                bnds += [(0, None)]

                        Ai = minimize(fun, x0=x0,
                                callback=self._callback, 
                                bounds=bnds, 
                                method='SLSQP', 
                                tol=1e-10).x
                        
                        for ii in range(self.preset.fg.params_foregrounds['bin_mixing_matrix']):
                            self.preset.acquisition.Amm_iter[ii*self.fsub:(ii+1)*self.fsub, i] = Ai[ii]                

            self.allAmm_iter = np.concatenate((self.allAmm_iter, np.array([self.preset.acquisition.Amm_iter])), axis=0)
        
            if self.preset.tools.rank == 0:
                print(f'Iteration k     : {previous_step.ravel()}')
                print(f'Iteration k + 1 : {self.preset.acquisition.Amm_iter[:self.preset.qubic.joint_out.qubic.nsub*2, 1:].ravel()}')
                print(f'Truth           : {self.preset.mixingmatrix.Amm_in[:self.preset.qubic.joint_out.qubic.nsub*2, 1:].ravel()}')
                print(f'Residuals       : {self.preset.mixingmatrix.Amm_in[:self.preset.qubic.joint_out.qubic.nsub*2, 1:].ravel() - self.preset.acquisition.Amm_iter[:self.preset.qubic.joint_out.qubic.nsub*2, 1:].ravel()}')
                self.plots.plot_sed(self.preset.qubic.joint_out.qubic.allnus, 
                                    self.allAmm_iter[:, :self.preset.qubic.joint_out.qubic.nsub*2, 1:], 
                                    ki=self._steps, truth=self.preset.mixingmatrix.Amm_in[:self.preset.qubic.joint_out.qubic.nsub*2, 1:])

            del tod_comp
            gc.collect()   
    def _give_me_intercal(self, D, d, _invn):   
        """
        Semi-analytical method for gains estimation. (cf CMM paper)

        Arguments:
        - D = HAc
            -> H: pointing matrix
            -> A: mixing matrix
            -> c: component vector
        - d = GD + n

        """
        
        _r = ReshapeOperator(self.preset.qubic.joint_out.qubic.ndets*self.preset.qubic.joint_out.qubic.nsamples, 
                             (self.preset.qubic.joint_out.qubic.ndets, self.preset.qubic.joint_out.qubic.nsamples))
        
        return (1/np.sum(_r(D) * _invn(_r(D)), axis=1)) * np.sum(_r(D) * _invn(_r(d)), axis=1)    
    def _update_gain(self):
        
        """
        
        Method that compute gains of each detectors using semi-analytical method g_i = TOD_obs_i / TOD_sim_i
        
        """
        
        self.H_i = self.preset.qubic.joint_out.get_operator(self.preset.acquisition.beta_iter, Amm=self.preset.acquisition.Amm_iter, gain=np.ones(self.preset.gain.gain_iter.shape), fwhm=self.preset.acquisition.fwhm_mapmaking, nu_co=self.preset.fg.nu_co)
        self.nsampling = self.preset.qubic.joint_out.qubic.nsamples
        self.ndets = self.preset.qubic.joint_out.qubic.ndets

        if self.preset.qubic.params_qubic['instrument'] == 'UWB':
            _r = ReshapeOperator(self.preset.qubic.joint_out.qubic.ndets*self.preset.joint.qubic.nsamples, (self.preset.joint.qubic.ndets, self.preset.joint.qubic.nsamples))

            TODi_Q = self.preset.acquisition.invN.operands[0](self.H_i.operands[0](self.preset.fg.components_iter)[:self.ndets*self.nsampling])
            self.preset.gain.gain_iter = self._give_me_intercal(TODi_Q, _r(self.preset.TOD_Q))
            self.preset.gain.gain_iter /= self.preset.gain.gain_iter[0]
            self.preset.allg = np.concatenate((self.preset.allg, np.array([self.preset.gain.gain_iter])), axis=0)
            
        elif self.preset.qubic.params_qubic['instrument'] == 'DB':
            
            TODi_Q_150 = self.H_i.operands[0](self.preset.fg.components_iter)[:self.ndets*self.nsampling]
            TODi_Q_220 = self.H_i.operands[0](self.preset.fg.components_iter)[self.ndets*self.nsampling:2*self.ndets*self.nsampling]
            
            g150 = self._give_me_intercal(TODi_Q_150, self.preset.TOD_Q[:self.ndets*self.nsampling], self.preset.acquisition.invN.operands[0].operands[1].operands[0])
            g220 = self._give_me_intercal(TODi_Q_220, self.preset.TOD_Q[self.ndets*self.nsampling:2*self.ndets*self.nsampling], self.preset.acquisition.invN.operands[0].operands[1].operands[1])
            
            self.preset.gain.gain_iter = np.array([g150, g220]).T
            self.preset.Gi = join_data(self.preset.tools.comm, self.preset.gain.gain_iter)
            self.preset.allg = np.concatenate((self.preset.allg, np.array([self.preset.gain.gain_iter])), axis=0)

            if self.preset.tools.rank == 0:
                print(np.mean(self.preset.gain.gain_iter - self.preset.g, axis=0))
                print(np.std(self.preset.gain.gain_iter - self.preset.g, axis=0))

        self.plots.plot_gain_iteration(self.preset.allg - self.preset.g, alpha=0.03, ki=self._steps)
    def _save_data(self, step):
        
        """
        
        Method that save data for each iterations. It saves components, gains, spectral index, coverage, seen pixels.
        
        """
        if self.preset.tools.rank == 0:
            if self.preset.tools.params['save_iter'] != 0:
                if (step+1) % self.preset.tools.params['save_iter'] == 0:
                    
                    if self.preset.tools.params['lastite']:
                    
                        if step != 0:
                            os.remove(self.preset.tools.params['foldername'] + '/' + self.preset.tools.params['filename']+  f"_seed{str(self.preset.tools.params['CMB']['seed'])}_{str(self.preset.job_id)}_k{step-1}.pkl")
                    
                    with open(self.preset.tools.params['foldername'] + '/' + self.preset.tools.params['filename'] + f"_seed{str(self.preset.tools.params['CMB']['seed'])}_{str(self.preset.job_id)}_k{step}.pkl", 'wb') as handle:
                        pickle.dump({'components':self.preset.fg.components_in, 
                                 'components_i':self.preset.fg.components_iter,
                                 'beta':self.preset.acquisition.allbeta,
                                 'beta_true':self.preset.mixingmatrix.beta_in,
                                 'index_beta':self.preset.mixingmatrix._index_seenpix_beta,
                                 'g':self.preset.gain.all_gain_in,
                                 'gi':self.preset.gain.all_gain,
                                 'allg':self.preset.gain.all_gain_iter,
                                 'A':self.preset.acquisition.Amm_iter,
                                 'Atrue':self.preset.mixingmatrix.Amm_in,
                                 'G':self.preset.gain.all_gain_in,
                                 'nus_in':self.preset.mixingmatrix.nus_eff_in,
                                 'nus_out':self.preset.mixingmatrix.nus_eff_out,
                                 'center':self.preset.sky.center,
                                 'coverage':self.preset.sky.coverage,
                                 'seenpix':self.preset.sky.seenpix,
                                 'fwhm':self.preset.acquisition.fwhm_tod,
                                 'acquisition.fwhm_reconstructed':self.preset.acquisition.fwhm_mapmaking}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def _compute_map_noise_qubic_patch(self):
        
        """
        
        Compute the rms of the noise within the qubic patch.
        
        """
        nbins = 1 #average over the entire qubic patch

        # if self.preset.fg.params_foregrounds['Dust']['nside_beta_out'] == 0:
        if self.preset.qubic.params_qubic['convolution_out']:
            residual = self.preset.fg.components_iter - self.preset.fg.components_convolved_out
        else:
            residual = self.preset.fg.components_iter - self.preset.fg.components_out
        # else:
        #     if self.preset.qubic.params_qubic['convolution_out']:
        #         residual = self.preset.fg.components_iter.T - self.preset.fg.components_convolved_out
        #     else:
        #         residual = self.preset.fg.components_iter.T - self.preset.fg.components_out.T
        rms_maxpercomp = np.zeros(len(self.preset.fg.components_name_out))

        for i in range(len(self.preset.fg.components_name_out)):
            angs,I,Q,U,dI,dQ,dU = get_angular_profile(residual[i],thmax=self.preset.angmax,nbins=nbins,doplot=False,allstokes=True,separate=True,integrated=True,center=self.preset.sky.center)
                
            ### Set dI to 0 to only keep polarization fluctuations 
            dI = 0
            rms_maxpercomp[i] = np.max([dI,dQ,dU])
        return rms_maxpercomp
    def _compute_maxrms_array(self):

        if self._steps <= self.preset.tools.params['PCG']['ites_to_converge']-1:
            self._rms_noise_qubic_patch_per_ite[self._steps,:] = self._compute_map_noise_qubic_patch()
        elif self._steps > self.preset.tools.params['PCG']['ites_to_converge']-1:
            self._rms_noise_qubic_patch_per_ite[:-1,:] = self._rms_noise_qubic_patch_per_ite[1:,:]
            self._rms_noise_qubic_patch_per_ite[-1,:] = self._compute_map_noise_qubic_patch()
    def _stop_condition(self):
        
        """
        
        Method that stop the convergence if there are more than k steps.
        
        """
        
        if self._steps >= self.preset.tools.params['PCG']['ites_to_converge']-1:
            
            deltarms_max_percomp = np.zeros(len(self.preset.fg.components_name_out))

            for i in range(len(self.preset.fg.components_name_out)):
                deltarms_max_percomp[i] = np.max(np.abs((self._rms_noise_qubic_patch_per_ite[:,i] - self._rms_noise_qubic_patch_per_ite[-1,i]) / self._rms_noise_qubic_patch_per_ite[-1,i]))

            deltarms_max = np.max(deltarms_max_percomp)
            if self.preset.tools.rank == 0:
                print(f'Maximum RMS variation for the last {self.preset.acquisition.ites_rms_tolerance} iterations: {deltarms_max}')

            if deltarms_max < self.preset.tools.params['PCG']['tol_rms']:
                print(f'RMS variations lower than {self.preset.acquisition.rms_tolerance} for the last {self.preset.acquisition.ites_rms_tolerance} iterations.')
                
                ### Update components last time with converged parameters
                #self._update_components(maxiter=100)
                self._info = False        

        if self._steps >= self.preset.tools.params['PCG']['n_iter_loop']-1:
            
            ### Update components last time with converged parameters
            #self._update_components(maxiter=100)
            
            ### Wait for all processes and save data inside pickle file
            #self.preset.tools.comm.Barrier()
            #self._save_data()
            
            self._info = False
            
        self._steps += 1