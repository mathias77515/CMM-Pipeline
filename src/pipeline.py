import os
from functools import partial

import numpy as np
import healpy as hp
from scipy.optimize import minimize, fmin_l_bfgs_b
import time

import pickle
import gc
from acquisition.Qacquisition import *

from simtools.mpi_tools import *
from simtools.noise_timeline import *
from simtools.analysis import *
 
from pyoperators import MPI
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

import preset

import fgb.mixing_matrix as mm
from solver.cg import (mypcg)
from plots.plots import *
from costfunc.chi2 import Chi2Parametric, Chi2Parametric_alt, Chi2Blind             
                
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
        self.plots = Plots(self.preset)
        if self.preset.fg.params_foregrounds['Dust']['method'] == 'blind' or self.preset.fg.params_foregrounds['Synchrotron']['method'] == 'blind':
           self.chi2 = Chi2Blind(self.preset)
        else:
           pass

        ### Define useful shapes 
        self.Nsub = self.preset.qubic.joint_out.qubic.Nsub
        self.fsub = int(self.Nsub*2 / self.preset.fg.params_foregrounds['bin_mixing_matrix'])
        self.Ncomp = len(self.preset.fg.components_name_out)
        self.Npix = 12*self.preset.sky.params_sky['nside']**2
        self.Nstokes = 3
        self.Ndet = self.preset.qubic.joint_out.qubic.ndets
        self.Nsamples = self.preset.qubic.joint_out.qubic.nsamples

        ### Create variables for stopping condition
        self._rms_noise_qubic_patch_per_ite = np.empty((self.preset.tools.params['PCG']['ites_to_converge'],self.Ncomp))
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

        ### Save time at the beginning of the map-making
        self.initial_time = time.time()
        
        ### Initialize loop variables
        self._info = True
        self._steps = 0
        
        while self._info:
            ### Display iteration
            self.preset.tools._display_iter(self._steps)

            ### Save time at the beginning of the iteration
            self.initial_iteration_time = time.time()
            
            ### Update self.fg.components_iter^{k} -> self.fg.components_iter^{k+1}
            self.preset.tools._print_message('    => Fitting component maps')
            self._update_components()
            
            ### Update self.preset.acquisition.mixingmatrix_iter^{k} -> self.preset.acquisition.mixingmatrix_iter^{k+1}
            if self.preset.fg.params_foregrounds['fit_spectral_index']:
                self.preset.tools._print_message('    => Fitting mixing matrix')
                self._update_mixing_matrix()
                
            ### Update self.gain.gain_iter^{k} -> self.gain.gain_iter^{k+1}
            if self.preset.qubic.params_qubic['GAIN']['fit_gain']:
                self.preset.tools._print_message('    => Fitting detectors gain')
                self._update_gain()
            
            ### Wait for all processes and save data inside pickle file
            self.preset.tools._print_message('    => Saving data')
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
        self.preset.tools._print_message(f'Fisher estimation : sigma(r) = {sigma_r:.6f}')

    def _call_pcg(self, max_iterations):
        """
        Method that calls the PCG in PyOperators.
        
        Args:
            max_iterations (int): Maximum number of iterations for the PCG algorithm.
        """

        ### Initialize PCG starting point
        if self.preset.tools.params['PCG']['fix_pixels_outside_patch']:
            initial_maps = self.preset.fg.components_iter[:, self.preset.sky.seenpix, :]
        elif self.preset.tools.params['PCG']['fixI']:
            initial_maps = self.preset.fg.components_iter[:, :, 1:]
        else:
            initial_maps = self.preset.fg.components_iter
        
        ### Run PCG
        result = mypcg(self.preset.A, 
                    self.preset.b, 
                    M=self.preset.acquisition.M, 
                    tol=self.preset.tools.params['PCG']['tol_pcg'], 
                    x0=initial_maps, 
                    maxiter=max_iterations, 
                    disp=True,
                    create_gif=False,
                    center=self.preset.sky.center, 
                    reso=self.preset.qubic.params_qubic['dtheta'], 
                    seenpix=self.preset.sky.seenpix, 
                    truth=self.preset.fg.components_out,
                    reuse_initial_state=False)['x']['x'] 
        
        ### Update components
        if self.preset.tools.params['PCG']['fix_pixels_outside_patch']:
            self.preset.fg.components_iter[:, self.preset.sky.seenpix, :] = result
        elif self.preset.tools.params['PCG']['fixI']:
            self.preset.fg.components_iter[:, :, 1:] = result
        else:
            self.preset.fg.components_iter = result.copy()
    
        ### Method to compute an approximation of sigma(r) using Fisher matrix at the end of the PCG
        self._fisher_compute_sigma_r()
        
        ### Plot if asked
        if self.preset.tools.rank == 0:
            self.plots.display_maps(self.preset.sky.seenpix, ki=self._steps)
            self.plots._display_allcomponents(self.preset.sky.seenpix, ki=self._steps)  
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
        
        ### Create acquisition operator
        H_i = self.preset.qubic.joint_out.get_operator(self.preset.acquisition.beta_iter, mixingmatrix=self.preset.acquisition.mixingmatrix_iter, gain=self.preset.gain.gain_iter, fwhm=self.preset.acquisition.fwhm_mapmaking, nu_co=self.preset.fg.nu_co)

        U = (
            ReshapeOperator((self.Ncomp * sum(self.preset.sky.seenpix) * self.Nstokes), (self.Ncomp, sum(self.preset.sky.seenpix), self.Nstokes)) *
            PackOperator(np.broadcast_to(self.preset.sky.seenpix[None, :, None], (self.Ncomp, self.preset.sky.seenpix.size, 3)).copy())
            ).T

        ### Update components when pixels outside the patch are fixed
        if self.preset.tools.params['PCG']['fix_pixels_outside_patch']:
            self.preset.A = U.T * H_i.T * self.preset.acquisition.invN * H_i * U

            if self.preset.qubic.params_qubic['convolution_out']:
                x_planck = self.preset.fg.components_convolved_out * (1 - self.preset.sky.seenpix[None, :, None])
            else:
                x_planck = self.preset.fg.components_out * (1 - self.preset.sky.seenpix[None, :, None])
            self.preset.b = U.T (  H_i.T * self.preset.acquisition.invN * (self.preset.acquisition.TOD_obs - H_i(x_planck)))

        ### Update components when intensity maps are fixed
        elif self.preset.tools.params['PCG']['fixI']:
            mask = np.ones((self.Ncomp, self.Npix, self.Nstokes))
            mask[:, :, 0] = 0
            P = (
                ReshapeOperator(PackOperator(mask).shapeout, (self.Ncomp, self.Npix, 2)) * 
                PackOperator(mask)
                ).T
            
            xI = self.preset.fg.components_convolved_out * (1 - mask)
            self.preset.A = P.T * H_i.T * self.preset.acquisition.invN * H_i * P
            self.preset.b = P.T (H_i.T * self.preset.acquisition.invN * (self.preset.acquisition.TOD_obs - H_i(xI)))
            
        ### Update components
        else:
            self.preset.A = H_i.T * self.preset.acquisition.invN * H_i
            self.preset.b = H_i.T * self.preset.acquisition.invN * self.preset.acquisition.TOD_obs
        
        ### Run PCG
        self._call_pcg(self.preset.tools.params['PCG']['n_iter_pcg'])

    def _get_TOD_component(self):
        """
        Method that produces Time-Ordered Data (TOD) using the component maps computed at the current iteration.

        This method initializes a zero-filled numpy array `TOD_component` with dimensions based on the number of components,
        the number of sub-components (multiplied by 2), and the product of the number of detectors and samples.
        It then iterates over each component and sub-component to compute the TOD by applying a convolution operator
        (if specified) and a mapping operator to the component maps.

        Returns:
            np.ndarray (Ncomp, Nsub, Ndet*Nsamples): A numpy array containing the computed TOD for each component and sub-component.
        """

        TOD_component = np.zeros((self.Ncomp, self.Nsub*2, self.Ndet*self.Nsamples))
        
        for i in range(self.Ncomp):
            for j in range(self.Nsub*2):
                if self.preset.qubic.params_qubic['convolution_out']:
                    C = HealpixConvolutionGaussianOperator(fwhm = self.preset.acquisition.fwhm_mapmaking[j], lmax=3*self.preset.sky.params_sky['nside'])
                else:
                    C = HealpixConvolutionGaussianOperator(fwhm = 0, lmax=3*self.preset.sky.params_sky['nside'])
                TOD_component[i, j] = self.preset.qubic.joint_out.qubic.H[j](C(self.preset.fg.components_iter[i])).ravel()
        
        return TOD_component
    
    def _callback(self, x, fitted_parameter):
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
            a. Every 5 iterations (when `self.iteration_number` is a multiple of 5), it prints the current iteration number and the parameter values rounded to 5 decimal places.
        3. Increments the iteration counter `self.iteration_number` by 1.
        """

        self.preset.tools.comm.Barrier()
        if self.preset.tools.rank == 0:
            if (self.iteration_number%5) == 0:
                print(f"Iter = {self.iteration_number:4d}      : {fitted_parameter} = {[np.round(x[i], 5) for i in range(len(x))]}")
            self.iteration_number += 1

    def _get_TOD_component_superpixel(self, index):
        if self.preset.tools.rank == 0:
            print('Computing contribution of each super-pixel')
        _index = np.zeros(12*self.preset.fg.params_foregrounds['Dust']['nside_beta_out']**2)
        _index[index] = index.copy()
        _index_nside = hp.ud_grade(_index, self.preset.sky.params_sky['nside'])
        TOD_component = np.zeros((len(index), self.Nsub*2, self.Ncomp, self.Ndet*self.Nsamples))
        
        maps_conv = self.preset.fg.components_iter.copy()

        for j in range(self.preset.qubic.params_qubic['nsub_out']):
            for icomp in range(self.Ncomp):
                if self.preset.qubic.params_qubic['convolution_out']:
                    C = HealpixConvolutionGaussianOperator(fwhm=self.preset.acquisition.fwhm_mapmaking[j], lmax=3*self.preset.sky.params_sky['nside'])
                else:
                    C = HealpixConvolutionGaussianOperator(fwhm=0, lmax=3*self.preset.sky.params_sky['nside'])
                maps_conv[icomp] = C(self.preset.fg.components_iter[icomp, :, :]).copy()
                for ii, i in enumerate(index):
                    maps_conv_i = maps_conv.copy()
                    _i = _index_nside == i
                    for stk in range(self.Nstokes):
                        maps_conv_i[:, :, stk] *= _i
                    TOD_component[ii, j, icomp] = self.preset.qubic.joint_out.qubic.H[j](maps_conv_i[icomp]).ravel()
        return TOD_component

    def _get_constrains(self):
        """
        Generate constraints readable by `scipy.optimize.minimize`.

        Return:
        constraints : list
            A list of constraint dictionaries for optimize.minimize.
        """

        constraints = []
        n = (self.preset.fg.params_foregrounds['bin_mixing_matrix']-1)*(self.Ncomp-1)

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
    
    def _get_mixing_matrix(self, beta, previous_mixingmatrix, icomp):
        """
        Method to update the mixing matrix using the current fitted value of the beta parameter and the parametric model associated.
        Only use when hybrid parametric-blind fit is selected !

        Arguments:
        - beta: int
        - previous_mixingmatrix: ndarray (Nsub_in + Nintegr * Nplanck, Ncomp)
        - icomp: int

        Return:
        - updated_mixingmatrix: ndarray (Nsub_in + Nintegr * Nplanck, Ncomp)
        """
        
        ### Build mixing matrix according to the choosen model and the beta parameter
        mixingmatrix = mm.MixingMatrix(*self.preset.fg.components_out)
        model_mixingmatrix = mixingmatrix.eval(self.preset.qubic.joint_out.qubic.allnus, *beta)

        ### Update the mixing matrix according to the one computed using the beta parameter
        updated_mixingmatrix = previous_mixingmatrix
        for ii in range(self.preset.fg.params_foregrounds['bin_mixing_matrix']):
            updated_mixingmatrix[ii*self.fsub: (ii + 1)*self.fsub, icomp] = model_mixingmatrix[ii*self.fsub: (ii + 1)*self.fsub, icomp]

        return updated_mixingmatrix
    
    def _get_mixing_matrix_fitting_method(self):
        """
        Function to check if the fitting method asked is implemented and determine which one will be used to fit the mixing matrix.
        Currently, the function is designed to work only for 'blind' or 'parametric' method for each component. 
        If different method are asked for the components, the hybrid 'parametric_blind' method will be applied.

        Return :
        - method: list(str) ('blind', 'parametric' or 'parametric_blind')
        """

        ### Build a list containing all the methods for the different components + Check if these methods are defiened 
        list_method = []
        for i in range(1, self.Ncomp):
            method_i = self.preset.fg.params_foregrounds[self.preset.fg.components_name_out[i]]['method']
            if method_i in ['parametric', 'blind']:
                list_method.append(method_i)
            else:
                raise TypeError(f'Fitting method {method_i} for {self.preset.fg.components_name_out[i]} is not implemented')
        
        ### Determine which method will be applied
        method_0 = list_method[0]
        for method_i in list_method[1:]:
            if method_i != method_0:
                return 'parametric_blind'
        return method_0

    def _update_mixing_matrix(self):
        """
        Method that fit the mixing matrix
        """
        
        ### Define the method used to fit the mixing matrix
        method = self._get_mixing_matrix_fitting_method()
        self.preset.tools._print_message(f'Fitting method   : {method}')

        ### Initialize fitting iteration number
        self.iteration_number = 0

        ### Get TOD from current component maps
        TOD_component = self._get_TOD_component()

        ### Parametric fitting
        if method == 'parametric':

            # Unique spectral index across the patch
            if self.preset.fg.params_foregrounds['Dust']['nside_beta_out'] == 0:

                ### Retrieve spectral index value at previous iteration
                previous_beta = self.preset.acquisition.beta_iter.copy()
                
                ### Build the cost function which will be minimize
                chi2 = Chi2Parametric(self.preset, TOD_component, previous_beta, seenpix_wrap=None)
                
                ### Fit the new value of the spectral index according to the component maps at the current iteration
                self.preset.acquisition.beta_iter = np.array([fmin_l_bfgs_b(chi2, 
                                                              x0=previous_beta, 
                                                              callback=lambda x: self._callback(x, 'beta'), 
                                                              approx_grad=True, 
                                                              epsilon=1e-6)[0]])
                
                ### Memory optimization
                del TOD_component
                gc.collect()
                
                if self.preset.tools.rank == 0:
                    # print the usuful informations about the fit
                    print(f'Iteration k - 1  : beta = {previous_beta}')
                    print(f'Iteration k      : beta = {self.preset.acquisition.beta_iter.copy()}')
                    print(f'Truth            : beta = {self.preset.mixingmatrix.beta_in.copy()}')
                    print(f'Residuals        : beta = {self.preset.mixingmatrix.beta_in - self.preset.acquisition.beta_iter}')
                    
                    # Plot
                    if self.Ncomp > 2:
                        self.plots.plot_beta_iteration(self.preset.acquisition.allbeta[:, 0], truth=self.preset.mixingmatrix.beta_in[0], ki=self._steps)
                    else:
                        self.plots.plot_beta_iteration(self.preset.acquisition.allbeta, truth=self.preset.mixingmatrix.beta_in, ki=self._steps)

                ### Wait until all processors end their task
                self.preset.tools.comm.Barrier()

                ### Add the lastest fitted spectral index
                self.preset.acquisition.allbeta = np.concatenate((self.preset.acquisition.allbeta, self.preset.acquisition.beta_iter), axis=0) 
            
            # Spatially distributed spectral index case
            else:
                ### Compute index of superpixels seen by QUBIC
                seenpix_qubic_superpixels = hp.ud_grade(self.preset.sky.seenpix_qubic, self.preset.fg.params_foregrounds['Dust']['nside_beta_out']) 
                self.preset.mixingmatrix._index_seenpix_beta = np.where(seenpix_qubic_superpixels == True)[0]

                ### Get TOD from current component maps
                TOD_component = self._get_TOD_component_superpixel(self.preset.mixingmatrix._index_seenpix_beta)

                ### Retrieve spectral index value at previous iteration
                previous_beta = self.preset.acquisition.beta_iter[self.preset.mixingmatrix._index_seenpix_beta, 0].copy()

                ### Build the cost function which will be minimize
                chi2 = Chi2Parametric(self.preset, TOD_component, self.preset.acquisition.beta_iter, seenpix_wrap=None)
                
                ### Fit the new value of the spectral index according to the component maps at the current iteration
                self.preset.acquisition.beta_iter[self.preset.mixingmatrix._index_seenpix_beta, 0] = np.array([fmin_l_bfgs_b(chi2, 
                                                                            x0=self.preset.acquisition.beta_iter[self.preset.mixingmatrix._index_seenpix_beta, 0], 
                                                                            callback=lambda x: self._callback(x, 'beta'), 
                                                                            approx_grad=True, 
                                                                            epsilon=1e-6, 
                                                                            maxls=5, 
                                                                            maxiter=5)[0]])
                
                ### Memory optimization
                del TOD_component
                gc.collect()
                            
                if self.preset.tools.rank == 0:
                    # print the usuful informations about the fit
                    print(fr'Iteration k - 1  : beta = {previous_beta}')
                    print(f'Iteration k      : beta = {self.preset.acquisition.beta_iter[self.preset.mixingmatrix._index_seenpix_beta, 0].copy()}')
                    print(f'Truth            : beta = {self.preset.mixingmatrix.beta_in[self.preset.mixingmatrix._index_seenpix_beta, 0].copy()}')
                    print(f'Residuals        : beta = {self.preset.mixingmatrix.beta_in[self.preset.mixingmatrix._index_seenpix_beta, 0] - self.preset.acquisition.beta_iter[self.preset.mixingmatrix._index_seenpix_beta, 0]}')
                    
                    # Plot
                    self.plots.plot_beta_iteration(self.preset.acquisition.allbeta[:, self.preset.mixingmatrix._index_seenpix_beta], 
                                                   truth=self.preset.mixingmatrix.beta_in[self.preset.mixingmatrix._index_seenpix_beta, 0], 
                                                   ki=self._steps)
                                    
                ### Wait until all processors end their task
                self.preset.tools.comm.Barrier()

                ### Add the lastest fitted spectral index
                self.preset.acquisition.allbeta = np.concatenate((self.preset.acquisition.allbeta, np.array([self.preset.acquisition.beta_iter])), axis=0) 
            
        ### Blind fitting
        elif method == 'blind':

            ### Retrieve mixing matrix at previous iteration
            previous_mixingmatrix = self.preset.acquisition.mixingmatrix_iter[:self.Nsub*2, 1:].copy()

            ### Initialize array that will contain the mixing matrix computed at each iteration
            if self._steps == 0:
                self.preset.acquisition.all_mixingmatrix = np.array([self.preset.acquisition.mixingmatrix_iter]) 

            ### Fit using scipy.optimize.minimize
            if self.preset.fg.params_foregrounds['blind_method'] == 'minimize' :

                # Build the cost function which will be minimize
                fun = partial(self.chi2._qu, TOD_component=TOD_component)
                
                # Initialize minimization starting point and boundaries
                x0 = []
                bnds = []
                for ii in range(self.preset.fg.params_foregrounds['bin_mixing_matrix']):
                    for i in range(1, self.Ncomp):
                        x0 += [np.mean(self.preset.acquisition.mixingmatrix_iter[ii*self.fsub:(ii+1)*self.fsub, i])]
                        bnds += [(0, None)]
                
                # Define random initial step for the mixing matrix
                if self._steps == 0:
                    x0 = np.random.normal(x0, self.preset.tools.params['INITIAL']['sig_mixingmatrix'])

                # Constraints on frequency evolution
                constraints = self._get_constrains()
                
                # Fit the new value of the mixing matrix according to the component maps at the current iteration
                Ai = minimize(fun, x0=x0, 
                            constraints=constraints, 
                            callback=lambda x: self._callback(x, 'A'),
                            bounds=bnds, 
                            method='L-BFGS-B', 
                            tol=1e-10).x
                
                # Update mixing matrix variable
                k=0
                for ii in range(self.preset.fg.params_foregrounds['bin_mixing_matrix']):
                    for i in range(1, self.Ncomp):
                        self.preset.acquisition.mixingmatrix_iter[ii*self.fsub:(ii+1)*self.fsub, i] = Ai[k]
                        k+=1

            ### Alternate fitting (one component after another) using scipy.optimize.minimize
            elif self.preset.fg.params_foregrounds['blind_method'] == 'alternate' :
                
                for i in range(self.Ncomp):
                    if self.preset.fg.components_name_out[i] != 'CMB':
                        print('I am fitting ', self.preset.fg.components_name_out[i])

                        # Build the cost function which will be minimize
                        fun = partial(self.chi2._qu_alt, TOD_component=TOD_component, A=self.preset.acquisition.mixingmatrix_iter, icomp=i)
                
                        # Starting point
                        x0 = []
                        bnds = []
                        for ii in range(self.preset.fg.params_foregrounds['bin_mixing_matrix']):
                            x0 += [np.mean(self.preset.acquisition.mixingmatrix_iter[ii*self.fsub:(ii+1)*self.fsub, i])]
                            bnds += [(0, None)]

                        # Define random initial step for the mixing matrix
                        if self._steps == 0:
                            x0 = np.random.normal(x0, self.preset.tools.params['INITIAL']['sig_mixingmatrix'])

                        # Fit the new value of the mixing matrix according to the component map i at the current iteration
                        Ai = minimize(fun, x0=x0,
                                callback=lambda x: self._callback(x, 'A'),
                                bounds=bnds, 
                                method='SLSQP', 
                                tol=1e-10).x
  
                        # Update mixing matrix variable
                        for ii in range(self.preset.fg.params_foregrounds['bin_mixing_matrix']):
                            self.preset.acquisition.mixingmatrix_iter[ii*self.fsub:(ii+1)*self.fsub, i] = Ai[ii]

            ### Fit using Preconditionned Conjugate Gradiant
            elif self.preset.fg.params_foregrounds['blind_method'] == 'PCG' :
                
                # Build binned TOD (Ncomp, Nsub_mixingmatrix, Ndet*Nsamples)
                TOD_component_binned = np.zeros((self.Ncomp, self.preset.fg.params_foregrounds['bin_mixing_matrix'], self.Ndet*self.Nsamples))
                for k in range(self.Ncomp):
                    for i in range(self.preset.fg.params_foregrounds['bin_mixing_matrix']):
                        TOD_component_binned[k, i] = np.sum(TOD_component[k, i*self.fsub:(i+1)*self.fsub], axis=0)

                # Observed TOD containing only CMB
                TOD_cmb150 = self.preset.tools.comm.allreduce(np.sum(TOD_component[0, :int(self.Nsub/2)], axis=0), op=MPI.SUM)
                TOD_cmb220 = self.preset.tools.comm.allreduce(np.sum(TOD_component[0, int(self.Nsub/2):int(self.Nsub)], axis=0), op=MPI.SUM)

                # Input TOD containing CMB + Foregrounds
                TOD_in_150 = self.preset.tools.comm.allreduce(self.preset.acquisition.TOD_qubic[:int(self.preset.acquisition.TOD_qubic.shape[0]/2)], op=MPI.SUM)
                TOD_in_220 = self.preset.tools.comm.allreduce(self.preset.acquisition.TOD_qubic[int(self.preset.acquisition.TOD_qubic.shape[0]/2):int(self.preset.acquisition.TOD_qubic.shape[0])], op=MPI.SUM)

                # Input foregrounds TOD and reshape it into 150 & 220 GHz signals
                TOD_foregrounds = np.r_[TOD_in_150 - TOD_cmb150, TOD_in_220 - TOD_cmb220]
                TOD_foregrounds_reshaped = np.sum(TOD_foregrounds.reshape(2, self.Ndet*self.Nsamples), axis=0)

                # Observed foregrounds TOD 
                dnu = self.preset.tools.comm.allreduce(TOD_component_binned[1:], op=MPI.SUM)
                dnu = dnu.reshape((dnu.shape[0]*dnu.shape[1], dnu.shape[2]))

                # Build the equation Ax=b which will be resolve using a PCG to fit the mixing matrix
                A = dnu @ dnu.T
                b = dnu @ TOD_foregrounds_reshaped

                # Run PCG to resolve the previous equation to fit the mixing matrix
                solution = mypcg(A, b, disp=False, tol=1e-20, maxiter=10000)['x']
            
                # Update mixing matrix variable
                k=0
                for i in range(1, self.Ncomp):
                    for ii in range(self.preset.fg.params_foregrounds['bin_mixing_matrix']):
                        self.preset.acquisition.mixingmatrix_iter[ii*self.fsub:(ii+1)*self.fsub, i] = solution['x'][k]
                        k+=1
                
            else:
                raise TypeError(f"{self.preset.fg.params_foregrounds['blind_method']} is not yet implemented..")           

            ### Wait until all processors end their task
            self.preset.tools.comm.Barrier()

            ### Add the lastest fitted mixing matrix
            self.preset.acquisition.all_mixingmatrix = np.concatenate((self.preset.acquisition.all_mixingmatrix, np.array([self.preset.acquisition.mixingmatrix_iter])), axis=0)
            
            ### Plot the evolution of the mixing matrix
            if self.preset.tools.rank == 0:
                print(fr'Iteration k - 1  : A = {previous_mixingmatrix.ravel()}')
                print(f'Iteration k      : A = {self.preset.acquisition.mixingmatrix_iter[:self.Nsub*2, 1:].ravel()}')
                print(f'Truth            : A = {self.preset.mixingmatrix.mixingmatrix_in[:self.Nsub*2, 1:].ravel()}')
                print(f'Residuals        : A = {self.preset.mixingmatrix.mixingmatrix_in[:self.Nsub*2, 1:].ravel() - self.preset.acquisition.mixingmatrix_iter[:self.Nsub*2, 1:].ravel()}')
                    
                self.plots.plot_sed(self.preset.qubic.joint_out.qubic.allnus, 
                                    self.preset.acquisition.all_mixingmatrix[:, :self.Nsub*2, 1:], 
                                    ki=self._steps, truth=self.preset.mixingmatrix.mixingmatrix_in[:self.Nsub*2, 1:])

            ### Memory optimization
            del TOD_component
            gc.collect()

        ### Hybrid Parametric-Blind fitting
        elif method == 'parametric_blind':

            ### Retrieve mixing matrix at previous iteration
            previous_mixingmatrix = self.preset.acquisition.mixingmatrix_iter[:self.Nsub*2, 1:].copy()

            ### Loop on the components, to fit one(s) with parametric approach and the rest with the blind one
            for i in range(self.Ncomp):
                # We don't fit the CMB part of the mixing matrix, which is set full of ones by definition
                if self.preset.fg.components_name_out[i] != 'CMB':

                    # Parametric fitting
                    if self.preset.fg.params_foregrounds[self.preset.fg.components_name_out[i]]['method'] == 'parametric':
                        print('I am fitting ', self.preset.fg.components_name_out[i], i)

                        # Retrieve the previous beta for the parametrix fitting
                        previous_beta = self.preset.acquisition.beta_iter.copy()
                            
                        # Build the cost function which will be minimize
                        chi2 = Chi2Parametric_alt(self.preset, TOD_component, self.preset.acquisition.mixingmatrix_iter, i, seenpix_wrap=None)

                        # Fit the new value of the spectral index according to the component maps at the current iteration
                        self.preset.acquisition.beta_iter[i-1] = np.array([fmin_l_bfgs_b(chi2, 
                                                                        x0 = self.preset.acquisition.beta_iter[i-1], callback=self._callback, approx_grad=True, epsilon=1e-6)[0]])

                        # Compute the mixing matrix associated with the fitted spectral index
                        self.preset.acquisition.mixingmatrix_iter = self._get_mixing_matrix(self.preset.acquisition.beta_iter, self.preset.acquisition.mixingmatrix_iter, i)
                    
                    # Blind fitting
                    else:
                        print('I am fitting ', self.preset.fg.components_name_out[i], i)

                        # Build the cost function which will be minimize
                        fun = partial(self.chi2._qu_alt, TOD_component=TOD_component, A=self.preset.acquisition.mixingmatrix_iter, icomp=i)
                
                        # Initialize minimization starting point and boundaries
                
                        # Define random initial step for the mixing matrix and boundaries
                        x0 = []
                        bnds = []
                        for ii in range(self.preset.fg.params_foregrounds['bin_mixing_matrix']):
                            for j in range(1, self.Ncomp):
                                x0 += [np.mean(self.preset.acquisition.mixingmatrix_iter[ii*self.fsub:(ii+1)*self.fsub, j])]
                                bnds += [(0, None)]
                        
                        # Define random initial step for the mixing matrix
                        if self._steps == 0:
                            x0 = np.random.normal(x0, self.preset.tools.params['INITIAL']['sig_mixingmatrix'])

                        # Fit the new value of the mixing matrix according to the component map i at the current iteration
                        Ai = minimize(fun, x0=x0,
                                callback=self._callback, 
                                bounds=bnds, 
                                method='SLSQP', 
                                tol=1e-10).x
                        
                        # Update mixing matrix variable
                        for ii in range(self.preset.fg.params_foregrounds['bin_mixing_matrix']):
                            self.preset.acquisition.mixingmatrix_iter[ii*self.fsub:(ii+1)*self.fsub, i] = Ai[ii]                

            ### Add the lastest fitted mixing matrix
            self.preset.acquisition.all_mixingmatrix = np.concatenate((self.preset.acquisition.all_mixingmatrix, np.array([self.preset.acquisition.mixingmatrix_iter])), axis=0)
            
            ### Pots
            if self.preset.tools.rank == 0:
                print(f'Iteration k     : {previous_mixingmatrix.ravel()}')
                print(f'Iteration k + 1 : {self.preset.acquisition.mixingmatrix_iter[:self.Nsub*2, 1:].ravel()}')
                print(f'Truth           : {self.preset.mixingmatrix.mixingmatrix_in[:self.Nsub*2, 1:].ravel()}')
                print(f'Residuals       : {self.preset.mixingmatrix.mixingmatrix_in[:self.Nsub*2, 1:].ravel() - self.preset.acquisition.mixingmatrix_iter[:self.Nsub*2, 1:].ravel()}')
                self.plots.plot_sed(self.preset.qubic.joint_out.qubic.allnus, 
                                    self.preset.acquisition.all_mixingmatrix[:, :self.Nsub*2, 1:], 
                                    ki=self._steps, truth=self.preset.mixingmatrix.mixingmatrix_in[:self.Nsub*2, 1:])

            ### Memory optimization
            del TOD_component
            gc.collect()   

    def _get_intercalibration(self, D, d, _invn):   
        """
        Semi-analytical method for gains estimation using the currently fitted mixing matrix. (cf CMM paper, Eq. 10)

        Arguments:
        - D = HAc: noiseless TOD
            -> H: pointing matrix
            -> A: mixing matrix
            -> c: component vector
        - d = GD + n: intercalibrated noisy TOD
            -> G: intercalibrations
            -> n: noise vector
        - invN: inverse noise-covariance matrix

        Result:
        - G = (D(N^-1)d) / (D(N^-1)D) = observed TOD / simulated TOD: intercalibrations
        """
        
        ### (Ndet*Nsamples) --> (Ndet, Nsamples)
        _r = ReshapeOperator(self.Ndet*self.Nsamples, 
                             (self.Ndet, self.Nsamples))

        return np.sum(_r(D) * _invn(_r(d)), axis=1) / (np.sum(_r(D) * _invn(_r(D)), axis=1))

    def _update_gain(self):
        """
        Method that compute gains of each detectors using semi-analytical method g_i = TOD_obs_i / TOD_sim_i
        """
        
        ### Retrieve gain at previous estimation
        previous_gain = self.preset.gain.gain_iter.copy()

        ### Acquisition operator
        self.H_i = self.preset.qubic.joint_out.get_operator(self.preset.acquisition.beta_iter, mixingmatrix=self.preset.acquisition.mixingmatrix_iter, gain=np.ones(self.preset.gain.gain_iter.shape), fwhm=self.preset.acquisition.fwhm_mapmaking, nu_co=self.preset.fg.nu_co)

        ### UWB case
        if self.preset.qubic.params_qubic['instrument'] == 'UWB':
            # (Ndet*Nsamples) --> (Ndet, Nsamples)
            _r = ReshapeOperator(self.Ndet*self.preset.joint.qubic.nsamples, (self.preset.joint.qubic.ndets, self.preset.joint.qubic.nsamples))

            # Compute noiseless TODs with current value of the mixing matrix : D = HAc
            TODi_qubic_noiseless = self.preset.acquisition.invN.operands[0](self.H_i.operands[0](self.preset.fg.components_iter)[:self.Ndet*self.Nsamples])
            
            # Compute gain using semi-analytical method : g = observed TOD / simulated TOD 
            self.preset.gain.gain_iter = self._get_intercalibration(TODi_qubic_noiseless, _r(self.preset.acquisition.TOD_qubic))

            # Update gain's variables
            self.preset.gain.gain_iter = join_data(self.preset.tools.comm, self.preset.gain.gain_iter)
            self.preset.gain.all_gain_iter = np.concatenate((self.preset.gain.all_gain, np.array([self.preset.gain.gain_iter])), axis=0)
        
        ### DB case
        elif self.preset.qubic.params_qubic['instrument'] == 'DB':
            # Compute noiseless TODs with current value of the mixing matrix : D = HAc
            TODi_qubic_noiseless_150 = self.H_i.operands[0](self.preset.fg.components_iter)[:self.Ndet*self.Nsamples]
            TODi_qubic_noiseless_220 = self.H_i.operands[0](self.preset.fg.components_iter)[self.Ndet*self.Nsamples:2*self.Ndet*self.Nsamples]
            
            # Compute gain using semi-analytical method : g = observed TOD / simulated TOD
            g150 = self._get_intercalibration(TODi_qubic_noiseless_150, self.preset.acquisition.TOD_qubic[:self.Ndet*self.Nsamples], self.preset.acquisition.invN.operands[0].operands[1].operands[0])
            g220 = self._get_intercalibration(TODi_qubic_noiseless_220, self.preset.acquisition.TOD_qubic[self.Ndet*self.Nsamples:2*self.Ndet*self.Nsamples], self.preset.acquisition.invN.operands[0].operands[1].operands[1])
            self.preset.gain.gain_iter = np.array([g150, g220]).T

            # Update gain's variables
            self.preset.gain.gain_iter = join_data(self.preset.tools.comm, self.preset.gain.gain_iter)
            self.preset.gain.all_gain = np.concatenate((self.preset.gain.all_gain, np.array([self.preset.gain.gain_iter])), axis=0)

            if self.preset.tools.rank == 0:
                print(fr'Iteration k - 1  : G = {previous_gain}')
                print(f'Iteration k      : G = {self.preset.gain.gain_iter}')
                print(f'Truth            : G = {self.preset.gain.gain_in}')
                print(f'Residuals        : G = {self.preset.gain.gain_in - self.preset.gain.gain_iter}')

                ### Plot gain evolution
                self.plots.plot_gain_iteration(self.preset.gain.all_gain - self.preset.gain.gain_in, ki=self._steps)

    def _save_data(self, step):
        """
        Method that determine time since the beginning of the current iteration and the beginning of the map-making, 
        and save data for each iterations. It saves components, gains, spectral index, coverage, seen pixels.
        
        Argument:
        - step: int
        """

        ### Compute the time since the beginning og the current iteration and since the beginning of the map-making
        iteration_duration = time.time() - self.initial_iteration_time
        duration = time.time() - self.initial_time

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
                                 'gain_in':self.preset.gain.gain_in,
                                 'gain_iter':self.preset.gain.gain_iter,
                                 'all_gain':self.preset.gain.all_gain,
                                 'A':self.preset.acquisition.mixingmatrix_iter,
                                 'Atrue':self.preset.mixingmatrix.mixingmatrix_in,
                                 'nus_in':self.preset.mixingmatrix.nus_eff_in,
                                 'nus_out':self.preset.mixingmatrix.nus_eff_out,
                                 'center':self.preset.sky.center,
                                 'coverage':self.preset.sky.coverage,
                                 'seenpix':self.preset.sky.seenpix,
                                 'fwhm':self.preset.acquisition.fwhm_tod,
                                 'acquisition.fwhm_reconstructed':self.preset.acquisition.fwhm_mapmaking,
                                 'iteration_duration':iteration_duration,
                                 'duration': duration}, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
        rms_maxpercomp = np.zeros(self.Ncomp)

        for i in range(self.Ncomp):
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
            
            deltarms_max_percomp = np.zeros(self.Ncomp)

            for i in range(self.Ncomp):
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