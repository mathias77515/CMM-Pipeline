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
from pyoperators import MPI
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
import os
import sys
from scipy.optimize import minimize, fmin, fmin_l_bfgs_b
from solver.cg import (mypcg)
from preset.preset import *
from plots.plots import *
from costfunc.chi2 import Chi2ConstantBeta, Chi2VaryingBeta, Chi2ConstantBlind
               
    
               
               
class Pipeline:


    """
    
    Main instance to create End-2-End pipeline for components reconstruction.
    
    Arguments :
    -----------
        - comm    : MPI common communicator (define by MPI.COMM_WORLD).
        - seed    : Int number for CMB realizations.
        - it      : Int number for noise realizations.
        
    """
    
    def __init__(self, comm, seed, it):
        
        self.sims = PresetSims(comm, seed, it)
        
        if self.sims.params['Foregrounds']['type'] == 'parametric':
            if self.sims.params['Foregrounds']['nside_fit'] == 0:
                self.chi2 = Chi2ConstantBeta(self.sims)
            else:
                self.chi2 = Chi2VaryingBeta(self.sims)
        elif self.sims.params['Foregrounds']['type'] == 'blind':
            self.chi2 = Chi2ConstantBlind(self.sims)
        else:
            raise TypeError(f"{self.sims.params['Foregrounds']['type']} is not yet implemented..")
        self.plots = Plots(self.sims, dogif=True)
        
        
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
            if self.sims.params['Foregrounds']['fit_spectral_index']:
                self._update_spectral_index()

            #self.sims.comm.Barrier()
            ### Update self.g_iter^{k} -> self.g_iter^{k+1}
            if self.sims.params['MapMaking']['qubic']['fit_gain']:
                self._update_gain()
            
            self.sims.comm.Barrier()
            ### Display maps
            if self.sims.rank == 0:
                self.plots.display_maps(self.sims.seenpix_plot, ngif=self._steps+1, ki=self._steps)
                self.plots._display_allcomponents(self.sims.seenpix_plot, ki=self._steps)

                #### Display convergence of beta
                if self.sims.params['Foregrounds']['type'] == 'parametric':
                    if self.sims.params['Foregrounds']['nside_fit'] == 0:
                        self.plots.plot_beta_iteration(self.sims.allbeta, truth=self.sims.beta, ki=self._steps)
                    else:
                        
                        self.plots.plot_beta_iteration(self.sims.allbeta, 
                                                   truth=self.sims.beta[np.where(self.sims.coverage_beta == 1)[0], 0], 
                                                   ki=self._steps)
                elif self.sims.params['Foregrounds']['type'] == 'blind':
                    self.plots.plot_sed(self.sims.joint.qubic.allnus, self.sims.allAmm_iter[:, :self.sims.joint.qubic.Nsub*2, 1], ki=self._steps, truth=self.sims.Amm[:self.sims.joint.qubic.Nsub*2, 1])
                
                else:
                    raise TypeError(f"{self.sims.params['Foregrounds']['type']} method is not yet implemented")
            
                #### Display convergence of beta
                self.plots.plot_gain_iteration(abs(self.sims.allg - self.sims.g), alpha=0.03, ki=self._steps)
            #stop
            ### Save data inside pickle file
            self.sims.comm.Barrier()
            self._save_data()

            ### Stop the loop when self._steps > k
            self._stop_condition()
            
            
    def _compute_maps_convolved(self):
        
        """
        
        Method to compute convolved maps for each FWHM of QUBIC.
        
        """
        
        ### We make the convolution before beta estimation to speed up the code, we avoid to make all the convolution at each iteration
        ### Constant spectral index
        if self.sims.params['Foregrounds']['nside_fit'] == 0:
            components_for_beta = np.zeros((self.sims.params['MapMaking']['qubic']['nsub'], len(self.sims.comps), 12*self.sims.params['MapMaking']['qubic']['nside']**2, 3))
            for i in range(self.sims.params['MapMaking']['qubic']['nsub']):

                for jcomp in range(len(self.sims.comps)):
                    if self.sims.params['MapMaking']['qubic']['convolution']:
                        C = HealpixConvolutionGaussianOperator(fwhm = self.fwhm_recon[i])
                    else:
                        C = HealpixConvolutionGaussianOperator(fwhm = 0)
                    components_for_beta[i, jcomp] = C(self.sims.components_iter[jcomp])
        else:
            components_for_beta = np.zeros((self.sims.params['MapMaking']['qubic']['nsub'], 3, 12*self.sims.params['MapMaking']['qubic']['nside']**2, len(self.sims.comps)))
            for i in range(self.sims.params['MapMaking']['qubic']['nsub']):
                for jcomp in range(len(self.sims.comps)):
                    if self.sims.params['MapMaking']['qubic']['convolution']:
                        C = HealpixConvolutionGaussianOperator(fwhm = self.sims.fwhm_recon[i])
                    else:
                        C = HealpixConvolutionGaussianOperator(fwhm = 0)

                    components_for_beta[i, :, :, jcomp] = C(self.sims.components_iter[:, :, jcomp].T).T
        return components_for_beta
    def _callback(self, x):
        
        """
        
        Method to make callback function readable by `scipy.optimize.minimize`.
        
        """
        
        self.sims.comm.Barrier()
        if self.sims.rank == 0:
            if self.nfev == 0:
                print(f"Iter = {self.nfev:4d}   beta = {[np.round(x[i], 5) for i in range(len(x))]}   -First(LogL) = {-self.chi2.chi2.copy():3.6f}")
            else:
                print(f"Iter = {self.nfev:4d}   beta = {[np.round(x[i], 5) for i in range(len(x))]}   -Delta(LogL) = {-self.chi2.chi2.copy():3.6f}")
            
            #print(f"{self.nfev:4d}   {x[0]:3.6f}   {self.chi2.chi2_P:3.6e}")
            self.nfev += 1
    def _update_spectral_index(self):
        
        """
        
        Method that perform step 3) of the pipeline for 2 possible designs : Two Bands and Wide Band
        
        """
        
        
        #self.sims.H_i = self.sims.joint.get_operator(self.sims.beta_iter, gain=self.sims.g_iter, fwhm=self.sims.fwhm_recon, nu_co=self.sims.nu_co)
        #print(self.sims.Amm)
        #print(self.sims.Amm_iter)
        if self.sims.params['Foregrounds']['type'] == 'parametric':
            if self.sims.params['Foregrounds']['nside_fit'] == 0:
                self._index_seenpix_beta = 0
                #print(_norm2(self.sims.TOD_Q, self.sims.comm))
            
            
                self.nfev = 0
                fun = partial(self.chi2.cost_function, solution=self.sims.components_iter)
                self.sims.beta_iter = np.array([fmin_l_bfgs_b(fun, x0=np.array([1.5]), callback=self._callback, approx_grad=True)[0]])
        
            
                if self.sims.rank == 0:
                    print(self.sims.beta_iter)
            
                self.sims.comm.Barrier()
                self.sims.allbeta = np.append(self.sims.allbeta, self.sims.beta_iter)
            else:
            
            
                self._index_seenpix_beta = np.where(self.sims.coverage_beta == 1)[0]
                previous_beta = self.sims.beta_iter[self._index_seenpix_beta, 0].copy()
            
                if self.sims.params['Foregrounds']['fit_all_at_same_time']:
                    self.nfev = 0
                    fun = partial(self.chi2.cost_function, 
                              solution=self.sims.components_iter,
                              patch_id=self._index_seenpix_beta, 
                              allbeta=self.sims.beta_iter)
                
                    self.sims.comm.Barrier()
                    self.sims.beta_iter[self._index_seenpix_beta, 0] = np.array([fmin_l_bfgs_b(fun, 
                                                                                           x0=self.sims.beta_iter[self._index_seenpix_beta, 0], 
                                                                                           callback=self._callback,
                                                                                           epsilon = 1e-5, 
                                                                                           approx_grad=True)[0]])
                                                
                else:
                    for iindex, index in enumerate(self._index_seenpix_beta):
                        self.nfev = 0
                        fun = partial(self.chi2.cost_function, 
                              solution=self.sims.components_iter,
                              patch_id=index, 
                              allbeta=self.sims.beta_iter)
                    
                        self.sims.comm.Barrier()
                        self.sims.beta_iter[index] = np.array([fmin_l_bfgs_b(fun, 
                                                                         x0=np.array([1.5]), 
                                                                         callback=self._callback,
                                                                         epsilon = 1e-5, 
                                                                         approx_grad=True)[0]])
             
                self.sims.comm.Barrier()
                self.sims.allbeta = np.concatenate((self.sims.allbeta, np.array([self.sims.beta_iter[self._index_seenpix_beta]])), axis=0)
            
                if self.sims.rank == 0:
                
                    print(f'Iteration k     : {previous_beta}')
                    print(f'Iteration k + 1 : {self.sims.beta_iter[self._index_seenpix_beta, 0].copy()}')
                    print(f'Truth           : {self.sims.beta[self._index_seenpix_beta, 0].copy()}')
                    print(f'Residuals       : {self.sims.beta[self._index_seenpix_beta, 0] - self.sims.beta_iter[self._index_seenpix_beta, 0]}')
        elif self.sims.params['Foregrounds']['type'] == 'blind':
            previous_step = self.sims.Amm_iter[:self.sims.joint.qubic.Nsub*2, 1].copy()
            self._index_seenpix_beta = None
            self.nfev = 0
            fun = partial(self.chi2._qu, solution=self.sims.components_iter)
            #print(self.sims.Amm_iter.shape)
            Ai = np.array([fmin_l_bfgs_b(fun, x0=self.sims.Amm_iter[:self.sims.joint.qubic.Nsub*2, 1], callback=self._callback, approx_grad=True)[0]])
            self.sims.Amm_iter[:self.sims.joint.qubic.Nsub*2, 1] = Ai.copy()
            self.sims.allAmm_iter = np.concatenate((self.sims.allAmm_iter, np.array([self.sims.Amm_iter])), axis=0)
            #print(self.sims.allAmm_iter.shape)
            if self.sims.rank == 0:
                    print(self.sims.Amm_iter[:, 1])
                    print(self.sims.Amm[:, 1])
                    print(f'Iteration k     : {previous_step}')
                    print(f'Iteration k + 1 : {self.sims.Amm_iter[:self.sims.joint.qubic.Nsub*2, 1]}')
                    print(f'Truth           : {self.sims.Amm[:self.sims.joint.qubic.Nsub*2, 1]}')
                    print(f'Residuals       : {self.sims.Amm[:self.sims.joint.qubic.Nsub*2, 1] - self.sims.Amm_iter[:self.sims.joint.qubic.Nsub*2, 1]}')
            #stop
        else:
            raise TypeError(f"{self.sims.params['Foregrounds']['type']} is not yet implemented..") 
    def _save_data(self):
        
        """
        
        Method that save data for each iterations. It saves components, gains, spectral index, coverage, seen pixels.
        
        """
        if self.sims.rank == 0:
            if self.sims.params['save'] != 0:
                if (self._steps+1) % self.sims.params['save'] == 0:
                    
                    if self.sims.params['lastite']:
                    
                        if self._steps != 0:
                            os.remove(self.sims.params['foldername'] + '/' + self.sims.params['filename']+f"_k{self._steps-1}_seed{str(self.sims.params['CMB']['seed'])}_iter{str(self.sims.params['CMB']['iter'])}.pkl")
                
                    with open(self.sims.params['foldername'] + '/' + self.sims.params['filename']+f"_k{self._steps}_seed{str(self.sims.params['CMB']['seed'])}_iter{str(self.sims.params['CMB']['iter'])}.pkl", 'wb') as handle:
                        pickle.dump({'components':self.sims.components, 
                                 'components_i':self.sims.components_iter,
                                 'beta':self.sims.allbeta,
                                 'beta_true':self.sims.beta,
                                 'index_beta':self._index_seenpix_beta,
                                 'g':self.sims.g,
                                 'gi':self.sims.g_iter,
                                 'allg':self.sims.allg,
                                 'A':self.sims.Amm_iter,
                                 'Atrue':self.sims.Amm,
                                 'allA':self.sims.allAmm_iter,
                                 'G':self.sims.G,
                                 'nus':self.sims.nus_eff,
                                 'center':self.sims.center,
                                 'coverage':self.sims.coverage,
                                 'seenpix':self.sims.seenpix}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def _update_components(self):
        
        """
        
        Method that solve the map-making equation ( H.T * invN * H ) * components = H.T * invN * TOD using OpenMP / MPI solver. 
        
        """
        
        self.sims.H_i = self.sims.joint.get_operator(self.sims.beta_iter, Amm=self.sims.Amm_iter, gain=self.sims.g_iter, fwhm=self.sims.fwhm_recon, nu_co=self.sims.nu_co)
        
        
        #if self.sims.params['Foregrounds']['nside_fit'] == 0:
        #    U = (
        #        ReshapeOperator((len(self.sims.comps_name) * sum(self.sims.seenpix) * 3), (len(self.sims.comps_name), sum(self.sims.seenpix), 3)) *
        #        PackOperator(np.broadcast_to(self.sims.seenpix[None, :, None], (len(self.sims.comps_name), self.sims.seenpix.size, 3)).copy())
        #    ).T
        #else:
        #    U = (
        #        ReshapeOperator((3 * len(self.sims.comps_name) * sum(self.sims.seenpix)), (3, sum(self.sims.seenpix), len(self.sims.comps_name))) *
        #        PackOperator(np.broadcast_to(self.sims.seenpix[None, :, None], (3, self.sims.seenpix.size, len(self.sims.comps_name))).copy())
        #    ).T
        
        #self.sims.A = U.T * self.sims.H_i.T * self.sims.invN * self.sims.H_i * U
        #x_planck = self.sims.components * (1 - self.sims.seenpix[None, :, None])
        #self.sims.b = U.T (  self.sims.H.T * self.sims.invN * (self.sims.TOD_obs - self.sims.H_i(x_planck)))
        
        self.sims.A = self.sims.H_i.T * self.sims.invN * self.sims.H_i
        self.sims.b = self.sims.H_i.T * self.sims.invN * self.sims.TOD_obs

        self._call_pcg()
    def _call_pcg(self):

        """
        
        Method that call the PCG in PyOperators.
        
        """
        
        mypixels = mypcg(self.sims.A, 
                                   self.sims.b, 
                                   M=self.sims.M, 
                                   tol=self.sims.params['MapMaking']['pcg']['tol'], 
                                   x0=self.sims.components_iter, 
                                   maxiter=self.sims.params['MapMaking']['pcg']['maxiter'], 
                                   disp=True,
                                   create_gif=False,
                                   center=self.sims.center, 
                                   reso=self.sims.params['MapMaking']['qubic']['dtheta'], 
                                   seenpix=self.sims.seenpix, 
                                   truth=self.sims.components)['x']['x']  
        self.sims.components_iter = mypixels.copy()
    def _stop_condition(self):
        
        """
        
        Method that stop the convergence if there are more than k steps.
        
        """
        
        if self._steps >= self.sims.params['MapMaking']['pcg']['k']-1:
            self._info = False
        
        if self.sims.rank == 0:
            self._steps += 1
    def _display_iter(self):
        
        """
        
        Method that display the number of a specific iteration k.
        
        """
        
        if self.sims.rank == 0:
            print('========== Iter {}/{} =========='.format(self._steps+1, self.sims.params['MapMaking']['pcg']['k']))
    def _update_gain(self):
        
        """
        
        Method that compute gains of each detectors using semi-analytical method g_i = TOD_obs_i / TOD_sim_i
        
        """
        
        self.H_i = self.sims.joint.get_operator(self.sims.beta_iter, Amm=self.sims.Amm_iter, gain=np.ones(self.sims.g_iter.shape), fwhm=self.sims.fwhm_recon, nu_co=self.sims.nu_co)
        
        if self.sims.params['MapMaking']['qubic']['type'] == 'wide':
            R2det_i = ReshapeOperator(self.sims.joint.qubic.ndets*self.sims.joint.qubic.nsamples, (self.sims.joint.qubic.ndets, self.sims.joint.qubic.nsamples))
            #print(R2det_i.shapein, R2det_i.shapeout)
            TOD_Q_ALL_i = R2det_i(self.H_i.operands[0](self.sims.components_iter))
        
            self.sims.g_iter = self._give_me_intercal(TOD_Q_ALL_i, R2det_i(self.sims.TOD_Q))
            self.sims.g_iter /= self.sims.g_iter[0]
            self.sims.allg = np.concatenate((self.sims.allg, np.array([self.sims.g_iter])), axis=0)
            
        elif self.sims.params['MapMaking']['qubic']['type'] == 'two':
            
            R2det_i = ReshapeOperator(2*self.sims.joint.qubic.ndets*self.sims.joint.qubic.nsamples, (2*self.sims.joint.qubic.ndets, self.sims.joint.qubic.nsamples))
            TODi_Q_150 = R2det_i(self.H_i.operands[0](self.sims.components_iter))[:self.sims.joint.qubic.ndets]
            TODi_Q_220 = R2det_i(self.H_i.operands[0](self.sims.components_iter))[self.sims.joint.qubic.ndets:2*self.sims.joint.qubic.ndets]
            
            g150 = self._give_me_intercal(TODi_Q_150, self.sims.TOD_Q_150)
            g220 = self._give_me_intercal(TODi_Q_220, self.sims.TOD_Q_220)
            g150 /= g150[0]
            g220 /= g220[0]
            
            self.sims.g_iter = np.array([g150, g220]).T
            self.sims.allg = np.concatenate((self.sims.allg, np.array([self.sims.g_iter])), axis=0)
    def _give_me_intercal(self, D, d):
        
        """
        
        Semi-analytical method for gains estimation.

        """
        
        return 1/np.sum(D[:]**2, axis=1) * np.sum(D[:] * d[:], axis=1)

