import os
from functools import partial

import numpy as np
import healpy as hp
from scipy.optimize import minimize, fmin_l_bfgs_b

import pickle
import gc
from acquisition.Qacquisition import *

from simtools.mpi_tools import *
from simtools.noise_timeline import *
#from simtools.foldertools import *
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

        if seed_noise == -1:
            if comm.Get_tools.rank() == 0:
                seed_noise = np.random.randint(100000000)
            else:
                seed_noise = None
        seed_noise = comm.bcast(seed_noise, root=0)
        ### Initialization
        self.preset = preset.PresetInitialisation(comm, seed, seed_noise).initialize()
        
        self.fsub = int(self.preset.qubic.joint_out.qubic.Nsub*2 / self.preset.fg.params_foregrounds['bin_mixing_matrix'])

        if self.preset.fg.params_foregrounds['Dust']['type'] == 'parametric' and self.preset.fg.params_foregrounds['Synchrotron']['type'] == 'parametric':
           pass
        elif self.preset.fg.params_foregrounds['Dust']['type'] == 'blind' and self.preset.fg.params_foregrounds['Synchrotron']['type'] == 'blind':
           self.chi2 = Chi2Blind(self.preset)
        elif self.preset.fg.params_foregrounds['Dust']['type'] == 'parametric' and self.preset.fg.params_foregrounds['Synchrotron']['type'] == 'blind':
           self.chi2 = Chi2Blind(self.preset)
        elif self.preset.fg.params_foregrounds['Dust']['type'] == 'blind' and self.preset.fg.params_foregrounds['Synchrotron']['type'] == 'parametric':
           self.chi2 = Chi2Blind(self.preset)
        else:
           raise TypeError(f"{self.preset.fg.params_foregrounds['type']} is not yet implemented..")
        
        self.plots = Plots(self.preset, dogif=True)
        self._rms_noise_qubic_patch_per_ite = np.empty((self.preset.tools.params['PCG']['ites_to_converge'],len(self.preset.fg.components_out)))
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
        
        while self._info:


            self._display_iter()
            
            ### Update self.fg.components_iter^{k} -> self.fg.components_iter^{k+1}
            self._update_components()
            
            ### Update self.preset.acquisition.beta_iter^{k} -> self.preset.acquisition.beta_iter^{k+1}
            if self.preset.fg.params_foregrounds['fit_spectral_index']:
                self._update_spectral_index()
            else:
                self._index_seenpix_beta = None
                
            #stop
            ### Update self.gain.gain_iter^{k} -> self.gain.gain_iter^{k+1}
            if self.preset.qubic.params_qubic['GAIN']['fit_gain']:
                self._update_gain()
            
            ### Wait for all processes and save data inside pickle file
            self.preset.tools.comm.Barrier()
            self._save_data()
            
            ### Compute the rms of the noise per iteration to later analyze its convergence in _stop_condition
            #self._compute_maxrms_array()

            ### Stop the loop when self._steps > k
            self._stop_condition()
    def _callback(self, x):
        
        """
        
        Method to make callback function readable by `scipy.optimize.minimize`.
        
        """
        
        self.preset.tools.comm.Barrier()
        if self.preset.tools.rank == 0:
            if (self.nfev%5) == 0:
                print(f"Iter = {self.nfev:4d}   A = {[np.round(x[i], 5) for i in range(len(x))]}")
            self.nfev += 1
    def _get_tod_comp(self):
        
        tod_comp = np.zeros((len(self.preset.fg.components_name_out), self.preset.qubic.joint_out.qubic.Nsub*2, self.preset.qubic.joint_out.qubic.ndets*self.preset.qubic.joint_out.qubic.nsamples))
        
        for i in range(len(self.preset.fg.components_name_out)):
            for j in range(self.preset.qubic.joint_out.qubic.Nsub*2):
                if self.preset.qubic.params_qubic['convolution_out']:
                    C = HealpixConvolutionGaussianOperator(fwhm = self.preset.acquisition.fwhm_mapmaking[j], lmax=3*self.preset.sky.params_sky['nside'])
                else:
                    C = HealpixConvolutionGaussianOperator(fwhm = 0, lmax=3*self.preset.sky.params_sky['nside'])
                tod_comp[i, j] = self.preset.qubic.joint_out.qubic.H[j](C(self.preset.fg.components_iter[i])).ravel()
        
        return tod_comp
    def _get_tod_comp_superpixel(self, index):
        if self.preset.tools.rank == 0:
            print('Computing contribution of each super-pixel')
        _index = np.zeros(12*self.preset.fg.params_foregrounds['Dust']['nside_beta_out']**2)
        _index[index] = index.copy()
        _index_nside = hp.ud_grade(_index, self.preset.qubic.joint_out.external.nside)
        tod_comp = np.zeros((len(index), self.preset.qubic.joint_out.qubic.Nsub*2, len(self.preset.fg.components_out), self.preset.qubic.joint_out.qubic.ndets*self.preset.qubic.joint_out.qubic.nsamples))
        
        maps_conv = self.preset.fg.components_iter.T.copy()

        for j in range(self.preset.qubic.params_qubic['nsub_out']):
            for co in range(len(self.preset.fg.components_out)):
                if self.preset.qubic.params_qubic['convolution_out']:
                    C = HealpixConvolutionGaussianOperator(fwhm=self.preset.acquisition.fwhm_mapmaking[j], lmax=3*self.preset.sky.params_sky['nside'])
                else:
                    C = HealpixConvolutionGaussianOperator(fwhm=0, lmax=3*self.preset.sky.params_sky['nside'])
                maps_conv[co] = C(self.preset.fg.components_iter[:, :, co].T).copy()
                for ii, i in enumerate(index):
        
                    maps_conv_i = maps_conv.copy()
                    _i = _index_nside == i
                    for stk in range(3):
                        maps_conv_i[:, :, stk] *= _i
                    tod_comp[ii, j, co] = self.preset.qubic.joint_out.qubic.H[j](maps_conv_i[co]).ravel()

        return tod_comp
    def _get_constrains(self):
        constraints = []
        n = (self.preset.fg.params_foregrounds['bin_mixing_matrix']-1)*(len(self.preset.fg.components_out)-1)
        
        if self.preset.fg.params_foregrounds['Dust']['Dust_out'] and (self.preset.fg.params_foregrounds['Synchrotron']['Synchrotron_out'] is False):
            for i in range(n):
                constraints.append(
                                    {'type': 'ineq', 'fun': lambda x, i=i: x[i+1] - x[i]}
                                  )
            return constraints
                
        elif (self.preset.fg.params_foregrounds['Dust']['Dust_out'] is False) and (self.preset.fg.params_foregrounds['Synchrotron']['Synchrotron_out']):
            for i in range(n):
                constraints.append(
                                    {'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[i+1]}
                                  )
            return constraints
        
        elif (self.preset.fg.params_foregrounds['Dust']['Dust_out'] is False) and (self.preset.fg.params_foregrounds['Synchrotron']['Synchrotron_out'] is False):
            return None

        elif (self.preset.fg.params_foregrounds['Dust']['Dust_out']) and (self.preset.fg.params_foregrounds['Synchrotron']['Synchrotron_out']):
            for i in range(n): 
                if i % 2 == 0:
                    constraints.append(
                                        {'type': 'ineq', 'fun': lambda x, i=i: x[i+2] - x[i]}
                                      )
                else:
                    constraints.append(
                                        {'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[i+2]}
                                      )
            return constraints
    def _update_mixing_matrix(self, beta, A, i):
        mixingmatrix = mm.MixingMatrix(*self.preset.fg.components_out)
        A_param = mixingmatrix.eval(self.preset.qubic.joint_out.qubic.allnus, *beta)
        A_blind = A
        for ii in range(self.preset.fg.params_foregrounds['bin_mixing_matrix']):
            A_blind[ii*self.fsub: (ii + 1)*self.fsub, i] = A_param[ii*self.fsub: (ii + 1)*self.fsub, i]
        return A_blind
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
        self._index_seenpix_beta = 0

        if method == 'parametric':
            if self.preset.fg.params_foregrounds['Dust']['nside_beta_out'] != 0:
                print('Not yet implemented')
            else :
                previous_beta = self.preset.acquisition.beta_iter.copy()

                chi2 = Chi2Parametric(self.preset, tod_comp, self.preset.acquisition.beta_iter, seenpix_wrap=None)
                
                self.preset.acquisition.beta_iter = np.array([fmin_l_bfgs_b(chi2, 
                                                              x0=self.preset.acquisition.beta_iter, 
                                                              callback=self._callback, 
                                                              approx_grad=True, 
                                                              epsilon=1e-6)[0]])
                
                del tod_comp
                gc.collect()
                
                if self.preset.tools.rank == 0:
                
                    print(f'Iteration k     : {previous_beta}')
                    print(f'Iteration k + 1 : {self.preset.acquisition.beta_iter.copy()}')
                    print(f'Truth           : {self.preset.mixingmatrix.beta_in.copy()}')
                    print(f'Residuals       : {self.preset.mixingmatrix.beta_in - self.preset.acquisition.beta_iter}')
                    
                    if len(self.preset.fg.components_out) > 2:
                        self.plots.plot_beta_iteration(self.preset.acquisition.allbeta[:, 0], truth=self.preset.mixingmatrix.beta_in[0], ki=self._steps)
                    else:
                        self.plots.plot_beta_iteration(self.preset.acquisition.allbeta, truth=self.preset.mixingmatrix.beta_in, ki=self._steps)
            
                self.preset.tools.comm.Barrier()

                self.preset.acquisition.allbeta = np.concatenate((self.preset.acquisition.allbeta, self.preset.acquisition.beta_iter), axis=0) 
            
        elif method == 'blind':
            previous_step = self.preset.acquisition.Amm_iter[:self.preset.qubic.joint_out.qubic.Nsub*2, 1:].copy()
            if self._steps == 0:
                self.allAmm_iter = np.array([self.preset.acquisition.Amm_iter]) 

            if self.preset.fg.params_foregrounds['blind_method'] == 'minimize' :
                ### Function to minimize
                fun = partial(self.chi2._qu, tod_comp=tod_comp)
                
                ### Starting point
                x0 = []
                bnds = []
                for ii in range(self.preset.fg.params_foregrounds['bin_mixing_matrix']):
                    for i in range(1, len(self.preset.fg.components_out)):
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
                    for i in range(1, len(self.preset.fg.components_out)):
                        self.preset.acquisition.Amm_iter[ii*self.fsub:(ii+1)*self.fsub, i] = Ai[k]
                        k+=1

            elif self.preset.fg.params_foregrounds['blind_method'] == 'PCG' :
                tod_comp_binned = np.zeros((tod_comp.shape[0], self.preset.fg.params_foregrounds['bin_mixing_matrix'], tod_comp.shape[-1]))
                for k in range(len(self.preset.fg.components_out)):
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
                for i in range(1, len(self.preset.fg.components_out)):
                    for ii in range(self.preset.fg.params_foregrounds['bin_mixing_matrix']):
                        #print(i, ii*fsub, (ii+1)*fsub, fsub)
                        self.preset.acquisition.Amm_iter[ii*self.fsub:(ii+1)*self.fsub, i] = s['x'][k]#Ai[k]
                        k+=1
                
            elif self.preset.fg.params_foregrounds['blind_method'] == 'alternate' :
                for i in range(len(self.preset.fg.components_out)):
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
                print(f'Iteration k + 1 : {self.preset.acquisition.Amm_iter[:self.preset.qubic.joint_out.qubic.Nsub*2, 1:].ravel()}')
                print(f'Truth           : {self.preset.mixingmatrix.Amm_in[:self.preset.qubic.joint_out.qubic.Nsub*2, 1:].ravel()}')
                print(f'Residuals       : {self.preset.mixingmatrix.Amm_in[:self.preset.qubic.joint_out.qubic.Nsub*2, 1:].ravel() - self.preset.acquisition.Amm_iter[:self.preset.qubic.joint_out.qubic.Nsub*2, 1:].ravel()}')
                self.plots.plot_sed(self.preset.qubic.joint_out.qubic.allnus, 
                                    self.allAmm_iter[:, :self.preset.qubic.joint_out.qubic.Nsub*2, 1:], 
                                    ki=self._steps, truth=self.preset.mixingmatrix.Amm_in[:self.preset.qubic.joint_out.qubic.Nsub*2, 1:])

            del tod_comp
            gc.collect()

        elif method == 'parametric_blind':
            previous_step = self.preset.acquisition.Amm_iter[:self.preset.qubic.joint_out.qubic.Nsub*2, 1:].copy()
            if self._steps == 0:
                self.allAmm_iter = np.array([self.preset.acquisition.Amm_iter]) 
            for i in range(len(self.preset.fg.components_out)):
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
                            for j in range(1, len(self.preset.fg.components_out)):
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
                print(f'Iteration k + 1 : {self.preset.acquisition.Amm_iter[:self.preset.qubic.joint_out.qubic.Nsub*2, 1:].ravel()}')
                print(f'Truth           : {self.preset.mixingmatrix.Amm_in[:self.preset.qubic.joint_out.qubic.Nsub*2, 1:].ravel()}')
                print(f'Residuals       : {self.preset.mixingmatrix.Amm_in[:self.preset.qubic.joint_out.qubic.Nsub*2, 1:].ravel() - self.preset.acquisition.Amm_iter[:self.preset.qubic.joint_out.qubic.Nsub*2, 1:].ravel()}')
                self.plots.plot_sed(self.preset.qubic.joint_out.qubic.allnus, 
                                    self.allAmm_iter[:, :self.preset.qubic.joint_out.qubic.Nsub*2, 1:], 
                                    ki=self._steps, truth=self.preset.mixingmatrix.Amm_in[:self.preset.qubic.joint_out.qubic.Nsub*2, 1:])

            del tod_comp
            gc.collect()   

    def _save_data(self):
        
        """
        
        Method that save data for each iterations. It saves components, gains, spectral index, coverage, seen pixels.
        
        """
        if self.preset.tools.rank == 0:
            if self.preset.tools.params['save_iter'] != 0:
                if (self._steps+1) % self.preset.tools.params['save_iter'] == 0:
                    
                    if self.preset.tools.params['lastite']:
                    
                        if self._steps != 0:
                            os.remove(self.preset.tools.params['foldername'] + '/' + self.preset.tools.params['filename']+f"_seed{str(self.preset.tools.params['CMB']['seed'])}_{str(self.preset.job_id)}_k{self._steps-1}.pkl")
                    
                    with open(self.preset.tools.params['foldername'] + '/' + self.preset.tools.params['filename']+f"_seed{str(self.preset.tools.params['CMB']['seed'])}_{str(self.preset.job_id)}_k{self._steps}.pkl", 'wb') as handle:
                        pickle.dump({'components':self.preset.fg.components_in, 
                                 'components_i':self.preset.fg.components_iter,
                                 'beta':self.preset.acquisition.allbeta,
                                 'beta_true':self.preset.mixingmatrix.beta_in,
                                 'index_beta':self._index_seenpix_beta,
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
    def _update_components(self, maxiter=None):
        
        """
        
        Method that solve the map-making equation ( H.T * invN * H ) * components = H.T * invN * TOD using OpenMP / MPI solver. 
        
        """
        H_i = self.preset.qubic.joint_out.get_operator(self.preset.acquisition.beta_iter, Amm=self.preset.acquisition.Amm_iter, gain=self.preset.gain.gain_iter, fwhm=self.preset.acquisition.fwhm_mapmaking, nu_co=self.preset.fg.nu_co)
        seenpix_var = self.preset.sky.seenpix
        
        #print(H_i.shapein, H_i.shapeout)
        #stop
        if self.preset.fg.params_foregrounds['Dust']['nside_beta_out'] == 0:
            U = (
                ReshapeOperator((len(self.preset.fg.components_name_out) * sum(seenpix_var) * 3), (len(self.preset.fg.components_name_out), sum(seenpix_var), 3)) *
                PackOperator(np.broadcast_to(seenpix_var[None, :, None], (len(self.preset.fg.components_name_out), seenpix_var.size, 3)).copy())
            ).T
        else:
            U = (
                ReshapeOperator((3 * len(self.preset.fg.components_name_out) * sum(seenpix_var)), (3, sum(seenpix_var), len(self.preset.fg.components_name_out))) *
                PackOperator(np.broadcast_to(seenpix_var[None, :, None], (3, seenpix_var.size, len(self.preset.fg.components_name_out))).copy())
            ).T
        
        if self.preset.tools.params['PLANCK']['fix_pixels_outside_patch']:
            self.preset.A = U.T * H_i.T * self.preset.acquisition.invN * H_i * U
            if self.preset.fg.params_foregrounds['Dust']['nside_beta_out'] == 0:
                if self.preset.qubic.params_qubic['convolution_out']:
                    x_planck = self.preset.fg.components_convolved_out * (1 - seenpix_var[None, :, None])
                else:
                    x_planck = self.preset.fg.components_out * (1 - seenpix_var[None, :, None])
            self.preset.b = U.T (  H_i.T * self.preset.acquisition.invN * (self.preset.acquisition.TOD_obs - H_i(x_planck)))
        elif self.preset.tools.params['PLANCK']['fixI']:
            mask = np.ones((len(self.preset.fg.components_out), 12*self.preset.sky.params_sky['nside']**2, 3))
            mask[:, :, 0] = 0
            P = (
                ReshapeOperator(PackOperator(mask).shapeout, (len(self.preset.fg.components_out), 12*self.preset.sky.params_sky['nside']**2, 2)) * 
                PackOperator(mask)
                ).T
            
            xI = self.preset.fg.components_convolved_out * (1 - mask)
            self.preset.A = P.T * H_i.T * self.preset.acquisition.invN * H_i * P
            self.preset.b = P.T (  H_i.T * self.preset.acquisition.invN * (self.preset.acquisition.TOD_obs - H_i(xI)))
        else:
            self.preset.A = H_i.T * self.preset.acquisition.invN * H_i
            self.preset.b = H_i.T * self.preset.acquisition.invN * self.preset.acquisition.TOD_obs
        
        self._call_pcg(maxiter=maxiter)
    def _call_pcg(self, maxiter=None):

        """
        
        Method that call the PCG in PyOperators.
        
        """
        if maxiter is None:
            maxiter=self.preset.tools.params['PCG']['n_iter_pcg']
        seenpix_var = self.preset.sky.seenpix
        #self.preset.fg.components_iter_minus_one = self.preset.fg.components_iter.copy()
        
        if self.preset.tools.params['PLANCK']['fix_pixels_outside_patch']:
            mypixels = mypcg(self.preset.A, 
                                    self.preset.b, 
                                    M=self.preset.acquisition.M, 
                                    tol=self.preset.tools.params['PCG']['tol_pcg'], 
                                    x0=self.preset.fg.components_iter[:, seenpix_var, :], 
                                    maxiter=maxiter, 
                                    disp=True,
                                    create_gif=False,
                                    center=self.preset.sky.center, 
                                    reso=self.preset.qubic.params_qubic['dtheta'], 
                                    seenpix=self.preset.sky.seenpix, 
                                    truth=self.preset.fg.components_out,
                                    reuse_initial_state=False)['x']['x']  
            self.preset.fg.components_iter[:, seenpix_var, :] = mypixels.copy()
        elif self.preset.tools.params['PLANCK']['fixI']:
            mypixels = mypcg(self.preset.A, 
                                    self.preset.b, 
                                    M=self.preset.acquisition.M, 
                                    tol=self.preset.tools.params['PCG']['tol_pcg'], 
                                    x0=self.preset.fg.components_iter[:, :, 1:], 
                                    maxiter=maxiter, 
                                    disp=True,
                                    create_gif=False,
                                    center=self.preset.sky.center, 
                                    reso=self.preset.qubic.params_qubic['dtheta'], 
                                    seenpix=self.preset.sky.seenpix, 
                                    truth=self.preset.fg.components_out,
                                    reuse_initial_state=False)['x']['x']  
            self.preset.fg.components_iter[:, :, 1:] = mypixels.copy()
        else:
            mypixels = mypcg(self.preset.A, 
                                    self.preset.b, 
                                    M=self.preset.acquisition.M, 
                                    tol=self.preset.tools.params['PCG']['tol_pcg'], 
                                    x0=self.preset.fg.components_iter, 
                                    maxiter=maxiter, 
                                    disp=True,
                                    create_gif=False,
                                    center=self.preset.sky.center, 
                                    reso=self.preset.qubic.params_qubic['dtheta'], 
                                    seenpix=self.preset.sky.seenpix, 
                                    truth=self.preset.fg.components_out,
                                    reuse_initial_state=False)['x']['x']  
            self.preset.fg.components_iter = mypixels.copy()
        #stop
        C = HealpixConvolutionGaussianOperator(fwhm=self.preset.acquisition.fwhm_reconstructed)
        map_to_namaster = C(self.preset.fg.components_iter[0] - self.preset.fg.components_out[0])
        map_to_namaster[~self.preset.sky.seenpix, :] = 0
        map_to_namaster[~self.preset.sky.seenpix, 0] = 0
        leff, dls, _ = self.preset.sky.namaster.get_spectra(map_to_namaster.T, beam_correction=np.rad2deg(self.preset.acquisition.fwhm_reconstructed), pixwin_correction=False, verbose=False)
        dl_BB = dls[:, 2] / self.preset.sky.cl2dl
        sigr = self._fisher(leff, dl_BB)
        self.preset.tools._print_message(f'sigma(r) = {sigr:.6f}')
        
        
        if self.preset.tools.rank == 0:
            self.plots.display_maps(self.preset.sky.seenpix, ki=self._steps)
            self.plots._display_allcomponents(self.preset.sky.seenpix, ki=self._steps)  
            self.plots.plot_rms_iteration(self.preset.acquisition.rms_plot, ki=self._steps) 
    def _fisher(self, ell, Nl):
        
        '''
        
        Fisher to compute sigma(r) for a given noise power spectrum.
        
        '''
        
        
        Dl = np.interp(ell, np.arange(1, 4001, 1), give_cl_cmb(r=1, Alens=0)[2])
        s = np.sum((ell + 0.5) * self.preset.sky.namaster.fsky * self.preset.tools.params['SPECTRUM']['dl'] * (Dl / (Nl))**2)
        

        return s**(-1/2)
    def _compute_map_noise_qubic_patch(self):
        
        """
        
        Compute the rms of the noise within the qubic patch.
        
        """
        nbins = 1 #average over the entire qubic patch

        if self.preset.fg.params_foregrounds['Dust']['nside_beta_out'] == 0:
            if self.preset.qubic.params_qubic['convolution_out']:
                residual = self.preset.fg.components_iter - self.preset.fg.components_convolved_out
            else:
                residual = self.preset.fg.components_iter - self.preset.fg.components_out
        else:
            if self.preset.qubic.params_qubic['convolution_out']:
                residual = self.preset.fg.components_iter.T - self.preset.fg.components_convolved_out
            else:
                residual = self.preset.fg.components_iter.T - self.preset.fg.components_out.T
        rms_maxpercomp = np.zeros(len(self.preset.fg.components_out))

        for i in range(len(self.preset.fg.components_out)):
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
            
            deltarms_max_percomp = np.zeros(len(self.preset.fg.components_out))

            for i in range(len(self.preset.fg.components_out)):
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
    def _display_iter(self):
        
        """
        
        Method that display the number of a specific iteration k.
        
        """
        
        if self.preset.tools.rank == 0:
            print('========== Iter {}/{} =========='.format(self._steps+1, self.preset.tools.params['PCG']['n_iter_loop']))
    def _update_gain(self):
        
        """
        
        Method that compute gains of each detectors using semi-analytical method g_i = TOD_obs_i / TOD_sim_i
        
        """
        
        self.H_i = self.preset.qubic.joint_out.get_operator(self.preset.acquisition.beta_iter, Amm=self.preset.acquisition.Amm_iter, gain=np.ones(self.preset.gain.gain_iter.shape), fwhm=self.preset.acquisition.fwhm_mapmaking, nu_co=self.preset.fg.nu_co)
        self.nsampling = self.preset.qubic.joint_out.qubic.nsamples
        self.ndets = self.preset.qubic.joint_out.qubic.ndets
        if self.preset.qubic.params_qubic['instrument'] == 'UWB':
            _r = ReshapeOperator(self.preset.qubic.joint_out.qubic.ndets*self.preset.joint.qubic.nsamples, (self.preset.joint.qubic.ndets, self.preset.joint.qubic.nsamples))
            #R2det_i = ReshapeOperator(self.preset.joint.qubic.ndets*self.preset.joint.qubic.nsamples, (self.preset.joint.qubic.ndets, self.preset.joint.qubic.nsamples))
            #print(R2det_i.shapein, R2det_i.shapeout)
            #TOD_Q_ALL_i = R2det_i(self.H_i.operands[0](self.preset.fg.components_iter))
            TODi_Q = self.preset.acquisition.invN.operands[0](self.H_i.operands[0](self.preset.fg.components_iter)[:self.ndets*self.nsampling])
            self.preset.gain.gain_iter = self._give_me_intercal(TODi_Q, _r(self.preset.TOD_Q))
            self.preset.gain.gain_iter /= self.preset.gain.gain_iter[0]
            self.preset.allg = np.concatenate((self.preset.allg, np.array([self.preset.gain.gain_iter])), axis=0)
            
        elif self.preset.qubic.params_qubic['instrument'] == 'DB':
            
            #R2det_i = ReshapeOperator(2*self.preset.joint.qubic.ndets*self.preset.joint.qubic.nsamples, (2*self.preset.joint.qubic.ndets, self.preset.joint.qubic.nsamples))
            TODi_Q_150 = self.H_i.operands[0](self.preset.fg.components_iter)[:self.ndets*self.nsampling]
            TODi_Q_220 = self.H_i.operands[0](self.preset.fg.components_iter)[self.ndets*self.nsampling:2*self.ndets*self.nsampling]
            
            g150 = self._give_me_intercal(TODi_Q_150, self.preset.TOD_Q[:self.ndets*self.nsampling], self.preset.acquisition.invN.operands[0].operands[1].operands[0])
            g220 = self._give_me_intercal(TODi_Q_220, self.preset.TOD_Q[self.ndets*self.nsampling:2*self.ndets*self.nsampling], self.preset.acquisition.invN.operands[0].operands[1].operands[1])
            #g150 /= g150[0]
            #g220 /= g220[0]
            
            self.preset.gain.gain_iter = np.array([g150, g220]).T
            self.preset.Gi = join_data(self.preset.tools.comm, self.preset.gain.gain_iter)
            self.preset.allg = np.concatenate((self.preset.allg, np.array([self.preset.gain.gain_iter])), axis=0)
            #print()
            #stop
            if self.preset.tools.rank == 0:
                print(np.mean(self.preset.gain.gain_iter - self.preset.g, axis=0))
                print(np.std(self.preset.gain.gain_iter - self.preset.g, axis=0))
            
        #stop
        #### Display convergence of beta
        self.plots.plot_gain_iteration(self.preset.allg - self.preset.g, alpha=0.03, ki=self._steps)
    def _give_me_intercal(self, D, d, _invn):
        
        """
        
        Semi-analytical method for gains estimation.

        """
        
        _r = ReshapeOperator(self.preset.qubic.joint_out.qubic.ndets*self.preset.qubic.joint_out.qubic.nsamples, (self.preset.qubic.joint_out.qubic.ndets, self.preset.qubic.joint_out.qubic.nsamples))
        
        return (1/np.sum(_r(D) * _invn(_r(D)), axis=1)) * np.sum(_r(D) * _invn(_r(d)), axis=1)
