######## This file perform a likelihood estimation on the tensor-to-scalar ratio. It take
######## as input a pickle file which have the ell, Dl and errors and return the same file 
######## with the likelihood on r.


import numpy as np
import pickle
import os
from pyoperators import MPI
import healpy as hp
import matplotlib.pyplot as plt
from functools import partial
from multiprocess import Pool
from schwimmbad import MPIPool
import emcee
from getdist import plots, MCSamples
import sys

nwalkers = int(sys.argv[1])
nsteps = int(sys.argv[2])

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

class Forecast:
    
    def __init__(self, ell, cov, bias, params, r_init=0, Alens_init=1, mu=[0.01, 1], sig=[0.001, 0.1]):
        
        self.job_id = os.environ.get('SLURM_JOB_ID')
        self.ell = ell
        self._f = self.ell * (self.ell + 1) / (2 * np.pi)
        self.bias = bias
        self.Dl = self.give_dl_cmb(r=r_init, Alens=Alens_init) + self.bias
        
        self.params = params
        
        self.nparams = 1
        
        self.cov = cov
        self.invcov = np.linalg.inv(self.cov)
        
        self.mu = mu
        self.sig = sig
        #self._plot_Dl()
        
    def _plot_Dl(self):
        plt.figure()
        
        plt.errorbar(self.ell, self.Dl, yerr=np.sqrt(np.diag(self.cov)), fmt='or', capsize=3)

        plt.plot(self.ell, self.give_dl_cmb(r=0, Alens=1), label='Theoretical r = 0 | Alens = 1')

        plt.yscale('log')
        plt.legend(frameon=False, fontsize=12)
        
        
        plt.savefig(f'Dl_{self.job_id}.png')
        plt.close()
        
    def _init_mcmc(self, nwalkers):
        
        x0 = np.zeros((nwalkers, self.nparams))

        for i in range(self.nparams):
            x0[:, i] = np.random.normal(self.mu[i], self.sig[i], (nwalkers))
        return x0
    def give_dl_cmb(self, r=0, Alens=1.):
        
        power_spectrum = hp.read_cl('/home/regnier/work/regnier/CMM-Pipeline/src/data/Cls_Planck2018_lensed_scalar.fits')[:,:4000]
        if Alens != 1.:
            power_spectrum[2] *= Alens
        if r:
            power_spectrum += r * hp.read_cl('/home/regnier/work/regnier/CMM-Pipeline/src/data/Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits')[:,:4000]
        return self._f * np.interp(self.ell, np.arange(1, 4001, 1), power_spectrum[2]) 
    def log_prior(self, x):
        
        r = x
        #if self.params[0] != None: 
        #    r = self.params[0]
        #    Alens = x
        #if self.params[1] != None: 
        #    r = x
        #    Alens = self.params[0]
        #if self.params[0] == None and self.params[0] == None: 
        #    r, Alens = x
            
        if r < -1 or r > 1:
            return -np.inf
        
        return 0     
    def likelihood_one_real(self, x, i):
        
        if self.params[0] != None: 
            r = self.params[0]
            Alens = x
        if self.params[1] != None: 
            r = x
            Alens = self.params[0]
        if self.params[0] == None and self.params[0] == None: 
            r, Alens = x
            
        ysim = self.give_dl_cmb(r=r, Alens=Alens)
        _r = (self.Dl_noisy[i] - self.Nl[i]) - ysim
        
        return self.log_prior(x) - 0.5 * (_r.T @ self.invcov @ _r)
    def likelihood(self, x):
        
        r = x
        Alens=1
        ysim = self.give_dl_cmb(r=r, Alens=Alens)
        yobs = self.Dl.copy()
        _r = yobs - ysim
        #print(ysim)
        #print(np.mean(self.Dl_noisy, axis=0))
        
        return self.log_prior(x) - 0.5 * (_r.T @ self.invcov @ _r)
    def _plot_chains(self, chains):
        
        plt.figure(figsize=(8, 6))  
        nsamp, nwalk, ndim = chains.shape
        
        for dim in range(ndim):
            plt.subplot(ndim, 1, dim+1)
            for i in range(nwalk):
                plt.plot(chains[:, i, dim], '-k', alpha=0.1)
        
            plt.plot(np.mean(chains, axis=1)[:, dim], '-k')
            
        plt.savefig(f'chains_{self.job_id}.png')
        plt.close()
    def _plot_1d(self, chainflat, label):
        
        labels = ['r']#, 'A_{lens}']
        names = ['r']#, 'Alens']

        s = MCSamples(samples=chainflat, names=names, labels=labels, label=label, ranges={'r':(0, None)})
        
        plt.figure(figsize=(12, 8))
        # 1D marginalized comparison plot
        g = plots.get_single_plotter(width_inch=3)
        g.plot_1d([s], 'r', title_limit=1)

        plt.savefig(f'plot1d_{self.job_id}.png')
        plt.close()  
    def _get_triangle(self, chainflat, label):
        
        labels = ['r']#, 'A_{lens}']
        names = ['r']#, 'Alens']

        s = MCSamples(samples=chainflat, names=names, labels=labels, label=label, ranges={'r':(0, None)})

        plt.figure(figsize=(12, 8))

        # Triangle plot
        g = plots.get_subplot_plotter(width_inch=10)
        g.triangle_plot([s], filled=True, title_limit=1)

        plt.savefig(f'triangle_{self.job_id}.png')
        plt.close()
        
    def run(self, nwalkers, nsteps, dis=0):
        
        x0 = self._init_mcmc(nwalkers)
        
        if size != 1:
            with MPIPool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, x0.shape[1], self.likelihood, pool=pool)
                sampler.run_mcmc(x0, nsteps, progress=True)
        else:    
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, x0.shape[1], self.likelihood, pool=pool)
                sampler.run_mcmc(x0, nsteps, progress=True)
        
        chainflat = sampler.get_chain(discard=dis, thin=15, flat=True)
        chains = sampler.get_chain()

        comm.Barrier()
        
        #self._get_triangle(chainflat, label='test')
        #self._plot_1d(chainflat, label='test')
        #self._plot_chains(chains)
        
        return chains, chainflat
    
    def run_one_real(self, nwalkers, nsteps, dis=0):
        
        x0 = self._init_mcmc(nwalkers)
        r = np.zeros(self.Dl_noisy.shape[0])
        Alens = np.zeros(self.Dl_noisy.shape[0]) 
        
        for i in range(r.shape[0]):
            
            like = partial(self.likelihood_one_real, i=i)
            if size != 1:
                raise TypeError('You should not use several MPI tasks...')
                    
            else:    
                with Pool() as pool:
                    sampler = emcee.EnsembleSampler(nwalkers, x0.shape[1], like, pool=pool)
                    sampler.run_mcmc(x0, nsteps, progress=False)
              
            chainflat = sampler.get_chain(discard=dis, thin=15, flat=True)

            r[i], Alens[i] = np.mean(chainflat, axis=0)
            
            print(f'\nRealization #{i+1} -> r = {r[i]:.6f} and Alens = {Alens[i]:.6f}')
            
            print(f'    Averaged r : {np.mean(r[:i+1])} +- {np.std(r[:i+1]):.6f}')
            print(f'    Averaged Alens : {np.mean(Alens[:i+1])} +- {np.std(Alens[:i+1]):.6f}')
            
            comm.Barrier()
            
        return r, Alens
            
            
def open_data(filename):      
    path = '/home/regnier/work/regnier/CMM-Pipeline/src/data_CMM/test_difference_noconvolution/'
    
    with open(path + '/' + filename, 'rb') as f:
        data = pickle.load(f)
    return data

instr = 'dual'
method = 'parametric'
#autospectrum_parametric_d0_forecastpaper_dualband_purcmb
filename = [f'autospectrum_{method}_d0_forecastpaper_fakeconv_sameseed.pkl', f'autospectrum_{method}_d0_forecastpaper_sameseed.pkl']
title = ['Fake convolution', 'True convolution']




bin_down = 0
bin_up = 16
onereal = False
nreal = 100
s = []
dis = 200

labels = ['r']#, 'A_{lens}']
names = ['r']#, 'Alens']

for iname, name in enumerate(filename):
    data = open_data(name)
    cov = np.cov(data['Dl'][:nreal, 0, bin_down:bin_up] - data['Nl'][:nreal, bin_down:bin_up], rowvar=False)

    ### Forecast
    forecast = Forecast(data['ell'][bin_down:bin_up], 
                    cov, 
                    bias=data['Dl_bias'][bin_down:bin_up]*0,
                    r_init=0, Alens_init=1,
                    params=[None, None], mu=[0.001, 0.1], sig=[0.0001, 0.01])

    chains, chainflat = forecast.run(nwalkers, nsteps, dis=dis)
    
    if rank == 0:
        
        s += [MCSamples(samples=chainflat, names=names, labels=labels, label=title[iname], ranges={'r':(0, None)})]
        
        with open("chains" + name[12:], 'wb') as handle:
            pickle.dump({'ell':data['ell'], 
                     'Dl':data['Dl'][:, 0, :],
                     'chains':chains, 
                     'chainflat':chainflat
                     }, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    
    comm.Barrier()


if rank == 0:
    plt.figure(figsize=(12, 8))

    # Triangle plot
    g = plots.get_subplot_plotter(width_inch=10)
    g.triangle_plot(s, filled=True, title_limit=1)

    plt.savefig(f'triangle_plot.png')
    plt.close()    