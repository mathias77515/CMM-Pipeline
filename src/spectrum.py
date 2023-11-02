import numpy as np
import matplotlib.pyplot as plt
import pickle
import healpy as hp
import emcee
from multiprocess import Pool
from getdist import plots, MCSamples
import os

import qubic
from qubic import NamasterLib as nam

#with open(path_to_data + 'CMM_MC_seed1/CMM_gain_bias_300_1.pkl', 'rb') as f:
#    data = pickle.load(f)
    
#seenpix = data['coverage']/data['coverage'].max() > 0.2

#namaster = nam.Namaster(seenpix, lmin=30, lmax=40, delta_ell=35, aposize=10)
#print(namaster.get_spectra)

class Likelihood:

    def __init__(self, ell, data, cov, ncomps=2):
        
        self.ncomps = ncomps
        self.ell = ell
        self._f = self.ell * (self.ell + 1) / (2 * np.pi)
        self.data = data
        self.cov = cov
        
    def give_cl_cmb(self, r, Alens):
        
        """
        
        Method to get theoretical CMB BB power spectrum according to Alens and r.


        """
        
        power_spectrum = hp.read_cl('data/Cls_Planck2018_lensed_scalar.fits')[:,:4000]
        if Alens != 1.:
            power_spectrum *= Alens
        if r:
            power_spectrum += r * hp.read_cl('data/Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits')[:,:4000]
        
        return np.interp(self.ell, np.linspace(1, 4000, 4000), power_spectrum[2])

    def foregrounds(self, ell, A, alpha):
        return A * (ell/80)**alpha
        
    def model(self, params):
        r, A, alpha = params

        #self.give_cl_cmb(r, Alens)
        ymodel = np.array([self._f * self.give_cl_cmb(r, 1), self.foregrounds(self.ell, A, alpha)])
        #ymodel = np.array([self._f * self.give_cl_cmb(r, 1)])
        
        return ymodel

    def log_prior(self, params):

        if params[0] < -1 or params[0] > 1:
            return -np.inf
        #if params[1] < 0 or params[1] > 2:
        #    return -np.inf
        if params[1] < 0 or params[1] > 10:
            return -np.inf
        if params[2] < -1 or params[2] > 0:
            return -np.inf

        return 0
    
    def _plot_chain(self, chains):

        plt.figure(figsize=(12, 5))
    
        for iparam in range(chains.shape[2]):
            plt.subplot(chains.shape[2], 1, iparam+1)
            for iwalk in range(chains.shape[1]):
                plt.plot(chains[:, iwalk, iparam], '-k', alpha=0.1)
            plt.plot(np.mean(chains[:, :, iparam], axis=1), '-r', alpha=1)
            plt.xlim(0, len(chains[:, 0, 0]))
        plt.xlabel('Iterations', fontsize=14)
        plt.savefig('chains.png')
        plt.close()
        
    def _get_triangle(self, chains):
        
        labels = []
        names = []

        for i in range(chains.shape[1]):
            names += [f'sig_{i}']
            labels += [f'\sigma_{i}']
            
        s = MCSamples(samples=chains, names=names, labels=labels, label=r'')#, ranges={'sig_0':(0, None)})
        plt.figure(figsize=(12, 8))
        # Triangle plot
        g = plots.get_subplot_plotter(width_inch=8)
        g.triangle_plot([s], filled=True, title_limit=1)
        plt.savefig('triangle.png')
        plt.close()

    def __call__(self, params):

        ### Residuals
        _r = self.data - self.model(params)
        chi2 = 0

        for i in range(self.cov.shape[0]):
            for j in range(self.cov.shape[3]):
                invcov = np.linalg.pinv(self.cov[i, :, :, j])
                chi2 += self.log_prior(params) - 0.5 * (_r[i].T @ invcov @ _r[j])
    
        return chi2
        
class ForecastCMM:
    
    def __init__(self, path_to_data, ncomps, nside, lmin=40, dl=30, aposize=10, type='varying'):
        
        self.files = os.listdir(path_to_data)
        self.type = type
        self.N = len(self.files)
        self.ncomps = ncomps
        self.nside = nside
        self.dl = dl
        self.lmin = lmin
        self.lmax = 2 * self.nside - 1
        self.aposize = aposize
        
        if self.type == 'varying':
            self.components = np.zeros((self.N, 3, 12*self.nside**2, self.ncomps))
            self.components_true = np.zeros((self.N, 3, 12*self.nside**2, self.ncomps))
            self.residuals = np.zeros((self.N, 3, 12*self.nside**2, self.ncomps))
        else:
            self.components = np.zeros((self.N, self.ncomps, 12*self.nside**2, 3))
            self.components_true = np.zeros((self.N, self.ncomps, 12*self.nside**2, 3))
            self.residuals = np.zeros((self.N, self.ncomps, 12*self.nside**2, 3))
        
        self.coverage = self._open_data(path_to_data+self.files[0], 'coverage')
        self.seenpix = hp.ud_grade(self.coverage / self.coverage.max() > 0.2, self.nside)
        
        print('======= Reading data =======')
        if self.type == 'varying':
            for i in range(self.N):
                print(f'Realization #{i+1} - {path_to_data+self.files[i]}')
                for j in range(self.ncomps):
                    for k in range(3):
                        
                        self.components[i, k, :, j] = hp.ud_grade(self._open_data(path_to_data+self.files[i], 'components_i')[k, :, j], self.nside)
                        self.components_true[i, k, :, j] = hp.ud_grade(self._open_data(path_to_data+self.files[i], 'components')[k, :, j], self.nside)
                        self.residuals[i, k, :, j] = hp.ud_grade(self.components[i, k, :, j] - self.components_true[i, k, :, j], self.nside)
        else:
            for i in range(self.N):
                print(f'Realization #{i+1} - {path_to_data+self.files[i]}')
                for j in range(self.ncomps):
                    for k in range(3):
                        
                        self.components[i, j, :, k] = hp.ud_grade(self._open_data(path_to_data+self.files[i], 'components_i')[j, :, k], self.nside)
                        self.components_true[i, j, :, k] = hp.ud_grade(self._open_data(path_to_data+self.files[i], 'components')[j, :, k], self.nside)
                        self.residuals[i, j, :, k] = hp.ud_grade(self.components[i, j, :, k] - self.components_true[i, j, :, k], self.nside)
                    
        self.components[:, :, ~self.seenpix, :] = 0
        self.residuals[:, :, ~self.seenpix, :] = 0
        print('======= Reading data - done =======')
        
        self.namaster = nam.Namaster(self.seenpix, lmin=lmin, lmax=self.lmax, delta_ell=self.dl, aposize=self.aposize)
        self.ell, _ = self.namaster.get_binning(self.nside)
        self._f = self.ell * (self.ell + 1) / (2 * np.pi)
        self.DlBB_1x1 = np.zeros((self.N, len(self.ell)))
        self.DlBB_2x2 = np.zeros((self.N, len(self.ell)))
        self.DlBB_1x2 = np.zeros((self.N, len(self.ell)))
        self.DlBB = np.zeros((self.N, self.ncomps, len(self.ell)))
        self.Nl = np.zeros((self.N, self.ncomps, len(self.ell))) 

    def _open_data(self, name, keyword):
        with open(name, 'rb') as f:
            data = pickle.load(f)
        return data[keyword]
    def _get_covar(self, X):

        """

        X -> (Nreal, Ncomps, Nbins)

        """

        nreals, ncomps, nbins = X.shape

        covariance = np.zeros((ncomps, nbins, nbins, ncomps))

        for icomps in range(ncomps):
            for jcomps in range(ncomps):
                c = np.cov(X[:, icomps, :].T, X[:, jcomps, :].T, rowvar=True)[icomps*nbins:(icomps+1)*nbins, jcomps*nbins:(jcomps+1)*nbins]
                covariance[icomps, :, :, jcomps] = c.copy()
        return covariance
    def _get_BB_spectrum(self, map1, map2=None, beam_correction=None, pixwin_correction=False):
        
        if map1.shape == (3, 12*self.nside**2):
            pass
        else:
            map1 = map1.T
        
        if map2 is not None:
            if map2.shape == (3, 12*self.nside**2):
                pass
            else:
                map2 = map2.T
                
        leff, BB, _ = self.namaster.get_spectra(map1, map2=map2, beam_correction=beam_correction, pixwin_correction=pixwin_correction, verbose=False)
        return BB[:, 2]
    def give_cl_cmb(self, r=0, Alens=1.):
        
        power_spectrum = hp.read_cl('data/Cls_Planck2018_lensed_scalar.fits')[:,:4000]
        if Alens != 1.:
            power_spectrum[2] *= Alens
        if r:
            power_spectrum += r * hp.read_cl('data/Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits')[:,:4000]
        return np.interp(self.ell, np.arange(1, 4001, 1), power_spectrum[2])
    def _get_x0(self, n, mu):
        
        x0 = np.zeros((n, len(mu)))
        for ii, i in enumerate(mu):
            x0[:, ii] = np.random.normal(i, 0.00001, (n))
        
        return x0
    def __call__(self):
        
        for i in range(self.N):
            print(f'\n========= Iteration {i+1}/{self.N} ========')
            if self.type == 'varying':
                print(f'     -> 1x1')
                self.DlBB_1x1[i] = self._get_BB_spectrum(self.residuals[i, :, :, 0].T)
                print(f'     -> 2x2')
                self.DlBB_2x2[i] = self._get_BB_spectrum(self.residuals[i, :, :, 1].T)
                print(f'     -> 1x2')
                self.DlBB_1x2[i] = self._get_BB_spectrum(self.residuals[i, :, :, 0].T, self.residuals[i, :, :, 1].T)
                print(f'     -> 1')
                self.DlBB[i, 0] = self._get_BB_spectrum(self.components[i, :, :, 0].T)
                print(f'     -> 2')
                self.DlBB[i, 1] = self._get_BB_spectrum(self.components[i, :, :, 1].T)
            else:
                print(f'     -> 1x1')
                self.DlBB_1x1[i] = self._get_BB_spectrum(self.residuals[i, 0])
                print(f'     -> 2x2')
                self.DlBB_2x2[i] = self._get_BB_spectrum(self.residuals[i, 1])
                print(f'     -> 1x2')
                self.DlBB_1x2[i] = self._get_BB_spectrum(self.residuals[i, 0], self.residuals[i, 1])
                print(f'     -> 1')
                self.DlBB[i, 0] = self._get_BB_spectrum(self.components[i, 0])
                print(f'     -> 2')
                self.DlBB[i, 1] = self._get_BB_spectrum(self.components[i, 1])
        return self.DlBB, self.DlBB_1x1, self.DlBB_2x2, self.DlBB_1x2
    
nside = 256
lmin = 40
dl = 30
ncomps = 2
path_to_data = '/home/regnier/work/regnier/CMM-Pipeline/src/d6_blind_forecastpaper_dualband/'

forecast = ForecastCMM(path_to_data, ncomps, nside, lmin=lmin, dl=dl, type='constant')
Dl, Nl_1x1, Dl_2x2, Nl_1x2 = forecast()

#print(Dl.shape)
mycl = forecast.give_cl_cmb()
_f = forecast.ell * (forecast.ell + 1) / (2 * np.pi)
plt.figure()
plt.errorbar(forecast.ell, np.mean(Dl[:, 0, :] - Nl_1x1, axis=0), yerr=np.std(Nl_1x1, axis=0), fmt='ko', capsize=3)
plt.plot(forecast.ell, _f * mycl)
plt.yscale('log')
plt.savefig('mydl2.png')
plt.close()

with open(f"forecast_d6_blind_spectrum_CMM_20reals_dual.pkl", 'wb') as handle:
    pickle.dump({'ell':forecast.ell, 
                 'Dl':Dl, 
                 'Dl_1x1':Nl_1x1,
                 'Dl_2x2':Dl_2x2, 
                 'Dl_1x2':Nl_1x2
                 }, handle, protocol=pickle.HIGHEST_PROTOCOL)