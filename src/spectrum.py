import numpy as np
import matplotlib.pyplot as plt
import pickle
import healpy as hp
import emcee
from multiprocess import Pool
from getdist import plots, MCSamples
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
import os
import sys

import qubic
from qubic import NamasterLib as nam
import data

t = 'varying'
nside = 256
lmin = 40
lmax = 2 * nside
aposize = 10
dl = 30
ncomps = 1
advanced_qubic = False
Alens = 0.1
#method = str(sys.argv[1])
#dustmodel = str(sys.argv[2])
#instr = str(sys.argv[3])

def extract_seed(filename):
    return int(filename.split('_')[1][4:])

class Spectrum:
    
    def __init__(self, path_to_data, lmin=40, lmax=512, dl=30, aposize=10, varying=True, center=qubic.equ2gal(0, -57)):

        self.files = sorted(os.listdir(path_to_data), key=extract_seed)
        self.N = len(self.files)
        if self.N % 2 != 0:
            self.N -= 1
            
        self.dl = dl
        self.lmin = lmin
        self.lmax = lmax
        self.aposize = aposize
        self.covcut = 0.3
        self.center = center
        self.jobid = os.environ.get('SLURM_JOB_ID')
        self.args_title = path_to_data.split('/')[-2].split('_')[:3]
        
        
        self.components_true = self._open_data(path_to_data+self.files[0], 'components')
        
        if varying:
            self.components_true = np.transpose(self.components_true, (2, 1, 0))
            self.nstk, self.npix, self.ncomps = self._open_data(path_to_data+self.files[0], 'components_i').shape
        else:
            self.ncomps, self.npix, self.nstk = self._open_data(path_to_data+self.files[0], 'components_i').shape
        self.nside = hp.npix2nside(self.npix)
        self.components = np.zeros((self.N, self.npix, 3))
        self.residuals = np.zeros((self.N, self.npix, 3))
        
        C = HealpixConvolutionGaussianOperator(fwhm=0.00415369, lmax=2*self.nside)
        C2 = HealpixConvolutionGaussianOperator(fwhm=np.sqrt(0.0078**2 - 0.00415369**2), lmax=2*self.nside)
        self.coverage = self._open_data(path_to_data+self.files[0], 'coverage')
        self.seenpix = self.coverage / self.coverage.max() > self.covcut
        
        
        for co in range(self.ncomps):
            self.components_true[co] = C(self.components_true[co])
        
        list_not_read = []
        print('    -> Reading data')
        for i in range(self.N):
            #print(self.nstk, self.npix, self.ncomps)
            try:
                            
                c = self._open_data(path_to_data+self.files[i], 'components_i')
                #print(c.shape)
                if varying:
                    self.components[i] = np.transpose(c, (2, 1, 0))[0]
                else:
                    self.components[i] = c[0].copy()
                
                #print(self.components.shape)
                #print(self.components_true.shape)
                #stop
                self.residuals[i] = self.components[i] - self.components_true[0]

                print(f'Realization #{i+1}')
                        
            except OSError as e:
                    
                list_not_read += [i]
                    
                print(f'Realization #{i+1} could not be read')
        
        print('    -> Reading data - done')
        ### Delete realizations still on going
        self.components = np.delete(self.components, list_not_read, axis=0)
        self.residuals = np.delete(self.residuals, list_not_read, axis=0)
        
        ### Set to 0 pixels not seen by QUBIC
        print('    -> Remove not seen pixels')
        self.components[:, ~self.seenpix, :] = 0
        self.components[:, :, 0] = 0
        self.components_true[:, ~self.seenpix, :] = 0
        self.residuals[:, ~self.seenpix, :] = 0
        
        ### Initiate spectra computation
        print('    -> Initialization of Namaster')
        self.N = self.components.shape[0]
        self.namaster = nam.Namaster(self.seenpix, lmin=self.lmin, lmax=self.lmax, delta_ell=self.dl, aposize=self.aposize)
        self.ell, _ = self.namaster.get_binning(self.nside)
        self._f = self.ell * (self.ell + 1) / (2 * np.pi)
        
        ### Average realizations over same CMB
        print('    -> Averaging realizations')
        _r = np.mean(self.components, axis=0) - self.components_true[0]
        
        plt.figure(figsize=(12, 4))
        nsig = 3
        hp.gnomview(np.mean(self.components[:, :, 1], axis=0), rot=self.center, reso=15, cmap='jet', min=-8, max=8,
                    title=f'Output averaged over realizations', notext=True, sub=(1, 3, 1))
        hp.gnomview(self.components_true[0, :, 1], rot=self.center, reso=15, cmap='jet', min=-8, max=8, notext=True,
                    title=f'Input', sub=(1, 3, 2))
        hp.gnomview(_r[:, 1], rot=self.center, reso=15, cmap='jet', notext=True,
                    title=f'RMS : {np.std(_r[self.seenpix, 1]):.3e}', sub=(1, 3, 3), min=-nsig*np.std(_r[self.seenpix, 1]), max=nsig*np.std(_r[self.seenpix, 1]))
        
        plt.suptitle(f'{self.args_title[0]} - {self.args_title[1]} - {self.args_title[2]}')
        plt.savefig(f'maps_{self.jobid}_Q.png')
        plt.close()
        
        
        plt.figure(figsize=(12, 4))
        nsig = 3
        hp.gnomview(np.mean(self.components[:, :, 2], axis=0), rot=self.center, reso=15, cmap='jet', min=-8, max=8,
                    title=f'Output averaged over realizations', notext=True, sub=(1, 3, 1))
        hp.gnomview(self.components_true[0, :, 2], rot=self.center, reso=15, cmap='jet', min=-8, max=8, notext=True,
                    title=f'Input', sub=(1, 3, 2))
        hp.gnomview(_r[:, 2], rot=self.center, reso=15, cmap='jet', notext=True,
                    title=f'RMS : {np.std(_r[self.seenpix, 2]):.3e}', sub=(1, 3, 3), min=-nsig*np.std(_r[self.seenpix, 2]), max=nsig*np.std(_r[self.seenpix, 2]))
        
        plt.suptitle(f'{self.args_title[0]} - {self.args_title[1]} - {self.args_title[2]}')
        plt.savefig(f'maps_{self.jobid}_U.png')
        plt.close()
        
        
        
        ### Create Dl array for bias, signal and noise
        print('    -> Computing bias Bl')
        self.BlBB = self._get_BB_spectrum(_r, beam_correction=np.rad2deg(0.00415369))
        
        plt.figure()
        #print(Alens)
        plt.errorbar(self.ell, self._f * self.give_cl_cmb(Alens=Alens), fmt='k-', capsize=3, label='Model')
        plt.errorbar(self.ell, self.BlBB, fmt='r-', capsize=3, label='Dl')
        plt.yscale('log')
        #plt.ylim(5e-4, 5e-2)
        plt.legend(frameon=False, fontsize=12)

        plt.savefig(f'bias_{os.environ.get("SLURM_JOB_ID")}.png')
        plt.close()
        print('Statistical bias -> ', self.BlBB)

        print(f'{self} initialized')
        #stop
    def main(self, cross=False):
        
        if cross:
            self.DlBB = np.zeros((int(self.N/2), len(self.ell)))
            k1=0
            k2=1
            for i in range(int(self.N/2)):
                print(i, k1, k2)
                self.DlBB[i] = self._get_BB_spectrum(self.components[k1], 
                                                     map2=self.components[k2], 
                                                     beam_correction=np.rad2deg(0.00415369))
                k1 += 2
                k2 += 2
            print(np.mean(self.DlBB, axis=0))
            print()
            print(np.std(self.DlBB, axis=0))
            
            plt.figure()

            plt.errorbar(self.ell, self._f * self.give_cl_cmb(Alens=1), fmt='k-', capsize=3, label='Model')
            plt.errorbar(self.ell, np.mean(self.DlBB, axis=0), yerr=np.std(self.DlBB, axis=0)*np.sqrt(2), fmt='bo', capsize=3, label='Bl')
            plt.yscale('log')
            #plt.ylim(5e-4, 5e-2)
            plt.legend(frameon=False, fontsize=12)

            plt.savefig(f'mydl_{os.environ.get("SLURM_JOB_ID")}.png')
            plt.close()
            stop
            return self.DlBB
        else:
            self.NlBB = np.zeros((self.N, len(self.ell)))
            for i in range(self.N):
                print(f'********* Iteration {i+1}/{self.N} *********')
                #print(f'     -> Dl')
                #self.DlBB[i] = self._get_BB_spectrum(self.components[i], beam_correction=np.rad2deg(0.00415369))
                #print(f'     -> Nl')
                self.NlBB[i] = self._get_BB_spectrum(self.residuals[i], beam_correction=np.rad2deg(0.00415369))
                    
                    
            return self.NlBB, self.BlBB
    def _open_data(self, name, keyword):
        with open(name, 'rb') as f:
            data = pickle.load(f)
        return data[keyword]
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
    def __repr__(self):
        return f"Spectrum class"

path = 'data_forecast_paper/CMM_paper_with_sync/'
foldername = f'parametric_d0_two_forecast_inCMBDustSync_outCMBDustSync_advanced'
path_to_data = os.getcwd() + '/' + foldername + '/'
spec = Spectrum(path_to_data, lmin=lmin, lmax=lmax, dl=dl, varying=False, aposize=aposize)
#DlBB = spec.main(cross=True)
NlBB, BlBB = spec.main(cross=False)
DlBB = None

plt.figure()

plt.errorbar(spec.ell, spec._f * spec.give_cl_cmb(Alens=Alens), fmt='k-', capsize=3, label='Model')
plt.errorbar(spec.ell, (spec._f * spec.give_cl_cmb(Alens=Alens))+BlBB, 
             yerr=np.std(NlBB, axis=0), fmt='r-', capsize=3, label='Dl')
plt.errorbar(spec.ell, BlBB, fmt='bo', capsize=3, label='Bl')
plt.yscale('log')
#plt.ylim(5e-4, 5e-2)
plt.legend(frameon=False, fontsize=12)

plt.savefig(f'mydl_{os.environ.get("SLURM_JOB_ID")}.png')
plt.close()


with open("autospectrum_" + foldername + ".pkl", 'wb') as handle:
    pickle.dump({'ell':spec.ell, 
                 'Dl':DlBB, 
                 'Nl':NlBB,
                 'Dl_bias':BlBB,
                 #'Dl_1x2':Nl_1x2
                 }, handle, protocol=pickle.HIGHEST_PROTOCOL)






class ForecastCMM:
    
    def __init__(self, path_to_data, ncomps, nside, center, lmin=40, lmax=512, dl=30, aposize=10, type='varying'):
        
        self.files = os.listdir(path_to_data)
        print(self.files)
        stop
        self.type = type
        self.N = len(self.files) - 190
        self.ncomps = ncomps
        self.nside = nside
        self.dl = dl
        self.lmin = lmin
        self.lmax = lmax
        self.aposize = aposize
        self.covcut = 0.2
        self.jobid = os.environ.get('SLURM_JOB_ID')
        self.center = center
        self.args_title = path_to_data.split('/')[-2].split('_')[:3]

        self.components = np.zeros((self.N, self.ncomps, 12*self.nside**2, 3))
        self.components_true = np.zeros((self.N, self.ncomps, 12*self.nside**2, 3))
        self.residuals = np.zeros((self.N, self.ncomps, 12*self.nside**2, 3))
        
        self.coverage = self._open_data(path_to_data+self.files[0], 'coverage')
        self.seenpix = hp.ud_grade(self.coverage / self.coverage.max() > self.covcut, self.nside)
        list_not_read = []
        
        print('======= Reading data =======')
        if self.type == 'varying':
            #print(self._open_data(path_to_data+self.files[0], 'components')[0, :, :].shape)
            self.components_true = self._open_data(path_to_data+self.files[0], 'components')[:, :, 0].T
            
            lmax = 2*self.nside
            #C1 = HealpixConvolutionGaussianOperator(fwhm=0.00415369, lmax=lmax)
            
            self.components_true = self.components_true.T
            for i in range(self.N):
                try:
                    for j in range(self.ncomps):
                        if j == 0:
                            for k in range(3):
                            
                                self.components[i, j, :, k] = self._open_data(path_to_data+self.files[i], 'components_i')[k, :, j].copy()
                            self.residuals[i, j] = self.components[i, j] - self.components_true
                            #self.components[i, j] = self.components[i, j]
                        else:
                            pass
                    print(f'Realization #{i+1}')
                        
                except OSError as e:
                    
                    list_not_read += [i]
                    
                    print(f'Realization #{i+1} could not be read')
        else:
            lmax = 2*self.nside
            C1 = HealpixConvolutionGaussianOperator(fwhm=0.00415369, lmax=lmax)
            
            #print(self._open_data(path_to_data+self.files[0], 'components')[0, :, :].shape)
            self.components_true = self._open_data(path_to_data+self.files[0], 'components')#[0, :, :]
            
            for j in range(self.ncomps):
                self.components_true[j] = C1(self.components_true[j])
            
            
            #C2 = HealpixConvolutionGaussianOperator(fwhm=np.sqrt(0.0078**2 - 0.00415369**2), lmax=lmax)
            
            #self.components_true = C1(self.components_true).copy()
            for i in range(self.N):
                try:
                    for j in range(self.ncomps):
                        #for k in range(3):
                            
                        self.components[i, j] = self._open_data(path_to_data+self.files[i], 'components_i')[j].copy()
                                
                        self.residuals[i, j] = self.components[i, j] - self.components_true[j]
                            #self.components[i, j] = self.components[i, j]
                    print(f'Realization #{i+1}')
                        
                except OSError as e:
                    
                    list_not_read += [i]
                    
                    print(f'Realization #{i+1} could not be read')
        
        
        ### Delete realizations still on going
        self.components = np.delete(self.components, list_not_read, axis=0)
        self.residuals = np.delete(self.residuals, list_not_read, axis=0)
        
        ### Set to 0 pixels not seen by QUBIC
        self.components[:, :, ~self.seenpix, :] = 0
        self.components_true[:, ~self.seenpix, :] = 0
        self.residuals[:, :, ~self.seenpix, :] = 0
        
        
        ### Plot input, average maps and residuals
        for j in range(self.ncomps):
            plt.figure(figsize=(12, 4))
            nsig = 3
            _r = np.mean(self.components, axis=0)[j] - self.components_true[j]
            hp.gnomview(np.mean(self.components, axis=0)[j, :, 1], rot=self.center, reso=15, cmap='jet', min=-8, max=8,
                    title=f'Output averaged over realizations', notext=True, sub=(1, 4, 1))
            hp.gnomview(self.components_true[j, :, 1], rot=self.center, reso=15, cmap='jet', min=-8, max=8, notext=True,
                    title=f'Input', sub=(1, 4, 2))
            hp.gnomview(_r[:, 1], rot=self.center, reso=15, cmap='jet', notext=True,
                    title=f'RMS : {np.std(_r[self.seenpix]):.3e}', sub=(1, 4, 3), min=-nsig*np.std(_r[self.seenpix]), max=nsig*np.std(_r[self.seenpix]))
            hp.gnomview(_r[:, 1], rot=self.center, reso=8, cmap='jet', notext=True,
                    title=f'', sub=(1, 4, 4), min=-nsig*np.std(_r[self.seenpix]), max=nsig*np.std(_r[self.seenpix]))
            plt.suptitle(f'{self.args_title[0]} - {self.args_title[1]} - {self.args_title[2]}')
            plt.savefig(f'maps_{self.jobid}_comp{j}.png')
            plt.close()
        
        self.stat_bias_map = np.mean(self.components, axis=0)[0] - self.components_true
        
        ### Initiate spectra computation
        print('======= Reading data - done =======')
        self.N = self.components.shape[0]
        self.namaster = nam.Namaster(self.seenpix, lmin=self.lmin, lmax=self.lmax, delta_ell=self.dl, aposize=self.aposize)
        self.ell, _ = self.namaster.get_binning(self.nside)
        self._f = self.ell * (self.ell + 1) / (2 * np.pi)
        _r = np.mean(self.components, axis=0)[0] - self.components_true[0]
        self.BlBB = self._get_BB_spectrum(_r, beam_correction=np.rad2deg(0.00415369))
        self.NlBB = np.zeros((self.N, self.ncomps**2, len(self.ell)))
        #self.DlBB = np.zeros((self.N, self.ncomps, len(self.ell)))
        self.DlBB = np.zeros((self.N, self.ncomps**2, len(self.ell)))
        self.Nl = np.zeros((self.N, self.ncomps, len(self.ell))) 
        print('Statistical bias -> ', self.BlBB)
        stop
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
            k=0
            print(f'\n********* Iteration {i+1}/{self.N} *********')
            for j in range(self.ncomps):
                #print(f'     -> Nl')
                #self.NlBB[i] = self._get_BB_spectrum(self.residuals[i, j], beam_correction=np.rad2deg(0.00415369))
                
                for jj in range(self.ncomps):
                    print(f'========= Components {j+1} x {jj+1} ========')
                    print(f'     -> Dl')
                    self.DlBB[i, k] = self._get_BB_spectrum(self.components[i, j], 
                                                            map2=self.components[i, jj])#, 
                                                            #beam_correction=np.rad2deg(0.00415369))
                    print(f'     -> Nl')
                    self.NlBB[i, k] = self._get_BB_spectrum(self.residuals[i, j], 
                                                            map2=self.residuals[i, jj])#, 
                                                            #beam_correction=np.rad2deg(0.00415369))
                    k+=1
                    
                    
        return self.DlBB, self.NlBB, self.BlBB, self.stat_bias_map
   
''' 
if advanced_qubic:
    foldername = f'{method}_{dustmodel}_{instr}_CMMpaper_withsync'
    path_to_data = os.getcwd() + '/' + foldername + '/'
    Alens = 0.1
    center = qubic.equ2gal(0, -45)
else:
    foldername = f'{method}_{dustmodel}_{instr}_CMMpaper_withsync'
    path_to_data = os.getcwd() + '/' + foldername + '/'
    Alens = 1
    center = qubic.equ2gal(0, -57)

forecast = ForecastCMM(path_to_data, ncomps, nside, center=center, lmin=lmin, lmax=lmax, dl=dl, type=t)
Dl, Nl, Bl, bias_map = forecast()
print(Dl.shape)
print(Nl.shape)
print(np.mean(Nl, axis=0))
print(np.std(Nl, axis=0))


mycl = forecast.give_cl_cmb(r=0, Alens=Alens)
_f = forecast.ell * (forecast.ell + 1) / (2 * np.pi)

plt.figure()

k=1
for _ in range(ncomps):
    for _ in range(ncomps):
        plt.subplot(ncomps, ncomps, k)
        plt.errorbar(forecast.ell, np.mean(Nl, axis=0)[k-1], fmt='ko', capsize=3)
        #plt.yscale('log')
        k+=1


plt.savefig(f'mydl_{os.environ.get("SLURM_JOB_ID")}.png')
plt.close()


with open("autospectrum_" + foldername + ".pkl", 'wb') as handle:
    pickle.dump({'ell':forecast.ell, 
                 'Dl':Dl, 
                 'Nl':Nl,
                 'Dl_bias':Bl,
                 'map_bias':bias_map,
                 #'Dl_1x2':Nl_1x2
                 }, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
'''
