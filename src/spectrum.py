import numpy as np
import matplotlib.pyplot as plt
import pickle
import healpy as hp
import emcee
from multiprocess import Pool
from getdist import plots, MCSamples
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
from pyoperators import *
import os
import sys

import qubic
from qubic import NamasterLib as nam
import data

t = 'varying'
nside = 256
lmin = 40
lmax = 2 * nside - 1
aposize = 10
dl = 30
ncomps = 1
advanced_qubic = False
Alens = 1

def extract_seed(filename):
    return int(filename.split('_')[1][4:])

class Spectrum:
    
    def __init__(self, path_to_data, lmin=40, lmax=512, dl=30, aposize=10, varying=True, center=qubic.equ2gal(0, -57)):

        self.files = os.listdir(path_to_data)[:50]
        self.N = len(self.files)
        if self.N % 2 != 0:
            self.N -= 1
            
        self.dl = dl
        self.lmin = lmin
        self.lmax = lmax
        self.aposize = aposize
        self.covcut = 0.2
        self.center = center
        self.jobid = os.environ.get('SLURM_JOB_ID')
        self.args_title = path_to_data.split('/')[-2].split('_')[:3]
        
        
        self.components_true = self._open_data(path_to_data+self.files[0], 'components')
        
        if varying:
            self.nstk, self.npix, self.ncomps = self._open_data(path_to_data+self.files[0], 'components_i').shape
            self.components_true = self.components_true[:, :, :self.ncomps].T
        else:
            self.ncomps, self.npix, self.nstk = self._open_data(path_to_data+self.files[0], 'components_i').shape
            self.components_true = self.components_true[:self.ncomps]
        self.nside = hp.npix2nside(self.npix)
        self.components = np.zeros((self.N, self.ncomps, self.npix, 3))
        self.residuals = np.zeros((self.N, self.ncomps, self.npix, 3))
        
        C = HealpixConvolutionGaussianOperator(fwhm=0, lmax=3*self.nside)#0.00415369, lmax=2*self.nside)
        #C2 = HealpixConvolutionGaussianOperator(fwhm=np.sqrt(0.0078**2 - 0.00415369**2), lmax=2*self.nside)
        self.coverage = self._open_data(path_to_data+self.files[0], 'coverage')
        self.seenpix = self.coverage / self.coverage.max() > self.covcut
        
        print('***** Configuration *****')
        print(f'    Nstk : {self.nstk}')
        print(f'    Nside : {self.nside}')
        print(f'    Ncomp : {self.ncomps}')
        print(f'    Nreal : {self.N}')
        
        for co in range(self.ncomps):
            self.components_true[co] = C(self.components_true[co])
        
        list_not_read = []
        print('    -> Reading data')
        for i in range(self.N):
            try:
                            
                c = self._open_data(path_to_data+self.files[i], 'components_i')
                
                if varying:
                    #ct = np.transpose(c, (2, 1, 0))
                    self.components[i] = c.T
                else:
                    for icomp in range(self.ncomps):
                        self.components[i, icomp] = c[icomp].copy()
                
                for icomp in range(self.ncomps):
                    self.residuals[i, icomp] = self.components[i, icomp] - self.components_true[icomp]
                print(f'Realization #{i+1}')
                
            except OSError as e:
                    
                list_not_read += [i]
                print(f'Realization #{i+1} could not be read')
                
        
        
        print(np.mean(np.std(self.residuals[:, 0, self.seenpix, 1], axis=1), axis=0))
        print(np.std(np.std(self.residuals[:, 0, self.seenpix, 1], axis=1), axis=0))

        print('    -> Reading data - done')
        ### Delete realizations still on going
        self.components = np.delete(self.components, list_not_read, axis=0)
        self.residuals = np.delete(self.residuals, list_not_read, axis=0)
        
        ### Set to 0 pixels not seen by QUBIC
        print('    -> Remove not seen pixels')
        self.components[:, :, ~self.seenpix, :] = 0
        #self.components[:, :, :, 0] = 0
        self.components_true[:, ~self.seenpix, :] = 0
        self.residuals[:, :, ~self.seenpix, :] = 0
        #self.residuals[:, :, :, 0] = 0
        
        ### Initiate spectra computation
        print('    -> Initialization of Namaster')
        self.N = self.components.shape[0]
        self.namaster = nam.Namaster(self.seenpix, lmin=self.lmin, lmax=self.lmax, delta_ell=self.dl, aposize=self.aposize)
        #print(self.namaster.mask_apo)
        #print(self.namaster.fsky, np.sum(self.seenpix), np.sum(self.namaster.mask_apo))
        #stop
        self.ell, _ = self.namaster.get_binning(self.nside)
        self._f = self.ell * (self.ell + 1) / (2 * np.pi)
        
        ### Average realizations over same CMB
        print('    -> Averaging realizations')
        mean_maps = np.mean(self.components, axis=0)    # shape (Ncomp, Npix, Nstk)
        _r = mean_maps# - self.components_true

        self._plot_maps(mean_maps, self.components_true, _r - self.components_true, istk=1)
        self._plot_maps(mean_maps, self.components_true, _r - self.components_true, istk=2)
        
        ### Create Dl array for bias, signal and noise
        print('    -> Computing bias Bl')
        self.BlBB = np.zeros((self.ncomps, len(self.ell)))
        for i in range(self.ncomps):
            self.BlBB[i] = self._get_BB_spectrum(_r[i], 
                                                 beam_correction=np.rad2deg(0.00415369), 
                                                 pixwin_correction=False)

        self._plot_bias(Alens)
        print('Statistical bias -> ', self.BlBB)

    def _plot_bias(self, Alens):
        t = ['-o', '--', ':']
        plt.figure()
        #print(Alens)
        plt.errorbar(self.ell, self._f * self.give_cl_cmb(Alens=Alens), fmt='k-', capsize=3, label='Model')
        plt.errorbar(self.ell, self._f * self.give_cl_cmb(r=0.01, Alens=Alens), fmt='k--', capsize=3, label='Model | r = 0.01')
        for i in range(self.ncomps):
            plt.errorbar(self.ell, self.BlBB[i], fmt=f'r{t[i]}', capsize=3, label='Dl')
        
        plt.yscale('log')
        #plt.ylim(5e-4, 5e-2)
        plt.legend(frameon=False, fontsize=12)

        plt.savefig(f'bias_{os.environ.get("SLURM_JOB_ID")}.png')
        plt.close()
    def _plot_maps(self, IN, OUT, _r, istk=1):
        stk = ['I', 'Q', 'U']
        plt.figure(figsize=(12, 8))
        nsig = 3
        k=1
        for i in range(self.ncomps):
            hp.gnomview(IN[i, :, istk], rot=self.center, reso=15, cmap='jet', min=-8, max=8,
                    title=f'Output averaged over realizations', notext=True, sub=(self.ncomps, 3, k))
            k+=1
            hp.gnomview(OUT[i, :, istk], rot=self.center, reso=15, cmap='jet', min=-8, max=8, notext=True,
                    title=f'Input', sub=(self.ncomps, 3, k))
            k+=1
            hp.gnomview(_r[i, :, istk], rot=self.center, reso=15, cmap='jet', notext=True,
                    title=f'RMS : {np.std(_r[i, self.seenpix, istk]):.3e}', sub=(self.ncomps, 3, k), 
                    min=-nsig*np.std(_r[i, self.seenpix, istk]), max=nsig*np.std(_r[i, self.seenpix, istk]))
            k+=1
        
        plt.suptitle(f'{self.args_title[0]} - {self.args_title[1]} - {self.args_title[2]}')
        plt.savefig(f'maps_{self.jobid}_{stk[istk]}.png')
        plt.close()
    def main(self, spec=False):
        
        if spec == False:
            self.NlBB = np.zeros((self.N, self.ncomps, self.ncomps, len(self.ell)))
            return self.NlBB, self.BlBB
        else:
            self.NlBB = np.zeros((self.N, self.ncomps, self.ncomps, len(self.ell)))
            self.DlBB = np.zeros((self.N, self.ncomps, self.ncomps, len(self.ell)))
            for i in range(self.N):
                print(f'********* Iteration {i+1}/{self.N} *********')

                for icomp in range(self.ncomps):
                    for jcomp in range(icomp, self.ncomps):
                        print(f'===== {icomp} x {jcomp} =====')
                        self.NlBB[i, icomp, jcomp] = self._get_BB_spectrum(self.residuals[i, icomp].T, map2=self.residuals[i, jcomp].T, 
                                                                        beam_correction=np.rad2deg(0),
                                                                        pixwin_correction=True)
                        
                        if icomp != jcomp:
                            self.NlBB[i, jcomp, icomp] = self.NlBB[i, icomp, jcomp].copy()


                #print(np.std(self.DlBB[:(i+1), :, :, 0], axis=0))
                print(np.std(self.NlBB[:(i+1), :, :, 0], axis=0))

                    
            return self.NlBB, self.BlBB, self.DlBB
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


#
#path_to_data = os.getcwd() + '/' + foldername + '/'
#spec = Spectrum(path_to_data, 
#                lmin=lmin, 
#                lmax=lmax, 
#                dl=dl, 
#                varying=False, 
#                aposize=aposize)

#NlBB, BlBB, DlBB = spec.main(spec=True)



comm = MPI.COMM_WORLD


class Spectrum:
    
    def __init__(self, folder, comm, lmin=40, lmax=512, dl=30, aposize=10):
        
        self.folder = folder
        
        self.lmin = lmin
        self.lmax = lmax
        self.dl = dl
        self.aposize = aposize
        
        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        
        self.allfiles = os.listdir(self.folder)[:200]
        self.N = len(self.allfiles)
        self.list_index_reals = np.array_split(np.arange(self.N), self.size)[self.rank]
        self.files = np.array_split(self.allfiles, self.size)[self.rank]
        self.components_true = self._open_data(self.folder + '/' + self.allfiles[0], 'components')
        self.ncomps, self.npix, self.nstk = self.components_true.shape
        self.nside = hp.npix2nside(self.npix)
        
        self.coverage = self._open_data(self.folder + '/' + self.allfiles[0], 'coverage')
        self.seenpix = self.coverage / self.coverage.max() > 0.2
        
        self.namaster = nam.Namaster(self.seenpix, lmin=self.lmin, lmax=self.lmax, delta_ell=self.dl, aposize=self.aposize)
        self.ell, _ = self.namaster.get_binning(self.nside)
        self._f = self.ell * (self.ell + 1) / (2 * np.pi)
        
        #self._print_config()
        
        self._read_files()
        #print(self.rank, self.residuals.shape)

        if self.rank == 0:
            DlBB = self._get_BB_spectrum(map1=self.average_components[0])
        else:
            DlBB = None
        
        self.DlBB = self.comm.bcast(DlBB, root=0)
        
        self._plot_bias(self.DlBB, Alens)
        
    def _open_data(self, name, keyword):
        with open(name, 'rb') as f:
            data = pickle.load(f)
        return data[keyword]
    def _print_config(self):
        if self.rank == 0:
            print('***** Configuration *****')
            print(f'    Nstk : {self.nstk}')
            print(f'    Nside : {self.nside}')
            print(f'    Ncomp : {self.ncomps}')
            print(f'    Nreal : {self.N}')
    def give_cl_cmb(self, r=0, Alens=1.):
        
        power_spectrum = hp.read_cl('data/Cls_Planck2018_lensed_scalar.fits')[:,:4000]
        if Alens != 1.:
            power_spectrum[2] *= Alens
        if r:
            power_spectrum += r * hp.read_cl('data/Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits')[:,:4000]
        return np.interp(self.ell, np.arange(1, 4001, 1), power_spectrum[2])
    def _plot_bias(self, Dl, Alens):
        t = ['-o', '--', ':']
        plt.figure()
        #print(Alens)
        plt.errorbar(self.ell, self._f * self.give_cl_cmb(Alens=Alens), fmt='k-', capsize=3, label='Model')
        plt.errorbar(self.ell, self._f * self.give_cl_cmb(r=0.01, Alens=Alens), fmt='k--', capsize=3, label='Model | r = 0.01')
        #for i in range(self.ncomps):
        plt.errorbar(self.ell, Dl, fmt=f'r{t[0]}', capsize=3, label='Dl')
        
        plt.yscale('log')
        #plt.ylim(5e-4, 5e-2)
        plt.legend(frameon=False, fontsize=12)

        plt.savefig(f'bias_{os.environ.get("SLURM_JOB_ID")}.png')
        plt.close()
    def _read_files(self):
        
        comps = np.zeros((self.N, self.ncomps, self.npix, 3))
        res = np.zeros((self.N, self.ncomps, self.npix, 3))
        self.components = np.zeros((1, self.ncomps, self.npix, 3))
        self.residuals = np.zeros((1, self.ncomps, self.npix, 3))
        
        list_not_read = [0]
        
        for ireal in self.list_index_reals:
            
            try :
                #print(f'Rank {self.rank} is reading real #{ireal}')
            
                ### Output
                c = self._open_data(self.folder + '/' + self.allfiles[ireal], 'components_i')
                self.components = np.concatenate((self.components, np.array([c])), axis=0)
                ### Output - Input
                res = c - self.components_true
                self.residuals = np.concatenate((self.residuals, np.array([res])), axis=0)

            except OSError as e:
                    
                #list_not_read += [ireal]
                print(f'Realization #{ireal} could not be read')
            
        
        ### Delete realizations still on going
        self.components = np.delete(self.components, list_not_read, axis=0)
        self.residuals = np.delete(self.residuals, list_not_read, axis=0)
        
        self.components[:, :, ~self.seenpix, :] = 0
        self.residuals[:, :, ~self.seenpix, :] = 0

        self.average_components = np.mean(self.components, axis=0)
        self.average_components = self.comm.allreduce(self.average_components, op=MPI.SUM) / self.size
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
    def main(self):
        
        self.NlBB = np.zeros((1, self.ncomps, self.ncomps, len(self.ell)))
        
        for ireal in range(self.residuals.shape[0]):
            print(f'********* Iteration {ireal+1}/{self.N} - Rank {self.rank}/{self.size} *********')

            Dl = np.zeros((self.ncomps, self.ncomps, len(self.ell)))
            for icomp in range(self.ncomps):
                for jcomp in range(icomp, self.ncomps):
                    print(f'===== {icomp} x {jcomp} =====')
                    Dl[icomp, jcomp] = self._get_BB_spectrum(self.residuals[ireal, icomp].T, map2=self.residuals[ireal, jcomp].T, 
                                                                        beam_correction=np.rad2deg(0),
                                                                        pixwin_correction=True)
                        
                    if icomp != jcomp:
                        Dl[jcomp, icomp] = Dl[icomp, jcomp].copy()

            self.NlBB = np.concatenate((self.NlBB, np.array([Dl])), axis=0)
            #print(np.std(self.NlBB[:(ireal+1), :, :, 0], axis=0))
            
        self.NlBB = np.delete(self.NlBB, [0], axis=0)
foldername = str(sys.argv[1])
     
spec = Spectrum(foldername, comm)
spec.main()
list_arrays = spec.comm.gather(spec.NlBB, root=0)

if spec.rank == 0:
    spec.NlBB = np.concatenate((list_arrays), axis=0)
    print(spec.DlBB)
    print(spec.NlBB.shape)
    print(np.std(spec.NlBB, axis=0))

    with open("autospectrum_" + foldername + ".pkl", 'wb') as handle:
        pickle.dump({'ell':spec.ell, 
                    'Dl':spec.DlBB, 
                    'Nl':spec.NlBB,
                    }, handle, protocol=pickle.HIGHEST_PROTOCOL)


