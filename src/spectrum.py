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
        self.nbins = len(self.ell)
        
        #self._print_config()
        
        self._read_files()
        #print(self.rank, self.residuals.shape)

        if self.rank == 0:
            DlBB = np.zeros((self.ncomps, self.ncomps, self.nbins))
            for icomp in range(self.ncomps):
                for jcomp in range(self.ncomps):
                    DlBB[icomp, jcomp] = self._get_BB_spectrum(map1=self.average_components[icomp], map2=self.average_components[jcomp])
        else:
            DlBB = None
        
        self.DlBB = self.comm.bcast(DlBB, root=0)
        
        ### Plot bias on spectrum
        self._plot_bias(self.DlBB[0, 0], 1)
        
        ### plot average residual map
        self._plot_map(icomp=0)
        stop
    
    def _plot_map(self, icomp=0, nsig=2):
        
        plt.figure()
        
        for istk in range(3):
            sig = np.std(self.average_residuals[icomp, self.seenpix, istk])
            hp.gnomview(self.average_residuals[icomp, :, istk], reso=15, rot=CENTER, cmap='jet', sub=(1, 3, istk+1), min=-nsig*sig, max=nsig*sig, notext=True, title=f'RMS = {sig:.6f}')
        
        plt.savefig(f'residuals_map_icomp{icomp}.png')
        plt.close()
        
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
                print(f'Rank {self.rank} is reading real #{ireal}')
            
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
        
        self.average_residuals = np.mean(self.residuals, axis=0)
        self.average_residuals = self.comm.allreduce(self.average_residuals, op=MPI.SUM) / self.size
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

NSIDE = 128
LMIN = 20
CENTER = qubic.equ2gal(0, -57)
DL = 15
APOSIZE = 10



spec = Spectrum(foldername, comm, lmin=LMIN, lmax=2*NSIDE, dl=DL, aposize=APOSIZE)
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


