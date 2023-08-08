import numpy as np
import pickle
import matplotlib.pyplot as plt
import healpy as hp
import imageio
import os
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator






class AnalysisParametricConstant:


    def __init__(self, folder_data, nside, N, ncomp):

        self.folder_data = folder_data + '/'
        self.allmaps = np.zeros((N, ncomp, 12*nside**2, 3))
        self.allite = np.arange(1, N+1, 1)
        self.beta = np.zeros((N, 2))
        self.coverage = self.open_pkl(f'Iter0_maps_beta_gain_rms_maps.pkl')['coverage'][0]
        self.seenpix = self.coverage / self.coverage.max() > 0.10
        for i in range(N):
            self.allmaps[i] = self.open_pkl(f'Iter{i}_maps_beta_gain_rms_maps.pkl')['maps']
            self.beta[i] = self.open_pkl(f'Iter{i}_maps_beta_gain_rms_maps.pkl')['beta']

    
    def plot_mollview(self, i, icomp, istk, min=-4, max=4, minr=-2, maxr=2):

        plt.figure(figsize=(15, 3.5))
        hp.mollview(self.allmaps[0, icomp, :, istk], cmap='jet', min=min, max=max, sub=(1, 3, 1),
                    title='Input')
        hp.mollview(self.allmaps[i, icomp, :, istk], cmap='jet', min=min, max=max, sub=(1, 3, 2),
                    title='Output')
        hp.mollview(self.allmaps[0, icomp, :, istk]-self.allmaps[i, icomp, :, istk], cmap='jet',
                    title='Residual', min=minr, max=maxr, sub=(1, 3, 3))
        
        plt.suptitle(f'Iteration = {self.allite[i]}')
        plt.show()

    def plot_gnomview(self, i, icomp, istk, rot, reso, min=-4, max=4, minr=-2, maxr=2):

        C=HealpixConvolutionGaussianOperator(fwhm=0.00)
        plt.figure(figsize=(10, 5))
        m_in = C(self.allmaps[0, icomp, :, istk])
        m_out = C(self.allmaps[i, icomp, :, istk])
        #print(m_in.shape)
        m_in[~self.seenpix] = hp.UNSEEN
        m_out[~self.seenpix] = hp.UNSEEN
        res = m_in - m_out
        res[~self.seenpix] = hp.UNSEEN
        hp.gnomview(m_in, cmap='jet', min=min, max=max, sub=(1, 3, 1),
                    title='Input', rot=rot, reso=reso, notext=True)
        hp.gnomview(m_out, cmap='jet', min=min, max=max, sub=(1, 3, 2),
                    title='Output', rot=rot, reso=reso, notext=True)
        hp.gnomview(res, cmap='jet',
                    title='Residual', min=minr, max=maxr, sub=(1, 3, 3), rot=rot, reso=reso, notext=True)
        
        plt.suptitle(f'Iteration = {self.allite[i]}', y=0.95)
        plt.savefig(f'iteration_{i}.png')
        plt.close()

        return f'iteration_{i}.png'
    
    def plot_beta(self, bar_ite=None, truth=None, log=False, **kwargs):

        plt.figure(figsize=(10, 6))

        plt.plot(self.allite, self.beta, '-k', **kwargs)
        if bar_ite is not None:
            plt.axvline(bar_ite+1, ls='-', color='black')

        if truth is not None:
            plt.axhline(truth, ls='--', color='black')
        if log:
            plt.yscale('log')
        #plt.ylim(1.52, 1.56)
        plt.title(f'Iteration = {self.allite[bar_ite]} | '+r'$\beta_d = $'+f'{self.beta[bar_ite]:.6f}')
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel(r'$\beta_d$', fontsize=14)
        if bar_ite is not None:
            plt.savefig(f'beta_{bar_ite}.png')
            plt.close()
            return f'beta_{bar_ite}.png'
        else:
            plt.savefig(f'beta.png')
            plt.close()
            return f'beta.png'
        
    def plot_all_components(self, i, center, reso, istk=1, name=['CMB', 'Dust'], nsig=3, fwhm=0):

        plt.figure(figsize=(8, 11))
        C = HealpixConvolutionGaussianOperator(fwhm=fwhm)
        k = 1
        for ic in range(self.allmaps.shape[1]):

            title_in = f'Input - {name[ic]}'
            m_in = self.allmaps[0, ic, :, istk]
            m_in[~self.seenpix] = hp.UNSEEN
            sig = np.std(m_in[self.seenpix])
            hp.gnomview(C(m_in), rot=center, reso=reso, cmap='jet', sub=(self.allmaps.shape[1], 3, k), notext=True, title=title_in,
                        min=-nsig*sig, max=nsig*sig)

            title_out = f'Output - {name[ic]}'
            m_out = self.allmaps[i, ic, :, istk]
            m_out[~self.seenpix] = hp.UNSEEN
            hp.gnomview(C(m_out), rot=center, reso=reso, cmap='jet', sub=(self.allmaps.shape[1], 3, k+1), notext=True, title=title_out,
                        min=-nsig*sig, max=nsig*sig)
            
            title_res = f'Residual - {name[ic]}'
            res = C(self.allmaps[i, ic, :, istk]) - C(self.allmaps[0, ic, :, istk])
            res[~self.seenpix] = hp.UNSEEN
            sig = np.std(res[self.seenpix])
            hp.gnomview(res, rot=center, reso=reso, cmap='jet', sub=(self.allmaps.shape[1], 3, k+2), notext=True, title=title_res,
                        min=-nsig*sig, max=nsig*sig)

            k+=3
        plt.savefig('allcomponents.png')
        plt.close()

    def make_gif_beta(self, duration, truth, **kwargs):

        allfig = []
        for i in self.allite:
            name = self.plot_beta(bar_ite=i-1, truth=truth, **kwargs)
            allfig += [name]
        #print(allfig)

        self.create_gif('beta.gif', duration, allfig)

        for filename in set(allfig):
            os.remove(filename)


    def create_gif(self, name, duration, filenames):
        
        with imageio.get_writer(name, mode='I', duration=duration) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

    def make_gif_gnomview(self, icomp, istk, rot, reso, duration, min=-4, max=4, minr=-2, maxr=2):
        
        allfig = []
        for i in self.allite:
            name = self.plot_gnomview(i-1, icomp, istk, rot, reso, min=min, max=max, minr=minr, maxr=maxr)
            allfig += [name]

        self.create_gif(f'maps_{icomp}.gif', duration, allfig)

        for filename in set(allfig):
            os.remove(filename)

    def plot_convergence_maps(self):

        plt.figure()
        C = HealpixConvolutionGaussianOperator(fwhm=0.0078)
        for icomp in range(self.allmaps.shape[1]):
            resI = np.mean(self.allmaps[:, icomp, self.seenpix, 1] - self.allmaps[0, icomp, self.seenpix, 1], axis=1)
            resI_std = np.std(self.allmaps[:, icomp, self.seenpix, 1] - self.allmaps[0, icomp, self.seenpix, 1], axis=1)
            plt.plot(self.allite-1, resI)
            plt.fill_between(self.allite-1, resI-resI_std/2, resI+resI_std/2, alpha=0.3)

        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel(r'Pixels value [$\mu K$]', fontsize=14)

        plt.axhline(0, ls='--', color='black')
        #plt.yscale('log')
        plt.xlim(0, np.max(self.allite))
        plt.savefig(f'convergence.png')
        plt.close()
    
    def open_pkl(self, filename):

        with open(self.folder_data + filename, 'rb') as f:
            data = pickle.load(f)
        
        return data
    

class AnalysisParametricVarying:

    def __init__(self, folder_data, nside, nside_fit, N, ncomp):

        self.folder_data = folder_data + '/'
        self.allmaps = np.zeros((N, 3, 12*nside**2, ncomp))
        self.allite = np.arange(1, N+1, 1)
        self.nside = nside
        self.nside_fit = nside_fit
        self.beta = np.zeros((N, 12*self.nside_fit**2, ncomp))
        for i in range(N):
            self.coverage = hp.ud_grade(self.open_pkl(f'Iter{i}_maps_beta_gain_rms_maps.pkl')['coverage'][0], self.nside_fit)
            self.allmaps[i] = self.open_pkl(f'Iter{i}_maps_beta_gain_rms_maps.pkl')['maps']
            self.beta[i] = self.open_pkl(f'Iter{i}_maps_beta_gain_rms_maps.pkl')['beta']
        self.beta[0, :, 0] = 1.45

        self.seenpix = self.coverage/self.coverage.max() > 0
        self.index_seenpix = np.where(self.seenpix == True)[0]

    def plot_gnomview(self, i, icomp, istk, rot, reso, min=-4, max=4, minr=-2, maxr=2):

        C=HealpixConvolutionGaussianOperator(fwhm=0.00)
        plt.figure(figsize=(10, 5))
        hp.gnomview(C(self.allmaps[0, istk, :, icomp]), cmap='jet', min=min, max=max, sub=(1, 3, 1),
                    title='Input', rot=rot, reso=reso, notext=True)
        hp.gnomview(C(self.allmaps[i, istk, :, icomp]), cmap='jet', min=min, max=max, sub=(1, 3, 2),
                    title='Output', rot=rot, reso=reso, notext=True)
        hp.gnomview(C(self.allmaps[0, istk, :, icomp])-C(self.allmaps[i, istk, :, icomp]), cmap='jet',
                    title='Residual', min=minr, max=maxr, sub=(1, 3, 3), rot=rot, reso=reso, notext=True)
        
        plt.suptitle(f'Iteration = {self.allite[i]}', y=0.95)
        plt.savefig(f'iteration_{i}.png')
        plt.close()

        return f'iteration_{i}.png'
    
    def plot_beta(self, bar_ite=None, truth=None, **kwargs):

        plt.figure(figsize=(10, 6))

        for i in self.index_seenpix:
            plt.plot(self.allite, self.beta[:, i, 0], '-k', **kwargs)
        if bar_ite is not None:
            plt.axvline(bar_ite+1, ls='-', color='black')

        if truth is not None:
            plt.axhline(truth, ls='--', color='black')
        
        #plt.ylim(1.52, 1.56)
        plt.title(f'Iteration = {self.allite[bar_ite]}')
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel(r'$\beta_d$', fontsize=14)
        if bar_ite is not None:
            plt.savefig(f'beta_{bar_ite}.png')
            plt.close()
            return f'beta_{bar_ite}.png'
        else:
            plt.savefig(f'beta.png')
            plt.close()
            return f'beta.png'
    
    def create_gif(self, name, duration, filenames):
        
        with imageio.get_writer(name, mode='I', duration=duration) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

    def make_gif_gnomview(self, icomp, istk, rot, reso, duration, min=-4, max=4, minr=-2, maxr=2):
        
        allfig = []
        for i in self.allite:
            name = self.plot_gnomview(i-1, icomp, istk, rot, reso, min=min, max=max, minr=minr, maxr=maxr)
            allfig += [name]

        self.create_gif(f'maps_{icomp}.gif', duration, allfig)

        for filename in set(allfig):
            os.remove(filename)

    def open_pkl(self, filename):

        with open(self.folder_data + filename, 'rb') as f:
            data = pickle.load(f)
        
        return data