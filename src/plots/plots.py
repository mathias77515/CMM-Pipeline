from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import os

def _plot_reconstructed_maps(maps, truth, name_file, center, num_iter, reso=12, figsize=(12, 8), fwhm=0, view='gnomview'):
    
    """
    
    Save a PNG with the actual maps at iteration i. It assumes that maps is 3-dimensional
    
    """
    
    plt.figure(figsize=figsize)
    
    _shape = maps.shape
    index = np.where(maps[0, :, 0] != hp.UNSEEN)[0]
    C = HealpixConvolutionGaussianOperator(fwhm=fwhm)
    
    k=0
    for inu in range(_shape[0]):
        mcomp = C(maps[inu])
        for istk in range(_shape[-1]):
            sig = np.std(truth[inu, index, istk])
            
            if inu > 0 and istk == 0:
                nsig = 7
            else:
                nsig = 3
                
            if view == 'gnomview':
                hp.gnomview(mcomp[:, istk], rot=center, reso=reso, cmap='jet', sub=(_shape[0], _shape[-1], k+1),
                            notext=True, min=-nsig*sig, max=nsig*sig, title='')
            elif view == 'mollview':
                hp.mollview(mcomp[:, istk], cmap='jet', sub=(_shape[0], _shape[-1], k+1),
                            notext=True, min=-nsig*sig, max=nsig*sig, title='')
            k+=1
    
    plt.suptitle(f'Iteration : {num_iter}', fontsize=15, y=0.99)
    #plt.tight_layout()
    plt.savefig(name_file)
    plt.close()
    
class Plots:

    """
    
    Instance to produce plots on the convergence. 
    
    Arguments : 
    ===========
        - jobid : Int number for saving figures.
        - dogif : Bool to produce GIF.
    
    """
    
    def __init__(self, preset, dogif=True):
        
        self.preset = preset
        self.job_id = self.preset.job_id
        self.dogif = dogif
        self.params = self.preset.tools.params
       
    def plot_sed(self, nus_in, A_in, nus_out, A_out, figsize=(8, 6), ki=0, gif=False):
        """
        Plots the Spectral Energy Distribution (SED) and saves the plot as a PNG file.

        Parameters:
        nus (array-like): Array of frequency values.
        A (array-like): Array of amplitude values.
        figsize (tuple, optional): Size of the figure. Defaults to (8, 6).
        truth (array-like, optional): Array of true values for comparison. Defaults to None.
        ki (int, optional): Iteration index for file naming. Defaults to 0.

        Returns:
        None
        """
        
        if self.params['Plots']['conv_beta']:
            
            nf_in, nc_in = A_in.shape
            nf_out, nc_out = A_out.shape
            fsub = int(nf_in / nf_out)
            plt.figure(figsize=figsize)
            
            for ic in range(nc_in):
                plt.plot(nus_in, A_in[:, ic], '-k')
            
            for inu in range(nf_out):
                plt.errorbar(nus_out[inu], np.mean(A_in[inu*fsub:(inu+1)*fsub]), fmt='og')
                
            for ic in range(nc_out):
                plt.errorbar(nus_out, A_out[:, ic], fmt='xb')
                
            
            plt.xlim(120, 260)
            eps = 0.1
            eps_max = A_in.max() * (1 + eps)
            eps_min = A_in.min() * (1 - eps)
            plt.ylim(eps_min, eps_max)
            plt.yscale('log')
            
            plt.savefig(f'jobs/{self.job_id}/A_iter/A_iter{ki+1}.png')

            if self.preset.tools.rank == 0:
                if ki > 0 and gif is False:
                    os.remove(f'jobs/{self.job_id}/A_iter/A_iter{ki}.png')
                
            plt.close() 
    def plot_beta_iteration(self, beta, figsize=(8, 6), truth=None, ki=0):
        """
        Method to plot beta as a function of iteration. Beta can have shape (niter) or (niter, nbeta).

        Parameters:
        beta : numpy.ndarray
            Array containing beta values for each iteration. Can be 1D or 2D.
        figsize : tuple, optional
            Size of the figure to be plotted. Default is (8, 6).
        truth : numpy.ndarray or float, optional
            True value(s) of beta to be plotted as a reference line. Default is None.
        ki : int, optional
            Iteration index for saving the plot. Default is 0.

        Returns:
        None
        """
        
        if self.params['Plots']['conv_beta']:
            niter = beta.shape[0]
            alliter = np.arange(0, niter, 1)
            
            plt.figure(figsize=figsize)
            plt.subplot(2, 1, 1)
            if np.ndim(beta) == 1:
                plt.plot(alliter[1:]-1, beta[1:])
                if truth is not None:
                    plt.axhline(truth, ls='--', color='red')
            else:
                print(beta.shape[1])
                print(truth.shape)
                for i in range(beta.shape[1]):
                    plt.plot(alliter, beta[:, i], '-k', alpha=0.3)
                    if truth is not None:
                        for j in range(truth.shape[1]):
                            plt.axhline(truth[i, j], ls='--', color='red')

            plt.subplot(2, 1, 2)
            if np.ndim(beta) == 1:
                plt.plot(alliter[1:]-1, abs(truth - beta[1:]))
            else:
                for i in range(beta.shape[1]):
                    plt.plot(alliter, abs(truth[i] - beta[:, i]), '-k', alpha=0.3)
            plt.yscale('log')
            plt.savefig(f'jobs/{self.job_id}/beta_iter{ki+1}.png')

            if ki > 0:
                os.remove(f'jobs/{self.job_id}/beta_iter{ki}.png')
            plt.close()
    def _display_allresiduals(self, map_i, seenpix, figsize=(14, 10), ki=0):
        """
        Display all components of the Healpix map with Gaussian convolution.

        Parameters:
        seenpix (array-like): Boolean array indicating the pixels that are seen.
        figsize (tuple): Size of the figure to be plotted. Default is (14, 10).
        ki (int): Iteration index for saving the figure. Default is 0.

        This function generates and saves a figure showing the output maps and residuals
        for each component and Stokes parameter (I, Q, U). The maps are convolved using
        a Gaussian operator and displayed using Healpix's gnomview function.
        """
        stk = ['I', 'Q', 'U']
        if self.params['Plots']['maps']:
            plt.figure(figsize=figsize)
            k = 0
            r = self.preset.A(map_i) - self.preset.b
            map_res = np.ones(self.preset.fg.components_iter.shape) * hp.UNSEEN
            map_res[:, seenpix, :] = r

            for istk in range(3):
                for icomp in range(len(self.preset.fg.components_name_out)):
                    
                    _reso = 15
                    nsig = 3
                    
                    hp.gnomview(map_res[icomp, :, istk], rot=self.preset.sky.center, reso=_reso, notext=True, title=f'{self.preset.fg.components_name_out[icomp]} - {stk[istk]} - r = A x - b',
                        cmap='jet', sub=(3, len(self.preset.fg.components_out), k+1), min=-nsig*np.std(r[icomp, :, istk]), max=nsig*np.std(r[icomp, :, istk]))
                    k += 1
            
            plt.tight_layout()
            plt.savefig(f'jobs/{self.job_id}/allcomps/allres_iter{ki+1}.png')
            
            #if self.preset.tools.rank == 0:
            #    if ki > 0:
            #        os.remove(f'jobs/{self.job_id}/allcomps/allres_iter{ki}.png')
            plt.close()
    def _display_allcomponents(self, seenpix, figsize=(14, 10), ki=0, gif=True, reso=15):
        """
        Display all components of the Healpix map with Gaussian convolution.

        Parameters:
        seenpix (array-like): Boolean array indicating the pixels that are seen.
        figsize (tuple): Size of the figure to be plotted. Default is (14, 10).
        ki (int): Iteration index for saving the figure. Default is 0.

        This function generates and saves a figure showing the output maps and residuals
        for each component and Stokes parameter (I, Q, U). The maps are convolved using
        a Gaussian operator and displayed using Healpix's gnomview function.
        """
        C = HealpixConvolutionGaussianOperator(fwhm=self.preset.acquisition.fwhm_reconstructed, lmax=3*self.params['SKY']['nside'])
        stk = ['I', 'Q', 'U']
        if self.params['Plots']['maps']:
            plt.figure(figsize=figsize)
            k = 0
            for istk in range(3):
                for icomp in range(len(self.preset.fg.components_name_out)):
                    
                    # if self.preset.fg.params_foregrounds['Dust']['nside_beta_out'] == 0:
                        
                    map_in = C(self.preset.fg.components_out[icomp, :, istk]).copy()
                    map_out = self.preset.fg.components_iter[icomp, :, istk].copy()
                        
                    sig = np.std(self.preset.fg.components_out[icomp, seenpix, istk])
                    map_in[~seenpix] = hp.UNSEEN
                    map_out[~seenpix] = hp.UNSEEN
                        
                    # else:
                    #     if self.preset.qubic.params_qubic['convolution_in']:
                    #         map_in = self.preset.fg.components_convolved_out[icomp, :, istk].copy()
                    #         map_out = self.preset.fg.components_iter[istk, :, icomp].copy()
                    #         sig = np.std(self.preset.fg.components_convolved_out[icomp, seenpix, istk])
                    #     else:
                    #         map_in = self.preset.fg.components_out[istk, :, icomp].copy()
                    #         map_out = self.preset.fg.components_iter[istk, :, icomp].copy()
                    #         sig = np.std(self.preset.fg.components_out[istk, seenpix, icomp])
                    #     map_in[~seenpix] = hp.UNSEEN
                    #     map_out[~seenpix] = hp.UNSEEN
                        
                    r = map_in - map_out
                    nsig = 2
                    hp.gnomview(map_out, rot=self.preset.sky.center, reso=reso, notext=True, title=f'{self.preset.fg.components_name_out[icomp]} - {stk[istk]} - Output',
                        cmap='jet', sub=(3, len(self.preset.fg.components_out)*2, k+1), min=-nsig*sig, max=nsig*sig)
                    k += 1
                    hp.gnomview(r, rot=self.preset.sky.center, reso=reso, notext=True, title=f'{self.preset.fg.components_name_out[icomp]} - {stk[istk]} - Residual',
                        cmap='jet', sub=(3, len(self.preset.fg.components_out)*2, k+1), min=-nsig*np.std(r[seenpix]), max=nsig*np.std(r[seenpix]))
                    k += 1
            
            plt.tight_layout()
            plt.savefig(f'jobs/{self.job_id}/allcomps/allcomps_iter{ki+1}.png')
            
            if self.preset.tools.rank == 0:
                if ki > 0 and gif is False:
                    os.remove(f'jobs/{self.job_id}/allcomps/allcomps_iter{ki}.png')
            plt.close()
    def display_maps(self, seenpix, figsize=(14, 8), nsig=6, ki=0, view='gnomview'):
        """
        
        Method to display maps at given iteration.
        
        Arguments:
        ----------
            - seenpix : array containing the id of seen pixels.
            - ngif    : Int number to create GIF with ngif PNG image.
            - figsize : Tuple to control size of plots.
            - nsig    : Int number to compute errorbars.
        
        """
        if self.params['Plots']['maps']:
            stk = ['I', 'Q', 'U']
            rms_i = np.zeros((1, 2))
            
            for istk, s in enumerate(stk):
                plt.figure(figsize=figsize)
                
                k=0

                for icomp in range(len(self.preset.fg.components_name_out)):

                    #if self.preset.fg.params_foregrounds['Dust']['nside_beta_out'] == 0:
                    if self.preset.qubic.params_qubic['convolution_in']:
                        map_in = self.preset.fg.components_convolved_out[icomp, :, istk].copy()
                        map_out = self.preset.fg.components_iter[icomp, :, istk].copy()
                    else:
                        map_in = self.preset.fg.components_out[icomp, :, istk].copy()
                        map_out = self.preset.fg.components_iter[icomp, :, istk].copy()
                            
                    # else:
                    #     if self.preset.qubic.params_qubic['convolution_in']:
                    #         map_in = self.preset.fg.components_convolved_out[icomp, :, istk].copy()
                    #         map_out = self.preset.fg.components_iter[istk, :, icomp].copy()
                    #     else:
                    #         map_in = self.preset.fg.components_out[istk, :, icomp].copy()
                    #         map_out = self.preset.fg.components_iter[istk, :, icomp].copy()
                    
                    sig = np.std(map_in[seenpix])
                    map_in[~seenpix] = hp.UNSEEN
                    map_out[~seenpix] = hp.UNSEEN
                    r = map_in - map_out
                    r[~seenpix] = hp.UNSEEN
                    if icomp == 0:
                        if istk > 0:
                            rms_i[0, istk-1] = np.std(r[seenpix])
                    
                    _reso = 15
                    nsig = 3
                    if view == 'gnomview':
                        hp.gnomview(map_in, rot=self.preset.sky.center, reso=_reso, notext=True, title='',
                            cmap='jet', sub=(len(self.preset.fg.components_out), 3, k+1), min=-nsig*sig, max=nsig*sig)
                        hp.gnomview(map_out, rot=self.preset.sky.center, reso=_reso, notext=True, title='',
                            cmap='jet', sub=(len(self.preset.fg.components_out), 3, k+2), min=-nsig*sig, max=nsig*sig)
                        hp.gnomview(r, rot=self.preset.sky.center, reso=_reso, notext=True, title=f"{np.std(r[seenpix]):.3e}",
                            cmap='jet', sub=(len(self.preset.fg.components_out), 3, k+3), min=-nsig*sig, max=nsig*sig)
                    elif view == 'mollview':
                        hp.mollview(map_in, notext=True, title='',
                            cmap='jet', sub=(len(self.preset.fg.components_out), 3, k+1), min=-nsig*sig, max=nsig*sig)
                        hp.mollview(map_out, notext=True, title='',
                            cmap='jet', sub=(len(self.preset.fg.components_out), 3, k+2), min=-nsig*sig, max=nsig*sig)
                        hp.mollview(r, notext=True, title=f"{np.std(r[seenpix]):.3e}",
                            cmap='jet', sub=(len(self.preset.fg.components_out), 3, k+3), min=-nsig*sig, max=nsig*sig)
                    k+=3
                    
                plt.tight_layout()
                plt.savefig(f'jobs/{self.job_id}/{s}/maps_iter{ki+1}.png')
                
                if self.preset.tools.rank == 0:
                    if ki > 0:
                        os.remove(f'jobs/{self.job_id}/{s}/maps_iter{ki}.png')

                plt.close()
            self.preset.acquisition.rms_plot = np.concatenate((self.preset.acquisition.rms_plot, rms_i), axis=0)
    def plot_gain_iteration(self, gain, figsize=(8, 6), ki=0):
        
        """
        
        Method to plot convergence of reconstructed gains.
        
        Arguments :
        -----------
            - gain    : Array containing gain number (1 per detectors). It has the shape (Niteration, Ndet, 2) for Two Bands design and (Niteration, Ndet) for Wide Band design
            - alpha   : Transparency for curves.
            - figsize : Tuple to control size of plots.
            
        """
        
        
        if self.params['Plots']['conv_gain']:
            
            plt.figure(figsize=figsize)

            
            
            niter = gain.shape[0]
            ndet = gain.shape[1]
            alliter = np.arange(1, niter+1, 1)

            #plt.hist(gain[:, i, j])
            if self.preset.qubic.params_qubic['type'] == 'two':
                color = ['red', 'blue']
                for j in range(2):
                    plt.hist(gain[-1, :, j], bins=20, color=color[j])
            #        plt.plot(alliter-1, np.mean(gain, axis=1)[:, j], color[j], alpha=1)
            #        for i in range(ndet):
            #            plt.plot(alliter-1, gain[:, i, j], color[j], alpha=alpha)
                        
            #elif self.preset.qubic.params_qubic['type'] == 'wide':
            #    color = ['--g']
            #    plt.plot(alliter-1, np.mean(gain, axis=1), color[0], alpha=1)
            #    for i in range(ndet):
            #        plt.plot(alliter-1, gain[:, i], color[0], alpha=alpha)
                        
            #plt.yscale('log')
            #plt.ylabel(r'|$g_{reconstructed} - g_{input}$|', fontsize=12)
            #plt.xlabel('Iterations', fontsize=12)
            plt.xlim(-0.1, 0.1)
            plt.ylim(0, 100)
            plt.axvline(0, ls='--', color='black')
            plt.savefig(f'jobs/{self.job_id}/gain_iter{ki+1}.png')

            if self.preset.tools.rank == 0:
                if ki > 0:
                    os.remove(f'jobs/{self.job_id}/gain_iter{ki}.png')

            plt.close()
    def plot_rms_iteration(self, rms, figsize=(8, 6), ki=0):
        
        if self.params['Plots']['conv_rms']:
            plt.figure(figsize=figsize)
            
            plt.plot(rms[1:, 0], '-b', label='Q')
            plt.plot(rms[1:, 1], '-r', label='U')
            
            plt.yscale('log')
            
            plt.tight_layout()
            plt.savefig(f'jobs/{self.job_id}/rms_iter{ki+1}.png')
                
            if self.preset.tools.rank == 0:
                if ki > 0:
                    os.remove(f'jobs/{self.job_id}/rms_iter{ki}.png')

            plt.close()