from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import os

class Plots:

    """
    
    Instance to produce plots on the convergence. 
    
    Arguments : 
    ===========
        - jobid : Int number for saving figures.
    
    """
    
    def __init__(self, preset):
        
        self.preset = preset
        self.job_id = self.preset.job_id
        self.params = self.preset.tools.params

    def display_stokes_maps(self, seenpix, figsize=(14, 10), nsig=6, ki=0):
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

                    if self.preset.qubic.params_qubic['convolution_in']:
                        map_in = self.preset.fg.components_convolved_out[icomp, :, istk].copy()
                        map_out = self.preset.fg.components_iter[icomp, :, istk].copy()
                    else:
                        map_in = self.preset.fg.components_out[icomp, :, istk].copy()
                        map_out = self.preset.fg.components_iter[icomp, :, istk].copy()

                    sig = np.std(map_in[seenpix])
                    map_in[~seenpix] = hp.UNSEEN
                    map_out[~seenpix] = hp.UNSEEN
                    residual = map_in - map_out
                    residual[~seenpix] = hp.UNSEEN
                    if icomp == 0:
                        if istk > 0:
                            rms_i[0, istk-1] = np.std(residual[seenpix])
                    
                    _reso = 15
                    nsig = 3
                    hp.gnomview(map_in, rot=self.preset.sky.center, reso=_reso, notext=True, title=f'{self.preset.fg.components_name_out[icomp]} - Input',
                        cmap='jet', sub=(len(self.preset.fg.components_out), 3, k+1), min=-nsig*sig, max=nsig*sig)
                    hp.gnomview(map_out, rot=self.preset.sky.center, reso=_reso, notext=True, title=f'{self.preset.fg.components_name_out[icomp]} - Output',
                        cmap='jet', sub=(len(self.preset.fg.components_out), 3, k+2), min=-nsig*sig, max=nsig*sig)
                    hp.gnomview(residual, rot=self.preset.sky.center, reso=_reso, notext=True, title=f'{self.preset.fg.components_name_out[icomp]} - Residual : {np.std(residual[seenpix]):.3e}',
                        cmap='jet', sub=(len(self.preset.fg.components_out), 3, k+3), min=-nsig*sig, max=nsig*sig)
                    k+=3
                    
                plt.tight_layout()
                plt.savefig(f'jobs/{self.job_id}/{s}/maps_iter{ki+1}.png')
                
                if self.preset.tools.rank == 0:
                    if ki > 0:
                        os.remove(f'jobs/{self.job_id}/{s}/maps_iter{ki}.png')

                plt.close()
            self.preset.acquisition.rms_plot = np.concatenate((self.preset.acquisition.rms_plot, rms_i), axis=0)

    def display_allcomponents(self, seenpix, figsize=(14, 10), ki=0):
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
                                            
                    map_in = C(self.preset.fg.components_out[icomp, :, istk]).copy()
                    map_out = self.preset.fg.components_iter[icomp, :, istk].copy()
                        
                    sig = np.std(self.preset.fg.components_out[icomp, seenpix, istk])
                    map_in[~seenpix] = hp.UNSEEN
                    map_out[~seenpix] = hp.UNSEEN
                        
                    residual = map_in - map_out
                    _reso = 15
                    nsig = 3
                    hp.gnomview(map_out, rot=self.preset.sky.center, reso=_reso, notext=True, title=f'{self.preset.fg.components_name_out[icomp]} - {stk[istk]} - Output',
                        cmap='jet', sub=(3, len(self.preset.fg.components_out)*2, k+1), min=-nsig*sig, max=nsig*sig)
                    k += 1
                    hp.gnomview(residual, rot=self.preset.sky.center, reso=_reso, notext=True, title=f'{self.preset.fg.components_name_out[icomp]} - {stk[istk]} - Residual',
                        cmap='jet', sub=(3, len(self.preset.fg.components_out)*2, k+1), min=-nsig*np.std(residual[seenpix]), max=nsig*np.std(residual[seenpix]))
                    k += 1
            
            plt.tight_layout()
            plt.savefig(f'jobs/{self.job_id}/allcomps/allcomps_iter{ki+1}.png')
            
            if self.preset.tools.rank == 0:
                if ki > 0:
                    os.remove(f'jobs/{self.job_id}/allcomps/allcomps_iter{ki}.png')
            plt.close()

    def plot_rms_iteration(self, rms, figsize=(8, 6), ki=0):
        
        if self.params['Plots']['conv_rms']:
            plt.figure(figsize=figsize)
            
            plt.plot(rms[1:, 0], '-b', label='Q')
            plt.plot(rms[1:, 1], '-r', label='U')
            plt.xlabel('Iterations')
            plt.ylabel("Residual RMS")
            plt.yscale('log')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'jobs/{self.job_id}/rms_iter{ki+1}.png')
                
            if self.preset.tools.rank == 0:
                if ki > 0:
                    os.remove(f'jobs/{self.job_id}/rms_iter{ki}.png')

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

            fig, axs = plt.subplots(2, 1, figsize=figsize)

            if np.ndim(beta) == 1:
                axs[0].plot(alliter[1:]-1, beta[1:], label = 'Reconstructed')
                if truth is not None:
                    axs[0].axhline(truth, ls='--', color='red', label = "True")
            else:
                for i in range(beta.shape[1]):
                    axs[0].plot(alliter, beta[:, i], '-k', alpha=0.3, label = 'Reconstructed')
                    if truth is not None:
                        axs[0].axhline(truth[i], ls='--', color='red', label = "True")
            axs[0].set_xlabel('Iteration')
            axs[0].set_ylabel('Spectral index')
            axs[0].legend()

            if np.ndim(beta) == 1:
                axs[1].plot(alliter[1:]-1, abs(truth - beta[1:]))
            else:
                for i in range(beta.shape[1]):
                    axs[1].plot(alliter, abs(truth[i] - beta[:, i]), '-k', alpha=0.3)
            axs[1].set_yscale('log')
            axs[1].set_xlabel('Iteration')
            axs[1].set_ylabel('Residual RMS')
            plt.savefig(f'jobs/{self.job_id}/beta_iter{ki+1}.png')

            if ki > 0:
                os.remove(f'jobs/{self.job_id}/beta_iter{ki}.png')
            plt.close()
       
    def plot_sed(self, nus, A, figsize=(8, 6), truth=None, ki=0):
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
            
            nf = truth.shape[0]
            fig, axs = plt.subplots(2, 1, figsize=figsize)
            fig.suptitle('Mixing matrix elements')
            for i in range(A[-1].shape[1]):
                axs[0].errorbar(nus, truth[:, i], fmt='ob', label = "True")
                axs[0].errorbar(nus, A[-1][:, i], fmt='xr', label = "Reconstructed")
                if i==0:
                    axs[0].legend()
            axs[0].set_xlim(120, 260)
            axs[0].set_xlabel('Frequency (GHz)')
            axs[0].set_ylabel('Mixing matrix')

            for j in range(A[-1].shape[1]):
                for i in range(nf):
                    _res = abs(truth[i, j] - A[:, i, j])
                    axs[1].plot(_res, '-r', alpha=0.5)
            axs[1].set_yscale('log')
            axs[1].set_xlabel('Iterations')
            axs[1].set_ylabel('Residual RMS')
            
            plt.savefig(f'jobs/{self.job_id}/A_iter{ki+1}.png')
            
            if ki > 0:
                os.remove(f'jobs/{self.job_id}/A_iter{ki}.png')
                
            plt.close()

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

            color = ['red', 'blue']
            label = ['150 GHz focal plane', '220 GHz focal plane']
            for j in range(2):
                plt.hist(gain[-1, :, j], bins=20, color=color[j], label=label[j])
            plt.axvline(0, ls='--', color='black', label='Expected gain')
            plt.title('Gain residual')
            plt.xlim(-0.5, 0.5)
            plt.xlabel('Gain value')
            plt.ylabel('# detectors')
            plt.legend()
            plt.savefig(f'jobs/{self.job_id}/gain_iter{ki+1}.png')

            if self.preset.tools.rank == 0:
                if ki > 0:
                    os.remove(f'jobs/{self.job_id}/gain_iter{ki}.png')

            plt.close()