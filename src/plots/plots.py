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
        - dogif : Bool to produce GIF.
    
    """
    
    def __init__(self, sims, dogif=False):
        
        self.sims = sims
        self.job_id = self.sims.job_id
        self.dogif = dogif
        self.params = self.sims.params
        
    def plot_beta_iteration(self, beta, figsize=(8, 6), truth=None, ki=0):

        """
        
        Method to plot beta as function of iteration. beta can have shape (niter) of (niter, nbeta)
        
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
                for i in range(beta.shape[1]):
                   
                    plt.plot(alliter, beta[:, i, 0], '-k', alpha=0.3)
                    if truth is not None:
                        plt.axhline(truth[i], ls='--', color='red')

            plt.subplot(2, 1, 2)
            
            if np.ndim(beta) == 1:
                plt.plot(alliter[1:]-1, truth - beta[1:])
                #if truth is not None:
                    #plt.axhline(truth, ls='--', color='red')
            else:
                for i in range(beta.shape[1]):
                   
                    plt.plot(alliter, abs(truth[i] - beta[:, i, 0]), '-k', alpha=0.3)
                    #if truth is not None:
                    #    plt.axhline(truth[i], ls='--', color='red')
            plt.yscale('log')
            plt.savefig(f'figures_{self.job_id}/beta_iter{ki+1}.png')

            if ki > 0:
                os.remove(f'figures_{self.job_id}/beta_iter{ki}.png')

            plt.close()
    def display_maps(self, seenpix, ngif=0, figsize=(14, 8), nsig=6, ki=0):
        
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
            C = HealpixConvolutionGaussianOperator(fwhm=self.params['Plots']['fake_conv'])
            for istk, s in enumerate(stk):
                plt.figure(figsize=figsize)

                k=0
                for icomp in range(len(self.sims.comps)):
                    
                    if self.params['Foregrounds']['nside_fit'] == 0:
                        map_in = C(self.sims.components[icomp, :, istk]).copy()
                        map_out = C(self.sims.components_iter[icomp, :, istk]).copy()
                        sig = np.std(self.sims.components[icomp, seenpix, istk])
                    else:
                        map_in = C(self.sims.components[istk, :, icomp]).copy()
                        map_out = C(self.sims.components_iter[istk, :, icomp]).copy()
                        sig = np.std(self.sims.components[istk, seenpix, icomp])
                    map_in[~seenpix] = hp.UNSEEN
                    map_out[~seenpix] = hp.UNSEEN
                    r = map_in - map_out
                    r[~seenpix] = hp.UNSEEN
                    
                    
                    
                    hp.gnomview(map_in, rot=self.sims.center, reso=13, notext=True, title='',
                        cmap='jet', sub=(len(self.sims.comps), 3, k+1), min=-2*sig, max=2*sig)
                    hp.gnomview(map_out, rot=self.sims.center, reso=13, notext=True, title='',
                        cmap='jet', sub=(len(self.sims.comps), 3, k+2), min=-2*sig, max=2*sig)
                    
                    hp.gnomview(r, rot=self.sims.center, reso=13, notext=True, title=f"{np.std(r[seenpix]):.3e}",
                        cmap='jet', sub=(len(self.sims.comps), 3, k+3), min=-1*sig, max=1*sig)

                    k+=3

                plt.tight_layout()
                plt.savefig(f'figures_{self.job_id}/{s}/maps_iter{ki+1}.png')

                plt.close()
        if self.dogif:
            if ngif%10 == 0:
                do_gif(f'figures_{self.job_id}/I/', ki+1)
                do_gif(f'figures_{self.job_id}/Q/', ki+1)
                do_gif(f'figures_{self.job_id}/U/', ki+1)
    def plot_gain_iteration(self, gain, alpha, figsize=(8, 6)):
        
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

            if self.params['MapMaking']['qubic']['type'] == 'two':
                color = ['--r', '--b']
                for j in range(2):
                    plt.plot(alliter-1, np.mean(gain, axis=1)[:, j], color[j], alpha=1)
                    for i in range(ndet):
                        plt.plot(alliter-1, gain[:, i, j], color[j], alpha=alpha)
                        
            elif self.params['MapMaking']['qubic']['type'] == 'wide':
                color = ['--g']
                plt.plot(alliter-1, np.mean(gain, axis=1), color[0], alpha=1)
                for i in range(ndet):
                    plt.plot(alliter-1, gain[:, i], color[0], alpha=alpha)
                        
            plt.yscale('log')
            plt.ylabel(r'|$g_{reconstructed} - g_{input}$|', fontsize=12)
            plt.xlabel('Iterations', fontsize=12)
            plt.savefig(f'figures_{self.job_id}/gain_iter{self._steps+1}.png')

            #if self._steps > 0:
            #    os.remove(f'figures_{self.job_id}/gain_iter{self._steps}.png')

            plt.close()
        
