import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pickle
import scipy
from tqdm import tqdm
import emcee
from multiprocess import Pool
from schwimmbad import MPIPool
from getdist import plots, MCSamples
import yaml

def open_data(filename):      

    """
    Open pickle file inside path with given filename
    """
        
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

dis = 400
nsteps = 500
nwalkers = 50

class FitTensor:
    
    def _check_free_param(self, line):
        if type(line[0]) is float or type(line[0]) is int:
            return line[0]
        elif line[0] is False:
            return False
        else:
            return True
    def __init__(self, ell, bias, Nl, samp_var=False, nsteps=100, nwalkers=10):
        
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        
        with open('params.yml', "r") as stream:
            self.params = yaml.safe_load(stream)
        
        self.names = list(self.params.keys())
        self.allnames = list(self.params.keys())
        self.free = []

        self.is_cmb = False
        self.is_dust = False
        self.is_sync = False
        
        for name in self.names:
            self.free += [self._check_free_param(self.params[name])]
        
        if self.free[0] is not False or self.free[1] is not False:
            self.is_cmb = True
        
        if self.free[2] is not False or self.free[3] is not False:
            self.is_dust = True
        
        if self.free[4] is not False or self.free[5] is not False:
            self.is_sync = True
        
        self.p0 = np.zeros((1, self.nwalkers))
        self.index_notfree_param = []
        for i in range(len(self.names)):
            if self.params[self.names[i]][0] is True:
                p0 = np.random.normal(self.params[self.names[i]][3], self.params[self.names[i]][4], (1, self.nwalkers))
                self.p0 = np.concatenate((self.p0, p0), axis=0)
            else:
                self.index_notfree_param += [i]
        
        self.p0 = np.delete(self.p0, 0, axis=0).T
        self.ndim = self.p0.shape[1]
        self.names = np.delete(self.names, self.index_notfree_param)
        
        self.ell = ell
        self.Nl = Nl
        self.bias = bias
        self.f = self.ell * (self.ell + 1) / (2 * np.pi)
        self.ncomps = int(np.sqrt(self.Nl.shape[1]))
        self.fsky = 0.015
        self.dl = 30
        self.samp_var = samp_var

        self.ncomps = np.sum(np.array([self.is_cmb, self.is_dust, self.is_sync]))
        self.Dl_obs = self._sky(r=0, Alens=1, Ad=1, alphad=-0.1)
        #self.Dl_obs = bias.copy()
        #print(self.Dl_obs.shape)
        #stop
  
        self.sampler = self()   
    def cmb(self, r=0, Alens=1.):
    
        power_spectrum = hp.read_cl('/Users/mregnier/Desktop/git/CMM-Pipeline/src/data/Cls_Planck2018_lensed_scalar.fits')[:,:4000]
        if Alens != 1.:
            power_spectrum[2] *= Alens
        if r:
            power_spectrum += r * hp.read_cl('/Users/mregnier/Desktop/git/CMM-Pipeline/src/data/Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits')[:,:4000]
        return self.f * np.interp(self.ell, np.arange(1, 4001, 1), power_spectrum[2])
    def dust(self, A=10, alpha=-0.1):
        return A * (self.ell/80)**alpha
    def _sky(self, r=0, Alens=1, Ad=10, alphad=-0.1, As=10, alphas=-0.1):
        
        s = np.zeros((1, len(self.ell)))
        if self.is_cmb:
            s = np.concatenate((s, np.array([self.cmb(r=r, Alens=Alens)])), axis=0)
        if self.is_dust:
            s = np.concatenate((s, np.array([self.dust(A=Ad, alpha=alphad)])), axis=0)
        if self.is_sync:
            s = np.concatenate((s, np.array([self.dust(A=As, alpha=alphas)])), axis=0)
        s = np.delete(s, 0, axis=0)

        return s
    def _sample_variance(self, cl):
        if self.samp_var:
            self.cov_sample_variance = np.zeros((len(self.ell), len(self.ell)))
            np.fill_diagonal(self.cov_sample_variance, (np.sqrt(2. / (2 * self.ell + 1) / self.fsky / self.dl) * cl)**2)
        else:
            self.cov_sample_variance = np.zeros((len(self.ell), len(self.ell)))
        
        return self.cov_sample_variance
    def _fill_params(self, x):

        for ii, i in enumerate(self.free):

            if i is True:
                pass
            elif type(i) is float or type(i) is int:
                x = np.insert(x, [ii], i)
            else:
                pass

        return x   
    def like(self, x):
        
        x = self._fill_params(x)

        Dl_true = self._sky(*x)
        _r = self.Dl_obs - Dl_true 
        
        k = 0
        Li = self.log_prob(x)

        for i in range(self.ncomps):
            for j in range(self.ncomps):
                if k == 0:
                    cov_sample = self._sample_variance(Dl_true[0])
                else:
                    cov_sample = 0
                covi = np.cov(self.Nl[:, k, :], rowvar=False)# * np.eye(len(self.ell))
                invcov_i = np.linalg.pinv(covi + cov_sample)
                
                #if i == j:
                Li -= 0.5 * (_r[i].T @ invcov_i @ _r[i])
                k+=1
        return Li
    def log_prob(self, x):
        
        for iparam, param in enumerate(self.free):
            
            
            if param is True:
                #print(self.params[self.allnames[iparam]])
                if x[iparam] < self.params[self.allnames[iparam]][1] or x[iparam] > self.params[self.allnames[iparam]][2]:
                    return -np.inf
            
            #if type(self.params[self.names[iparam]]) is list:
            #    if self.params[self.names[iparam]][0]:
            #        if param < self.params[self.names[iparam]][1] or param > self.params[self.names[iparam]][2]:
            #            return -np.inf
        return 0
    def __call__(self):
        
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(self.nwalkers, self.p0.shape[1], self.like, pool=pool)
            sampler.run_mcmc(self.p0, self.nsteps, progress=True)
        
        return sampler

folder = ''
files = ['autospectrum_parametric_d0_two_inCMB_outCMB_ndet0_nyrs6.pkl',
         #'autospectrum_parametric_d0_two_inCMB_outCMB_ndet0_7_nyrs1_5.pkl',
         #'autospectrum_parametric_d0_two_inCMB_outCMB_ndet0_5_nyrs1_5.pkl',
         #'autospectrum_parametric_d0_two_inCMB_outCMB_ndet0_3_nyrs1_5.pkl',
         #'autospectrum_parametric_d0_two_inCMB_outCMB_ndet0_1_nyrs1_5.pkl'
         ]
for iname, name in enumerate(files):
    
    d = open_data(folder + name)
    
    dnoise = open_data(name)
    fit = FitTensor(d['ell'][:-1], d['Dl_bias'][:, :-1], dnoise['Nl'][:, :, :-1], samp_var=False,
                    nsteps=nsteps, nwalkers=nwalkers)
    
    chains = fit.sampler.get_chain()
    chains_flat = fit.sampler.get_chain(discard=dis, flat=True, thin=15)

    with open("chains" + name[12:], 'wb') as handle:
        pickle.dump({'ell':d['ell'], 
                     'Nl':d['Nl'],
                     'bias':d['Dl_bias'],
                     'chains':chains, 
                     'chainflat':chains_flat,
                     'discard':dis
                     }, handle, protocol=pickle.HIGHEST_PROTOCOL)

    _p = fit._fill_params(np.mean(chains_flat, axis=0))
    #print(_p)
    mysky = fit._sky(*_p)
    
    plt.figure(figsize=(15, 8))
    c = ['red', 'blue', 'green']
    for i in range(fit.ncomps):
        if i == 0:
            plt.plot(fit.ell, fit.cmb(r=0, Alens=_p[1]), '-k', label=f'BB power spectra assuming r = 0 | Alens = {_p[1]}')
        plt.plot(fit.ell, mysky[i], color=c[i])
        plt.scatter(fit.ell, fit.Dl_obs[i], s=5, color=c[i], marker='o')
        
    plt.plot(fit.ell*1000, mysky[i], '-k', label='Model')
    plt.scatter(fit.ell*1000, fit.Dl_obs[i], s=5, color='black', marker='o', label = 'Data')
    
    plt.xlim(30, 512)
    plt.legend(frameon=False, fontsize=12)
    plt.yscale('log')
    plt.savefig('Dl_fit.png')
    plt.close()
    
    print('Average : ', np.mean(chains_flat, axis=0))
    print('Std     : ', np.std(chains_flat, axis=0))
    
    plt.figure()

    names = fit.names.copy()
    labels =  fit.names.copy()

    for i in range(fit.ndim):
        plt.subplot(fit.ndim, 1, i+1)
        plt.plot(chains[:, :, i], 'k', alpha=0.3)
        #plt.plot(np.mean(chains[:, :, i], axis=1), '-b')
        #plt.plot(np.std(chains[:, :, i], axis=1), '-r')

    plt.savefig('chains.png')   
    plt.close() 



print(np.mean(chains_flat, axis=0))
print(np.std(chains_flat, axis=0))




samples = MCSamples(samples=chains_flat, names=names, labels=labels, ranges={'r':(0, None)})

plt.figure()
# Triangle plot
g = plots.get_subplot_plotter(width_inch=8)
g.triangle_plot([samples], filled=True, title_limit=1)

plt.savefig('test.png')   
plt.close()

