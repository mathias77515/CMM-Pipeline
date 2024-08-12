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
        self.nspecs = (self.ncomps * (self.ncomps + 1)) // 2
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
            self.cov_sample_variance = np.zeros((self.ncomps, self.ncomps, len(self.ell), len(self.ell)))
            
            indices_tr = np.triu_indices(self.ncomps)
            matrix = np.zeros((self.nspecs, len(self.ell), self.nspecs, len(self.ell)))
            factor_modecount = 1/((2 * self.ell + 1) * self.fsky * self.dl)
            for ii, (i1, i2) in enumerate(zip(indices_tr[0], indices_tr[1])):
                for jj, (j1, j2) in enumerate(zip(indices_tr[0], indices_tr[1])):
                    print(ii, jj, i1, i2, j1, j2)
            stop
            
            for icomp in range(self.ncomps):
                for jcomp in range(self.ncomps):
                    np.fill_diagonal(self.cov_sample_variance[icomp, jcomp], (1/factor_modecount) * (cl[:, icomp, icomp, :] * cl[:, jcomp, jcomp, :] + cl[:, icomp, jcomp, :]**2))
            
            #np.fill_diagonal(self.cov_sample_variance, (np.sqrt(2. / (2 * self.ell + 1) / self.fsky / self.dl) * cl)**2)
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
                    print(Dl_true.shape)
                    
                    cov_sample = self._sample_variance(Dl_true)
                    stop
                else:
                    cov_sample = 0
                covi = np.cov(self.Nl[:, i, j, :], rowvar=False)
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
files = ['autospectrum_parametric_d0_DB_test_sigr_noplanck_conv.pkl']

class Fit:
    
    def __init__(self, ell):
        
        self.ell = ell
        self.nbins = len(self.ell)
        self.f = self.ell * (self.ell + 1) / (2 * np.pi)
    
    def _check_free_param(self, line):
        if type(line[0]) is float or type(line[0]) is int:
            return line[0]
        elif line[0] is False:
            return False
        else:
            return True
    def _initialize(self, nwalkers):
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
        
        self.p0 = np.zeros((1, nwalkers))
        self.index_notfree_param = []
        for i in range(len(self.names)):
            if self.params[self.names[i]][0] is True:
                p0 = np.random.normal(self.params[self.names[i]][3], self.params[self.names[i]][4], (1, nwalkers))
                self.p0 = np.concatenate((self.p0, p0), axis=0)
            else:
                self.index_notfree_param += [i]
        
        self.p0 = np.delete(self.p0, 0, axis=0).T
        self.ndim = self.p0.shape[1]
        self.names = np.delete(self.names, self.index_notfree_param)
    def cmb(self, r=0, Alens=1.):
    
        power_spectrum = hp.read_cl('/Users/mregnier/Desktop/git/CMM-Pipeline/src/data/Cls_Planck2018_lensed_scalar.fits')[:,:4000]
        if Alens != 1.:
            power_spectrum[2] *= Alens
        if r:
            power_spectrum += r * hp.read_cl('/Users/mregnier/Desktop/git/CMM-Pipeline/src/data/Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits')[:,:4000]
        return self.f * np.interp(self.ell, np.arange(1, 4001, 1), power_spectrum[2])
    def dust(self, A=10, alpha=-0.1):
        return A * (self.ell/80)**alpha
  
class FitTensor(Fit):
    
    def __init__(self, ell, Dl, Nl, samp_var=False, nwalkers=10):
        
        Fit.__init__(self, ell)
        
        self.nwalkers = nwalkers
        self.N, self.ncomps, _ = Nl.shape
        self.Dl = Dl.reshape((self.ncomps*self.nbins)) # shape (Ncomp, Nbins)
        
        self.Nl = Nl.reshape((self.N, self.ncomps*self.nbins)) # shape (Nreals, Ncomps, Nbins)
        
        self.cov = np.cov(self.Nl, rowvar=False)
        self.corr = np.corrcoef(self.Nl, rowvar=False)
        print(self.corr.shape)
        #print(self.Dl.shape)
        #stop
        #print(Dl.shape)
        #stop
        self.cov += self._fill_sample_variance(Dl)
        #plt.figure()
        #plt.imshow(self.corr, vmin=-1, vmax=1, cmap='bwr')
        #plt.show()
        #stop
        self.invN = np.linalg.pinv(self.cov)
        
        self._initialize(nwalkers=nwalkers)
        
    def like(self, x):
        
        Dl_i = np.array([self.cmb(*x), np.zeros(self.nbins)]).reshape((self.ncomps*self.nbins))
        _r = Dl_i - self.Dl
        
        L = -0.5 * (_r.T @ self.invN @ _r)
        
        return L
    def _fill_sample_variance(self, bandpower):
        
        indices_tr = np.triu_indices(self.ncomps)

        matrix = np.zeros((self.ncomps, len(self.ell), self.ncomps, len(self.ell)))
        factor_modecount = 1/((2 * self.ell + 1) * 0.015 * 30)
        for ii in range(self.ncomps):
            for jj in range(self.ncomps):
                covar = (2 * bandpower[ii, :] * bandpower[jj, :]) * factor_modecount
                matrix[ii, :, jj, :] = np.diag(covar)

        return matrix.reshape((self.ncomps*len(self.ell), self.ncomps*len(self.ell)))
    def __call__(self, nsteps):
        
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(self.nwalkers, self.p0.shape[1], self.like, pool=pool)
            sampler.run_mcmc(self.p0, nsteps, progress=True)
        
        return sampler
        

'''
class FitTensor(Fit):
    
    def __init__(self, ell, Dl, Nl, samp_var=False, nwalkers=10):
        
        Fit.__init__(self, ell)
        
        self.nwalkers = nwalkers
        self._initialize(self.nwalkers)

        self.ell = ell
        self.f = self.ell * (self.ell + 1) / (2 * np.pi)
        
        self.Nl = Nl
        self.N, self.ncomps, _, self.nbins = self.Nl.shape
        self.nspecs = (self.ncomps * (self.ncomps + 1)) // 2
        
        
        #stop
        x = self._fill_params(np.array([0, 0.001, -0.1]))
        Dl = np.array([[self.cmb(r=0, Alens=1), self.cmb()*0], 
                       [self.cmb()*0, self.dust(A=1, alpha=-0.02)]])
        
        #self._sky(*x).reshape((3, self.nbins))
        
        #Dl = np.array([[Dl[0], Dl[0]*0]])

        self.Dl = np.zeros((self.nspecs, self.nbins))
        nl = np.zeros((self.N, self.nspecs, self.nbins))
        print(Dl.shape)
        k=0
        for i in range(self.ncomps):
            for j in range(i, self.ncomps):
                nl[:, k, :] = self.Nl[:, i, j, :]
                self.Dl[k, :] = Dl[i, j, :]
                k+=1
        
        self.Dl = self.Dl.reshape((self.nspecs*self.nbins))
        nl = nl.reshape((self.N, self.nspecs*self.nbins))
        self.sample_variance = self._fill_sample_variance(Dl)
        
        self.cov = np.cov(nl, rowvar=False)
        self.corr = np.corrcoef(nl, rowvar=False)
        
        if samp_var:
            self.cov += self.sample_variance

        self.invN = np.linalg.pinv(self.cov)

    def _fill_sample_variance(self, bandpower):
        
        indices_tr = np.triu_indices(self.ncomps)

        matrix = np.zeros((self.nspecs, len(self.ell), self.nspecs, len(self.ell)))
        factor_modecount = 1/((2 * self.ell + 1) * 0.015 * 30)
        for ii, (i1, i2) in enumerate(zip(indices_tr[0], indices_tr[1])):
            for jj, (j1, j2) in enumerate(zip(indices_tr[0], indices_tr[1])):
                if ii == jj:
                    covar = (bandpower[i1, j1, :] * bandpower[i2, j2, :] + bandpower[i1, j2, :] * bandpower[i2, j1, :]) * factor_modecount
                    matrix[ii, :, jj, :] = np.diag(covar)

        return matrix.reshape((self.nspecs*len(self.ell), self.nspecs*len(self.ell)))
    def _initialize(self, nwalkers):
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
        
        self.p0 = np.zeros((1, nwalkers))
        self.index_notfree_param = []
        for i in range(len(self.names)):
            if self.params[self.names[i]][0] is True:
                p0 = np.random.normal(self.params[self.names[i]][3], self.params[self.names[i]][4], (1, nwalkers))
                self.p0 = np.concatenate((self.p0, p0), axis=0)
            else:
                self.index_notfree_param += [i]
        
        self.p0 = np.delete(self.p0, 0, axis=0).T
        self.ndim = self.p0.shape[1]
        self.names = np.delete(self.names, self.index_notfree_param)
    def _check_free_param(self, line):
        if type(line[0]) is float or type(line[0]) is int:
            return line[0]
        elif line[0] is False:
            return False
        else:
            return True
    def _fill_params(self, x):

        for ii, i in enumerate(self.free):

            if i is True:
                pass
            elif type(i) is float or type(i) is int:
                x = np.insert(x, [ii], i)
            else:
                pass

        return x
    def _sky(self, r=0, Alens=1, Ad=10, alphad=-0.1, As=10, alphas=-0.1):
        
        s = np.zeros((self.nspecs, len(self.ell)))
        
        s[0] = self.cmb(r=r, Alens=Alens)
        s[2] = self.dust(A=Ad, alpha=alphad)

        return s.reshape((self.nspecs*self.nbins))
    def like(self, x):
        x = self._fill_params(x)

        Dl_true = self._sky(*x)
        #print(Dl_true.shape, self.Dl.shape, self.invN.shape)
        #stop
        _r = self.Dl - Dl_true 
        
        Li = self.log_prob(x)
        Li -= 0.5 * (_r.T @ self.invN @ _r)
        
        return Li
    def log_prob(self, x):
        
        for iparam, param in enumerate(self.free):
            if param is True:
                if x[iparam] < self.params[self.allnames[iparam]][1] or x[iparam] > self.params[self.allnames[iparam]][2]:
                    return -np.inf

        return 0
    def __call__(self, nsteps):
        
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(self.nwalkers, self.p0.shape[1], self.like, pool=pool)
            sampler.run_mcmc(self.p0, nsteps, progress=True)
        
        return sampler
''' 

DISCARD = 200
NSTEPS = 300
NWALKERS = 100

for iname, name in enumerate(files):
    
    d = open_data(folder + name)
    
    #fit = FitTensor(d['ell'][:-1], d['Dl_bias'][:, :-1], dnoise['Nl'][:, :, :, :-1], samp_var=True,
    #                nsteps=nsteps, nwalkers=nwalkers)
    
    fit = FitTensor(d['ell'][:], d['Dl'][:, 0, :], d['Nl'][:200, 0, :, :], samp_var=True, nwalkers=NWALKERS)
    #print(fit.like(np.array([0])))
    #stop
    sampler = fit(NSTEPS)
    chains = sampler.get_chain()
    chains_flat = sampler.get_chain(discard=DISCARD, flat=True, thin=15)

    
    print(np.mean(chains_flat, axis=0))
    print(np.std(chains_flat, axis=0))
    
    #with open("chains" + name[12:], 'wb') as handle:
    #    pickle.dump({'ell':d['ell'], 
    #                 'Nl':d['Nl'],
    #                 'bias':d['Dl_bias'],
    #                 'chains':chains, 
    #                 'chainflat':chains_flat,
    #                 'discard':dis
    #                 }, handle, protocol=pickle.HIGHEST_PROTOCOL)


labels = ['r']#, 'A_d', 'alpha_d']
names = ['r']#, 'ad', 'alphad']

plt.figure()

plt.plot(chains[..., 0], '-k', alpha=0.1)
plt.plot(np.mean(chains, axis=1)[:, 0], '-b', alpha=1)
plt.plot(np.std(chains, axis=1)[:, 0], '-r', alpha=1)
plt.axhline(0, ls='--', color='black')

plt.savefig('chains.png')
plt.close()

samples = MCSamples(samples=chains_flat, names=names, labels=labels, ranges={'r':(0, None)})

plt.figure()
# Triangle plot
g = plots.get_subplot_plotter(width_inch=8)
g.triangle_plot([samples], filled=True, title_limit=1)

plt.savefig('triangle.png')   
plt.close()

