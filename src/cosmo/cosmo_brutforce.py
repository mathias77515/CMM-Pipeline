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

dis = 200
nsteps = 600
nwalkers = 30


class FitTensor:
    
    def _check_free_param(self, line):
        if type(line[0]) is float or type(line[0]) is int:
            return line[0]
        elif line[0] is False:
            return False
        else:
            return True
    def __init__(self, ell, Nl, bias, samp_var=False, nsteps=100, nwalkers=10, r_init=0, Alens_init=1., A_init=1, alpha_init=-0.1):
        
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
        
        #if type(self.params['r']) is list:
        #    if self.params['r'][0]:
        
        for name in self.names:
            self.free += [self._check_free_param(self.params[name])]
        #stop
        
        
        if self.free[0] is not False or self.free[1] is not False:
            self.is_cmb = True
        
        if self.free[2] is not False or self.free[3] is not False:
            self.is_dust = True


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
        
        self.fsky = 0.01
        self.dl = 30
        self.samp_var = samp_var

        self.ncomps = np.sum(np.array([self.is_cmb, self.is_dust, self.is_sync]))
        self.Dl_obs = np.zeros((self.ncomps, len(self.ell)))

        for i in range(self.ncomps):
            if i == 0:
                self.Dl_obs[i] = self.cmb(r=r_init, Alens=Alens_init)
            elif i == 1:
                self.Dl_obs[i] = self.dust(A=A_init, alpha=alpha_init)
        self.Dl_obs = self._sky(r=0, Alens=1, A=1, alpha=-0.1)#self.bias[i]
            
        self.L, self.sigmar = self()   
    def cmb(self, r=0, Alens=1.):
    
        power_spectrum = hp.read_cl('/Users/mregnier/Desktop/git/CMM-Pipeline/src/data/Cls_Planck2018_lensed_scalar.fits')[:,:4000]
        if Alens != 1.:
            power_spectrum[2] *= Alens
        if r:
            power_spectrum += r * hp.read_cl('/Users/mregnier/Desktop/git/CMM-Pipeline/src/data/Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits')[:,:4000]
        return self.f * np.interp(self.ell, np.arange(1, 4001, 1), power_spectrum[2])
    def dust(self, A=10, alpha=-0.1):
        return A * (self.ell/80)**alpha
    def _sky(self, r=0, Alens=1, A=10, alpha=-0.1):
        
        s = np.zeros((1, len(self.ell)))
        if self.is_cmb:
            s = np.concatenate((s, np.array([self.cmb(r=r, Alens=Alens)])), axis=0)
        if self.is_dust:
            s = np.concatenate((s, np.array([self.dust(A=A, alpha=alpha)])), axis=0)
        
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
                elif k == 3:
                    cov_sample = self._sample_variance(Dl_true[1])
                else:
                    cov_sample = 0
                covi = np.cov(self.Nl[:, k, :], rowvar=False) * np.eye(len(self.ell))
                invcov_i = np.linalg.pinv(covi + cov_sample)
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
    def _get_one_sigma(self, rv, like):
        cumint = scipy.integrate.cumtrapz(like, x=rv)
        cumint = cumint / np.max(cumint)
        return np.interp(0.68, cumint, rv[1:])
    def __call__(self):
        
        self.rv = np.linspace(0, 0.1, 300)
        chi2 = np.zeros(len(self.rv))
        for i in range(len(self.rv)):
            chi2[i] = self.like(np.array([self.rv[i]]))
        L = np.exp(chi2)
        sig = self._get_one_sigma(self.rv, L)
        return L/L.max(), sig



files = [
        #'autospectrum_parametric_d0_wide_inCMBDust_outCMBDust_ndet0.pkl',
        'autospectrum_parametric_d0_wide_inCMBDust_outCMBDust_ndet0_1.pkl',
        'autospectrum_parametric_d0_wide_inCMBDust_outCMBDust_ndet0_3.pkl',
        'autospectrum_parametric_d0_wide_inCMBDust_outCMBDust_ndet0_5.pkl',
        'autospectrum_parametric_d0_wide_inCMBDust_outCMBDust_ndet0_7.pkl',
        'autospectrum_parametric_d0_wide_CMMpaper_inCMBDust_outCMBDust_ndet1.pkl'
         ]

plt.figure(figsize=(8, 6))
for iname, name in enumerate(files):
    
    d = open_data(name)
    dnoise = open_data(name)
    
    fit = FitTensor(d['ell'][:-1], dnoise['Nl'][:, :, :-1], d['Dl_bias'][:, :-1], samp_var=True,
                    nsteps=nsteps, nwalkers=nwalkers, r_init=0, Alens_init=1, A_init=10, alpha_init=-0.1)

    maxL = fit.rv[np.where(fit.L == fit.L.max())[0]][0]

    print('sigma(r) =', fit.sigmar)
    
    plt.plot(fit.rv, fit.L, label=r'$\sigma(r)$ = ' + f'{fit.sigmar:.4f}')


    
    
    
    
    
    
    
    
    
    
    
plt.legend(frameon=False, fontsize=12)
plt.xlim(0, np.max(fit.rv))
plt.ylim(0, 1.05)
plt.savefig('like.png')
plt.close()


plt.figure()
plt.plot(fit.ell, fit.cmb(r=0, Alens=1), '-k')
plt.plot(fit.ell, (fit.bias[0]), '-r')
plt.plot(fit.ell, (fit.cmb(r=maxL, Alens=1)))
plt.yscale('log')
plt.savefig('dl.png')
plt.close()




#plt.figure()
#plt.plot(fit.ell, fit.cmb(r=0, Alens=1) / fit.f)
#plt.errorbar(fit.ell, np.std(fit.Nl/fit.f, axis=0)[0], yerr=np.std(fit.Nl/fit.f, axis=0)[0] / np.sqrt(200), fmt='-b', capsize=5)
#plt.errorbar(fit.ell, np.std(fit.Nl/fit.f, axis=0)[3], yerr=np.std(fit.Nl/fit.f, axis=0)[3] / np.sqrt(200), fmt='-r', capsize=5)
#plt.axhline(9.938330e-6, ls='--', color='b')
#plt.axhline(2.0893e-06, ls='--', color='r')
#plt.plot(fit.ell, fit.cmb(r=0.001, Alens=1) / fit.f)
#plt.errorbar(fit.ell, fit.bias[0] / fit.f)
#plt.plot(fit.ell, (fit.cmb(r=0, Alens=1) + fit.bias[0]) / fit.f)
#plt.yscale('log')
#plt.savefig('cl.png')
#plt.close()
    

