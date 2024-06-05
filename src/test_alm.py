import numpy as np
from pyoperators import *
import healpy as hp
import pysm3
import pysm3.utils as utils
import pysm3.units as u
import matplotlib.pyplot as plt
import qubic
from acquisition.systematics import *
import pysimulators
from solver.cg import mypcg

NSIDE = 128
NSUB = 4
NREC = 2
NPOINTINGS=100
LMAX = 2 * NSIDE - 1
MMAX = LMAX

mapin = np.array(pysm3.Sky(nside=NSIDE, preset_strings=['c1']).get_emission(150 * u.GHz)).T
mapin[20000:] = 0
@pyoperators.flags.linear
class Alm2Map(Operator):
    
    def __init__(self, nside, lmax=None, mmax=None, pixwin=False, fwhm=0.0, sigma=None, pol=True, inplace=False, verbose=True, **keywords):        
        
        self.nside=nside
        self.npix=12*self.nside**2
        self.lmax=lmax
        if mmax is None:
            self.mmax=lmax
        else:
            self.mmax=mmax
        self.pixwin=pixwin
        self.fwhm=fwhm
        self.sigma=sigma 
        self.pol=pol 
        self.inplace=inplace
        self.verbose=verbose
        
        self.size = self.mmax * (2 * self.lmax + 1 - self.mmax) / 2 + self.lmax + 1
        Operator.__init__(self, shapein=(self.size, 3), shapeout=(self.npix, 3), dtype=np.complex64, **keywords)
    
    
    def direct(self, input, output):
        if input.ndim > 1:
            input = [_ for _ in input.T]
        output[...] = hp.alm2map(alms=input, 
                              nside=self.nside, 
                              lmax=self.lmax, 
                              mmax=self.mmax, 
                              pixwin=self.pixwin, 
                              fwhm=self.fwhm, 
                              sigma=self.sigma, 
                              pol=self.pol, 
                              inplace=self.inplace, 
                              verbose=self.verbose).T
    def transpose(self, input, output):
        if input.ndim > 1:
            input = [_ for _ in input.T]
        output[...] = hp.map2alm(maps=input, 
                              lmax=self.lmax, 
                              mmax=self.mmax, 
                              pol=self.pol, 
                              verbose=self.verbose).T

def give_cl_cmb(ell, r=0, Alens=1.):
        
    power_spectrum = hp.read_cl('data/Cls_Planck2018_lensed_scalar.fits')[:,:4000]
    if Alens != 1.:
        power_spectrum[2] *= Alens
    if r:
        power_spectrum += r * hp.read_cl('data/Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits')[:,:4000]
    return np.interp(ell, np.arange(1, 4001, 1), power_spectrum[2])
def _get_ultrawideband_config():
        
    nu_up = 247.5
    nu_down = 131.25
    nu_ave = np.mean(np.array([nu_up, nu_down]))
    delta = nu_up - nu_ave
    
    return nu_ave, 2*delta/nu_ave
def _get_dict():
    
    nu_ave, delta_nu_over_nu = _get_ultrawideband_config()

    args = {'npointings':NPOINTINGS, 
                'nf_recon':1, 
                'nf_sub':int(NSUB/2), 
                'nside':NSIDE, 
                'MultiBand':True, 
                'period':1, 
                'RA_center':0, 
                'DEC_center':-57,
                'filter_nu':nu_ave*1e9, 
                'noiseless':False, 
                'comm':None, 
                'kind':'IQU',
                'config':'FI',
                'verbose':False,
                'dtheta':15,
                'nprocs_sampling':1, 
                'nprocs_instrument':1,
                'photon_noise':True, 
                'nhwp_angles':3, 
                'effective_duration':3, 
                'filter_relative_bandwidth':delta_nu_over_nu, 
                'type_instrument':'wide', 
                'TemperatureAtmosphere150':None, 
                'TemperatureAtmosphere220':None,
                'EmissivityAtmosphere150':None, 
                'EmissivityAtmosphere220':None, 
                'detector_nep':4.7e-17, 
                'synthbeam_kmax':1,
                'synthbeam_fraction':1,
                'detector_tau':0}

    ### Get the default dictionary
    dictfilename = 'dicts/pipeline_demo.dict'
    d = qubic.qubicdict.qubicDict()
    d.read_from_file(dictfilename)
    for i in args.keys():
        
        d[str(i)] = args[i]

    
    return d


_, _, nus150, _, _, _ = qubic.compute_freq(150, Nfreq=int(NSUB/2))
_, _, nus220, _, _, _ = qubic.compute_freq(220, Nfreq=int(NSUB/2))
nus = np.array(list(nus150) + list(nus220))
d = _get_dict()

acq = QubicFullBandSystematic(d, NSUB, Nrec=1, comp=[], kind='two', nu_co=None, H=None, effective_duration150=3, effective_duration220=3)
invN = acq.get_invntt_operator()
#print(invN.shapein)
#h = []
#for i in range(len(acq.H)):
almop = Alm2Map(NSIDE, lmax=LMAX, fwhm=0.01)
alm_in = almop.T(mapin)
h = []
for i in range(len(acq.H)):
    with rule_manager(inplace=True):
        h += [CompositionOperator([acq.H[i], almop])]
    #h = CompositionOperator([acq.H[0], almop])
H = BlockColumnOperator([AdditionOperator(h[:int(NSUB/2)]), AdditionOperator(h[int(NSUB/2):])], axisout=0) 

A = H.T * invN * H
b = H.T * invN * H(alm_in)

shape = alm_in.shape

x0 = np.zeros(shape, dtype=np.complex64)
x0 += 1j * x0


solution = mypcg(A, b, disp=True, tol=1e-20, maxiter=50, x0=alm_in + 0.01, M=None)


cls = hp.alm2cl(alms1=solution['x']['x'][:, 2], lmax=LMAX, mmax=MMAX)

ell = np.arange(1, 2 * NSIDE + 1, 1)
_f = ell * (ell + 1) / (2 * np.pi)
Dls_true = give_cl_cmb(ell, r=0, Alens=1)
plt.figure()
plt.plot(ell, cls*_f)
plt.plot(ell, Dls_true*_f)
plt.show()
