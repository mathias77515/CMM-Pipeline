import numpy as np
from preset.preset import *
from pipeline import *
from pyoperators import *
from fgb.component_model import *
from fgb.mixing_matrix import *
import healpy as hp
from scipy.optimize import minimize, fsolve

### MPI common arguments
comm = MPI.COMM_WORLD


pip = Pipeline(comm, 1, 1)
tod_obs = pip.sims.TOD_Q

def _get_tod_comp_superpixel_planck():
    _index = np.zeros(12*pip.sims.params['Foregrounds']['nside_fit']**2)
    _index_nside = hp.ud_grade(_index, pip.sims.joint_out.external.nside)
    tod_comp = np.zeros((12*pip.sims.params['Foregrounds']['nside_fit']**2, len(pip.sims.comps_out), len(pip.sims.external_nus)*pip.sims.joint_out.external.npix*3))
    maps_conv = pip.sims.components_iter.T.copy()
    print(maps_conv.shape)
    for j in range(len(pip.sims.external_nus)):
        for co in range(len(pip.sims.comps_out)):
            for ii, i in enumerate(np.arange(12*pip.sims.params['Foregrounds']['nside_fit']**2)):
        
                maps_conv_i = maps_conv.copy()
                _i = _index_nside == i
                for stk in range(3):
                    maps_conv_i[:, :, stk] *= _i
                tod_comp[ii, co] = maps_conv_i[co].ravel()
    return tod_comp

    
    tod_comp = np.zeros((12*pip.sims.params['Foregrounds']['nside_fit']**2, len(nus), len(pip.sims.comps_out), tod_q.shape[-1] + tod_p.shape[-1]))


#stop

index_num = hp.ud_grade(pip.sims.seenpix_qubic, pip.sims.params['Foregrounds']['nside_fit'])    #
index = np.where(index_num == True)[0]
#print(tod_obs.shape)
d = pip._get_tod_comp_superpixel(index)


Np, Nsub, Ncomp, Nt = d.shape

dcmb = np.sum(np.sum(d[:, :, 0, :], axis=1), axis=0)             # (Np, Nsub, Ncomp, Nt) -> (Np, Nt) -> (Nt)
dnu = d[:, :, 1, :]                                              # (Np, Nsub, Ncomp, Nt) -> (Np, Nsub, Nt)

print(dcmb.shape)
print(dnu.shape)
dnu_reshaped = dnu.reshape((Np*Nsub, Nt))
print(dnu_reshaped.shape)



def _get_A_from_beta(nus, beta):
    comp = Dust(nu0=150, temp=20)
    sed = comp.eval(nus, beta)
    sed_reshaped = sed.reshape((sed.shape[0]*sed.shape[1]))
    return sed_reshaped


#_get_A_from_beta(pip.sims.joint_in.qubic.allnus, np.array([1.54, 1.54]))

invN_q = pip.sims.invN.operands[0]
fact = 1e30
A = (dnu_reshaped @ dnu_reshaped.T) * fact
b = (dnu_reshaped @ (tod_obs - dcmb)) * fact

print(A.shape)
print()
print(b.shape)


def _mychi2(x):

    Amm = _get_A_from_beta(pip.sims.joint_in.qubic.allnus, x)
    chi2 = np.sum((Amm @ A) - b)**2
    
    return chi2
    
#_mychi2(pip.sims.beta_in[index, 0])    
    
 

    
def chi2(x):
    #print(dcmb.shape)
    #print(dnu_reshaped.shape)
    x = _get_A_from_beta(pip.sims.joint_in.qubic.allnus, x)
    #print(x.shape)
    #stop
    dsim = dcmb + x @ dnu_reshaped
    
    return (dsim - tod_obs).T @ invN_q(dsim - tod_obs)

#stop
sol = minimize(_mychi2, x0=np.ones(len(index))*1.53, tol=1e-20, method='BFGS')

print(sol.x)
print()
print(pip.sims.beta_in[index, 0])
print()
print(sol.x - pip.sims.beta_in[index, 0])
