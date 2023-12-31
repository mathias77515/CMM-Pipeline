from preset.preset import PresetSims
from pyoperators import *

import fgb.mixing_matrix as mm
import fgb.component_model as c

from acquisition.systematics import *

from simtools.mpi_tools import *
from simtools.noise_timeline import *
from simtools.foldertools import *


def _norm2(x, comm):
    x = x.ravel()
    n = np.array(np.dot(x, x))
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, n)
    return n
def _dot(x, y, comm):
    d = np.array(np.dot(x.ravel(), y.ravel()))
    
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, d)
    return d



class Chi2Parametric:
    
    def __init__(self, sims, d, betamap):
        
        self.sims = sims
        self.d = d
        self.betamap = betamap
        if len(self.d.shape) == 3:
            self.nc, self.nf, self.nsnd = self.d.shape
            self.constant = True
        else:
            self.npix, self.nf, self.nc, self.nsnd = self.d.shape
            index_num = hp.ud_grade(self.sims.seenpix_qubic, self.sims.params['Foregrounds']['nside_fit'])    #
            index = np.where(index_num == True)[0]
            self._index = index#np.where(np.sum(self.d[:, 0, 0, :], axis=-1) != 0)[0]
            self.constant = False

    def _get_mixingmatrix(self, x):
        mixingmatrix = mm.MixingMatrix(*self.sims.comps_out)
        if self.constant:
            return mixingmatrix.eval(self.sims.joint_out.qubic.allnus, *x)
        else:
            return mixingmatrix.eval(self.sims.joint_out.qubic.allnus, x)

    def __call__(self, x):

        A = self._get_mixingmatrix(x)
        #print(A.shape)

        if self.constant:
            self.betamap = x.copy()
            if self.sims.params['MapMaking']['qubic']['type'] == 'wide':
                ysim = np.zeros(self.nsnd)
                for ic in range(self.nc):
                    ysim += A[0, :, ic] @ self.d[ic]
            else:
                ysim = np.zeros(int(self.nsnd*2))
                for ic in range(self.nc):
                    ysim[:int(self.nsnd)] += A[:int(self.nf/2), ic] @ self.d[ic, :int(self.nf/2)]
                    ysim[int(self.nsnd):int(self.nsnd*2)] += A[int(self.nf/2):int(self.nf), ic] @ self.d[ic, int(self.nf/2):int(self.nf)]
        else:
            self.betamap[self._index, 0] = x.copy()
            if self.sims.params['MapMaking']['qubic']['type'] == 'wide':
                ysim = np.zeros(self.nsnd)
                for ic in range(self.nc):
                    for ip, p in enumerate(self._index):
                        ysim += A[ip, :, ic] @ self.d[ip, :, ic]
            else:
                ysim = np.zeros(int(self.nsnd*2))
                for ic in range(self.nc):
                    for ip, p in enumerate(self._index):
                        ysim[:int(self.nsnd)] += A[ip, :int(self.nf/2), ic] @ self.d[ip, :int(self.nf/2), ic]
                        ysim[int(self.nsnd):int(self.nsnd*2)] += A[ip, int(self.nf/2):int(self.nf), ic] @ self.d[ip, int(self.nf/2):int(self.nf), ic]

        
        _r = ysim - self.sims.TOD_Q
        H_planck = self.sims.joint_out.get_operator(self.betamap,  gain=self.sims.g_iter, fwhm=self.sims.fwhm_recon, nu_co=self.sims.nu_co).operands[1]
        tod_pl_s = H_planck(self.sims.components_iter)
        
        _r_pl = self.sims.TOD_E - tod_pl_s
        _r = np.r_[_r, _r_pl]
        
        #return _dot(_r.T, self.sims.invN.operands[0](_r), self.sims.comm) + _r_pl.T @ self.sims.invN.operands[1](_r_pl)
        return _dot(_r.T, self.sims.invN(_r), self.sims.comm)


'''
class Chi2ConstantParametric:
    
    def __init__(self, sims):
        
        self.sims = sims
        self.nc = len(self.sims.comps)
        self.nsnd = self.sims.joint.qubic.ndets*self.sims.joint.qubic.nsamples
        self.nsub = self.sims.joint.qubic.Nsub
        self.mixingmatrix = mm.MixingMatrix(*self.sims.comps)
        #print(self.sims.comps)
        
    def _qu(self, x, tod_comp, components, nus):
        
        A = self.mixingmatrix.eval(nus, *x)

        if self.sims.params['MapMaking']['qubic']['type'] == 'two':
            ysim = np.zeros(2*self.nsnd)
            
            for i in range(self.nc):
                ysim[:self.nsnd] += A[:self.nsub, i] @ tod_comp[i, :self.nsub]
                ysim[self.nsnd:self.nsnd*2] += A[self.nsub:self.nsub*2, i] @ tod_comp[i, self.nsub:self.nsub*2]
        
        elif self.sims.params['MapMaking']['qubic']['type'] == 'wide':
            ysim = np.zeros(self.nsnd)
            
            for i in range(self.nc):
                ysim[:self.nsnd] += A[:self.nsub*2, i] @ tod_comp[i, :self.nsub*2]
        _r = self.sims.TOD_Q - ysim 
        
        H_planck = self.sims.joint.get_operator(x, 
                                           gain=self.sims.g_iter, 
                                           fwhm=self.sims.fwhm_recon, 
                                           nu_co=self.sims.nu_co).operands[1]
        
        tod_pl_s = H_planck(components) 
        _r_pl = self.sims.TOD_E - tod_pl_s
        _r = np.r_[_r, _r_pl]
        #self.chi2 = _dot(_r.T, self.sims.invN.operands[0](_r), self.sims.comm)
        #self.chi2 = _dot(_r.T, self.sims.invN.operands[0](_r), self.sims.comm) + (_r_pl.T @ self.sims.invN.operands[1](_r_pl))
        self.chi2 = _dot(_r.T, self.sims.invN(_r), self.sims.comm)# + (_r_pl.T @ self.sims.invN.operands[1](_r_pl))
        self.sims.comm.Barrier()
        return self.chi2
'''
class Chi2ConstantBlindJC:
    
    def __init__(self, sims):
        
        self.sims = sims
        self.nc = len(self.sims.comps_out)
        self.nf = self.sims.joint_out.qubic.Nsub
        self.nsnd = self.sims.joint_out.qubic.ndets*self.sims.joint_out.qubic.nsamples
        self.nsub = self.sims.joint_out.qubic.Nsub
    def _reshape_A(self, x):
        nf, nc = x.shape
        x_reshape = np.array([])
        for i in range(nc):
            x_reshape = np.append(x_reshape, x[:, i].ravel())
        return x_reshape
    def _reshape_A_transpose(self, x, nf):

        nc = int(x.shape[0] / nf)
        x_reshape = np.ones((nf, nc))
        for i in range(nc):
            x_reshape[:, i] = x[i*nf:(i+1)*nf]
        return x_reshape
    def _qu(self, x, tod_comp, A, icomp):
        
        x = self._reshape_A_transpose(x, self.nsub*2)
        if self.sims.params['MapMaking']['qubic']['type'] == 'two':
            ysim = np.zeros(2*self.nsnd)
            ysim[:self.nsnd] += np.sum(tod_comp[0, :self.nsub], axis=0)
            ysim[self.nsnd:self.nsnd*2] += np.sum(tod_comp[0, self.nsub:self.nsub*2], axis=0)
            #print(x.shape)
            #print(tod_comp.shape)
            for i in range(self.nc-1):
                #print(i, icomp)
                if i+1 == icomp:
                    ysim[:self.nsnd] += x[:self.nsub, 0] @ tod_comp[i+1, :self.nsub]
                    ysim[self.nsnd:self.nsnd*2] += x[self.nsub:self.nsub*2, 0] @ tod_comp[i+1, self.nsub:self.nsub*2]
                else:
                    ysim[:self.nsnd] += A[:self.nsub, i+1] @ tod_comp[i+1, :self.nsub]
                    ysim[self.nsnd:self.nsnd*2] += A[self.nsub:self.nsub*2, i+1] @ tod_comp[i+1, self.nsub:self.nsub*2]
        elif self.sims.params['MapMaking']['qubic']['type'] == 'wide':
            ysim = np.zeros(self.nsnd)
            ysim += np.sum(tod_comp[0, :self.nsub*2], axis=0)
            for i in range(nc-1):
                if i+1 == icomp:
                    ysim[:self.nsnd] += x[:self.nsub*2, 0] @ tod_comp[i+1, :self.nsub*2]
                else:
                    ysim[:self.nsnd] += A[:self.nsub*2, i+1] @ tod_comp[i+1, :self.nsub*2]
        
        _r = ysim - self.sims.TOD_Q
        self.chi2 = _dot(_r.T, self.sims.invN.operands[0](_r), self.sims.comm)
        
        return self.chi2
    



'''
class Chi2VaryingParametric:
    
    def __init__(self, sims):
        
        self.sims = sims
        self.nc = len(sims.comps)
        self.nsnd = self.sims.joint.qubic.ndets*self.sims.joint.qubic.nsamples
        self.nsub = self.sims.joint.qubic.Nsub
        self.mixingmatrix = mm.MixingMatrix(*self.sims.comps)
    
    def _qu(self, x, patch_id, betamap, solution, tod_comp):
        
        
        betamap[patch_id] = x
        
        index = np.arange(len(betamap))

        A = np.zeros((len(betamap), self.nsub*2, len(self.sims.comps)))
        for co in range(len(self.sims.comps)):
            if self.sims.comps_name[co] == 'CMB':
                A[:, :, co] = self.sims.comps[co].eval(self.sims.nus_eff[:self.nsub*2])#[0]
            elif self.sims.comps_name[co] == 'Dust':
                A[:, :, co] = self.sims.comps[co].eval(self.sims.nus_eff[:self.nsub*2], betamap)#[0]
        
        if self.sims.params['MapMaking']['qubic']['type'] == 'two':
            ysim = np.zeros(self.nsnd*2)
            
            for co in range(len(self.sims.comps)):
                for ii, i in enumerate(index):
                    
                    ysim[:self.nsnd] += A[ii, :self.nsub, co] @ tod_comp[i, :self.nsub, co, :self.nsnd]
                    #print(ysim.shape, A.shape, tod_comp.shape)
                    #stop
                    ysim[self.nsnd:2*self.nsnd] += A[ii, self.nsub:2*self.nsub, co] @ tod_comp[i, self.nsub:self.nsub*2, co, :]
        else:
            ysim = np.zeros(self.nsnd)
            for co in range(len(self.sims.comps)):
                for ii, i in enumerate(index):
                    
                    ysim += A[ii, :self.nsub*2, co] @ tod_comp[i, :, co, :]
                
        
        H_planck = self.sims.joint.get_operator(np.array([betamap]).T,  gain=self.sims.g_iter, fwhm=self.sims.fwhm_recon, nu_co=self.sims.nu_co).operands[1]
        tod_pl_s = H_planck(solution)
        
        _r = self.sims.TOD_Q - ysim
        _r_pl = self.sims.TOD_E - tod_pl_s
        #_r = np.r_[_r, _r_pl]
        
        #return _dot(_r.T, self.sims.invN(_r), self.sims.comm)# + (_r_pl.T @ self.sims.invN.operands[1](_r_pl))
        return  _dot(_r.T, self.sims.invN.operands[0](_r), self.sims.comm) + (_r_pl.T @ self.sims.invN.operands[1](_r_pl))
'''
