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
    
    def __init__(self, sims, d, betamap, seenpix_wrap=None):
        
        self.sims = sims
        self.d = d 
        self.betamap = betamap
        
        if np.ndim(self.d) == 3:
            self.nc, self.nf, self.nsnd = self.d.shape
            self.constant = True
        else:
            
            if self.sims.params['QUBIC']['instrument'] == 'UWB':
                pass
            else:
                self.nf = self.d.shape[1]
                self.d150 = self.d[:, :int(self.nf/2)].copy()
                self.d220 = self.d[:, int(self.nf/2):int(self.nf)].copy()
                _sh = self.d150.shape
                _rsh = ReshapeOperator(self.d150.shape, (_sh[0]*_sh[1], _sh[2], _sh[3]))
                self.d150 = _rsh(self.d150)
                self.d220 = _rsh(self.d220)
                self.dcmb150 = np.sum(self.d150[:, 0, :], axis=0).copy()
                self.dfg150 = self.d150[:, 1, :].copy()
                self.dcmb220 = np.sum(self.d220[:, 0, :], axis=0).copy()
                self.dfg220 = self.d220[:, 1, :].copy()
                self.npixnf, self.nc, self.nsnd = self.d150.shape
                
            index_num = hp.ud_grade(self.sims.seenpix_qubic, self.sims.params['Foregrounds']['nside_fit'])    #
            index = np.where(index_num == True)[0]
            self._index = index
            self.seenpix_wrap = seenpix_wrap
            self.constant = False
    def _get_mixingmatrix(self, x):
        mixingmatrix = mm.MixingMatrix(*self.sims.comps_out)
        if self.constant:
            return mixingmatrix.eval(self.sims.joint_out.qubic.allnus, *x)
        else:
            return mixingmatrix.eval(self.sims.joint_out.qubic.allnus, x)
    def __call__(self, x):
        if self.constant:
            A = self._get_mixingmatrix(x)
            self.betamap = x.copy()

            if self.sims.params['QUBIC']['instrument'] == 'UWB':
                ysim = np.zeros(self.nsnd)
                for ic in range(self.nc):
                    ysim += A[:, ic] @ self.d[ic]
            else:
                ysim = np.zeros(int(self.nsnd*2))
                for ic in range(self.nc):
                    ysim[:int(self.nsnd)] += A[:int(self.nf/2), ic] @ self.d[ic, :int(self.nf/2)]
                    ysim[int(self.nsnd):int(self.nsnd*2)] += A[int(self.nf/2):int(self.nf), ic] @ self.d[ic, int(self.nf/2):int(self.nf)]
        else:
            if self.seenpix_wrap is None:
                self.betamap[self._index, 0] = x.copy()
            else:
                self.betamap[self.seenpix_wrap, 0] = x.copy()
   
            if self.sims.params['QUBIC']['instrument'] == 'UWB':
                ysim = np.zeros(self.nsnd)
                for ic in range(self.nc):
                    for ip, p in enumerate(self._index):
                        ysim += A[ip, :, ic] @ self.d[ip, :, ic]
            else:
                ysim = np.zeros(int(self.nsnd*2))
                Atot = self._get_mixingmatrix(self.betamap[self._index])
                A150 = Atot[:, 0, :int(self.nf/2), 1].ravel()
                A220 = Atot[:, 0, int(self.nf/2):int(self.nf), 1].ravel()
                
                ysim[:int(self.nsnd)] = (A150 @ self.dfg150) + self.dcmb150
                ysim[int(self.nsnd):int(self.nsnd*2)] = (A220 @ self.dfg220) + self.dcmb220
                #stop
                #ysim[:int(self.nsnd)] = A150 @ 
                #for ic in range(self.nc):
                #    for ip, p in enumerate(self._index):
                #        ysim[:int(self.nsnd)] += A[ip, :int(self.nf/2), ic] @ self.d[ip, :int(self.nf/2), ic]
                #        ysim[int(self.nsnd):int(self.nsnd*2)] += A[ip, int(self.nf/2):int(self.nf), ic] @ self.d[ip, int(self.nf/2):int(self.nf), ic]

        _r = ysim - self.sims.TOD_Q
        H_planck = self.sims.joint_out.get_operator(self.betamap, 
                                                    gain=self.sims.g_iter, 
                                                    fwhm=self.sims.fwhm_recon, 
                                                    nu_co=self.sims.nu_co).operands[1]
        tod_pl_s = H_planck(self.sims.components_iter)
        
        _r_pl = self.sims.TOD_E - tod_pl_s

        LLH = _dot(_r.T, self.sims.invN_beta.operands[0](_r), self.sims.comm) + _r_pl.T @ self.sims.invN_beta.operands[1](_r_pl)
        #LLH = _r.T @ self.sims.invN.operands[0](_r)
        
        #return _dot(_r.T, self.sims.invN.operands[0](_r), self.sims.comm) + _r_pl.T @ self.sims.invN.operands[1](_r_pl)
        return LLH
    
class Chi2Parametric_alt:
    
    def __init__(self, sims, d, A_blind, icomp, seenpix_wrap=None):
        
        self.sims = sims
        self.d = d   
        #self.betamap = betamap
        self.A_blind = A_blind
        self.icomp = icomp
        self.nsub = self.sims.joint_out.qubic.Nsub
        self.fsub = int(self.nsub*2/self.sims.params['Foregrounds']['bin_mixing_matrix'])
        self.nc = len(self.sims.comps_out)

        self.constant = True

        # if np.ndim(self.d) == 3:
        #     self.nc, self.nf, self.nsnd = self.d.shape
        #     self.constant = True
        # else:
            
        # if self.sims.params['QUBIC']['instrument'] == 'UWB':
        #     pass
        # else:
        #     self.nf = self.d.shape[1]
        #     self.d150 = self.d[:, :int(self.nf/2)].copy()
        #     self.d220 = self.d[:, int(self.nf/2):int(self.nf)].copy()
        #     _sh = self.d150.shape
        #     _rsh = ReshapeOperator(self.d150.shape, (_sh[0]*_sh[1], _sh[2], _sh[3]))
        #     self.d150 = _rsh(self.d150)
        #     self.d220 = _rsh(self.d220)
        #     self.dcmb150 = np.sum(self.d150[:, 0, :], axis=0).copy()
        #     self.dfg150 = self.d150[:, 1, :].copy()
        #     self.dcmb220 = np.sum(self.d220[:, 0, :], axis=0).copy()
        #     self.dfg220 = self.d220[:, 1, :].copy()
        #     self.npixnf, self.nc, self.nsnd = self.d150.shape
            
        #     index_num = hp.ud_grade(self.sims.seenpix_qubic, self.sims.params['Foregrounds']['nside_fit'])    #
        #     index = np.where(index_num == True)[0]
        #     self._index = index
        #     self.seenpix_wrap = seenpix_wrap
        #     self.constant = False
    def _get_mixingmatrix(self, x):
        mixingmatrix = mm.MixingMatrix(self.sims.comps_out[self.icomp])
        if self.constant:
            return mixingmatrix.eval(self.sims.joint_out.qubic.allnus, *x)
        else:
            return mixingmatrix.eval(self.sims.joint_out.qubic.allnus, x)
    def get_mixingmatrix_comp(self, x):
        A_comp = self._get_mixingmatrix(x)
        A_blind = self.A_blind
        print('test', A_comp.shape, A_blind.shape)
        for ii in range(self.sims.params['Foregrounds']['bin_mixing_matrix']):
            A_blind[ii*self.fsub: (ii + 1)*self.fsub, self.icomp] = A_comp[ii*self.fsub: (ii + 1)*self.fsub]
        return A_blind
    def __call__(self, x):
        
        if self.constant:
            if self.sims.params['QUBIC']['instrument'] == 'DB':
                ### CMB contribution
                tod_cmb_150 = np.sum(self.d[0, :self.nsub, :], axis=0)
                tod_cmb_220 = np.sum(self.d[0, self.nsub:2*self.nsub, :], axis=0)

                ### FG contributions
                tod_comp_150 = self.d[1:, :self.nsub, :].copy()
                tod_comp_220 = self.d[1:, self.nsub:2*self.nsub, :].copy()

                ### Describe the data as d = d_cmb + A . d_fg
                d_150 = tod_cmb_150.copy()
                d_220 = tod_cmb_220.copy()
            A = self.get_mixingmatrix_comp(x)

            for i in range(self.nc-1):
                for j in range(self.nsub):
                    d_150 += A[:self.nsub, (i+1)][j] * tod_comp_150[i, j]
                    d_220 += A[self.nsub:self.nsub*2, (i+1)][j] * tod_comp_220[i, j]
            
            ### Residuals
            _r = np.r_[d_150, d_220] - self.sims.TOD_Q
            
            ### Chi^2
            self.chi2 = _dot(_r.T, self.sims.invN.operands[0](_r), self.sims.comm)

        # else:
        #     if self.seenpix_wrap is None:
        #         self.betamap[self._index, 0] = x.copy()
        #     else:
        #         self.betamap[self.seenpix_wrap, 0] = x.copy()

        #     if self.sims.params['QUBIC']['instrument'] == 'UWB':
        #         ysim = np.zeros(self.nsnd)
        #         for ic in range(self.nc):
        #             for ip, p in enumerate(self._index):
        #                 ysim += A[ip, :, ic] @ self.d[ip, :, ic]
        #     else:
        #         ysim = np.zeros(int(self.nsnd*2))
        #         Atot = self._get_mixingmatrix(self.betamap[self._index])
        #         A150 = Atot[:, 0, :int(self.nf/2), 1].ravel()
        #         A220 = Atot[:, 0, int(self.nf/2):int(self.nf), 1].ravel()
                
        #         ysim[:int(self.nsnd)] = (A150 @ self.dfg150) + self.dcmb150
        #         ysim[int(self.nsnd):int(self.nsnd*2)] = (A220 @ self.dfg220) + self.dcmb220
                
        # _r = ysim - self.sims.TOD_Q
        # H_planck = self.sims.joint_out.get_operator(self.betamap, 
        #                                             gain=self.sims.g_iter, 
        #                                             fwhm=self.sims.fwhm_recon, 
        #                                             nu_co=self.sims.nu_co).operands[1]
        # tod_pl_s = H_planck(self.sims.components_iter)
        
        # _r_pl = self.sims.TOD_E - tod_pl_s
        # LLH = _dot(_r.T, self.sims.invN_beta.operands[0](_r), self.sims.comm) + _r_pl.T @ self.sims.invN_beta.operands[1](_r_pl)

        return self.chi2

class Chi2Blind:
    
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
    def _fill_A(self, x):
        
        fsub = int(self.nsub*2/self.sims.params['Foregrounds']['bin_mixing_matrix'])
        A = np.ones((self.nsub*2, self.nc-1))
        k=0
        for i in range(self.sims.params['Foregrounds']['bin_mixing_matrix']):
            for j in range(self.nc-1):
                A[i*fsub:(i+1)*fsub, j] = np.array([x[k]]*fsub)
                k+=1
        return A.ravel()
    def _reshape_A_transpose(self, x):
        
        fsub = int(self.nsub*2/self.sims.params['Foregrounds']['bin_mixing_matrix'])    
        x_reshape = np.ones(self.nsub*2)

        for i in range(self.sims.params['Foregrounds']['bin_mixing_matrix']):
            x_reshape[i*fsub:(i+1)*fsub] = np.array([x[i]]*fsub)
        return x_reshape
    
    def _qu(self, x, tod_comp):
        ### Fill mixing matrix if fsub different to 1
        x = self._fill_A(x)

        ### CMB contribution
        tod_cmb_150 = np.sum(tod_comp[0, :self.nsub, :], axis=0)
        tod_cmb_220 = np.sum(tod_comp[0, self.nsub:2*self.nsub, :], axis=0)

        if self.sims.params['QUBIC']['instrument'] == 'DB':
            
            ### Mixing matrix element for each nus
            A150 = x[:self.nsub*(self.nc-1)].copy()
            A220 = x[self.nsub*(self.nc-1):self.nsub*2*(self.nc-1)].copy()
            
            ### FG contributions
            tod_comp_150 = tod_comp[1:, :self.nsub, :].copy()
            tod_comp_220 = tod_comp[1:, self.nsub:2*self.nsub, :].copy()

            ### Describe the data as d = d_cmb + A . d_fg
            d_150 = tod_cmb_150.copy()# + A150 @ tod_comp_150
            d_220 = tod_cmb_220.copy()# + A220 @ tod_comp_220
            k=0
   
            ### Recombine data with MM amplitude
            for i in range(self.nsub):
                for j in range(self.nc-1):
                    d_150 += A150[k] * tod_comp_150[j, i]
                    d_220 += A220[k] * tod_comp_220[j, i]
                    k+=1

            ### Residuals
            _r = np.r_[d_150, d_220] - self.sims.TOD_Q
            
            ### Chi^2
            self.chi2 = _dot(_r.T, self.sims.invN.operands[0](_r), self.sims.comm)
        
        return self.chi2
    
    def _qu_alt(self, x, tod_comp, A, icomp):
        
        x = self._reshape_A_transpose(x)
        
        if self.sims.params['QUBIC']['instrument'] == 'DB':
            ### CMB contribution
            tod_cmb_150 = np.sum(tod_comp[0, :self.nsub, :], axis=0)
            tod_cmb_220 = np.sum(tod_comp[0, self.nsub:2*self.nsub, :], axis=0)

            ### FG contributions
            tod_comp_150 = tod_comp[1:, :self.nsub, :].copy()
            tod_comp_220 = tod_comp[1:, self.nsub:2*self.nsub, :].copy()

            ### Describe the data as d = d_cmb + A . d_fg
            d_150 = tod_cmb_150.copy()
            d_220 = tod_cmb_220.copy()

            for i in range(self.nc-1):
                for j in range(self.nsub):
                    if i+1 == icomp:
                        d_150 += x[:self.nsub][j] * tod_comp_150[i, j]
                        d_220 += x[self.nsub:self.nsub*2][j] * tod_comp_220[i, j]
                    else:
                        d_150 += A[:self.nsub, (i+1)][j] * tod_comp_150[i, j]
                        d_220 += A[self.nsub:self.nsub*2, (i+1)][j] * tod_comp_220[i, j]
        
        ### Residuals
        _r = np.r_[d_150, d_220] - self.sims.TOD_Q
        
        ### Chi^2
        self.chi2 = _dot(_r.T, self.sims.invN.operands[0](_r), self.sims.comm)
        
        return self.chi2
    
