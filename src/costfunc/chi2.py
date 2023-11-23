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
        
        _r = self.sims.TOD_Q - ysim 
        
        H_planck = self.sims.joint.get_operator(x, 
                                           gain=self.sims.g_iter, 
                                           fwhm=self.sims.fwhm_recon, 
                                           nu_co=self.sims.nu_co).operands[1]
        
        tod_pl_s = H_planck(components) 
        _r_pl = self.sims.TOD_E - tod_pl_s
        _r = np.r_[_r, _r_pl]
        
        #self.chi2 = _dot(_r.T, self.sims.invN.operands[0](_r), self.sims.comm) + (_r_pl.T @ self.sims.invN.operands[1](_r_pl))
        self.chi2 = _dot(_r.T, self.sims.invN(_r), self.sims.comm)# + (_r_pl.T @ self.sims.invN.operands[1](_r_pl))
        self.sims.comm.Barrier()
        return self.chi2

class Chi2ConstantBlindJC:
    
    def __init__(self, sims, d, invn):
        
        self.sims = sims
        self.d = d
        self.invn = invn
        self.nc = len(self.sims.comps)
        self.nf = self.sims.joint.qubic.Nsub
        self.nsnd = self.sims.joint.qubic.ndets*self.sims.joint.qubic.nsamples
        self.nsub = self.sims.joint.qubic.Nsub
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
        
        nc = tod_comp.shape[0]
        x = self._reshape_A_transpose(x, self.nsub*2)
        if self.sims.params['MapMaking']['qubic']['type'] == 'two':
            ysim = np.zeros(2*self.nsnd)
            ysim[:self.nsnd] += np.sum(tod_comp[0, :self.nsub], axis=0)
            ysim[self.nsnd:self.nsnd*2] += np.sum(tod_comp[0, self.nsub:self.nsub*2], axis=0)
            #print(x.shape)
            #print(tod_comp.shape)
            for i in range(nc-1):
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
                
            
        _r = ysim - self.d
        self.chi2 = _dot(_r, self.invn(_r), self.sims.comm)
        
        return self.chi2
    

class Chi2ConstantBeta:
    
    def __init__(self, sims):
        
        self.sims = sims
    
    def cost_function(self, x, solution):

        """
    
        Method to define chi^2 function for all experience :

            chi^2 = chi^2_QUBIC + chi^2_external

        """
        
        if self.sims.rank == 0:
            pass
        else:
            x = None
            
        x = self.sims.comm.bcast(x, root=0)
        
        #self.chi2_P = self.chi2_external(x, solution)
        if self.sims.params['MapMaking']['qubic']['type'] == 'wide':
            self.chi2 = self.wide(x, solution)# + self.chi2_P
        elif self.sims.params['MapMaking']['qubic']['type'] == 'two':
            self.chi2 = self.two(x, solution)# + self.chi2_P

        return self.chi2# / 1e10
    def chi2_external(self, x, solution):

        """
    
        Define chi^2 function for external data with shape :

            chi^2 = (TOD_true - sum_nsub(H * A * c))^2 

        """

        Hexternal = self.sims.joint.get_operator(x, 
                                                 gain=self.sims.g_iter, 
                                                 fwhm=self.sims.fwhm_recon, 
                                                 nu_co=self.sims.nu_co).operands[1]

        tod_s_i = Hexternal(solution)
        
        _r = tod_s_i.ravel() - self.sims.TOD_E.ravel()
        
        return _dot(_r, self.sims.invN_beta.operands[1](_r), None)
    def two(self, x, solution):

        H_i = self.sims.joint.get_operator(x, 
                                           gain=self.sims.g_iter, 
                                           fwhm=self.sims.fwhm_recon, 
                                           nu_co=self.sims.nu_co)
        tod_sims = H_i(solution)

        _r = self.sims.TOD_obs.ravel() - tod_sims.ravel()
        
        return _dot(_r, self.sims.invN(_r), self.sims.comm)
    def two_grad(self, x, solution):
        
        H_i = self.sims.joint.get_operator(x, 
                                           gain=self.sims.g_iter, 
                                           fwhm=self.sims.fwhm_recon, 
                                           nu_co=self.sims.nu_co)
        tod_sims = H_i(solution)

        _r = self.sims.TOD_obs.ravel() - tod_sims.ravel()
        
        return _dot(tod_sims.ravel(), self.sims.invN(_r), self.sims.comm)
    def two220(self, x, solution):
        
        tod_s_i = self.sims.TOD_Q_220.ravel() * 0

        G = DiagonalOperator(self.sims.g_iter[:, 1], broadcast='rightward', shapein=(self.sims.joint.qubic.ndets, self.sims.joint.qubic.nsamples))
        k=0
        for ii, i in enumerate(self.sims.array_of_operators220[:self.sims.params['MapMaking']['qubic']['nsub']]):
            
            mynus = np.array([self.sims.joint.qubic.allnus[k+int(self.sims.params['MapMaking']['qubic']['nsub']/2)]])
            A = get_mixing_operator(x, nus=mynus, comp=self.sims.comps, nside=self.sims.params['MapMaking']['qubic']['nside'], active=False)
            Hi = G * i.copy()
            Hi.operands[-1] = A
            
            tod_s_i += Hi(solution[ii+int(self.sims.params['MapMaking']['qubic']['nsub']/2)]).ravel()
            k+=1

        if self.sims.nu_co is not None:
            A = get_mixing_operator(x, nus=np.array([self.sims.nu_co]), comp=self.comps, nside=self.sims.params['MapMaking']['qubic']['nside'], active=True)
            Hi = self.sims.array_of_operators[-1].copy()
            Hi.operands[-1] = A

            tod_s_i += Hi(solution[-1])


        _r = ReshapeOperator((self.sims.joint.qubic.ndets, self.sims.joint.qubic.nsamples), (self.sims.joint.qubic.ndets*self.sims.joint.qubic.nsamples))

        invn = CompositionOperator([_r, self.sims.invN.operands[0].operands[1].operands[1], _r.T])
        
        #self.sims.comm.Barrier()
        #tod_sim_norm = self.sims.comm.allreduce(tod_s_i, op=MPI.SUM)
        #tod_obs_norm = self.sims.comm.allreduce(self.sims.TOD_Q_220, op=MPI.SUM)
        _r = self.sims.TOD_Q_220.ravel() - tod_s_i.ravel()
           
        return _r.T @ invn(_r)
    def wide(self, x, solution):

        """
    
        Define chi^2 function for Wide Band TOD with shape :

            chi^2 = (TOD_true - sum_nsub(H * A * c))^2 

        """

        H_i = self.sims.joint.get_operator(x, 
                                           gain=self.sims.g_iter, 
                                           fwhm=self.sims.fwhm_recon, 
                                           nu_co=self.sims.nu_co)
        tod_sims = H_i(solution)

        _r = self.sims.TOD_obs.ravel() - tod_sims.ravel()

        return _dot(_r, self.sims.invN(_r), self.sims.comm)


'''
class Chi2VaryingBeta:

    """
    
    Instance that define Chi^2 function for many configurations. The instance initialize first the PresetSims instance and knows every parameters.

    Arguments : 
    ===========
        - comm    : MPI common communicator (define by MPI.COMM_WORLD).
        - seed    : Int number for CMB realizations.
        - it      : Int number for noise realizations.

    """
    
    def __init__(self, sims):
        
        self.sims = sims
        
    def cost_function(self, x, patch_id, allbeta, solution):

        """
    
        Method to define chi^2 function for all experience :

            chi^2 = chi^2_QUBIC + chi^2_external

        """
        #self.chi2_P = self.chi2_external_varying(x, patch_id, allbeta, solution)
        if self.sims.params['MapMaking']['qubic']['type'] == 'wide':
            self.chi2 = self.wide_varying(x, patch_id, allbeta, solution)# + self.chi2_P
            #print(xi2_w, xi2_external)
        elif self.sims.params['MapMaking']['qubic']['type'] == 'two':
            self .chi2 = self.two_varying(x, patch_id, allbeta, solution)# + self.chi2_P
            #self.chi2 = self.chi2_P
        #print(f'{self.two_varying(x, patch_id, allbeta, solution):.3e}  {self.chi2_P:.3e}')
        self.sims.comm.Barrier()
        return self.chi2 # + self.chi2_P
    def two_varying(self, x, patch_id, allbeta, solution):
        
        allbeta[patch_id, 0] = x.copy()
        
        H = self.sims.joint.get_operator(allbeta, 
                                         gain=self.sims.g_iter, 
                                         fwhm=self.sims.fwhm_recon)
        
        d_sims = H(solution).ravel()
        _r = d_sims - self.sims.TOD_obs.ravel()
        
        return _dot(_r.T, self.sims.invN_beta(_r), self.sims.comm)
        #return _rP @ self.sims.invN_beta.operands[1](_rP)
    def two220_varying(self, x, patch_id, allbeta, solution):
        
        allbeta[patch_id, 0] = x
        
        tod_s_i = self.sims.TOD_Q_220.ravel() * 0
        G = DiagonalOperator(self.sims.g_iter[:, 1], broadcast='rightward', shapein=(self.sims.joint.qubic.ndets, self.sims.joint.qubic.nsamples))
        
        k=0
        for ii, i in enumerate(self.sims.array_of_operators220[:self.sims.params['MapMaking']['qubic']['nsub']]):

            mynus = np.array([self.sims.joint.qubic.allnus[k+int(self.sims.params['MapMaking']['qubic']['nsub']/2)]])

            A = get_mixing_operator(allbeta, nus=mynus, comp=self.sims.comps, nside=self.sims.params['MapMaking']['qubic']['nside'], active=False)
            Hi = G * i.copy()
            Hi.operands[-1] = A
            
            tod_s_i += Hi(solution[ii+int(self.sims.params['MapMaking']['qubic']['nsub']/2)]).ravel()
            k+=1

        if self.sims.nu_co is not None:
            A = get_mixing_operator(x, nus=np.array([self.sims.nu_co]), comp=self.sims.comps, nside=self.sims.params['MapMaking']['qubic']['nside'], active=True)
            Hi = self.sims.array_of_operators[-1].copy()
            Hi.operands[-1] = A

            tod_s_i += Hi(solution[-1])
        
        _r = ReshapeOperator((self.sims.joint.qubic.ndets, self.sims.joint.qubic.nsamples), (self.sims.joint.qubic.ndets*self.sims.joint.qubic.nsamples))

        invn = CompositionOperator([_r, self.sims.invN.operands[0].operands[1].operands[1], _r.T])
        

        _r = self.sims.TOD_Q_220.ravel() - tod_s_i.ravel()
        
        return _r.T @ invn(_r)   
    def chi2_external_varying(self, x, patch_id, allbeta, solution):

        """
    
        Define chi^2 function for external data with shape :

            chi^2 = (TOD_true - sum_nsub(H * A * c))^2 

        """
        allbeta[patch_id, 0] = x
        Hexternal = self.sims.joint.get_operator(allbeta, 
                                                 gain=self.sims.g_iter, 
                                                 fwhm=self.sims.fwhm_recon, 
                                                 nu_co=self.sims.nu_co).operands[1]
        
        _r = Hexternal(solution).ravel() - self.sims.TOD_E.ravel()
        chi2_P = _r.T @ (self.sims.invN.operands[1](_r))

        return chi2_P
    def wide_varying(self, x, patch_id, allbeta, solution):

        """
    
        Define chi^2 function for Wide Band TOD with shape :

            chi^2 = (TOD_true - sum_nsub(H * A * c))^2 

        """

        allbeta[patch_id, 0] = x.copy()
        
        H = self.sims.joint.get_operator(allbeta, 
                                         gain=self.sims.g_iter, 
                                         fwhm=self.sims.fwhm_recon)
        
        d_sims = H(solution).ravel()
        _r = d_sims - self.sims.TOD_obs.ravel()
        
        return _dot(_r.T, self.sims.invN_beta(_r), self.sims.comm)
        #H_Q = H.operands[0]
        #H_P = H.operands[1]
        #d_sims_Q = H_Q(solution).ravel()
        #d_sims_P = H_P(solution).ravel()
    
        #_rQ = d_sims_Q - self.sims.TOD_Q.ravel()
        #_rP = d_sims_P - self.sims.TOD_E.ravel()
        #return _dot(_rQ.T, self.sims.invN_beta.operands[0](_rQ), self.sims.comm) + _rP.T @ self.sims.invN_beta.operands[1](_rP)
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