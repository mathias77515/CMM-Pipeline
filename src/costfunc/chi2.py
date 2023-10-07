from preset.preset import PresetSims
from pyoperators import *

import fgb.mixing_matrix as mm
import fgb.component_model as c

from acquisition.systematics import *

from simtools.mpi_tools import *
from simtools.noise_timeline import *
from simtools.foldertools import *





class Chi2ConstantBeta:
    
    def __init__(self, sims):
        
        self.sims = sims
    
    def cost_function(self, x, solution):

        """
    
        Method to define chi^2 function for all experience :

            chi^2 = chi^2_QUBIC + chi^2_external

        """
        #print(x)
        chi2_P = self.chi2_external(x, solution)
        if self.sims.params['MapMaking']['qubic']['type'] == 'wide':
            self.wide(x, solution)
        elif self.sims.params['MapMaking']['qubic']['type'] == 'two':
            chi2_Q_150 = self.two150(x, solution)
            chi2_Q_220 = self.two220(x, solution)
         
            chi2_Q = chi2_Q_150 + chi2_Q_220

        return chi2_Q + chi2_P
    def chi2_external(self, x, solution):

        """
    
        Define chi^2 function for external data with shape :

            chi^2 = (TOD_true - sum_nsub(H * A * c))^2 

        """

        tod_s_i = self.sims.TOD_E.copy() * 0

        Hexternal = self.sims.joint.external.get_operator(beta=x, convolution=False)

        tod_s_i = Hexternal(solution[-1])
        
        _r = tod_s_i.ravel() - self.sims.TOD_E.ravel()
        return _r.T @ self.sims.invN_beta.operands[1](_r) 
    def two150(self, x, solution):
        
        tod_s_i = self.sims.TOD_Q_150.ravel() * 0

        G = DiagonalOperator(self.sims.g_iter[:, 0], broadcast='rightward', shapein=(self.sims.joint.qubic.ndets, self.sims.joint.qubic.nsamples))
        k=0
        for ii, i in enumerate(self.sims.array_of_operators150):
        
            A = get_mixing_operator(x, nus=np.array([self.sims.joint.qubic.allnus[k]]), comp=self.sims.comps, nside=self.sims.params['MapMaking']['qubic']['nside'], active=False)
            Hi = G * i.copy()
            Hi.operands[-1] = A
            
            tod_s_i += Hi(solution[ii]).ravel()
            k+=1
            
        _r = ReshapeOperator((self.sims.joint.qubic.ndets, self.sims.joint.qubic.nsamples), (self.sims.joint.qubic.ndets*self.sims.joint.qubic.nsamples))

        invn = CompositionOperator([_r, self.sims.invN.operands[0].operands[1].operands[0], _r.T])
        
        #tod_sim_norm = self.sims.comm.reduce(tod_s_i, op=MPI.SUM, root=0)
        #tod_obs_norm = self.sims.comm.reduce(self.sims.TOD_Q_150, op=MPI.SUM, root=0)
        
        _r = self.sims.TOD_Q_150.ravel() - tod_s_i.ravel()
        
        return _r.T @ invn(_r)
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

        tod_s_i = self.TOD_Q * 0

        G = DiagonalOperator(self.g_iter, broadcast='rightward', shapein=(self.joint.qubic.ndets, self.joint.qubic.nsamples))

        k=0
        for ii, i in enumerate(self.array_of_operators[:self.params['MapMaking']['qubic']['nsub']]):
        
            A = get_mixing_operator(x, nus=np.array([self.joint.qubic.allnus[k]]), comp=self.comps, nside=self.params['MapMaking']['qubic']['nside'], active=False)
            Hi = G * i.copy()
            Hi.operands[-1] = A
            
            tod_s_i += Hi(solution[ii]).ravel()
            k+=1

        if self.nu_co is not None:
            A = get_mixing_operator(x, nus=np.array([self.nu_co]), comp=self.comps, nside=self.params['MapMaking']['qubic']['nside'], active=True)
            Hi = self.array_of_operators[-1].copy()
            Hi.operands[-1] = A

            tod_s_i += Hi(solution[-1])
        

        tod_sim_norm = self.invN.operands[0](tod_s_i.ravel()**2)
        tod_obs_norm = self.invN.operands[0](self.TOD_Q.ravel()**2)
        tod_sim_norm = self.comm.allreduce(tod_sim_norm, op=MPI.SUM)
        tod_obs_norm = self.comm.allreduce(tod_obs_norm, op=MPI.SUM)
        
        diff = tod_obs_norm - tod_sim_norm
        
        self.chi2_Q = np.sum(diff**2)  




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
        chi2_P = self.chi2_external_varying(x, patch_id, allbeta, solution)
        if self.sims.params['MapMaking']['qubic']['type'] == 'wide':
            chi2_Q = self.wide_varying(x, patch_id, allbeta, solution)
            #print(xi2_w, xi2_external)
        elif self.sims.params['MapMaking']['qubic']['type'] == 'two':
            chi2_150 = self.two150_varying(x, patch_id, allbeta, solution)
            chi2_220 = self.two220_varying(x, patch_id, allbeta, solution)
            chi2_Q = chi2_150 + chi2_220
            
            
        
        return chi2_Q + chi2_P
    def two150_varying(self, x, patch_id, allbeta, solution):
        
        allbeta[patch_id, 0] = x
    
        tod_s_i = self.sims.TOD_Q_150.ravel() * 0

        G = DiagonalOperator(self.sims.g_iter[:, 0], broadcast='rightward', shapein=(self.sims.joint.qubic.ndets, self.sims.joint.qubic.nsamples))
        
        k=0
        for ii, i in enumerate(self.sims.array_of_operators150):
            
            A = get_mixing_operator(allbeta, nus=np.array([self.sims.joint.qubic.allnus[k]]), comp=self.sims.comps, nside=self.sims.params['MapMaking']['qubic']['nside'], active=False)
            Hi = G * i.copy()
            Hi.operands[-1] = A
            
            tod_s_i += Hi(solution[ii]).ravel()
            k+=1
        
        _r = ReshapeOperator((self.sims.joint.qubic.ndets, self.sims.joint.qubic.nsamples), (self.sims.joint.qubic.ndets*self.sims.joint.qubic.nsamples))

        invn = CompositionOperator([_r, self.sims.invN.operands[0].operands[1].operands[0], _r.T])
        
        _r = self.sims.TOD_Q_150.ravel() - tod_s_i.ravel()
        
        return _r.T @ invn(_r)
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
        Hexternal = self.sims.joint.external.get_operator(beta=allbeta, convolution=False)
        #if self.sims.size != 1:
        #    Hexternal = self.joint.external.get_operator(beta=allbeta, convolution=False, comm=self.joint.qubic.mpidist, nu_co=None)
        #else:
        #    Hexternal = self.joint.external.get_operator(beta=allbeta, convolution=False, comm=None, nu_co=None)
        
        tod_s_i = Hexternal(solution[-1])
            
        _r = tod_s_i.ravel() - self.sims.TOD_E.ravel()
        chi2_P = _r.T @ self.sims.invN.operands[1](_r)

        return chi2_P
    def wide_varying(self, x, patch_id, allbeta, solution):

        """
    
        Define chi^2 function for Wide Band TOD with shape :

            chi^2 = (TOD_true - sum_nsub(H * A * c))^2 

        """

        allbeta[patch_id, 0] = x
    
        tod_s_i = self.TOD_Q_ALL.copy() * 0
        G = DiagonalOperator(self.g_iter, broadcast='rightward', shapein=(self.joint.qubic.ndets, self.joint.qubic.nsamples))
        
        k=0
        for ii, i in enumerate(self.array_of_operators[:self.params['MapMaking']['qubic']['nsub']]):
        
            A = get_mixing_operator(allbeta, nus=np.array([self.joint.qubic.allnus[k]]), comp=self.comps, nside=self.params['MapMaking']['qubic']['nside'], active=False)
            Hi = G * i.copy()
            Hi.operands[-1] = A

            tod_s_i += Hi(solution[ii]).ravel()
            k+=1
    
        if self.nu_co is not None:
            A = Acq.get_mixing_operator(x, nus=np.array([self.nu_co]), comp=self.comps, nside=self.params['MapMaking']['qubic']['nside'], active=True)
            Hi = array_of_operators[-1].copy()
            Hi.operands[-1] = A

            tod_s_i += Hi(solution[-1]).ravel()


        
        invn = self.invN.operands[0]
        tod_sim_norm = self.comm.allreduce(tod_s_i, op=MPI.SUM)
        tod_obs_norm = self.comm.allreduce(self.TOD_Q, op=MPI.SUM)
        _r = tod_sim_norm.ravel() - tod_obs_norm.ravel()
        chi2_w = _r.T @ invn(_r)
        self.comm.Barrier()
    
        return chi2_w
    
    