import numpy as np
import pysm3
import pysm3.units as u
import pysm3.utils as utils
from pyoperators import pcg, ReshapeOperator, DenseOperator, BlockColumnOperator, AdditionOperator, CompositionOperator, rule_manager
import qubic
from  acquisition.systematics import *
from solver.cg import mypcg

NSIDE = 64
NSUB = 4
NREC = 2
NPOINTINGS=50
### MPI common arguments
comm = MPI.COMM_WORLD

sky_cmb = pysm3.Sky(NSIDE, preset_strings=['c1'])
sky_dust = pysm3.Sky(NSIDE, preset_strings=['d0'])
    
def _get_map_1nu(sky, nu):
    return np.array(sky.get_emission(nu * u.GHz, None).T * utils.bandpass_unit_conversion(nu * u.GHz, None, u.uK_CMB))
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
                'comm':comm, 
                'kind':'IQU',
                'config':'FI',
                'verbose':False,
                'dtheta':15,
                'nprocs_sampling':1, 
                'nprocs_instrument':comm.Get_size(),
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
                'synthbeam_fraction':1}

    ### Get the default dictionary
    dictfilename = 'dicts/pipeline_demo.dict'
    d = qubic.qubicdict.qubicDict()
    d.read_from_file(dictfilename)
    for i in args.keys():
        
        d[str(i)] = args[i]

    
    return d
def _get_matrix_Acomp(nsub, nc):
    nc_plus_one = nc + 1
    A = np.zeros((nsub, nc_plus_one))
    A[:, 0] = 1
    fsub = int(nsub/nc)
    for i in range(nc):
        A[i*fsub:(i+1)*fsub, i+1] = i+1
    return A

_, _, nus150, _, _, _ = qubic.compute_freq(150, Nfreq=int(NSUB/2))
_, _, nus220, _, _, _ = qubic.compute_freq(220, Nfreq=int(NSUB/2))
nus = np.array(list(nus150) + list(nus220))

maps = np.zeros((NREC+1, 12*NSIDE**2, 3))
maps[0] = _get_map_1nu(sky_cmb, 150)
for i in range(NREC):
    maps[i+1] = _get_map_1nu(sky_dust, nus[i])
    

d = _get_dict()

acq = QubicFullBandSystematic(d, NSUB, Nrec=1, comp=[], kind='two', nu_co=None, H=None, effective_duration150=3, effective_duration220=3)
seenpix = acq.coverage/acq.coverage.max() > 0.2
A = _get_matrix_Acomp(NSUB, NREC)
h = []
for i in range(len(acq.H)):
    r = ReshapeOperator(((1, 12*NSIDE**2, 3)), ((12*NSIDE**2, 3)))
    Acomp = r * DenseOperator(A[i], broadcast='rightward', shapein=(A.shape[1], 12*NSIDE**2, 3), shapeout=(1, 12*NSIDE**2, 3))
    with rule_manager(inplace=True):
        h += [CompositionOperator([acq.H[i], Acomp])]
    
H = BlockColumnOperator([AdditionOperator(h[:int(NSUB/2)]), AdditionOperator(h[int(NSUB/2):])], axisout=0)    
