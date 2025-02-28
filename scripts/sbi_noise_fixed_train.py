import argparse
import os
import time
try:
    import pickle5 as pickle
except:
    import pickle
    
import numpy as np
import torch

from coup_corr_dist import CoupDist
from sbi_util import train_posterior

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser()

parser.add_argument('--obs_file', '-o',  help='file name of the observed data', required=True)
parser.add_argument('--tE', '-tE',  help='excitatory time constant (s)', type=float, default=0.02)
parser.add_argument('--tI', '-tI',  help='inhibitory time constant (s)', type=float, default=0.01)
parser.add_argument('--max_coup', '-maxW',  help='maximum effective coupling magnitude', type=float, default=200)
parser.add_argument('--num_sim', '-n',  help='number of simulations', type=int, default=10000000)

args = vars(parser.parse_args())
print(parser.parse_args())

obs_file = args['obs_file']
tE = args['tE']
tI = args['tI']
maxW = args['max_coup']
num_simulations = args['num_sim']

t = torch.tensor([tE,tI])

with open('../results/'+obs_file+'.pkl', 'rb') as handle:
    obs_dict = pickle.load(handle)
    c = obs_dict['c']
    a = obs_dict['a']

def lfp_sign_func(f, A, p0, q0, q2):
    return A * (p0 + f**2) / (q0 + q2*f**2 + f**4)

def simulator(theta):
    '''
    theta[0] = Weff_EE
    theta[1] = Weff_EI
    theta[2] = Weff_IE
    theta[3] = Weff_II
    
    returns: [fr,wr,Ar]
    fr = peak frequency
    wr = width of peak
    Ar = amplitude of peak (relative to amplitude at 50 Hz)
    '''
    tr = (theta[0]-1)/t[0] + (theta[3]-1)/t[1]
    det = ((theta[0]-1)*(theta[3]-1) - theta[1]*theta[2])/t[0]/t[1]
    
    lam = 0.5*(tr + torch.sqrt(torch.maximum(torch.tensor(0),tr**2-4*det)))
    
    if lam >= 0: # prevent unstable models
        return torch.tensor([torch.nan,torch.nan,torch.nan])
    
    p0 = (a**2*theta[1]**2 - 2*a*c*theta[1]*(theta[3]-1) +\
        (theta[3]-1)**2)/t[1]**2/(2*np.pi)**2
    q0 = (theta[1]*theta[2] - (theta[0]-1)*(theta[3]-1))**2/t[0]**2/t[1]**2/(2*np.pi)**4
    q2 = ((theta[0]-1)*t[1]**2+2*theta[1]*theta[2]*t[0]*t[1]+\
        (theta[3]-1)**2*t[0]**2)/t[0]**2/t[1]**2/(2*np.pi)**2
    
    if q2**2 > 4*q0: # prevent unphysical solutions
        return torch.tensor([torch.nan,torch.nan,torch.nan])
    
    fr = torch.sqrt(torch.sqrt(p0**2-p0*q2+q0) - p0)
    # fr = torch.sqrt(torch.maximum(torch.tensor(0),torch.sqrt(p0**2-p0*q2+q0) - p0))
    Ar = 1/(q2+2*fr**2) / lfp_sign_func(50,1,p0,q0,q2)
    # Ar = lfp_sign_func(fr,1,p0,q0,q2) / lfp_sign_func(50,1,p0,q0,q2)
    wr = torch.sqrt(-(p0+fr**2)*(q2**2-4*q0)**2/8/(q2+2*fr**2)/\
        (4*p0*(2*p0*q2-2*q0+3*q2*fr**2)-(4*q0+q2**2-4*q2*fr**2)*fr**2))
    # wr = torch.sqrt(-Ar/lfp_sign_func_d2(fr,1,p0,q0,q2))
    
    return torch.tensor([fr,wr,Ar])

prior = CoupDist(torch.tensor([0.05],device=device),
                 torch.tensor([maxW],device=device),True)

posterior = train_posterior(prior,simulator,num_simulations,device)

with open('./../results/gamma_posterior_tE={:.3f}_tI={:.3f}_o={:s}_n={:d}_d={:s}.pkl'.format(tE,tI,obs_file,num_simulations,str(device)), 'wb') as handle:
    pickle.dump(posterior,handle)
