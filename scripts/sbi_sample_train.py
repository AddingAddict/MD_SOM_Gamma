import argparse
import os
import time
try:
    import pickle5 as pickle
except:
    import pickle
    
import numpy as np
import torch

from coup_corr_dist import CoupCorrDist
from sbi_util import train_posterior

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser()

parser.add_argument('--tE', '-tE',  help='excitatory time constant (s)', type=float, default=0.02)
parser.add_argument('--tI', '-tI',  help='inhibitory time constant (s)', type=float, default=0.01)
parser.add_argument('--max_coup', '-maxW',  help='maximum effective coupling magnitude', type=float, default=200)
parser.add_argument('--max_corr', '-maxc',  help='maximum correlation coefficient for E/I noise', type=float, default=1)
parser.add_argument('--max_Iamp', '-maxa',  help='maximum ratio of I to E noise amplitude', type=float, default=2)
parser.add_argument('--num_sim', '-n',  help='number of simulations', type=int, default=10000000)

args = vars(parser.parse_args())
print(parser.parse_args())

tE = args['tE']
tI = args['tI']
maxW = args['max_coup']
maxc = args['max_corr']
maxa = args['max_Iamp']
num_simulations = args['num_sim']

t = torch.tensor([tE,tI])

def lfp_sign_func(f, A, p0, q0, q2):
    return A * (p0 + f**2) / (q0 + q2*f**2 + f**4)

def simulator(theta):
    '''
    theta[0] = (τ_E*|Weff_EE|+τ_I*|Weff_II|)/√(τ_E^2+τ_I^2)
    theta[1] = |Weff_EI|
    theta[2] = |Weff_IE|
    theta[3] = (τ_I*|Weff_EE|-τ_E*|Weff_II|)/√(τ_E^2+τ_I^2)
    theta[4] = r(eta_E,eta_I)
    theta[5] = |eta_I|/|eta_E|
    
    returns: [fr,wr,Ar]
    fr = peak frequency
    wr = width of peak
    Ar = amplitude of peak (relative to amplitude at 50 Hz)
    '''
    Weff_EE =  torch.maximum(torch.tensor([0]),theta[0]*t[0] + theta[3]*t[1])/torch.sqrt(t[0]**2+t[1]**2)
    Weff_EI = -theta[1]
    Weff_IE =  theta[2]
    Weff_II = -torch.maximum(torch.tensor([0]),theta[0]*t[1] - theta[3]*t[0])/torch.sqrt(t[0]**2+t[1]**2)
    c = theta[4]
    a = theta[5]
    
    tr = (Weff_EE-1)/t[0] + (Weff_II-1)/t[1]
    det = ((Weff_EE-1)*(Weff_II-1) - Weff_EI*Weff_IE)/t[0]/t[1]
    
    lam = 0.5*(tr + torch.sqrt(torch.maximum(torch.tensor(0),tr**2-4*det)))
    
    if lam >= 0: # prevent unstable models
        return torch.tensor([torch.nan,torch.nan,torch.nan])
    
    p0 = (a**2*Weff_EI**2 - 2*a*c*Weff_EI*(Weff_II-1) +\
        (Weff_II-1)**2)/t[1]**2/(2*np.pi)**2
    q0 = (Weff_EI*Weff_IE - (Weff_EE-1)*(Weff_II-1))**2/t[0]**2/t[1]**2/(2*np.pi)**4
    q2 = ((Weff_EE-1)**2*t[1]**2+2*Weff_EI*Weff_IE*t[0]*t[1]+\
        (Weff_II-1)**2*t[0]**2)/t[0]**2/t[1]**2/(2*np.pi)**2
    
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

prior = CoupCorrDist(torch.tensor([0.01],device=device),
                     torch.tensor([maxW],device=device),
                     torch.tensor([0.0,0.0,0.0],device=device),
                     torch.tensor([1.5,maxc,maxa],device=device),
                     num_coups=3,validate_args=True)

posterior = train_posterior(prior,simulator,num_simulations,device)

with open('./../results/gamma_posterior_tE={:.3f}_tI={:.3f}_n={:d}_d={:s}.pkl'.format(tE,tI,num_simulations,str(device)), 'wb') as handle:
    pickle.dump(posterior,handle)
