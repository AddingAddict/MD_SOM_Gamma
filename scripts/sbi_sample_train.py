import argparse
import os
import time
try:
    import pickle5 as pickle
except:
    import pickle
    
import numpy as np
import torch
from sbi.analysis import pairplot
from sbi.inference import NPE
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

from coup_corr_dist import CoupCorrDist

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
parser.add_argument('--num_samp', '-n',  help='number of parameter samples', type=int, default=10000000)

args = vars(parser.parse_args())
print(parser.parse_args())

tE = args['tE']
tI = args['tI']
maxW = args['max_coup']
maxc = args['max_corr']
maxa = args['max_Iamp']
num_simulations = args['num_samp']

t = torch.tensor([tE,tI])

def lfp_sign_func(f, A, p0, q0, q2):
    return A * (p0 + f**2) / (q0 + q2*f**2 + f**4)

def simulator(theta):
    '''
    theta[0] = Weff_EE
    theta[1] = Weff_EI
    theta[2] = Weff_IE
    theta[3] = Weff_II
    theta[4] = r(eta_E,eta_I)
    theta[5] = |eta_I|/|eta_E|
    
    returns: [fr,wr,Ar]
    fr = peak frequency
    wr = width of peak
    Ar = amplitude of peak (relative to amplitude at 50 Hz)
    '''
    p0 = (theta[5]**2*theta[1]**2 - 2*theta[5]*theta[4]*theta[1]*(theta[3]-1) +\
        (theta[3]-1)**2)/t[1]**2/(2*np.pi)**2
    q0 = (theta[1]*theta[2] - (theta[0]-1)*(theta[3]-1))**2/t[0]**2/t[1]**2/(2*np.pi)**4
    q2 = ((theta[0]-1)*t[1]**2+2*theta[1]*theta[2]*t[0]*t[1]+(theta[3]-1)*t[0]**2)/t[0]**2/t[1]**2/(2*np.pi)**2
    
    fr = torch.sqrt(torch.sqrt(p0**2-p0*q2+q0) - p0)
    # fr = torch.sqrt(torch.maximum(torch.tensor(0),torch.sqrt(p0**2-p0*q2+q0) - p0))
    Ar = 1/(q2+2*fr**2) / lfp_sign_func(50,1,p0,q0,q2)
    # Ar = lfp_sign_func(fr,1,p0,q0,q2) / lfp_sign_func(50,1,p0,q0,q2)
    wr = torch.sqrt(-(p0+fr**2)*(q2**2-4*q0)**2/8/(q2+2*fr**2)/\
        (4*p0*(2*p0*q2-2*q0+3*q2*fr**2)-(4*q0+q2**2-4*q2*fr**2)*fr**2))
    # wr = torch.sqrt(-Ar/lfp_sign_func_d2(fr,1,p0,q0,q2))
    
    return torch.tensor([fr,wr,Ar])

prior = CoupCorrDist(torch.tensor([0.05],device=device),
                     torch.tensor([maxW],device=device),
                     torch.tensor([0.0,0.0],device=device),
                     torch.tensor([maxc,maxa],device=device),True)

# Check prior, return PyTorch prior.
prior, num_parameters, prior_returns_numpy = process_prior(prior)

# Check simulator, returns PyTorch simulator able to simulate batches.
simulator = process_simulator(simulator, prior, prior_returns_numpy)

# Consistency check after making ready for sbi.
check_sbi_inputs(simulator, prior)

inference = NPE(prior=prior,device=device)

start = time.process_time()

theta = prior.sample((num_simulations,)).to(device)
x = simulator(theta).to(device)

print('Sampling and simulating took',time.process_time()-start,'s')

start = time.process_time()

inference = inference.append_simulations(theta, x)
density_estimator = inference.train()

print()
print('Inference training took',time.process_time()-start,'s')

posterior = inference.build_posterior(density_estimator)

with open('./../results/gamma_posterior_tE={:.3f}_tI={:.3f}_n={:d}_d={:s}.pkl'.format(tE,tI,num_simulations,str(device)), 'wb') as handle:
    pickle.dump(posterior,handle)
