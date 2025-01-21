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
from sbi.utils.user_input_checks import (
    process_prior,
    process_simulator,
)

from coup_corr_dist import CoupCorrDist

# if torch.cuda.is_available():
#     device = torch.device('cuda')
# elif torch.backends.mps.is_available():
#     device = torch.device('mps')
# else:
device = torch.device('cpu')

parser = argparse.ArgumentParser()

parser.add_argument('--obs_file', '-o',  help='file name of the observed data', required=True)
parser.add_argument('--tE', '-tE',  help='excitatory time constant (s)', type=float, default=0.02)
parser.add_argument('--tI', '-tI',  help='inhibitory time constant (s)', type=float, default=0.01)
parser.add_argument('--num_sim', '-n',  help='number of simulations', type=int, default=10000000)
parser.add_argument('--num_samp', '-p',  help='number of posterior samples', type=int, default=10000000)

args = vars(parser.parse_args())
print(parser.parse_args())

obs_file = args['obs_file']
tE = args['tE']
tI = args['tI']
num_simulations = args['num_sim']
num_samples = args['num_samp']

test_samples = 100000

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
    tr = (theta[0]-1)/t[0] + (theta[3]-1)/t[1]
    det = ((theta[0]-1)*(theta[3]-1) - theta[1]*theta[2])/t[0]/t[1]
    
    lam = 0.5*(tr + torch.sqrt(torch.maximum(torch.tensor(0),tr**2-4*det)))
    
    if lam >= 0:
        return torch.tensor([torch.nan,torch.nan,torch.nan])
    
    p0 = (theta[5]**2*theta[1]**2 - 2*theta[5]*theta[4]*theta[1]*(theta[3]-1) +\
        (theta[3]-1)**2)/t[1]**2/(2*np.pi)**2
    q0 = (theta[1]*theta[2] - (theta[0]-1)*(theta[3]-1))**2/t[0]**2/t[1]**2/(2*np.pi)**4
    q2 = ((theta[0]-1)*t[1]**2+2*theta[1]*theta[2]*t[0]*t[1]+\
        (theta[3]-1)**2*t[0]**2)/t[0]**2/t[1]**2/(2*np.pi)**2
    
    if q2**2 > 4*q0:
        return torch.tensor([torch.nan,torch.nan,torch.nan])
    
    fr = torch.sqrt(torch.sqrt(p0**2-p0*q2+q0) - p0)
    # fr = torch.sqrt(torch.maximum(torch.tensor(0),torch.sqrt(p0**2-p0*q2+q0) - p0))
    Ar = 1/(q2+2*fr**2) / lfp_sign_func(50,1,p0,q0,q2)
    # Ar = lfp_sign_func(fr,1,p0,q0,q2) / lfp_sign_func(50,1,p0,q0,q2)
    wr = torch.sqrt(-(p0+fr**2)*(q2**2-4*q0)**2/8/(q2+2*fr**2)/\
        (4*p0*(2*p0*q2-2*q0+3*q2*fr**2)-(4*q0+q2**2-4*q2*fr**2)*fr**2))
    # wr = torch.sqrt(-Ar/lfp_sign_func_d2(fr,1,p0,q0,q2))
    
    return torch.tensor([fr,wr,Ar])

def max_re_eigval(theta):
    '''
    theta[0] = Weff_EE
    theta[1] = Weff_EI
    theta[2] = Weff_IE
    theta[3] = Weff_II
    theta[4] = r(eta_E,eta_I)
    theta[5] = |eta_I|/|eta_E|
    
    returns: lam
    lam = max(Re(eig(Weff)))
    '''
    tr = (theta[0]-1)/t[0] + (theta[3]-1)/t[1]
    det = ((theta[0]-1)*(theta[3]-1) - theta[1]*theta[2])/t[0]/t[1]
    
    lam = 0.5*(tr + torch.sqrt(torch.maximum(torch.tensor(0),tr**2-4*det)))
    
    return lam

prior = CoupCorrDist(torch.tensor([0.05],device=device),
                     torch.tensor([200],device=device),
                     torch.tensor([0.0,0.0],device=device),
                     torch.tensor([1.0,2.0],device=device),True)

# Check prior, return PyTorch prior.
prior, num_parameters, prior_returns_numpy = process_prior(prior)

# Check simulator, returns PyTorch simulator able to simulate batches.
simulator = process_simulator(simulator, prior, prior_returns_numpy)
max_re_eigval = process_simulator(max_re_eigval, prior, prior_returns_numpy)

with open('../results/'+obs_file+'.pkl', 'rb') as handle:
    obs_dict = pickle.load(handle)
    frn = obs_dict['frn']
    frs = obs_dict['frs']
    wrn = obs_dict['wrn']
    wrs = obs_dict['wrs']
    Arn = obs_dict['Arn']
    Ars = obs_dict['Ars']

with open('./../results/gamma_posterior_tE={:.3f}_tI={:.3f}_n={:d}_d={:s}.pkl'.format(tE,tI,num_simulations,str(device)), 'rb') as handle:
    posterior = pickle.load(handle)

x_obs = torch.tensor([frn,wrn,Arn])
x_err = torch.tensor([frs,wrs,Ars])

start = time.process_time()

samples = posterior.sample((test_samples,), x=x_obs)
fit_idx = torch.sqrt((((simulator(samples)-x_obs[None,:])/x_err[None,:])**2).sum(-1)) < 1
print(fit_idx.sum().item() / test_samples)
print((max_re_eigval(samples) < 0).sum().item() / test_samples)
fit_idx = torch.logical_and(fit_idx,max_re_eigval(samples) < 0)
fit_frac = fit_idx.sum().item() / test_samples
required_samples = int(num_samples/fit_frac*1.05)
print('required samples:', required_samples)

print('Test sampling took',time.process_time()-start,'s')

start = time.process_time()

samples = posterior.sample((required_samples,), x=x_obs)
fit_idx = torch.sqrt((((simulator(samples)-x_obs[None,:])/x_err[None,:])**2).sum(-1)) < 1
fit_idx = torch.logical_and(fit_idx,max_re_eigval(samples) < 0)
samples = samples[fit_idx,:]
print(samples.shape)

print('Sampling the required number for desired sample size took',time.process_time()-start,'s')

start = time.process_time()

print('Inference training took',time.process_time()-start,'s')

with open('./../results/gamma_sample_tE={:.3f}_tI={:.3f}_o={:s}_n={:d}_d={:s}.pkl'.format(tE,tI,obs_file,num_samples,str(device)), 'wb') as handle:
    pickle.dump(samples,handle)
