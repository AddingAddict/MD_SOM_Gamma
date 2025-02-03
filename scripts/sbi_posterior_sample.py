import argparse
import os
import time
try:
    import pickle5 as pickle
except:
    import pickle

import numpy as np
import torch
from sbi.utils.user_input_checks import (
    process_prior,
    process_simulator,
)

from coup_corr_dist import CoupCorrDist
from sbi_util import sample_posterior

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
parser.add_argument('--max_coup', '-maxW',  help='maximum effective coupling magnitude', type=float, default=100)
parser.add_argument('--max_corr', '-maxc',  help='maximum correlation coefficient for E/I noise', type=float, default=1)
parser.add_argument('--max_Iamp', '-maxa',  help='maximum ratio of I to E noise amplitude', type=float, default=2)
parser.add_argument('--num_sim', '-n',  help='number of simulations', type=int, default=10000000)
parser.add_argument('--num_samp', '-p',  help='number of posterior samples', type=int, default=10000000)

args = vars(parser.parse_args())
print(parser.parse_args())

obs_file = args['obs_file']
tE = args['tE']
tI = args['tI']
maxW = args['max_coup']
maxc = args['max_corr']
maxa = args['max_Iamp']
num_simulations = args['num_sim']
num_samples = args['num_samp']

test_samples = 1000000

t = torch.tensor([tE,tI])
eps = 0.01
tol = 0.05

def lfp_sign_func(f, A, p0, q0, q2):
    return A * (p0 + f**2) / (q0 + q2*f**2 + f**4)

def simulator(theta):
    '''
    theta[:,0] = (τ_E*|Weff_EE|+τ_I*|Weff_II|)/√(τ_E^2+τ_I^2)
    theta[:,1] = |Weff_EI|
    theta[:,2] = |Weff_IE|
    theta[:,3] = (τ_I*|Weff_EE|-τ_E*|Weff_II|)/√(τ_E^2+τ_I^2)
    theta[:,4] = r(eta_E,eta_I)
    theta[:,5] = |eta_I|/|eta_E|
    
    returns: [fr,wr,Ar]
    fr = peak frequency
    wr = width of peak
    Ar = amplitude of peak (relative to amplitude at 50 Hz)
    '''
    perts = torch.cat([ax_mesh.flatten()[None,:] for ax_mesh in torch.meshgrid((torch.linspace(-1,1,3),)*4)],dim=0)
    perts *= eps/torch.sum(perts**2,dim=0)**0.5
    perts[:,81//2] = 0
    perts += 1

    Weff_EE =  torch.maximum(torch.tensor([0]),theta[:,0]*t[0] + theta[:,3]*t[1])/torch.sqrt(t[0]**2+t[1]**2)
    Weff_EI = -theta[:,1]
    Weff_IE =  theta[:,2]
    Weff_II = -torch.maximum(torch.tensor([0]),theta[:,0]*t[1] - theta[:,3]*t[0])/torch.sqrt(t[0]**2+t[1]**2)
    c = theta[:,4][:,None]
    a = theta[:,5][:,None]
    
    Weff_EE = Weff_EE[:,None] * perts[0][None,:]
    Weff_EI = Weff_EI[:,None] * perts[1][None,:]
    Weff_IE = Weff_IE[:,None] * perts[2][None,:]
    Weff_II = Weff_II[:,None] * perts[3][None,:]
    
    out = torch.zeros((theta.shape[0],3,81),dtype=theta.dtype).to(theta.device)
    
    tr = (Weff_EE-1)/t[0] + (Weff_II-1)/t[1]
    det = ((Weff_EE-1)*(Weff_II-1) - Weff_EI*Weff_IE)/t[0]/t[1]
    
    lam = 0.5*(tr + torch.sqrt(torch.maximum(torch.tensor(0),tr**2-4*det)))
    
    p0 = (a**2*Weff_EI**2 - 2*a*c*Weff_EI*(Weff_II-1) +\
        (Weff_II-1)**2)/t[1]**2/(2*np.pi)**2
    q0 = (Weff_EI*Weff_IE - (Weff_EE-1)*(Weff_II-1))**2/t[0]**2/t[1]**2/(2*np.pi)**4
    q2 = ((Weff_EE-1)**2*t[1]**2+2*Weff_EI*Weff_IE*t[0]*t[1]+\
        (Weff_II-1)**2*t[0]**2)/t[0]**2/t[1]**2/(2*np.pi)**2
    
    valid_idx = torch.logical_and(lam < 0,q2**2 < 4*q0)
    
    #### fr
    out[:,0,:] = torch.sqrt(torch.sqrt(p0**2-p0*q2+q0) - p0)
    # fr = torch.sqrt(torch.maximum(torch.tensor(0),torch.sqrt(p0**2-p0*q2+q0) - p0))
    
    #### Ar
    out[:,2,:] = 1/(q2+2*out[:,0,:]**2) / lfp_sign_func(50,1,p0,q0,q2)
    # Ar = lfp_sign_func(out[:,0,:],1,p0,q0,q2) / lfp_sign_func(50,1,p0,q0,q2)
    
    #### wr
    out[:,1,:] = torch.sqrt(-(p0+out[:,0,:]**2)*(q2**2-4*q0)**2/8/(q2+2*out[:,0,:]**2)/\
        (4*p0*(2*p0*q2-2*q0+3*q2*out[:,0,:]**2)-(4*q0+q2**2-4*q2*out[:,0,:]**2)*out[:,0,:]**2))
    # wr = torch.sqrt(-Ar/lfp_sign_func_d2(fr,1,p0,q0,q2))
    
    valid_idx = torch.logical_and(lam < 0,q2**2 < 4*q0)
    
    torch.where(valid_idx[:,None,:],out,torch.tensor([torch.nan])[:,None,None],out=out)
    
    valid_idx = torch.sqrt(torch.sum((out/out[:,:,81//2][:,:,None] - 1)**2,dim=(1,2))/80) < tol
    
    return torch.where(valid_idx[:,None],out[:,:,81//2],torch.tensor([torch.nan])[:,None])

prior = CoupCorrDist(torch.tensor([0.02],device=device),
                     torch.tensor([maxW],device=device),
                     torch.tensor([0.0,0.0,0.0],device=device),
                     torch.tensor([1.5,maxc,maxa],device=device),
                     num_coups=3,validate_args=True)

# Check prior, return PyTorch prior.
prior, num_parameters, prior_returns_numpy = process_prior(prior)

# Check simulator, returns PyTorch simulator able to simulate batches.
simulator = process_simulator(simulator, prior, prior_returns_numpy)

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

def criterion_fn(samples):
    return torch.logical_and(torch.sqrt((((simulator(samples)-x_obs[None,:])/x_err[None,:])**2).sum(-1)) < 1,
                             simulator_robustness(samples,eps=0.01,tol=0.05))

samples = sample_posterior(posterior,x_obs,num_samples,test_samples,criterion_fn)

posterior = posterior.set_default_x(x_obs)

start = time.process_time()

logL = posterior.log_prob(samples) - prior.log_prob(samples)
samples = torch.concatenate((samples,logL[:,None]),dim=1)

print('Computing log likelihoods took',time.process_time()-start,'s')

with open('./../results/gamma_sample_tE={:.3f}_tI={:.3f}_o={:s}_n={:d}_d={:s}.pkl'.format(tE,tI,obs_file,num_samples,str(device)), 'wb') as handle:
    pickle.dump(samples,handle)
