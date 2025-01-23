import argparse
import os
import time
try:
    import pickle5 as pickle
except:
    import pickle

import torch
from sbi.utils.user_input_checks import (
    process_prior,
)

from coup_corr_dist import CoupCorrDist

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
parser.add_argument('--num_sim', '-n',  help='number of simulations', type=int, default=10000000)
parser.add_argument('--num_samp', '-p',  help='number of posterior samples', type=int, default=10000000)

args = vars(parser.parse_args())
print(parser.parse_args())

obs_file = args['obs_file']
tE = args['tE']
tI = args['tI']
num_simulations = args['num_sim']
num_samples = args['num_samp']

t = torch.tensor([tE,tI])

with open('../results/'+obs_file+'.pkl', 'rb') as handle:
    obs_dict = pickle.load(handle)
    frn = obs_dict['frn']
    frs = obs_dict['frs']
    wrn = obs_dict['wrn']
    wrs = obs_dict['wrs']
    Arn = obs_dict['Arn']
    Ars = obs_dict['Ars']
    
x_obs = torch.tensor([frn,wrn,Arn])

with open('./../results/gamma_posterior_tE={:.3f}_tI={:.3f}_n={:d}_d={:s}.pkl'.format(tE,tI,num_simulations,str(device)), 'rb') as handle:
    posterior = pickle.load(handle).set_default_x(x_obs)

with open('./../results/gamma_sample_tE={:.3f}_tI={:.3f}_o={:s}_n={:d}_d={:s}.pkl'.format(tE,tI,obs_file,num_samples,str(device)), 'rb') as handle:
    samples = pickle.load(handle)

prior = CoupCorrDist(torch.tensor([0.05],device=device),
                     torch.tensor([200],device=device),
                     torch.tensor([0.0,0.0],device=device),
                     torch.tensor([1.0,2.0],device=device),True)

# Check prior, return PyTorch prior.
prior, num_parameters, prior_returns_numpy = process_prior(prior)

start = time.process_time()

logL = posterior.log_prob(samples) - prior.log_prob(samples)

print('Computing log likelihoods took',time.process_time()-start,'s')

start = time.process_time()

top_one_pc = torch.topk(logL,k=num_samples//100,largest=False,sorted=False)
top_one_pc = torch.concatenate((samples[top_one_pc.indices,:],top_one_pc.values[:,None]),dim=1)

print('Selecting 1%% highest likelihood samples took',time.process_time()-start,'s')

with open('./../results/gamma_likelihood_tE={:.3f}_tI={:.3f}_o={:s}_n={:d}_d={:s}.pkl'.format(tE,tI,obs_file,num_samples,str(device)), 'wb') as handle:
    pickle.dump(top_one_pc,handle)
