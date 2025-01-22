import time

import torch
from sbi.inference import NPE
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

def train_posterior(prior,simulator,num_simulations,device=torch.device('cpu')):
    # Check prior, return PyTorch prior.
    prior, _, prior_returns_numpy = process_prior(prior)

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
    
    return posterior

def sample_posterior(posterior,x_obs,num_samples,test_samples=100000,criterion_fn=None):
    if criterion_fn is not None:
        start = time.process_time()
        
        samples = posterior.sample((test_samples,), x=x_obs)
        fit_idx = criterion_fn(samples)
        fit_frac = fit_idx.sum().item() / test_samples
        required_samples = int(num_samples/fit_frac*1.05)
        print('required samples:', required_samples)

        print('Test sampling took',time.process_time()-start,'s')

        start = time.process_time()

        samples = posterior.sample((required_samples,), x=x_obs)
        fit_idx = criterion_fn(samples)
        samples = samples[fit_idx,:]
        print(samples.shape)

        print('Sampling the required number for desired sample size took',time.process_time()-start,'s')
    else:
        start = time.process_time()
        
        samples = posterior.sample((num_samples,), x=x_obs)

        print('Sampling the required number for desired sample size took',time.process_time()-start,'s')
        
    return samples