from multiprocessing import Process
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

def sample_posterior_batch(posterior,x_obs_batch,num_samples,test_samples=100000,
                           timeout_samples=5,timeout_per_samp=4,criterion_fn=None):
    assert x_obs_batch.dim() == 2, 'x_obs_batch must have a batch dimension'
    n_obs = x_obs_batch.shape[0]

    start = time.process_time()
    
    fit_fracs = torch.zeros(n_obs)
    for batch_idx in range(n_obs):
        # Check which observations allow successful sampling of posterior
        p = Process(target=sample_posterior, args=(posterior,x_obs_batch[batch_idx],timeout_samples))
        p.start()
        
        p.join(timeout_samples*timeout_per_samp)

        if p.is_alive():
            p.terminate()
            p.join()
            print('Batch',batch_idx,'failed')
            continue
        
        p.close()
        
        print('Batch',batch_idx,'succeeded, proceeding')
        
        # Compute how many samples per observation are needed to get the desired number of total samples
        samples = sample_posterior(posterior,x_obs_batch[batch_idx],test_samples)
        if criterion_fn is not None:
            fit_idx = criterion_fn(samples)
            
            print('\tBatch',batch_idx,'fit the criteria with',fit_idx.sum().item(),'samples')
            fit_fracs[batch_idx] = fit_idx.sum().item() / test_samples
        else:
            fit_fracs[batch_idx] = 1
    
    if criterion_fn is not None:
        required_samples = int(num_samples/fit_fracs.sum()*1.05)
    else:
        required_samples = int(num_samples/fit_fracs.sum())
    print('required samples:', required_samples)

    print('Test sampling took',time.process_time()-start,'s')

    start = time.process_time()
    
    samples = None
    for batch_idx in range(n_obs):
        if fit_fracs[batch_idx] == 0:
            continue
        
        this_samples = posterior.sample((required_samples,), x=x_obs_batch[batch_idx])
        fit_idx = criterion_fn(this_samples)
        samples = this_samples[fit_idx,:]
        if samples is None:
            samples = this_samples
        else:
            samples = torch.cat((samples,this_samples),dim=0)

    print('Sampling the required number for desired sample size took',time.process_time()-start,'s')
        
    return samples
