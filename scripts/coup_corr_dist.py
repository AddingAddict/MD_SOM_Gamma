import torch

from torch.distributions import Uniform,constraints

from bounded_exponential import BoundedExponential

class CoupDist(torch.distributions.Distribution):
    def __init__(self, rate, coup_mag_bound, num_coups=4, validate_args=None):
        self.device = rate.device
        self.rate = rate
        self.coup_mag_bound = coup_mag_bound
        self.num_coups = num_coups
        self.coup_mag_dist = BoundedExponential(rate,torch.zeros(self.num_coups,device=self.device),
                                                coup_mag_bound*torch.ones(self.num_coups,device=self.device))
        
        self.lower_bound = torch.zeros(self.num_coups,device=self.device)
        self.upper_bound = coup_mag_bound*torch.ones(self.num_coups,device=self.device)
        
        super().__init__(validate_args=validate_args)
    
    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.lower_bound, self.upper_bound)

    def sample(self, sample_shape=torch.Size()):
        return self.coup_mag_dist.sample(sample_shape)

    def log_prob(self, value):
        return self.coup_mag_dist.log_prob(value)

class CoupCorrDist(torch.distributions.Distribution):
    def __init__(self, rate, coup_mag_bound, corr_lo_bound, corr_up_bound, num_coups=4, validate_args=None):
        self.device = rate.device
        self.rate = rate
        self.coup_mag_bound = coup_mag_bound
        self.num_coups = num_coups
        self.corr_lo_bound, self.corr_up_bound = corr_lo_bound, corr_up_bound
        self.corr_dim = corr_lo_bound.size(0)
        self.coup_dist = CoupDist(rate,coup_mag_bound,num_coups)
        self.corr_dist = Uniform(self.corr_lo_bound,self.corr_up_bound)
        
        self.lower_bound = torch.cat((self.coup_dist.lower_bound,corr_lo_bound))
        self.upper_bound = torch.cat((self.coup_dist.upper_bound,corr_up_bound))
        
        super().__init__(validate_args=validate_args)
    
    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.lower_bound, self.upper_bound)

    def sample(self, sample_shape=torch.Size()):
        coup_samples = self.coup_dist.sample(sample_shape)
        corr_samples = self.corr_dist.sample(sample_shape)
        return torch.concat([coup_samples,corr_samples],dim=-1)

    def log_prob(self, value):
        coup_log_probs = self.coup_dist.log_prob(value[...,:self.num_coups])
        corr_log_probs = self.corr_dist.log_prob(value[...,self.num_coups:])
        return coup_log_probs + torch.sum(corr_log_probs,-1)
