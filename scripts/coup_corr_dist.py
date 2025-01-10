import torch

from torch.distributions import Uniform,constraints

from bounded_exponential import BoundedExponential

class CoupCorrDist(torch.distributions.Distribution):
    def __init__(self, rate, coup_mag_bound, corr_lo_bound, corr_up_bound, validate_args=None):
        self.device = rate.device
        self.rate = rate
        self.coup_mag_bound = coup_mag_bound
        self.corr_lo_bound, self.corr_up_bound = corr_lo_bound, corr_up_bound
        self.corr_dim = corr_lo_bound.size(0)
        self.coup_dist = BoundedExponential(rate,torch.zeros(4,device=self.device),
                                            coup_mag_bound*torch.ones(4,device=self.device))
        self.corr_dist = Uniform(self.corr_lo_bound,self.corr_up_bound)
        
        self.lower_bound = torch.cat((torch.tensor([0,-coup_mag_bound,0,-coup_mag_bound],device=self.device),corr_lo_bound))
        self.upper_bound = torch.cat((torch.tensor([ coup_mag_bound,0, coup_mag_bound,0],device=self.device),corr_up_bound))
        
        super().__init__(validate_args=validate_args)
    
    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.lower_bound, self.upper_bound)

    def sample(self, sample_shape=torch.Size()):
        coup_samples = self.coup_dist.sample(sample_shape)
        coup_samples[...,1] *= -1
        coup_samples[...,3] *= -1
        corr_samples = self.corr_dist.sample(sample_shape)
        return torch.concat([coup_samples,corr_samples],dim=-1)

    def log_prob(self, value):
        coup_log_probs = value[...,:4].detach().clone()
        coup_log_probs[...,1] *= -1
        coup_log_probs[...,3] *= -1
        coup_log_probs = self.coup_dist.log_prob(coup_log_probs)
        corr_log_probs = value[...,4:].detach().clone()
        corr_log_probs = self.corr_dist.log_prob(corr_log_probs)
        return torch.concat([coup_log_probs,corr_log_probs],dim=-1)
