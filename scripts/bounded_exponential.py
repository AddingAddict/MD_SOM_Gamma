import torch

from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

class BoundedExponential(torch.distributions.Distribution):
    arg_constraints = {
        "rate": constraints.positive,
        "low": constraints.dependent(is_discrete=False, event_dim=0),
        "high": constraints.dependent(is_discrete=False, event_dim=0),
    }
    
    def __init__(self, rate, low, high, validate_args=None):
        self.rate, self.low, self.high = broadcast_all(rate, low, high)
        
        if self._validate_args:
            if not torch.gt(self.low, 0).all():
                raise ValueError("BoundedExponential is not defined when low<0")
            if not torch.gt(self.high, 0).all():
                raise ValueError("BoundedExponential is not defined when high<0")
            if not torch.lt(self.low, self.high).all():
                raise ValueError("BoundedExponential is not defined when low>= high")
            
        self.low_exp_cdi = 1 - torch.exp(-self.rate * low)
        self.high_exp_cdi = 1 - torch.exp(-self.rate * high)
        
        super().__init__(validate_args=validate_args)
    
    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.low, self.high)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return self.icdi(self.rate.new(shape + self.rate.size()).uniform_())

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.rate.log() - self.rate * value - (self.high_exp_cdi - self.low_exp_cdi).log()
    
    def cdi(self, value):
        if self._validate_args:
            self._validate_sample(value)
        exp_cdi = 1 - torch.exp(-self.rate * value)
        return (exp_cdi - self.low_exp_cdi) / (self.high_exp_cdi - self.low_exp_cdi)
    
    def icdi(self, value):
        return -torch.log1p(-self.low_exp_cdi-value*(self.high_exp_cdi - self.low_exp_cdi)) / self.rate