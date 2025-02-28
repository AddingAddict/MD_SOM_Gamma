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
from sbi.utils import BoxUniform

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser()

parser.add_argument('--tE', '-tE',  help='excitatory time constant (s)', type=float, default=0.005)
parser.add_argument('--tI', '-tI',  help='inhibitory time constant (s)', type=float, default=0.007)
# parser.add_argument('--max_coup', '-maxW',  help='effective coupling order of magnitude', type=float, default=2)
# parser.add_argument('--max_corr', '-maxc',  help='maximum correlation coefficient for E/I noise', type=float, default=1)
# parser.add_argument('--max_Iamp', '-maxa',  help='ratio of I to E noise order of magnitude', type=float, default=1)
parser.add_argument('--num_sim', '-n',  help='number of simulations', type=int, default=10000000)

args = vars(parser.parse_args())
print(parser.parse_args())

tE = args['tE']
tI = args['tI']
# maxW = args['max_coup']
# maxc = args['max_corr']
# maxa = args['max_Iamp']
num_simulations = args['num_sim']

t = torch.tensor([tE,tI])
eps = 0.01
tol = 0.05

def skew_func(x):
    return torch.tan(x*1.07)/1.07

def simulator(theta):
    '''
    theta[:,0] = log10[(τ_E*|Weff_EE|+τ_I*|Weff_II|)/√(τ_E^2+τ_I^2) - min_value]
    theta[:,1] = (τ_I*|Weff_EE|-τ_E*|Weff_II|)/√(τ_E^2+τ_I^2)
    theta[:,2] = (log10[|Weff_EI|] + log10[|Weff_IE|]) / 2
    theta[:,3] = (log10[|Weff_EI|] - log10[|Weff_IE|]) / 2
    theta[:,4] = r(eta_E,eta_I)
    theta[:,5] = log10[|eta_I|/|eta_E|]
    
    returns: [f,g,l]
    f = peak frequency
    g = width of peak
    l = skewness of peak
    '''
    perts = torch.cat([ax_mesh.flatten()[None,:] for ax_mesh in torch.meshgrid((torch.linspace(-1,1,3),)*4)],dim=0)
    perts *= eps/torch.sum(perts**2,dim=0)**0.5
    perts[:,81//2] = 0
    perts += 1

    tE = t[0]
    tI = t[1]
    min_theta0 = torch.where(theta[:,1] > 0, tE/tI*theta[:,1], -tI/tE*theta[:,1])
    Weff_EE =  ((10**theta[:,0]+min_theta0)*tE + theta[:,1]*tI)/torch.sqrt(tE**2+tI**2)
    Weff_EI = -10**(theta[:,2] + theta[:,3])
    Weff_IE =  10**(theta[:,2] - theta[:,3])
    Weff_II = -((10**theta[:,0]+min_theta0)*tI - theta[:,1]*tE)/torch.sqrt(tE**2+tI**2)
    c = theta[:,4][:,None]
    a = 10**theta[:,5][:,None]
    
    Weff_EE = Weff_EE[:,None] * perts[0][None,:]
    Weff_EI = Weff_EI[:,None] * perts[1][None,:]
    Weff_IE = Weff_IE[:,None] * perts[2][None,:]
    Weff_II = Weff_II[:,None] * perts[3][None,:]
    
    out = torch.zeros((theta.shape[0],3,81),dtype=theta.dtype).to(theta.device)
    
    tr = (Weff_EE-1)/tE + (Weff_II-1)/tI
    det = ((Weff_EE-1)*(Weff_II-1) - Weff_EI*Weff_IE)/tE/tI
    
    lam = 0.5*(tr + torch.sqrt(torch.maximum(torch.tensor(0),tr**2-4*det)))
    
    p0 = (a**2*Weff_EI**2 - 2*a*c*Weff_EI*(Weff_II-1) +\
        (Weff_II-1)**2)/tI**2/(2*np.pi)**2
    q0 = (Weff_EI*Weff_IE - (Weff_EE-1)*(Weff_II-1))**2/tE**2/tI**2/(2*np.pi)**4
    q2 = ((Weff_EE-1)**2*tI**2+2*Weff_EI*Weff_IE*tE*tI+\
        (Weff_II-1)**2*tE**2)/tE**2/tI**2/(2*np.pi)**2
    
    #### f
    out[:,0,:] = torch.sqrt(torch.sqrt(p0**2-p0*q2+q0) - p0)
    
    #### g
    out[:,1,:] = torch.pow((p0 + out[:,0,:]**2)*(q2 + 2*out[:,0,:]**2),0.25)
    
    #### l
    out[:,2,:] = skew_func((4*p0 - q2 + 2*out[:,0,:]**2) / (p0 + out[:,0,:]**2)**2 / (4*q0 - q2**2) * out[:,1,:]**6)
    
    valid_idx = ((lam < 0) & (q2**2 < 4*q0) & (Weff_EE > 0) & (Weff_II < 0)).all(dim=1)

    torch.where(valid_idx[:,None,None],out,torch.tensor([torch.nan])[:,None,None],out=out)
    
    err = out/out[:,:,81//2][:,:,None] - 1
    err[:,2,:] = (out[:,2,:] - out[:,2,81//2][:,None])/0.1
    valid_idx = torch.sqrt(torch.sum(err**2,dim=(1,2))/80) < tol
    
    return torch.where(valid_idx[:,None],out[:,:,81//2],torch.tensor([torch.nan])[:,None])

# prior = CoupCorrDist(torch.tensor([0.02],device=device),
#                      torch.tensor([maxW],device=device),
#                      torch.tensor([0.0,0.0,0.0],device=device),
#                      torch.tensor([1.5,maxc,maxa],device=device),
#                      num_coups=3,validate_args=True)

prior = BoxUniform(low =torch.tensor([-2.0,-0.5,np.log10(35*np.sqrt(t[0]*t[1])*2*np.pi),-4.0,0.0,-1.0],device=device),
                   high=torch.tensor([ 0.5, 1.5,np.log10(55*np.sqrt(t[0]*t[1])*2*np.pi), 3.0,1.0, 1.0],device=device),)

posterior = train_posterior(prior,simulator,num_simulations,device)

with open('./../results/gamma_posterior_tE={:.3f}_tI={:.3f}_n={:d}_d={:s}.pkl'.format(tE,tI,num_simulations,str(device)), 'wb') as handle:
    pickle.dump(posterior,handle)
