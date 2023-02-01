import argparse
import gc
import numpy as np
from numpy.linalg import solve
import time

import sim_fun as sfun
import sim_util as sutil
import rpl as rpl

parser = argparse.ArgumentParser(description=('This python script samples parameters for a Pyr+PV+SOM network, '
    'simulates the network, and saves the rates and LFP of the network.'))
parser.add_argument('-j',    '--jobnumber',help='job number',type=int,required=True)
parser.add_argument('-nrep', '--nrep',     help='number of parameters per job', type=int, default=1000)

args = vars(parser.parse_args())
jn = int(args['jobnumber'])
nrep = int(args['nrep'])

rp = rpl.RPL([0.1, 0.01,0.1 ],
             [1.6, 2.4, 1.6 ],
             [100, 200, 100 ],
             [0.03,0.01,0.03])

syn_taus = np.array([0.005,0.100,0.007])
corr_tau = 0.005

T = np.arange(0,10,rp.taus[1]/3)

resultsdir = './../results/'
name_results = 'sim_results'+'-'+str(jn)+'.txt'
this_results = resultsdir+name_results
print('Saving all results in '+  name_results)
print(' ')

c0 = 5
c1 = 100
dc = 5
cs = np.linspace(c0,c1,(c1-c0)//dc+1)

f0 = 5
f1 = 80
df = 1
fs = np.linspace(f0,f1,(f1-f0)//df+1)

init = True
try:
    this_loaded_results = np.loadtxt(this_results)
    first_rep = this_loaded_results.shape[0]
except:
    first_rep = 0

for idx_rep in range(first_rep,nrep):
    start = time.process_time()
    print('-------------------Computing and saving network response for rep '+'{:4d}'.format(idx_rep)+' out of '+\
        str(nrep)+'-------------------')

    seed = jn*10000+idx_rep
    param_dict = sutil.gen_rand_param(seed)

    gE = param_dict['gE']
    gP = param_dict['gP']
    gS = param_dict['gS']
    bS = param_dict['bS']
    WEE = param_dict['WEE']
    WEP = param_dict['WEP']
    WES = param_dict['WES']
    WPE = param_dict['WPE']
    WPP = param_dict['WPP']
    WPS = param_dict['WPS']
    WSE = param_dict['WSE']
    WSP = param_dict['WSP']

    print('Parameters used seed = {:d} // gE = {:.2f} // gP = {:.2f} // gS = {:.2f} // bS = {:.2f}' \
    .format(seed,gE,gP,gS,bS))
    print('                                WEE = {:.2f} // WEP = {:.2f} // WES = {:.2f} // WPE = {:.2f}'\
    .format(WEE,WEP,WES,WEP))
    print('                                WPP = {:.2f} // WPS = {:.2f} // WSE = {:.2f} // WSP = {:.2f}'\
    .format(WPP,WPS,WSE,WSP))
    print()

    g = np.array([gE,gP,gS])
    b = np.array([0.0,0.0,bS])
    W = np.array([[WEE,-WEP,-WES],[WPE,-WPP,-WPS],[WSE,-WSP,0.0]])

    Wlfp = np.vstack((np.hstack((W[:,0:1]/2,np.zeros((rp.n_types,rp.n_types-1)))),
                      np.hstack((W[:,0:1]/2,np.zeros((rp.n_types,rp.n_types-1)))),
                      np.hstack((np.zeros((rp.n_types,1)),W[:,1:]))))
    hs = g*cs[:,np.newaxis] + b

    vs,rs,_ = sfun.sim_rates_mult(rp,T,W,hs)
    As = sfun.calc_lfp_mult(rp,Wlfp,hs,syn_taus,corr_tau,vs,fs)

    print(vs)
    print()
    print(rs)
    print()
    print(As)

    print()
    print('Simulations took ',time.process_time() - start,' s')
    print()
    print('------------------------------------------Saving results------------------------------------------')
    print()

    start = time.process_time()
    sim_params = np.array(list(param_dict.values()))
    sim_results = np.concatenate((vs.flatten(),rs.flatten(),As.flatten()))

    if init:
        results = np.zeros((nrep,len(sim_params)+len(sim_results)))
        try:
            results[:first_rep,:] = np.loadtxt(this_results)
        except:
            pass
        init = False

    results[idx_rep,0:len(sim_params)] = sim_params
    results[idx_rep,len(sim_params):(len(sim_params)+len(sim_results))] = sim_results

    mask_rep = results[:,0]>0
    
    #------------------------------------------------------------------------------------------------------
    # save
    #------------------------------------------------------------------------------------------------------
    
    # Clean file to print results
    f_handle = open(this_results,'w')
    np.savetxt(f_handle,results[mask_rep,:],fmt='%.6f', delimiter='\t')
    f_handle.close()
    print('Saving results took ',time.process_time() - start,' s')
    print()
    
    gc.collect()
