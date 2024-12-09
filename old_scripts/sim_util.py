import os

import numpy as np

import sim_fun as sfun
import rpl as rpl

c0 = 5
c1 = 100
dc = 5
cs = np.linspace(c0,c1,(c1-c0)//dc+1)

f0 = 1
f1 = 80
df = 1
fs = np.linspace(f0,f1,(f1-f0)//df+1)

prm_idxs = {
    'gE'  : 0,
    'gP'  : 1,
    'gS'  : 2,
    'bS'  : 3,
    'WEE' : 4,
    'WEP' : 5,
    'WES' : 6,
    'WPE' : 7,
    'WPP' : 8,
    'WPS' : 9,
    'WSE' :10,
    'WSP' :11
}

def gen_rand_prm(seed):
    np.random.seed(seed)

    rand_prm_dict = {}

    rand_prm_dict['seed'] = seed
    rand_prm_dict['gE'] = np.random.uniform(0.0,8.0)
    rand_prm_dict['gP'] = np.random.uniform(0.0,16.0)
    rand_prm_dict['gS'] = np.random.uniform(0.0,4.0)
    rand_prm_dict['bS'] = np.random.uniform(0.0,16.0)
    rand_prm_dict['WEE'] = np.random.uniform(0.0,16.0)
    rand_prm_dict['WEP'] = np.random.uniform(0.0,16.0)
    rand_prm_dict['WES'] = np.random.uniform(0.0,16.0)
    rand_prm_dict['WPE'] = np.random.uniform(0.0,16.0)
    rand_prm_dict['WPP'] = np.random.uniform(0.0,16.0)
    rand_prm_dict['WPS'] = np.random.uniform(0.0,16.0)
    rand_prm_dict['WSE'] = np.random.uniform(0.0,16.0)
    rand_prm_dict['WSP'] = np.random.uniform(0.0,16.0)

    return rand_prm_dict

def load_results(name_end):
    results_dir = './../results/'
    results_file = 'sim_'+name_end+'-'
    file_idxs = []
    for file in os.listdir(results_dir):
        if file.startswith(results_file):
            file_idxs.append(int(file.replace(results_file,'').replace('.txt','')))
    file_idxs.sort()

    init = True
    results = None
    for i in file_idxs:
        this_loaded_results=np.loadtxt(results_dir+results_file+str(i)+'.txt')
        try:
            if init:
                results = this_loaded_results
                init = False
            else:
                results = np.vstack((results,this_loaded_results))
        except:
            pass

    seeds = results[:,0]
    prms = results[:,1:13]
    vs = results[:,13:13+len(cs)*3].reshape((-1,len(cs),3))
    rs = results[:,13+len(cs)*3:13+len(cs)*6].reshape((-1,len(cs),3))
    return seeds,prms,vs,rs

def get_prms_dict(prms):
    prm_dict = {}

    prm_dict['gE']  = prms[0]
    prm_dict['gP']  = prms[1]
    prm_dict['gS']  = prms[2]
    prm_dict['bS']  = prms[3]
    prm_dict['WEE'] = prms[4]
    prm_dict['WEP'] = prms[5]
    prm_dict['WES'] = prms[6]
    prm_dict['WPE'] = prms[7]
    prm_dict['WPP'] = prms[8]
    prm_dict['WPS'] = prms[9]
    prm_dict['WSE'] = prms[10]
    prm_dict['WSP'] = prms[11]

    return prm_dict

def sim_rates(rpl,T,prm_dict,cs=cs,c0=10):
    b = np.array([0.0,0.0,prm_dict['bS']])
    g = np.array([prm_dict['gE'],prm_dict['gP'],prm_dict['gS']])
    W = np.array([[prm_dict['WEE'],-prm_dict['WEP'],-prm_dict['WES']],
        [prm_dict['WPE'],-prm_dict['WPP'],-prm_dict['WPS']],
        [prm_dict['WSE'],-prm_dict['WSP'],0.0]])

    h = b+g*(c0+cs[:,np.newaxis])

    return sfun.sim_rates_mult(rpl,T,W,h)

def sim_lfp(rpl,vs,prm_dict,c0=10):
    b = np.array([0.0,0.0,prm_dict['bS']])
    g = np.array([prm_dict['gE'],prm_dict['gP'],prm_dict['gS']])
    W = np.array([[prm_dict['WEE'],-prm_dict['WEP'],-prm_dict['WES']],
        [prm_dict['WPE'],-prm_dict['WPP'],-prm_dict['WPS']],
        [prm_dict['WSE'],-prm_dict['WSP'],0.0]])

    Wlfp = np.vstack((np.hstack((W[:,:1]/2,np.zeros((rpl.n_types,rpl.n_types-1)))),
                      np.hstack((W[:,:1]/2,np.zeros((rpl.n_types,rpl.n_types-1)))),
                      np.hstack((np.zeros((rpl.n_types,1)),W[:,1:]))))

    h = b+g*(c0+cs[:,np.newaxis])

    syn_taus = np.array([0.005,0.100,0.007])
    corr_tau = 0.005

    return sfun.calc_lfp_mult(rpl,Wlfp,[np.identity(3) for c in cs],syn_taus,corr_tau,vs,fs)

def sim_jac(rpl,vs,prm_dict):
    W = np.array([[prm_dict['WEE'],-prm_dict['WEP'],-prm_dict['WES']],
        [prm_dict['WPE'],-prm_dict['WPP'],-prm_dict['WPS']],
        [prm_dict['WSE'],-prm_dict['WSP'],0.0]])

    return sfun.calc_jac_mult(rpl,W,vs)
