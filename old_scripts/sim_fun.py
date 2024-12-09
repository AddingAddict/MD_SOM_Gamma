import numpy as np
from numpy.linalg import solve,inv
from scipy.integrate import solve_ivp
import time

def sim_rates(rpl,T,W,h,v0=None,max_min=2):
    r = np.zeros(len(h))
    F = np.zeros(len(h))
    start = time.process_time()
    max_time = max_min*60
    timeout = False

    # This function computes the dynamics of the rate model
    def system_RK45(t,v):
        r = rpl.phi_vec(v)
        F = -v + h + np.matmul(W,r)
        F /= rpl.taus
        return F

    # This function determines if the system is stationary or not
    # def stat_event(t,v):
    #     meanF = np.mean(np.abs(F)/np.maximum(v,1e-2)) - 1e-2
    #     if meanF < 0: meanF = 0
    #     return meanF
    # stat_event.terminal = True

    # This function forces the integration to stop after 15 minutes
    def time_event(t,v):
        int_time = (start + max_time) - time.process_time()
        if int_time < 0: int_time = 0
        return int_time
    time_event.terminal = True

    vs=np.zeros((len(h),len(T)));
    if v0 is not None:
        vs[:,0] = v0
    sol = solve_ivp(system_RK45,[np.min(T),np.max(T)],vs[:,0],method='RK45',t_eval=T,events=[time_event])
    if sol.t.size < len(T):
        print("      Integration stopped after " + str(np.around(T[sol.t.size-1],2)) + "s of simulation time")
        if time.process_time() - start > max_time:
            print("            Integration reached time limit")
            timeout = True
        vs[:,0:sol.t.size] = sol.y
        vs[:,sol.t.size:] = sol.y[:,-1:]
    else:
        vs=sol.y;
    
    r = rpl.phi_vec(vs[:,-1])
    return vs[:,-1],r,timeout

def sim_rates_mult(rpl,T,W,hs,max_min=2):
    vs = np.zeros(np.shape(hs))
    rs = np.zeros(np.shape(hs))
    timeouts = np.zeros(len(hs))

    v0 = np.zeros(np.shape(hs)[1])
    for idx in range(len(hs)):
        vs[idx,:],rs[idx,:],timeouts[idx] = sim_rates(rpl,T,W,hs[idx],v0=v0,max_min=max_min)
        v0 = vs[idx,:]

    return vs,rs,timeouts

def calc_lfp(rpl,Ws,eta_mat,syn_taus,corr_tau,v,fs):
    nc = rpl.n_types
    ns = len(syn_taus)

    ceta = np.zeros((ns*nc,ns*nc))
    ceta[:nc,:nc] = eta_mat

    e = np.zeros(ns*nc)
    e[::ns] = 1

    T = np.kron(np.diag(syn_taus),np.identity(nc))

    dr = np.diag(rpl.dphi_vec(v))

    if len(Ws) == ns*nc:
        W = Ws
    else:
        W = np.zeros((ns*nc,nc))
        for i in range(ns):
            W[i*nc:(i+1)*nc,:] = Ws[i]

    As = np.zeros(len(fs))
    for i in range(len(fs)):
        G = -2*np.pi*1j*fs[i]*T + np.identity(ns*nc) - W@dr@np.kron(np.ones(ns)[np.newaxis,:],np.identity(nc))
        Gdage = inv(G).conj().T@e
        As[i] = np.vdot(Gdage,ceta@Gdage)*2*corr_tau/((2*np.pi*fs[i]*corr_tau)**2+1)
    return As

def calc_lfp_mult(rpl,Ws,eta_mats,syn_taus,corr_tau,vs,fs):
    As = np.zeros((len(fs),len(vs)))

    for idx in range(len(vs)):
        As[:,idx] = calc_lfp(rpl,Ws,eta_mats[idx],syn_taus,corr_tau,vs[idx],fs)

    return As

def calc_jac(rpl,W,v):
    nc = rpl.n_types
    dr = np.diag(rpl.dphi_vec(v))

    jac = inv(np.identity(nc) - W@dr)

    return jac

def calc_jac_mult(rpl,W,vs):
    nc = rpl.n_types
    jacs = np.zeros((len(vs),nc,nc))

    for idx in range(len(vs)):
        jacs[idx] = calc_jac(rpl,W,vs[idx])

    return jacs
