import numpy as np
from scipy.integrate import solve_ivp

class RPL(object):
    def __init__(self,ks,ns,maxrs,taus):
        # Parameters defined by ale
        self.ks = np.array(ks)
        self.ns = np.array(ns)
        self.maxrs = np.array(maxrs)
        self.taus = np.array(taus)

        self.n_types = len(ks)
        
    def phi(self,mu,type_idx):
        r = np.minimum(self.ks[type_idx]*np.maximum(mu,0)**self.ns[type_idx],self.maxrs[type_idx])
        return r

    def phi_tens(self,mu):
        if np.size(mu) == 1:
            r = np.zeros((self.n_types))
            for type_idx in range(self.n_types):
                r[type_idx] = self.phi(mu,type_idx)
        else:
            r = np.zeros((self.n_types,*np.shape(mu)))
            for type_idx in range(self.n_types):
                r[type_idx] = self.phi(mu,type_idx)
        return r

    def phi_vec(self,mu):
        r = np.zeros((self.n_types))
        for type_idx in range(self.n_types):
            r[type_idx] = self.phi(mu[type_idx],type_idx)
        return r

    def dphi(self,mu,type_idx):
        dmu = 0.01
        rpdmu = self.phi(mu+dmu,type_idx)
        rmdmu = self.phi(mu-dmu,type_idx)
        return (rpdmu-rmdmu)/(2*dmu)

    def dphi_tens(self,mu):
        if np.size(mu) == 1:
            dr = np.zeros((self.n_types))
            for type_idx in range(self.n_types):
                dr[type_idx] = self.dphi(mu,type_idx)
        else:
            dr = np.zeros((self.n_types,*np.shape(mu)))
            for type_idx in range(self.n_types):
                dr[type_idx] = self.dphi(mu,type_idx)
        return dr

    def dphi_vec(self,mu):
        dr = np.zeros((self.n_types))
        for type_idx in range(self.n_types):
            dr[type_idx] = self.dphi(mu[type_idx],type_idx)
        return dr
