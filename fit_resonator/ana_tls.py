Tc = 9.288
import scipy.constants as cs
import numpy as np
from scipy import special

# Power in W 
def pow_res(p,atten): return 10**((p+atten)/10)*1e-3

# Photons 
def n(p,f, q,qc,atten): return pow_res(p,atten)*q**2/qc/(cs.h*f**2*np.pi)

# Boltzmann
def tp(f, T): return np.tanh(cs.h*f/(2*cs.k*T))

# T: temperature, nc: critical phonon number, f: frequency, Qtls0: TLS limit, beta: power law, n: photon number
def Qtls(n, T, f,  Qtls0, nc, beta): 
    return Qtls0/tp(f, T)*np.sqrt(1+(n/nc)**beta*tp(f, T))

# MB fit; Quasiparticle quality 
def Qqp(T, f, Qqp0, Tc): return Qqp0*np.exp(1.764*Tc/T)/np.sinh(cs.h * f/2/cs.k/T)/special.kn(0,cs.h*f/2/cs.k/T)

# Quality including TLS, QP, other 
def Qtot(n, T, f, Qqp0, Qtls0, Qoth, Tc, beta, nc): 
    return 1/(1/Qqp(T, f, Qqp0, Tc) + 1/Qtls(n, T, f, Qtls0, beta, nc)+1/Qoth)

# Quality including TLS and other
def Qtotn(n, T, f, Qtls0, Qoth, nc, beta): 
    return 1/(1/Qtls(n, T, f, Qtls0, nc,beta)+1/Qoth)

# Houck lab TLS model 
def Qtls2(n, T, f, Qtls0, b1, b2, D): 
     return Qtls0 * np.sqrt(1 + n**b2 / (D*T**b1) * tp(f, T) / tp(f, T))