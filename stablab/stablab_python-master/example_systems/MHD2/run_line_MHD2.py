import numpy as np
import matplotlib.pyplot as plt

# from contour import semicirc, semicirc2, winding_number, Evans_plot 
# from evans import emcset, LdimRdim, Evans_compute, reflect_image, radius
# from root_finding import moments_roots
# from wave_profile import profile_flux, soln
# from Struct import Struct

from stablab import (semicirc, winding_number, Evans_plot, emcset,
                      Evans_compute, Struct, reflect_image)
from stablab.wave_profile import profile_flux, soln
from stablab.root_finding import moments_roots
from stablab.evans import LdimRdim

from MHD2 import profile_ode, profile_jacobian
from MHD2 import A_evans, A_lop, A_jump, Ak

from importlib import import_module
import pickle
import sys
import os
from math import floor

np.set_printoptions(suppress=True)

# Initialize Parameters

# Parameters
def setParams(h1,vp):
    cnu = 1
    mu = .1
    Gamma = 2/3
    p = Struct({'h1': h1,
            'Gamma': Gamma, 
            'v_plus': vp,
            'mu': .1,
            'eta': -2/3*mu,
            'cnu': cnu,
            'R': cnu*Gamma,
            'alpha':1,
            'kappa': .1 })
    p.update({
            'v_star': p['Gamma']/(p['Gamma']+2) # 3.14
            })
    p.update({
            'e_plus': p['v_plus']*(p['Gamma']+2-p['Gamma']*p['v_plus'])/(2*p['Gamma']*(p['Gamma']+1)), # 3.15
            'e_minus': (p['Gamma']+2)*(p['v_plus']-p['v_star'])/(2*p['Gamma']*(p['Gamma']+1)), # 3.16
            'v_minus': 1,
            'nu': p['kappa']/p['cnu'] # see below 2.25
            })
    p.update({
            'UR': [1/p['v_plus'],p['v_plus'],0,p['h1'],0,p['e_plus']/p['cnu']],
            'UL': [1/p['v_minus'],p['v_minus'],0,p['h1'],0,p['e_minus']/p['cnu']]
            })
    return p

def initialize_sol(p):
    # Initialising sol, the Struct with solution values
    sol = Struct({
        'n': 2, # this is the dimension of the profile ode
        # we divide the domain in half to deal with the
        # non-uniqueness caused by translational invariance
        # sol.side = 1 means we are solving the profile on the interval [0,X]
        'side': 1,
        'F': profile_ode, # F is the profile ode
        'Flinear': profile_jacobian, # J is the profile ode Jacobian
        'UL': np.array([p['v_minus'],p['e_minus']]), # These are the endstates of the profile and its derivative at x = -infty
        'UR': np.array([p['v_plus'],p['e_plus']]), # These are the endstates of the profile and its derivative at x = +infty
        'tol': 1e-7,
        'xi': 0.1,
        'system' : 'parallel',
        'mat_type' : 'mbfv',
        'tol':1e-5
        })
    sol.update({
        'phase': 0.5*(sol['UL']+sol['UR']), # this is the phase condition for the profile at x = 0
        'order': [1], # this indicates to which component the phase conditions is applied
        'stats': 'on', # this prints data and plots the profile as it is solved
        'bvp_options': {'Tol': 1e-5, 'Nmax': 200},
        'L_max': 10000,
        'R_max': 10000
        })
    return sol

# Solve Profile
def solve_profile(p,sol):
    sol['stats'] = 'off'
    p,s = profile_flux(p,sol)
    return p,s

# Plot the profile
def plot_profile(p,s):
    x = np.linspace(s['L'],s['R'],10)
    y = soln(x,s)

    plt.figure("Profile")
    plt.plot(x,y)
    plt.show()
    
def set_up(h1,vp):
    p = setParams(h1,vp)
    sol = initialize_sol(p)
    p,s = solve_profile(p,sol)
    
    # plot_profile(p,s)
    return p,s

# Evaluate Evans Function
def evaluate_Evans(p,s,xi):
    R,tol = .01,.01
    # R is actually redefined later on
    ksteps = 2**8;
    lamsteps = 0;

    # Set Fourier variable
    s['xi'] = xi

    # Set up Stablab structures
    # s, e, m, c = emcset(s,'front',[5,6],'reg_reg_polar',A)
    Ldim,Rdim = LdimRdim(A_evans, s, p)
    s, e, m, c = emcset(s,'front',[Ldim,Rdim],'default',A_evans)

    # set error tolerance
    m['options'] = {}
    m['options']['RelTol'] = 1e-10
    m['options']['AbsTol'] = 1e-12

    # refine the Evans function computation to achieve set relative error
    c['refine'] = 'on';
    c['ksteps'] = ksteps
    c['lambda_steps'] = lamsteps
    c['tol'] = tol
    c['check'] = 'off'

    # display a waitbar
    c['stats'] = 'off'

    r = 1e-12
    R = 1e-4
    pnts = 20
    preimage = np.linspace(R,r,pnts+(pnts-1)*ksteps)

    # Compute the Evans function
    out, domain = Evans_compute(preimage,c,s,p,m,e)

    # Normalize and plot the Evans function
    out = out/out[0]
    
    return domain,out

def get_root(d,o):
    l_ind = 0
    r_ind = len(o)-1
    
    count = 0
    while True:
        if np.real(o[l_ind])*np.real(o[r_ind]) > 0:
            found_root = False
            break
            
        if abs(l_ind-r_ind) == 1:
            found_root = True
            break
        
        new_ind = floor((r_ind+l_ind)/2)
        
        if np.real(o[l_ind])*np.real(o[new_ind]) > 0:
            l_ind = new_ind
        else:
            r_ind = new_ind
            
    l_val = np.real(o[l_ind])
    r_val = np.real(o[r_ind])
    
    l_dom = d[l_ind]
    r_dom = d[r_ind]
    
    x_val = l_dom - l_val * (r_dom-l_dom)/(r_val-l_val)
    
    return x_val, found_root

def find_roots(h1_vals,up,xi_vals):
    # starting h1 val
    # fixed uplus and xi vals
    d = {}
    
    for h1 in h1_vals:
        d[h1] = {}
        for xi in xi_vals:
            print(h1,xi)
            try:
                p,s = set_up(h1,up)
                domain,out = evaluate_Evans(p,s,xi)
                loc,success = get_root(domain,out)
                d[h1][xi] = (loc,success)
            except:
                print('\t did not work')
        
    return d
