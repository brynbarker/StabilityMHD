import numpy as np
import matplotlib.pyplot as plt
from stablab import (semicirc, winding_number, Evans_plot, emcset,
                       Evans_compute, Struct, reflect_image)
from stablab.wave_profile import profile_flux, soln
from stablab.root_finding import moments_roots
from stablab.evans import LdimRdim

from MHD2 import profile_ode, profile_jacobian
from MHD2 import A_evans, A_lop, A_jump, Ak

import pickle

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
    ksteps = 2**6;
    lamsteps = 2**8;

    # Set Fourier variable
    s['xi'] = xi


    # Set up Stablab structures
    # s, e, m, c = emcset(s,'front',[5,6],'reg_reg_polar',A)
    Ldim,Rdim = LdimRdim(A_evans, s, p)
    s, e, m, c = emcset(s,'front',[Ldim,Rdim],'default',A_evans)

    # refine the Evans function computation to achieve set relative error
    c['refine'] = 'on';
    c['ksteps'] = ksteps
    c['lambda_steps'] = lamsteps
    c['tol'] = tol
    c['check'] = 'off'

    # display a waitbar
    c['stats'] = 'off'

    # Set up the preimage
    circpnts, imagpnts, innerpnts = 20, 20, 5
    spread = 4
    zerodist = 0#10**(-4)
    
    preimage = semicirc(circpnts, imagpnts, c['ksteps'], R, spread, zerodist,c['lambda_steps'])

    # Compute the Evans function
    out, domain = Evans_compute(preimage,c,s,p,m,e)

    # Normalize and plot the Evans function
    out = out/out[0]
    w = reflect_image(out)
    prew = reflect_image(domain)
    
    # Evans_plot(w,titlestring='Evans Function')
    # plt.show()

    wnd = winding_number(w)
    rts = moments_roots(prew,w).real

#     print('Evans Function Winding Number: {:f}'.format(wnd))
    
    output = {'w':w,'wnd':wnd,'prew':prew,'xi':xi,'R':R,
              'tol':tol,'p':dict(p),'rts':rts}
    return output



# Evaluate Lopatinski Determinant
def evaluate_Lopatinski(p,s):
    R,tol = .1,.1
    ksteps = 2**8
    
    # Set Fourier variable
    s['xi'] = 1

    # Set up Stablab structures
    Ldim,Rdim = LdimRdim(A_lop, s, p)
    s, e, m, c = emcset(s,'lopatinski',[Ldim,Rdim],'default',A_lop)

    # set the lopatinski jump condition
    e['jump'] = A_jump


    s['R'] = 1
    s['L'] = -1

    # refine the Evans function computation to achieve set relative error
    c['refine'] = 'on';
    c['ksteps'] = ksteps
    c['tol'] = tol
    c['check'] = 'off'
    
    # display a waitbar
    c['stats'] = 'off'

    # Set up the preimage
    circpnts, imagpnts, innerpnts = 20, 80, 50
    shift = 1e-5
    spread = 4
    zerodist = 0
    preimage = semicirc(circpnts, imagpnts, c['ksteps'], R, spread, zerodist)+shift

    # Compute the Evans function
    out, domain = Evans_compute(preimage,c,s,p,m,e)

    # Normalize and plot the Evans function
    out = out/out[0]
    w = reflect_image(out)
    prew = reflect_image(domain)
    # Evans_plot(w,titlestring='Lopatinski Determinant')

    wnd = winding_number(w)
    rts = moments_roots(prew,w).real

#     print('Lopatinski Winding Number: {:f}'.format(wnd))

    output = {'w':w,'wnd':wnd,'prew':prew,'xi':1,'R':R,
              'tol':tol,'p':dict(p),'rts':rts}
    return output


    
# # contour radius
# R = 0.01;

# # contour tol
# tol = 0.01;

if __name__ == "__main__":

    h1_vals = [0,.99]
    vp_vals = [.3,.8]
    data = {}

    for h1 in h1_vals:
        for vp in vp_vals:
            # key for data storage
            key = str(h1)+' '+str(vp)
            data[key] = {}
            print(key)

            # set up profile and params
            p,s = set_up(h1,vp)

            # lopatinski
            print('\tlopatinki')
            try:
                res = evaluate_Lopatinski(p,s)
            except:
                res = None
            data[key]['lop'] = res

            # evans
            print('\tevans')
            data[key]['evans'] = {}
            for xi in [.1,.2]:
                print('\t\t{}'.format(xi))
                try:
                    res = evaluate_Evans(p,s,xi)
                except:
                    res = None
                data[key]['evans'][xi] = res
                
    pickle.dump(data, open("output/test1.pkl","wb"))

