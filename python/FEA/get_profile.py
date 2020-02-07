from wave_profile import profile_flux, soln
from Struct import Struct
import numpy as np
import matplotlib.pyplot as plt

def profile_jacobian(y, p):

    v = y[0]
    e = y[1]

    J = np.array([[(1/(2*p['mu']+p['eta']))*(1-p['Gamma']*e/v**2),
                   p['Gamma']/(2*p['mu']+p['eta'])/v],
                  [(1/(p['kappa']/p['cnu']))*(-(v-1) +  p['e_minus']*p['Gamma']),
                   1/(p['kappa']/p['cnu'])]])

    return J

def profile_ode(x, y, sol, p):

    v = y[0]
    e = y[1]

    # multiply by u to tranform to pseudo lagrangian coordinates
    y = v*np.vstack([(1/(2*p['mu']+p['eta']))*((v-1)+p['Gamma']*(e/v-p['e_minus'])),
                   (1/(p['kappa']/p['cnu']))*(-((v-1)**2)/2+e-p['e_minus']+(v-1)*p['Gamma']*p['e_minus'])])

    return y

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

# evaluate the profile 
def evaluate_profile(s,n):
    x = np.linspace(s['L'],s['R'],n+1)
    y = soln(x,s)
    return y

def profile(h1=1.62,vp=.8,n=100):
    p = setParams(h1,vp)
    sol = initialize_sol(p)
    p,s = solve_profile(p,sol)

    y = evaluate_profile(s,n)
    # plot_profile(p,s)
    return y[:,0], y[:,1]


