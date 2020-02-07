
from stablab.bvp_fsolve import bvp_fsolve
from stablab.bvp_fsolve import bvp_fsolve_options
import numpy as np

def profile_solve_guess(s,p,num_inf = 30):
   """[s,p] = profile(s,p)
   Guesses what the profile solution is and puts in the boundary conditions.
   Then calls the bvp_fsolve function with wave speed zero."""

   #Set the initial solution data
   s['Flam'] = lambda x,y: s['F'](x,y,s,p)
   s['I'] = num_inf
   s['L'] = -s['I']
   s['R'] = s['I']

   #solve the ODE
   a = 0.5*(s['UL']+s['UR'])
   c = 0.5*(s['UL']-s['UR'])

   # Set up solinit, parameters used in fsolve
   solinit = {}
   solinit['guess'] = lambda x: a - c*np.tanh(x)
   solinit['boundaries'] = [s['L'],s['R']]

   # Set up the boundary conditions.
   bc_fun = lambda fun: s['BC'](fun, s, p)
   solinit['BClen'] = s['BClen']

   # Call to bvp_fsolve, which will solve the bvp.
   options = bvp_fsolve_options(algorithm_stats = 'off')
   sol = bvp_fsolve(s['Flam'],bc_fun,solinit,options)

   # Update the error.
   [ode_err,bc_err ] = sol['check_err'](1001)
   sol['ode_err'] = ode_err
   sol['bc_err'] = bc_err
   s['sol'] = sol

   return (s,p)
