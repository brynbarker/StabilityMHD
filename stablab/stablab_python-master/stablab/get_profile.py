
# Imports
import numpy as np
from stablab.profile_solve_guess import profile_solve_guess


"""get_profile gets the profile of a """
def get_profile(p, s, tol=1e-8, sL=-10, sR=10, num_inf=2, timeout = 10):

    # Set up the passed arguments into s.
    s.update({'L': sL, 'R': sR})
    max_err = 1+tol

    # Begin solving until the error is small enough.
    while max_err > tol:

        # Solve the profile using the pseudo method.
        [s,p] = profile_solve_guess(s,p,num_inf)

        # Update the max_err to reflect the current profile.
        max_err = max(abs(s['sol']['deval'](s['L'])[0] - np.array([s['UL']]).T))
        max_err = max(max_err, max(abs(s['sol']['deval'](s['R'])[0] - np.array([s['UR']]).T)))

        # Update numerical infinity to attempt a wider domain.
        num_inf = 2*num_inf

        # Update the timeout variable.
        timeout -= 1
        if timeout == 0:
            print('Failed to solve the profile with timeout = '+str(timeout))
            return s,p

        print()
        print('Boundary Error', max_err)
        print('Ode Error', s['sol']['ode_err'])
        print('Tol', tol)
        #print(s['sol']['ode_err_full'])

    # Return s, the solution and p, the parameters.
    return s,p
