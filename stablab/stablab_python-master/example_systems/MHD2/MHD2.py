import numpy as np
import matplotlib.pyplot as plt
from stablab import (semicirc, winding_number, Evans_plot, emcset,
                       Evans_compute, Struct, reflect_image)
from stablab.wave_profile import profile_flux,soln

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

def A_evans(x,lam,sol,p):

    # solution to profile
    v = soln(x,sol)
    

    rho = 1/v[0]
    u1 = v[0]
    u2 = 0
    h1 = p['h1']
    h2 = 0
    T = v[1]/p['cnu']
    
    # extra parameter
    Rnu = p['R']+p['cnu']
    xi = sol['xi']
    
    # use profile system to get profile derivatives
    u1_x = v[0]*(1/(2*p['mu']+p['eta']))*((v[0]-1)+p['Gamma']*(v[1]/v[0]-p['e_minus']))
    u2_x = 0
    h1_x = 0
    h2_x = 0
    rho_x = -u1_x/v[0]**2
    T_x = v[0]/p['cnu']*((1/(p['kappa']/p['cnu']))*(-((v[0]-1)**2)/2+v[1]-p['e_minus']+(v[0]-1)*p['Gamma']*p['e_minus']))
    
    # define r to create the sharp variables
    
    # choose matrix type
    if sol['mat_type'] == 'mbfv':
        r = np.linalg.norm(xi)+lam
    else:
        r = 1
        
    # make transformation to sharp variables
    xi /= r
    lam /= r
    
    # \bar{A}^0 or equivalently A0(Ubar)
    A0bar = np.array([[1,0,0,0,0,0],
                   [u1,rho,0,0,0,0],
                   [u2,0,rho,0,0,0],
                   [0,0,0,1,0,0],
                   [0,0,0,0,1,0],
                   [p['cnu']*T+(u1**2+u2**2)/2,rho*u1,rho*u2,h1,h2,p['cnu']*rho]])
    
    # \bar{A}^1 or equivalently A1(Ubar)
    A1 = np.array([[u1,rho,0,0,0,0],
                   [p['R']*T+u1**2,2*rho*u1,0,-h1,h2,p['R']*rho],
                   [u1*u2,rho*u2,rho*u1,-h2,-h1,0],
                   [0,0,0,p['alpha'],0,0],
                   [0,h2,-h1,-u2,u1,0],
                   [Rnu*u1*T+u1*(u1**2+u2**2)/2,Rnu*rho*T+h2**2+rho*(3*u1**2+u2**2)/2,-h1*h2+rho*u1*u2,-h2*u2,2*h2*u1-h1*u2,Rnu*rho*u1]])
    
    # \bar{A}^2 or equivalently A2(Ubar)
    A2 = np.array([[u2,0,rho,0,0,0],
                   [u1*u2,rho*u2,rho*u1,-h2,-h1,0],
                   [p['R']*T+u2**2,0,2*rho*u2,h1,-h2,p['R']*rho],
                   [0,-h2,h1,u2,p['alpha']-u1,0],
                   [0,0,0,0,0,0],
                   [Rnu*u2*T+u2*(u1**2+u2**2)/2,-h1*h2+rho*u1*u2, Rnu*rho*T+h1**2+rho*(3*u2**2+u1**2)/2,2*h1*u2-h2*u1,-h1*u1,Rnu*rho*u2]])
    
    # extra to subtract off \bar{A}^1
    B11Ux = np.array([[0,0,0,0,0,0],
                    [0,0,0,0,0,0],
                    [0,0,0,0,0,0],
                    [0,0,0,0,0,0],
                    [0,0,0,0,0,0],
                    [0,(2*p['mu']+p['eta'])*u1_x,0,0,0,0]])
    
    # extra to subtract off \bar{A}^2
    B21Ux = np.array([[0,0,0,0,0,0],
                    [0,0,0,0,0,0],
                    [0,0,0,0,0,0],
                    [0,0,0,0,0,0],
                    [0,0,0,0,0,0],
                    [0,0,p['eta']*u1_x,0,0,0]])
    
    # finish defining A1bar
    A1bar = A1 - B11Ux
    
    # finish defining A2bar
    A2bar = A2 - B21Ux
    
    # \bar{B}^{11} or equivalently B11(Ubar)
    B11bar = np.array([[0,0,0,0,0,0],
                    [0,p['eta']+2*p['mu'],0,0,0,0],
                    [0,0,p['mu'],0,0,0],
                    [0,0,0,p['nu'],0,0],
                    [0,0,0,0,p['nu'],0],
                    [0,u1*(p['eta']+2*p['mu']),p['mu']*u2,0,h2*p['nu'],p['kappa']]])
    
    # \bar{B}^{21} or equivalently B21(Ubar)
    B21bar = np.array([[0,0,0,0,0,0],
                    [0,0,0,0,0,0],
                    [0,p['eta']+p['mu'],0,0,0,0],
                    [0,0,0,0,0,0],
                    [0,0,0,0,0,0],
                    [0,u2*p['eta'],p['mu']*u1,0,-h1*p['nu'],0]])
    
    # \bar{B}^{12} or equivalently B12(Ubar)
    B12bar = np.array([[0,0,0,0,0,0],
                    [0,0,p['eta']+p['mu'],0,0,0],
                    [0,0,0,0,0,0],
                    [0,0,0,0,0,0],
                    [0,0,0,0,0,0],
                    [0,u2*p['mu'],p['eta']*u1,-h2*p['nu'],0,0]])
    
    # \bar{B}^{22} or equivalently B22(Ubar)
    B22bar = np.array([[0,0,0,0,0,0],
                    [0,p['mu'],0,0,0,0],
                    [0,0,p['eta']+2*p['mu'],0,0,0],
                    [0,0,0,p['nu'],0,0],
                    [0,0,0,0,p['nu'],0],
                    [0,u1*p['mu'],(p['eta']+2*p['mu'])*u2,h1*p['nu'],0,p['kappa']]])
    
    # \bar{B}^{21}' or equivalently B21(Ubar)'
    B21bar_prime = np.array([[0,0,0,0,0,0],
                    [0,0,0,0,0,0],
                    [0,0,0,0,0,0],
                    [0,0,0,0,0,0],
                    [0,0,0,0,0,0],
                    [0,u2_x*p['eta'],p['mu']*u1_x,0,-h1_x*p['nu'],0]])
    
    A0bar_11 = A0bar[0,0]                   
    A0bar_12 = A0bar[0,1:].reshape((5,1)).T
    A0bar_21 = A0bar[1:,0].reshape((5,1))
    A0bar_22 = A0bar[1:,1:]

    A1bar_11 = A1bar[0,0]
    A1bar_12 = A1bar[0,1:].reshape((5,1)).T
    A1bar_21 = A1bar[1:,0].reshape((5,1))
    A1bar_22 = A1bar[1:,1:]

    a = 1/A1bar_11
    a_der = -u1_x/u1**2
    
    A1bar_der_12 = np.array([rho_x,0,0,0,0]).reshape((5,1)).T
    
    b21 = B11bar[1:,0].reshape((5,1))

    b11bar = B11bar[1:,1:]

    b11bar_inv = np.linalg.inv(b11bar)
    
    A2tilde = A2bar + B21bar_prime
    B2bar = B21bar + B12bar
    
    Bxi_bar = xi*B2bar
    bxibar = Bxi_bar[1:,1:]
    bxibar_21 = Bxi_bar[1:,0].reshape((5,1))
    
    Axitilde = xi*A2tilde
    
    Bxixi_bar = xi*xi*B22bar
    bxixi_bar_22 = Bxixi_bar[1:,1:]
    bxixi_bar_21 = Bxixi_bar[1:,0].reshape((5,1))
    
    Axitilde_11 = Axitilde[0,0]                      
    Axitilde_12 = Axitilde[0,1:].reshape((5,1)).T
    Axitilde_21 = Axitilde[1:,0].reshape((5,1))
    Axitilde_22 = Axitilde[1:,1:]
    
    # following the flux form paper
    m11 = r*np.array([[-lam*A0bar_11*a - 1j*Axitilde_11*a]])
    m12 = np.zeros((1,5))
    m13 = lam*(A0bar_12-A0bar_11*a*A1bar_12)-1j*Axitilde_11*a*A1bar_12+1j*Axitilde_12

    m21 = r*(-lam*A0bar_21*a-1j*Axitilde_21*a)
    m22 = np.zeros((5,5))
    m23 = lam*(A0bar_22-a*A0bar_21@A1bar_12) - 1j*a*Axitilde_21@A1bar_12 + 1j*Axitilde_22+r*bxixi_bar_22

    m31 = -r*a*b11bar_inv@A1bar_21
    m32 = r*b11bar_inv
    m33 = b11bar_inv@(A1bar_22-a*A1bar_21@A1bar_12-1j*r*bxibar)

    row1 = np.hstack((m11,m12,m13))
    row2 = np.hstack((m21,m22,m23))
    row3 = np.hstack((m31,m32,m33))
    A = np.vstack((row1,row2,row3))
        
    # multiply by u1 to tranfrom to pseudo lagrangian coordinates
    return u1*A

"""    else:
        # following Barker's code
        temp1 = -lam*A0bar_11*a - 1j*Axitilde_11*a
        m11 = r*np.array([[temp1]])
        m12 = np.zeros((1,5))
        m13 = lam*(A0bar_12-A0bar_11*a*A1bar_12)-1j*Axitilde_11*a*A1bar_12+1j*Axitilde_12

        m21 = r*(-lam*A0bar_21*a-1j*Axitilde_21*a) - r*r*bxixi_bar_21*a
        m22 = np.zeros((5,5))
        m23 = lam*(A0bar_22-a*A0bar_21@A1bar_12) - 1j*a*Axitilde_21@A1bar_12 + 1j*Axitilde_22+r*bxixi_bar_22 - r*a*bxixi_bar_21@A1bar_12

        m31 = r*b11bar_inv@(a_der*b21+r*a*b21*temp1+1j*r*a*bxibar_21-a*A1bar_21)
        m32 = r*b11bar_inv
        m33 = b11bar_inv@(a_der*b21@A1bar_12+a*b21@A1bar_der_12-1j*r*bxibar+A1bar_22+r*a*b21@m13+1j*r*a*bxibar_21@A1bar_12-a*A1bar_21@A1bar_12)

        row1 = np.hstack((m11,m12,m13))
        row2 = np.hstack((m21,m22,m23))
        row3 = np.hstack((m31,m32,m33))
        A = np.vstack((row1,row2,row3))
        
        # multiply by u1 to tranfrom to pseudo lagrangian coordinates
        return u1*A"""
    
def A_lop(x,lam,sol,p):

    # solution to profile
    if x > 0:
        rho, u1, u2, h1, h2, T = p['UR']
    if x < 0:
        rho, u1, u2, h1, h2, T = p['UL']
    
    # extra parameter
    Rnu = p['R']+p['cnu']
    xi = sol['xi']
    
    # df^0(u)
    df0 = np.array([[1,0,0,0,0,0],
                   [u1,rho,0,0,0,0],
                   [u2,0,rho,0,0,0],
                   [0,0,0,1,0,0],
                   [0,0,0,0,1,0],
                   [p['cnu']*T+(u1**2+u2**2)/2,rho*u1,rho*u2,h1,h2,p['cnu']*rho]])
    
    # df^1(u)
    df1 = np.array([[u1,rho,0,0,0,0],
                   [p['R']*T+u1**2,2*rho*u1,0,-h1,h2,p['R']*rho],
                   [u1*u2,rho*u2,rho*u1,-h2,-h1,0],
                   [0,0,0,p['alpha'],0,0],
                   [0,h2,-h1,-u2,u1,0],
                   [Rnu*u1*T+u1*(u1**2+u2**2)/2,Rnu*rho*T+h2**2+rho*(3*u1**2+u2**2)/2,-h1*h2+rho*u1*u2,-h2*u2,2*h2*u1-h1*u2,Rnu*rho*u1]])
    
    # df^2(u)
    df2 = np.array([[u2,0,rho,0,0,0],
                   [u1*u2,rho*u2,rho*u1,-h2,-h1,0],
                   [p['R']*T+u2**2,0,2*rho*u2,h1,-h2,p['R']*rho],
                   [0,-h2,h1,u2,p['alpha']-u1,0],
                   [0,0,0,0,0,0],
                   [Rnu*u2*T+u2*(u1**2+u2**2)/2,-h1*h2+rho*u1*u2, Rnu*rho*T+h1**2+rho*(3*u2**2+u1**2)/2,2*h1*u2-h2*u1,-h1*u1,Rnu*rho*u2]])
    
    df1_inv = np.linalg.inv(df1)
    I = np.eye(6)
    A = (lam*df0 + 1j*xi*df2) @ df1_inv # Barker uses df0 instead of I
    return -A # negate to find correct stability

def A_jump(x,lam,sol,p):

    # solution to profile
    if x > 0:
        rho, u1, u2, h1, h2, T = p['UR']
    if x < 0:
        rho, u1, u2, h1, h2, T = p['UL']
    
    # extra parameter
    Rnu = p['R']+p['cnu']
    xi = sol['xi']
    
    # f^0(u)
    f0 = np.array([rho,rho*u1,rho*u2,h1,h2,(h1**2+h2**2)/2+rho*(p['cnu']*T+(u1**2+u2**2)/2)])
    
    # \tilde{f} = f^2
    f2 = np.array([rho*u2,rho*u1*u2-h1*h2,p['R']*rho*T+(h1**2+h2**2)/2+rho*u2**2,p['alpha']*h2+h1*u2-h2*u1,0,Rnu*rho*u2*T+h1*(h1*u2+u1*h2)+rho*u2*(u1**2+u2**2)/2])
    
    result = lam*f0+1j*xi*f2
    return result.reshape((6,1))

def Ak(x,lam,sol,p):
    
    return None


    v = soln(x,sol)
    v = v[0]
    mu = p['mu']
    sigma = p['sigma']
    B = p['B']
    mu0 = p['mu0']

    out = np.array([
        [ v/mu, 0, -B*sigma*v, 0, 0, 0],
        [ 0, 0, mu0*sigma*v, 1/mu, 0, 0],
        [-B*v/mu, lam*v, mu0*sigma*(v**2), 0, 1/mu, 0],
        [ 0, lam*v, 0, v/mu, mu0*sigma*v, B*sigma*v],
        [ 0, 0, lam*v, lam*v, mu0*sigma*v**2 + v/mu, 0],
        [ 0, 0, 0, B*v/mu, 0, mu0*sigma*(v**2)]],dtype=np.complex)

    return out

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # parameters
    # -------------------------------------------------------------------------

    p = Struct()

    p.B = 2
    p.gamma = 5/3
    p.mu0 = 1
    p.sigma = 1
    p.vp = 0.0001
    p.mu = 1
    p.eta = -2*p.mu/3

    # -------------------------------------------------------------------------
    # dependent parameters
    # -------------------------------------------------------------------------

    p.a = p.vp**p.gamma*((1-p.vp)/(1-p.vp**p.gamma))

    # Initialising sol, the dict with solution values
    sol = Struct({
        'n': 1, # this is the dimension of the profile ode
        # we divide the domain in half to deal with the
        # non-uniqueness caused by translational invariance
        # sol.side = 1 means we are solving the profile on the interval [0,X]
        'side': 1,
        'F': profile_ode, # F is the profile ode
        'Flinear': profile_jacobian, # J is the profile ode Jacobian
        'UL': np.array([1]), # These are the endstates of the profile and its derivative at x = -infty
        'UR': np.array([p.vp]), # These are the endstates of the profile and its derivative at x = +infty
        'tol': 1e-6
        })
    sol.update({
        'phase': 0.5*(sol['UL']+sol['UR']), # this is the phase condition for the profile at x = 0
        'order': [0], # this indicates to which component the phase conditions is applied
        'stats': 'on', # this prints data and plots the profile as it is solved
        'bvp_options': {'Tol': 1e-6, 'Nmax': 200}
        })

    # Solve the profile
    p,s = profile_flux(p,sol)

    x = np.linspace(s['L'],s['R'],200)
    y = soln(x,s)

    # Plot the profile
    plt.figure("Profile")
    plt.plot(x,y)
    plt.show()

    s, e, m, c = emcset(s,'front',[2,2],'reg_adj_compound',A,Ak)

    circpnts, imagpnts, innerpnts = 30, 30, 32
    r = 1
    spread = 4
    zerodist = 10**(-4)
    # ksteps, lambda_steps = 32, 0
    preimage = semicirc(circpnts, imagpnts, c['ksteps'], r, spread, zerodist)

    out, domain = Evans_compute(preimage,c,s,p,m,e)
    out = out/out[0]
    w = reflect_image(out)
    windnum = winding_number(w)

    print('Winding Number: {:f}\n'.format(windnum))

    Evans_plot(w)

