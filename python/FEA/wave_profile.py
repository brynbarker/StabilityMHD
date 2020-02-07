#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:02:46 2017

@author: Taylor Paskett

This file is a translation into PYTHON of the folder bin_profile from the
MATLAB version of STABLAB
"""
import numpy as np
import scipy.linalg as scipylin
from scipy.integrate import solve_bvp

def profile_flux(p,s,s_old=None):
    """
    # Solves the profile for the flux formulation.
    # If s.s_old exists, then this
    # is used for the initial guess. Otherwise, a tanh
    # solution is used for an initial guess. Left and right
    # numerical infinity are expanded as needed to assure
    # the end state error is within s.tol, though s.L = s.R
    # in this program, which may not be necessary. Uneeded
    # mesh points of the solution are removed to speed up
    # interpolation and continuation routines.
    #
    # The user must include in the input structures the following:
    #
    # s.phase - a vector of phase conditions
    # s.order - a vector indicating the order in which the phase
    #               conditions should be applied.
    # s.F - a function handle to the profile, e.g. s.F = @F,  F(x,y,s,p) = ...
    # s.UL, UR - end states of the n-r equations that need to be solved in
    #                   flux formulation. UL at - infinity, UR at + infinity
    # s.n = n-r in flux formulation (number of profile equations to integrate)
    #
    # Optional input:
    # s.tol - profile endstate maxim absolute error, (defaults to 1e-4)
    # s.R_max - maximum allowed interval length on right (defaults to 100)
    # s.L_max - maximum allowed interval length on left (defaults to 100)
    """

    #------------------------------------------------------------
    # End states
    #------------------------------------------------------------

    # end state tolerance
    if 'tol' not in s:
        s['tol'] = 1e-4

    # default for maximum value of R
    if 'R_max' not in s:
        s['R_max'] = 100

    # default for maximum value of L
    if 'L_max' not in s:
        s['L_max'] = 100

    # numerical infinity
    s['I'] = 1
    # profile solved on right half domain
    s['side'] = 1
    # array for right hand side
    s['rarray'] = np.array(range(s['n']))
    # array for left hand side
    s['larray'] = np.array(range(s['n'],2*s['n']))

    # bvp solver projections
    A_min = s['Flinear'](s['UL'],p)
    s['LM'] = scipylin.orth(projection(A_min,-1,0)) # Removed .T inside orth
    A_plus = s['Flinear'](s['UR'],p)
    s['LP'] = scipylin.orth(projection(A_plus,1,0)) # Removed .T inside orth

    # This gives us the number of phase conditions needed
    s['n_phs'] = s['n']-s['LM'].shape[1]-s['LP'].shape[1]

    if s['n_phs'] < 1:
        print("Eigenvalues at negative infinity: ")
        print(np.linalg.eigvals(A_min))
        print("Eigenvalues at positive infinity: ")
        print(np.linalg.eigvals(A_plus))
        raise ValueError("profile_flux.m does not solve undercompressive profiles")

    # bvp tolerances
    if 'bvp_options' not in s:
        s['bvp_options'] = {'Tol': 1e-5,'Nmax': 2000}
    elif 'Nmax' not in s['bvp_options']:
       s['bvp_options']['Nmax'] = 2000
    elif 'Tol' not in s['bvp_options']:
       s['bvp_options']['Tol'] = 1e-5
    # tol and max_nodes
    # positive numerical infinity

    # -----------------------------------------------------------
    # solve the profile initially
    # -----------------------------------------------------------

    p,s = profile(p,s,s_old)

    # -----------------------------------------------------------
    # take out extra mesh points
    # -----------------------------------------------------------

    # stride = how many points to take out of solution to
    # minimize points in final solution.
    stride = 3
    s['stride'] = stride
    s_old = s
    mesh = len(s_old['sol'].x)
    mesh_old = mesh+1
    while mesh < mesh_old:
        p,s = profile(p,s,s_old)
        s_old = s
        mesh_old = mesh
        mesh = len(s_old['sol'].x)

    s['stride'] = stride

    return p,s

def profile(p,s,s_old):

    #--------------------------------------------------------------------------
    # provide initial guess
    #--------------------------------------------------------------------------

    if isinstance(s_old, dict):
        if 'solver' in s_old:
            if s_old['solver'] == 'bvp':
                pre_guess = lambda x: continuation_guess(x,s_old,s)
            else:
                pre_guess = lambda x: ode_to_bvp_guess(x,s_old,s)
                s['stride'] = 3
        else:
            pre_guess = lambda x: continuation_guess(x,s_old,s)

        stride = s_old['stride']

        x_dom = s_old['sol'].x[::stride].copy() # do I need the .copy() ?

        if (len(s_old['sol'].x)-1) % stride != 0:
           x_dom[-1] = s_old['sol'].x[-1]

        s['I'] = s_old['I']
        s['L']= s_old['L']
        s['R'] = s_old['R']

    else:

        s['I'] = 1
        if 'R' not in s:
            s['R'] = 5

        s['L'] = -s['R']

        pre_guess = lambda x: guess(x,s)
        x_dom = np.linspace(0,1,30)

    #--------------------------------------------------------------------------
    # convergence to endstates tolerance
    #--------------------------------------------------------------------------

    err = s['tol'] + 1
    while err > s['tol']:
        pre_bc = lambda x,y: bc(x,y,s)
        pre_ode = lambda x,y: double_F(x,y,s,p)

        initGuess = np.array([pre_guess(x) for x in x_dom],dtype=np.complex).T

        s['sol'] = solve_bvp(pre_ode,pre_bc,x_dom,initGuess,
                             tol=s['bvp_options']['Tol'],
                             max_nodes=s['bvp_options']['Nmax'])

        err1 = np.max(np.abs(s['sol'].y[s['rarray'],-1] - s['UR']))
        err2 = np.max(np.abs(s['sol'].y[s['larray'],-1] - s['UL']))
        err  = max(err1,err2)

        if 'stats' in s:
            if s['stats'] == 'on':
                print("Profile boundary error: ",err)

        if err > s['tol']:
           s_old = s
        if err1 > s['tol']:
           s['R'] *= 1.1#*s['R']
           s['L'] = -s['R']
        if err2 > s['tol']:
           s['L'] *= 1.1#*s.L;
           s['R'] = -s['L']
        if abs(s['L']) > s['L_max']:
           raise ValueError("""Could not meet specified tolerance in profile solver
                without exceeding the maximum allowed value of negative infinity.""")
        if abs(s['R']) > s['R_max']:
           raise ValueError("""Could not meet specified tolerance in profile solver
                without exceeding the maximum allowed value of positive infinity.""")
        if err > s['tol']:
            pre_guess = lambda x: continuation_guess(x,s_old,s)
            x_dom = s_old['sol'].x

    return p,s

def guess(x,s):
    # guess using tanh solution
    a = 0.5*(s['UL']+s['UR'])
    c = 0.5*(s['UL']-s['UR'])

    outVector = np.concatenate([a-c*np.tanh((s['R']/s['I'])*x),
                               a-c*np.tanh((s['L']/s['I'])*x)])
    return outVector

def bc(ya,yb,s):
    # Boundary conditions. We split the problem in half and reflect onto
    # the right side
    outVector = np.concatenate([
        ya[s['rarray']]-ya[s['larray']], #  matching conditions
        np.dot(s['LM'].T , (yb[s['larray']] - s['UL'])), # projection at - infinity
        np.dot(s['LP'].T , (yb[s['rarray']] - s['UR'])), # projection at + infinity
        ya[s['order'][0:s['n_phs']]]-s['phase'][s['order'][:s['n_phs']]] # Phase conditions
        ])

    return outVector

def ode_to_bvp_guess(x,s_old,s):
    out = np.array([[ deval(s_old['sol'],(s.R/s.I)*x)],
                    [ deval(s_old['sol'],(s.L/s.I)*x)]],dtype=np.complex)
    return out

def continuation_guess(x,s_old,s_new):
    """
    # Ouput gives initial guess for boundary value solver at x where s_old is
    # the standard stablab structure s for the previously solved boundary value
    # solution and s_new is that for the solution being solved. If v is the
    # solution corresponding to s_new and y is the solution corresponding to
    # s_old, continuation_guess yields as output v=a*y+b done componenet wise
    # to allow phase conditions to be specified and so v matches its end
    # states.
    """

    y = deval(x,s_old['sol'])
    out = np.zeros((len(y)),dtype=np.complex)

    # coefficients for the guess for the new function v \approx a*y+b done
    # componentswise. Positive infinity corresponds to the first column of the
    # coefficeint matrices, and negative infinity corresponds to the second
    # column of coefficient matrices.
    a = np.zeros((len(s_old['rarray']),2),dtype=np.complex)
    b = np.zeros((len(s_old['rarray']),2),dtype=np.complex)

    # find scaling coefficients
    for j in range(len(s_old['rarray'])):

        # determine if the phase condition should be specified for the jth
        # component
        specify_phase = False
        for k in range(len(s_new['order'])):
            if j == s_new['order'][k]:
                specify_phase = True
                phase_index = j # Changed to j from k

        # determine coefficients based on type
        vminus = s_new['UL'][j]
        vplus  = s_new['UR'][j]
        yminus = s_old['UL'][j]
        yplus  = s_old['UR'][j]
        if specify_phase: # case where the phase condition is specified
            vnot = s_new['phase'][phase_index]
            ynot = s_old['phase'][phase_index]
            vec = np.dot(np.linalg.inv(np.array([[yplus,1,0,0],
                    [ynot,1,0,0],[0,0,yminus,1],[0,0,ynot,1]])),
                    np.array([[vplus],[vnot],[vminus],[vnot]],
                    dtype=np.complex))
            a[j,0] = vec[0]
            b[j,0] = vec[1]
            a[j,1] = vec[2]
            b[j,1] = vec[3]
        else: # case where the phase condition is not specified
            if yplus == yminus:
                a[j,0]=1
                b[j,0]=0
                a[j,1]=1
                b[j,1]=0
            else:
                vec = np.dot(np.linalg.inv(np.array([[yplus,1],
                        [yminus,1]],dtype=np.complex)),
                        np.array([[vplus],[vminus]],dtype=np.complex))
                a[j,0] = vec[0]
                b[j,0] = vec[1]
                a[j,1] = vec[0]
                b[j,1] = vec[1]

    # make the affine transformation, v=a*y+b
    out[s_old['rarray']] = a[:,0]*y[s_old['rarray']]+b[:,0]
    out[s_old['larray']] = a[:,1]*y[s_old['larray']]+b[:,1]

    return out

def double_F(x,y,s,p,otherargs=None):
    """
    # out = double_F(x,y,s,p)
    #
    # Returns the split domain for the ode given in the function F.
    #
    # Input "x" and "y" are provided by the ode solver.Note that s.rarray
    # should be [1,2,...,k] and s.larray should be [k+1,k+2,...,2k]. See
    # STABLAB documentation for more information about the structure s.
    """
    if otherargs is not None:
        out = np.vstack([(s['R']/s['I'])*s['F']((s['R']/s['I'])*x,
                                            y[s['rarray'],:],s,p,otherargs),
                         (s['L']/s['I'])*s['F']((s['L']/s['I'])*x,
                                            y[s['larray'],:],s,p,otherargs)])
    else:
        out = np.vstack([(s['R']/s['I'])*s['F']((s['R']/s['I'])*x,
                                                        y[s['rarray'],:],s,p),
                         (s['L']/s['I'])*s['F']((s['L']/s['I'])*x,
                                                        y[s['larray'],:],s,p)])

    return out

def projection(matrix,posneg,eps):
    """
    """
    D,R = np.linalg.eig(matrix)
    L = np.linalg.inv(R)
    P = np.zeros(R.shape,dtype=np.complex)

    if posneg == 1:
        index = np.where(np.real(D) > eps)
    elif posneg == -1:
        index = np.where(np.real(D) < eps)
    elif posneg == 0:
        index = np.where(np.abs(np.real(D)) < eps)

    for j in index:
        P = P + np.dot(R[:,j],L[j,:])

    Q = np.concatenate([np.dot(P,R[:,j]) for j in index])
    out = np.concatenate([P,Q],axis=1)
    return P

def deval(x,solStruct):
    """
    Takes two inputs, x and solStruct, and returns the y values corresponding
    to the x values
    """
    return solStruct.sol(x).real

def soln(xArray,s):
    """
    # out = soln(x,s)
    #
    # Returns the solution of bvp problem where the domain was split in half
    #
    # Input "x" is the value where the solution is evaluated and "s" is a
    # stucture described in the STABLAB documenation
    """
    if isinstance(xArray,(float, int)):
        outVector = np.zeros((1,s['n']))
        if xArray < 0:
            xArray = s['side']*s['I']/s['L']*xArray
            temp = deval(xArray,s['sol'])
            outVector = temp[s['larray']]
        else:
            xArray = s['side']*s['I']/s['R']*xArray
            temp = deval(xArray,s['sol'])
            outVector = temp[s['rarray']]
    else:
        outVector = np.zeros((len(xArray),s['n']))
        for index,x in enumerate(xArray):
            if x < 0:
                x = s['side']*s['I']/s['L']*x
                temp = deval(x,s['sol'])
                outVector[index] = temp[s['larray']]
            else:
                x = s['side']*s['I']/s['R']*x
                temp = deval(x,s['sol'])
                outVector[index] = temp[s['rarray']]

    return outVector.real