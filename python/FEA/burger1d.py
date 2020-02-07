import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root, fsolve
import scipy.linalg as la
from time import time
from tqdm import tqdm

import matplotlib.animation as animation

from solve1d import solve
from linear1d import solve as lsolve
from crank_nick import solveCN

k = .05

import sympy as sp
from sympy.utilities.lambdify import lambdify
nu = .07
xval,tval = sp.symbols('xval tval')
phi = sp.exp(-(xval-4*tval)**2/(4*nu*(tval+1)))+sp.exp(-(xval-4*tval-2*np.pi)**2/(4*nu*(tval+1)))
phi_der = phi.diff(xval)
u = -2*nu*phi_der/phi+4
ufunc = lambdify((xval,tval),u)


nx = 11
nt = 11
xgrid,h = np.linspace(0,2*np.pi,nx,retstep=True)
t_final = nt*h*nu

u = np.asarray([ufunc(0,x) for x in xgrid])
plt.plot(xgrid,u)
plt.title('initial condition')
plt.show()

true_sol = np.zeros((nt,nx))
tgrid = np.linspace(0,t_final,nt)
for i,t in enumerate(tgrid):
    true_sol[i] = np.asarray([ufunc(t,x) for x in xgrid])

#animation.writer = animation.writers['ffmpeg']

# Define our update function that will get our solution at each time t and plot it against x
#def update(i):
#    curve.set_data(xgrid,true_sol[i])
#    return curve

# Create our animation base figure
#fig = plt.figure()

# display results
#ax = fig.add_subplot(1,1,1)
#ax.set_ylim((0,4.5))
#ax.set_xlim((0,2*np.pi))
#curve, = plt.plot([],[], color='k')
#plt.xlabel("x")
#plt.title('Finite Element Solution')

# save animation
#ani = animation.FuncAnimation(fig,update,frames=nt,interval=100)
#ani.save('burgers.mp4')
#plt.show()


# function for initial condition
def initial(x):
    if isinstance(x,np.ndarray):
        return np.asarray([ufunc(0,_) for _ in x])
    else:
        return ufunc(0,x)

# flux form functions for heat equation
def f0(u):
    return u

def f1(u):
    return .5*u**2

def b(u):
    try:
        n = len(u)
    except:
        n = 1
    y = np.eye(n)
    return nu*y

# boundary conditions
left = [(True,0)]
right = [(True,0)]
bnds = np.array([[0, 0, 0, 0]])

# discretization scheme

sol = solve(initial,f0,f1,b,nt-1,nx-1,bnds)

# store solutions
U = sol.U

for i in range(nt):
    plt.plot(xgrid,true_sol[i],'r')
    plt.plot(xgrid,U[i],'k.')
plt.show()
print(np.linalg.norm(U-true_sol))
print(np.allclose(U,true_sol))
raise ValueError('end')

# compare results

# define solution domain
x = np.linspace(0,1,sol.nx+1)
t = np.linspace(0,1,sol.nt+1)

# compare against true solution
true = np.copy(U)
y = lambda x,t: 6*np.sin(np.pi*x)*np.exp(-k*np.pi**2*t) 
for i in range(sol.nt+1):
    true[i] = y(x,t[i])

# display error for nonlinear solution
print('Error:',np.linalg.norm(true-U))

filename = 'ani_heat.mp4'
# Create our animation writer
animation.writer = animation.writers['ffmpeg']

# Define our update function that will get our solution at each time t and plot it against x
def update(i):
    curve1.set_data(x,U[i])
    curve2.set_data(x,true_sol[i])
    return curve1, curve2

# Create our animation base figure
plt.ioff()
fig = plt.figure()

# display results
ax = fig.add_subplot(1,1,1)
ax.set_ylim((-.2,6.5))
ax.set_xlim((0,1))
curve1, = plt.plot([],[], marker='o',color='r')
curve2, = plt.plot([],[], color='k')
plt.xlabel("x")
plt.title('Finite Element Solution')

# save animation
ani = animation.FuncAnimation(fig,update,frames=nt+1,interval=100)
ani.save(filename)
plt.close(fig)
