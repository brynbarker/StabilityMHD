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


# function for initial condition
def initial(x):
    return 6*np.sin(np.pi*x)
    sol = []
    for _ in x:
        sol.append(2*np.maximum(.2-np.abs(_-.5),0))
    return np.array(sol)

# flux form functions for heat equation
def f0(u):
    return u

def f1(u):
    return np.zeros_like(u)
    try:
        n = len(u)
    except:
        n = 1
    y = np.zeros(n)
    return y

def b(u):
    try:
        n = len(u)
    except:
        n = 1
    y = np.eye(n)
    return k*y

# boundary conditions
left = [(True,0)]
right = [(True,0)]
bnds = np.array([[1, 0, 1, 0]])

# discretization scheme
nt = 50
nx = 5

# solve using Crank Nicholson
solCN = solveCN(initial,f0,f1,b,nt,nx,bnds)
sol = solve(initial,f0,f1,b,nt,nx,bnds)
lin_sol = lsolve(initial,f0,f1,b,nt,nx,bnds)

# store solutions
UL = lin_sol.U
UCN = solCN.U
U = sol.U

# compare results
print('Linear matches nonlinear solver: {}'.format(np.allclose(UL,U)))
print('CN matches Linear solver: {}'.format(np.allclose(UCN,UL)))
#print(U.round(3))

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
    curve2.set_data(x,true[i])
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
