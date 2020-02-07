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

# function for initial condition
alpha,beta = 3,4
def initial(x):
    return alpha*x+beta

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
    y = np.zeros((n,n))
    return y

# boundary conditions
left = [(True,0)]
right = [(True,0)]
bnds = np.array([[0, 0, 0, 0]])

# discretization scheme
nt = 50
nx = 10
sol = solve(initial,f0,f1,b,nt,nx,bnds)

# store solutions
U = sol.U

# compare results

# define solution domain
x = np.linspace(0,1,sol.nx+1)
t = np.linspace(0,1,sol.nt+1)

# compare against true solution
true = np.copy(U)
y = lambda x,t: (alpha*x+beta) / (alpha*t+1)
for i in range(sol.nt+1):
    true[i] = y(x,t[i])

# display error for nonlinear solution
print('Error:',np.linalg.norm(true-U))

filename = 'inviscid_burgers.mp4'
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
