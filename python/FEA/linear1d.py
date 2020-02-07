import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root, fsolve
import scipy.linalg as la
from mesh1d import mesh

class solve():
    def __init__(self,init,f0,f1,b,nt,nx,bnds):
        self.init = init
        self.f0 = f0
        self.f1 = f1
        self.b = b
        self.nt = nt
        self.nx = nx
        self.dt = 1./nt
        self.bnds = bnds

        # initial data
        self._setup()

        # solve
        self._solve()

    def _setup(self):
        self.dom = np.linspace(0,1,self.nx+1)
        U0 = self.init(self.dom)
        self.numeq = int(len(U0.flatten())/(self.nx+1))

        self.mesh = mesh(self.nx)
        self.elements = self.mesh.elements

        self.U = np.zeros((self.nt+1,self.numeq*(self.nx+1)))
        self.U[0] = U0

    def _solve(self):

        num_eqs, nx, dt = self.numeq, self.nx, self.dt
        w = [5/9, 8/9, 5/9]

        # function that solves the same system but assuming linearity (used KU = F)

        def func2(D_):
            # initialize K and F
            K = np.zeros((nx+1,nx+1))
            F = np.zeros(nx+1)
            
            # Make sure D_ is correct shape
            D_ = D_.reshape((num_eqs,nx+1))
            
            for e in self.elements:
                
                # get gauss quad vals on the element
                vals, derivs, globs = e.quad_vals()
                
                for j in range(2):
                    fapprox = 0
                    for i in range(2):
                        approx = 0
                        for k in range(3):
                            # evaluate shape func at xi
                            v = vals[j][k]
                            dv = derivs[j][k]

                            # evaluate U approx at xi
                            U = vals[i][k]
                            U_ = D_[:,e.id+i]*vals[i][k]
                            dU = derivs[i][k]

                            # compute the integral
                            val = self.f0(U)*v/dt  - self.f1(U)*dv + (self.b(U)*dU)*dv
                            fval = self.f0(U_)*v/dt
                            approx += val*w[k]
                            fapprox += fval*w[k]
                        
                        # add to the appropriate equation in the system
                        K[e.id+j,e.id+i] += approx
                    F[e.id+j] += fapprox
                    
            # incorporate boundary data
            K[-1] *= 0
            K[:,-1] *= 0
            K[-1,-1] = 1
            F[-1] = 0
            
            K[0] *= 0
            K[:,0] *= 0
            K[0,0] = 1
            F[0] = 0
            
            # solve the system
            d = la.solve(K,F)
            return d

        for i in range(1,self.nt+1):
            sol = func2( self.U[i-1])
            self.U[i] = sol
