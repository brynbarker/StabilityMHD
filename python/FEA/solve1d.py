import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root, fsolve
import scipy.linalg as la
from mesh1d import mesh

class solve():
    def __init__(self,init,f0,f1,b,nt,nx,bnds,Df0=None):
        self.init = init
        self.f0 = f0
        self.f1 = f1
        self.b = b
        self.nt = nt
        self.nx = nx
        self.dt = 1./nt
        self.bnds = bnds
        self.Df0 = Df0

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

        # function that we want to find the root of
        def func(D, D_):
            res = np.zeros((num_eqs,nx+1))
            D = D.reshape((num_eqs,nx+1))
            D_ = D_.reshape((num_eqs,nx+1))

            for e in self.elements:
                # get gauss quad vals on the element
                vals, derivs, globs = e.quad_vals()

                for j in range(2):
                    approx = 0
                    for k in range(3):
                        # evaluate shape func at xi
                        v = vals[j][k]
                        dv = derivs[j][k]

                        # evaluate U approx at xi
                        U = D[:,e.ind1]*vals[0][k] + D[:,e.ind2]*vals[1][k]
                        U_ = D_[:,e.ind1]*vals[0][k] + D_[:,e.ind2]*vals[1][k]
                        dU = D[:,e.ind1]*derivs[0][k] + D[:,e.ind2]*derivs[1][k]

                        # compute the integral
                        if self.Df0 is not None:
                            tder = self.Df0(U)@(U-U_)
                        else:
                            tder = self.f0(U)-self.f0(U_)
                        #val = self.f0(U)*v/dt - self.f0(U_)*v/dt - self.f1(U)*dv + (self.b(U)@dU)*dv
                        val = tder*v/dt - self.f1(U)*dv + (self.b(U)@dU)*dv
                        
                        
                        approx += val*w[k]

                    # add to the appropriate equation in the system
                    res[:,e.id+j] += approx

#            left = (self.bnds[:,0] == True)
#            right = (self.bnds[:,2] == True)

#            res[left,0] = self.bnds[left,1] - D[left,0]
#            res[~left,0] = D[~left,0] - D[~left,1]

#            res[right,-1] = self.bnds[right,3] - D[right,-1]
#            res[~right,-1] = D[~right,-1] - D[~right,-2]

#             res[:,0] *= 0
#             res[:,-1] *= 0
#             print(D[:,0],D[:,-1])
            return res.flatten()



        for i in range(1,self.nt+1):
            sol = root(func, self.U[i-1], self.U[i-1],tol=1e-12)
#            print(la.norm(func(sol.x,self.U[i-1])));
            print(la.norm(func(sol.x,self.U[i-1])));
            
            if not sol.success:
                print(i)
                print(la.norm(func(sol.x,self.U[i-1])));
            if i % int(self.nt/10) == 0:
                print('{}/{}'.format(i,self.nt))
#            print(sol.success)
            self.U[i] = sol.x
