import numpy as np


class element():
    def __init__( self, id, n1, n2 ):
        self.id = id           # global element id
        self.ind1 = id
        self.ind2 = id+1
        self.n1 = n1           # loc of left node
        self.n2 = n2           # loc of right node
        self.m = 1/(n2-n1)     # shape func slope

    def shape_vals(self, x):
        # returns the value of
        # each shape function at x

        v1 = self.m * (self.n1-x) + 1
        v2 = self.m * (x-self.n2) + 1
        return v1, v2

    def shape_d_vals(self,x):
        # returns the derivative of
        # each shape function at x

        v1 = -self.m * np.ones_like(x)
        v2 = self.m * np.ones_like(x)
        return v1, v2

    def quad_vals(self):
        # returns the value and the
        # derivative of each shape
        # function at the 3 quad pts

        # quad points
        xi =[ -np.sqrt(3/5), 0, np.sqrt(3/5) ]

        # maps local quad points to global points
        to_glob = lambda x: (x+1)/(self.m*2)+self.n1
        globs = np.array([to_glob(i) for i in xi])

        # get shape func vals
        vals = self.shape_vals(globs)

        # get shape func derivs
        derivs = self.shape_d_vals(globs)

        # return vals, derivs, and global points
        # note row1 of vals is phi_1 at xis
        return np.array(vals), np.array(derivs), globs
