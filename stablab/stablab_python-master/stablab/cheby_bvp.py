#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 23:07:56 2017

@author: Taylor Paskett
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as scipylin

def bvpinit(A, mesh, degree, Proj_L, B_L, Proj_R, B_R):
    # Begin with some error checking to make sure that everything will work

    # Check that A and the projection conditions are callable. If not (i.e.
    #  they are arrays), then make them callable so that the code in solve can
    #  treat them as if they are callable objects. This way, the user can
    #  either pass in arrays or functions
    if not callable(A):
        A_copy = A.copy()
        def A(x): return A_copy
    if not callable(Proj_L):
        Proj_L_copy = Proj_L.copy()
        def Proj_L(x): return Proj_L_copy
    if not callable(Proj_R):
        Proj_R_copy = Proj_R.copy()
        def Proj_R(x): return Proj_R_copy
    if not callable(B_L):
        B_L_copy = B_L.copy()
        def B_L(x): return B_L_copy
    if not callable(B_R):
        B_R_copy = B_R.copy()
        def B_R(x): return B_R_copy

    # Get our system dimensions, and the number of intervals
    sys_dim = len(A(mesh[0]))
    num_intervals = len(mesh)-1

    # Check that mesh is in increasing order
    if np.all(mesh != sorted(mesh)):
        raise ValueError("To initialize cheby_bvp, mesh must be in order from"+
            " least to greatest.")

    # Check that A returns a square shape
    dimA = A(1).shape
    if (len(dimA) != 2) or (dimA[0] != dimA[1]):
        raise ValueError("To initialize cheby_bvp, A must be square, but "+
            "has shape "+str(dimA))

    # Check that Proj_L and Proj_R return arrays that have 2 dimensions
    dimPR = Proj_R(mesh[-1]).shape
    dimPL = Proj_L(mesh[0]).shape
    if len(dimPR) != 2 or len(dimPL) != 2:
        raise ValueError("To initialize cheby_bvp, Proj_L and Proj_R must"+
            " have 2 dimensions. However, Proj_L has "+str(len(dimPL))+
            " dimensions, and Proj_R has "+str(len(dimPR))+" dimensions.")

    # Check that the number of boundary conditions do not exceed the dimensions
    #  of our system
    dimL = B_L(mesh[0]).shape[0]
    dimR = B_R(mesh[-1]).shape[0]
    if dimL + dimR != sys_dim:
        raise ValueError("""Cannot initialize cheby_bvp because the system's
            boundary conditions are overdetermined or underdetermined.
            You must have len(B_L)+len(B_R) == sys_dim, where sys_dim is
            A.shape[0] (e.g. if A is a 4x4, len(B_L)+len(B_R) = 4) Currently,
            you have len(B_L) = """+str(dimL)+", len(B_R) = "+str(dimR)+""",
            and sys_dim = """+str(sys_dim)+".")

    # Initialize d, the dict with subinterval information
    d = Struct(
        { 'A':       A,
          'Proj_L':  Proj_L,
          'B_L':     B_L,
          'Proj_R':  Proj_R,
          'B_R':     B_R,
          'N':       np.zeros((num_intervals), dtype=np.intp),  #Is this field necessary?
          'a':       np.zeros((num_intervals)),
          'b':       np.zeros((num_intervals)),
          'theta':   np.zeros((num_intervals, degree)),
          'nodes_0': np.zeros((num_intervals, degree)), #Is this field necessary?
          'nodes':   np.zeros((num_intervals, degree)),
          'Tcf':     np.zeros((num_intervals, degree, degree),dtype=np.complex),
          'Ta':      np.zeros((num_intervals, degree),dtype=np.complex),
          'Tb':      np.zeros((num_intervals, degree),dtype=np.complex),
          'Ta_x':    np.zeros((num_intervals, degree),dtype=np.complex),
          'Tb_x':    np.zeros((num_intervals, degree),dtype=np.complex),
          'T':       np.zeros((num_intervals, degree, degree),dtype=np.complex),
          'T_x':     np.zeros((num_intervals, degree, degree),dtype=np.complex),
          'T_xcf':   np.zeros((num_intervals, degree, degree),dtype=np.complex),
          'cf':      np.zeros((num_intervals, sys_dim, degree),dtype=np.complex),
          'cfx':     np.zeros((num_intervals, sys_dim, degree),dtype=np.complex),
          'err':     np.zeros((num_intervals)),
          'x':       [],
          'y':       [],
          'dim':     sys_dim
        })

    # Filling in subinterval values [a,b]
    d['a'] = mesh[:-1]
    d['b'] = mesh[1:]

    # Degree of polynomials
    d['N'].fill(degree)

    out = cheby_bvp(d)

    return out

class Struct(dict):
    """
    Struct inherits from dict and adds this functionality:
        Instead of accessing the keys of struct by typing
        struct['key'], one may instead type struct.key.
    These two options will do exactly the same thing. A new
    Struct object can also be created with a dict as an input
    parameter, and the resulting Struct object will have the
    same data members as the dict passed to it.
    """
    def __init__(self,inpt={}):
        super(Struct,self).__init__(inpt)

    def __getattr__(self, name):
        return self.__getitem__(name)

    def __setattr__(self,name,value):
        self.__setitem__(name,value)

class cheby_bvp(Struct):
    """
    The cheby_bvp class inherits from Struct. When a user calls bvpinit, a
    cheby_bvp object is returned which will have the methods below, as well as
    the data fields in bvpinit, as its attributes.
    """
    def __init__(self,startStruct):
        super(cheby_bvp,self).__init__(startStruct)

    def solve(self, max_err=None, xSize=25):
        """
        solve takes a cheby_bvp object and solves the boundary value problem
        that is contained in it.
        """
        #d = dict(d) # copies d so that init could be reused if wanted; should it be copied?
        sys_dim = len(self['A'](self['a'][0]))
        num_intervals = len(self['a'])

        # total number of nodes we solve for; iterates through each interval
        equ_dim = 0;

        # interval specific objects
        for j in range(num_intervals):

            # degree of polynomial
            degreeRange = np.array(range(self['N'][j]))

            equ_dim = equ_dim + self['N'][j]

            # Chebyshev nodes
            self['theta'][j] = (degreeRange + 0.5)*np.pi/self['N'][j]

            # nodes in [-1,1]
            self['nodes_0'][j] = np.cos(self['theta'][j])

            # nodes in [a,b]
            self['nodes'][j] = 0.5*(self['a'][j]+self['b'][j])+0.5*(self['a'][j]-self['b'][j])*self['nodes_0'][j]

            # Transformation to get Chebyshev coefficients
            Id2 = (2/self['N'][j])*np.eye(self['N'][j])
            Id2[0,0] = Id2[0,0]/2
            self['Tcf'][j] = Id2.dot(np.cos(np.outer(self['theta'][j], degreeRange)).T)

            # Chebyshev polynomials at the end points
            self['Ta'][j] = np.cos(np.zeros((1,self['N'][j])))
            self['Tb'][j] = np.cos(np.pi*degreeRange)

            # Derivative of Chebyshev polynomials at end points
            self['Ta_x'][j] = np.square(degreeRange)
            self['Ta_x'][j] = (2/(self['a'][j]-self['b'][j]))*self['Ta_x'][j]
            self['Tb_x'][j] = self['Ta_x'][j]
            self['Tb_x'][j][::2] = -1*self['Tb_x'][j][::2]

            # Matrix to evaluate Chebyshev polynomials
            self['T'][j] = np.cos(np.outer(self['theta'][j], degreeRange));

            # derivative of d(j).T
            self['T_x'][j] = ((2/(self['a'][j]-self['b'][j])) * (np.tile(degreeRange,(self['N'][j],1))
                            * (np.sin(np.outer(self['theta'][j], degreeRange))))
                            / np.sin(np.outer(self['theta'][j], np.ones((self['N'][j])))))

            # matrix to obtain chebyshev coefficients of derivative
            self['T_xcf'][j] = np.dot(self['Tcf'][j], self['T_x'][j])

            self['Tx'] = np.dot(self['T'],self['T_xcf'])


        # This creates two matrices used to solve for the coefficients of the system
        # We will insert function values and derivative conditions
        equ_dim = equ_dim*sys_dim
        C = np.zeros((equ_dim,equ_dim),dtype=np.complex)
        B = np.zeros((equ_dim),dtype=np.complex)

        # Fill with Left B.C. values
        row_L = slice(0,np.shape(self['B_L'](self['nodes'][0]))[0])
        for i in range(np.shape(self['Proj_L'](self['nodes'][0]))[1]):
            begin_L = i*self['N'][0]
            end_L = (i+1)*self['N'][0]
            col_L = slice(begin_L,end_L)
            C[row_L,col_L] = np.outer(self['Proj_L'](self['nodes'][0]).T[i], self['Ta'][0])
        B[row_L] = self['B_L'](self['nodes'][0])


        #Fill with Right B.C. values
        row_R  = slice((num_intervals-1)*self['N'][-1]*sys_dim+(self['N'][-1]-1)*sys_dim + sys_dim-len(self['B_R'](self['nodes'][-1])),
                       (num_intervals-1)*self['N'][-1]*sys_dim+(self['N'][-1]-1)*sys_dim + sys_dim)
        for i in range(1,np.shape(self['Proj_R'](self['nodes'][-1]))[1]+1):
            begin_R = (num_intervals-1)*self['N'][-1]*sys_dim+(i-1)*self['N'][-1] # begin_R = (num_intervals-1)*self['N'][-1]*sys_dim + sys_dim*self['N'][-1] - (i)*self['N'][-1] #
            end_R = (num_intervals-1)*self['N'][-1]*sys_dim+(i)*self['N'][-1] # end_R = (num_intervals-1)*self['N'][-1]*sys_dim + sys_dim*self['N'][-1] - (i-1)*self['N'][-1]
            col_R = slice(begin_R,end_R)
            C[row_R,col_R] = np.outer(self['Proj_R'](self['nodes'][-1]).T[i-1], self['Tb'][-1])
        B[row_R] = self['B_R'](self['nodes'][-1])


        #Fill with left interval values
        for poly in range(1,num_intervals):
            for dim in range(sys_dim):
                row = poly*self['N'][poly]*sys_dim + dim
                begin1 = sys_dim*self['N'][poly]*(poly-1) + (dim)*self['N'][poly]
                end1   = sys_dim*self['N'][poly]*(poly-1) + (dim)*self['N'][poly] + self['N'][poly]
                begin2 = sys_dim*self['N'][poly]*(poly-1) + (sys_dim)*self['N'][poly] + (dim-1)*self['N'][poly] + self['N'][poly]
                end2   = sys_dim*self['N'][poly]*(poly-1) + (sys_dim)*self['N'][poly] + (dim-1)*self['N'][poly] + 2*self['N'][poly]
                col1 = slice(begin1,end1)
                col2 = slice(begin2,end2)
                C[row,col1] = self['Tb_x'][poly-1]  # Here we are imposing a condition so that Tb_x == Ta_x
                C[row,col2] = -self['Ta_x'][poly]


        #Fill with right interval values
        for poly in range(num_intervals-1):
            for dim in range(sys_dim):
                row  = poly*self['N'][poly]*sys_dim + (self['N'][poly]-1)*sys_dim + dim
                begin1 = sys_dim*self['N'][poly]*poly + (dim)*self['N'][poly]
                end1   = sys_dim*self['N'][poly]*poly + (dim)*self['N'][poly] + self['N'][poly]
                begin2 = sys_dim*self['N'][poly]*poly + (sys_dim)*self['N'][poly] + (dim-1)*self['N'][poly] + self['N'][poly]
                end2   = sys_dim*self['N'][poly]*poly + (sys_dim)*self['N'][poly] + (dim-1)*self['N'][poly] + 2*self['N'][poly]
                col1 = slice(begin1,end1)
                col2 = slice(begin2,end2)
                C[row, col1] = self['Tb'][poly]
                C[row, col2] = -self['Ta'][poly+1]

        #Fill with typical derivative conditions
        for poly in range(num_intervals):
            # This deals with the edge case of the far left interval
            if poly == 0 and len(self['B_L'](self['nodes'][0])) < sys_dim:
                nodeRange = range(self['N'][poly]-1)
                # This if statement will deal with the case where a single interval is taken
                if poly == num_intervals-1 and len(self['B_R'](self['nodes'][-1])) < sys_dim:
                    nodeRange = range(self['N'][poly])
            # This deals with the edge case of the far right interval
            elif poly == num_intervals-1 and len(self['B_R'](self['nodes'][-1])) < sys_dim:
                nodeRange = range(1,self['N'][poly])
            # Typical Cases
            else:
                nodeRange = range(1,self['N'][poly]-1)
            for node in nodeRange:
                # Far left interval edge case
                if node == 0:
                    dimRange = range(sys_dim-1,len(self['B_L'](self['nodes'][0]))-1,-1)
                # Far right interval edge case
                elif node == self['N'][poly]-1:
                    dimRange = range(sys_dim - len(self['B_R'](self['nodes'][0])))
                # Typical Cases
                else:
                    dimRange = range(sys_dim)

                for dim in dimRange:
                    row  = poly*self['N'][poly]*sys_dim + node*sys_dim + dim
                    begin = sys_dim*self['N'][poly]*poly
                    end = sys_dim*self['N'][poly]*poly + sys_dim*self['N'][poly]
                    col = slice(begin,end)
                    yVals = np.zeros((sys_dim, sys_dim*self['N'][poly]), dtype=np.complex)
                    derivConditions = np.zeros((sys_dim*self['N'][poly]), dtype=np.complex)
                    for i in range(sys_dim):
                        yVals[i,self['N'][poly]*i:self['N'][poly]*i+self['N'][poly]] = self['T'][poly][node,:]

                    derivConditions[self['N'][poly]*dim:self['N'][poly]*dim+self['N'][poly]] = self['T_x'][poly][node,:]
                    yValConditions = -self['A'](self['nodes'][poly][node])
                    yValConditions = np.dot(yValConditions[dim,:],yVals)

                    C[row, col] = derivConditions + yValConditions

        # Solves for coefficients and writes them to key 'cf' in d
        pre_cf = np.linalg.solve(C,B)
        self['cf'] = np.reshape(pre_cf,(num_intervals,sys_dim,self['N'][0]))
        # 'cfx' are the coefficients which, when multiplied by T, give us the derivative values
        self['cfx'] = np.inner(self['cf'], self['T_xcf'][0])
        if max_err is not None:
            self.reduceError(max_err, xSize)

    def reduceError(self, max_err, xSize):
        """
        This function reduces the error on a given solver so that it is within
        the tolerance specified by the user. It does so by solving repeatedly,
        subdividing into smaller and smaller intervals as needed.
        """
        # Sets the values of x to be the same number of points in each interval
        x = [np.linspace(a,b,xSize) for a,b in zip(self['a'],self['b'])]

        # Calls getError using the given x values
        x,y = self.getError(np.concatenate(x))

        index = [i+1 for i,err in enumerate(self['err']) if err > max_err*1e-2]
        if np.all(self['err'] <= max_err):
            index = []

        if len(index) > 0:
            mesh = np.zeros(len(self['a'])+1)
            mesh[:-1] = self['a']
            mesh[-1] = self['b'][-1]
            insertVals = [(mesh[i]+mesh[i-1])/2 for i in index]
            mesh = np.insert(mesh,index,insertVals)
            d = bvpinit(self['A'], mesh, self['N'][0], self['Proj_L'], self['B_L'], self['Proj_R'], self['B_R'])
            d.solve(max_err, xSize)
            self = d

    def getError(self, x):
        """
        getError calculates the residual error in the solution which has been
        calculated for the cheby_bvp object on the x values which are given as
        the input parameter x.
        """
        # This line takes the array in x, and creates a new array,
        # xIntervals, which has the same values, but separated into individual
        # arrays (1 array for each interval)
        xIntervals = [np.unique(x[[xVal>=a and xVal<=b for xVal in x]]) for a,b in zip(self['a'],self['b'])]

        xtilde = [(2*xVals - (a+b))/(a-b) for a,b,xVals in zip(self['a'],self['b'],xIntervals)]

        # This block will fix round-off error that can occur when calculating xtilde
        for i,xVals in enumerate(xtilde):
            if xVals[0] > 1:
                xtilde[i][0] = 1
            if xVals[-1] < -1:
                xtilde[i][-1] = -1

        theta = [np.arccos(xVals) for xVals in xtilde]

        # Creates T, the array which holds the values of our chebyshev polynomials
        # evaluated at theta (ie T = cos(n(theta)) )
        T = [np.cos(np.outer(t, np.array(range(N)))) for N,t in zip(self['N'],theta)]

        y = [np.zeros((len(self['cf'][0]),len(t)),dtype=np.complex) for t in theta]
        for i,interval in enumerate(self['cf']):
            for j,cf in enumerate(interval):
                y[i][j] = np.dot(T[i],cf)

        # Sets the keys self['y'] and self['x'] so the user can plot them
        self['y'] = np.concatenate(y,1)
        self['x'] = np.concatenate(xIntervals)

        # Initializes y_x, the derivative of y
        y_x = [np.zeros((len(self['cf'][0]),len(t)),dtype=np.complex) for t in theta]

        # Fills y_x with appropriate values
        for i,interval in enumerate(self['cfx']):
            for j,cfx in enumerate(interval):
                y_x[i][j] = np.dot(T[i],cfx)

        # Calculates Residual Error
        for i,interval in enumerate(self['err']):
            res_err = 0
            for j,xVal in enumerate(xIntervals[i]):
                res_err = max(res_err, np.nanmax(np.abs(y_x[i][:,j]-np.dot(self['A'](xVal),y[i][:,j]))))
            self['err'][i] = res_err
        print()
        #print("Error:", self['err'])
        print("MAX ERROR:", np.max(self['err']))

        return (xIntervals,y)

    def deval(self, x):
        """
        deval takes the array x which is given as input and calculates the
        solution values of the bvp corresponding to these x values, writing
        them to the key 'y' in the cheby_bvp object.
        """
        # This line takes the array in x, and creates a new array,
        # xIntervals, which has the same values, but separated into individual
        # arrays (1 array for each interval)
        xIntervals = [np.unique(x[[xVal>=a and xVal<=b for xVal in x]]) for a,b in zip(self['a'],self['b'])]

        # Change of variable so that in each interval, xtilde is in [-1,1]
        xtilde = [(2*xVals - (a+b))/(a-b) for a,b,xVals in zip(self['a'],self['b'],xIntervals)]

        # This block will fix round-off error that can occur when calculating xtilde
        for i,xVals in enumerate(xtilde):
            if xVals[0] > 1:
                xtilde[i][0] = 1
            if xVals[-1] < -1:
                xtilde[i][-1] = -1

        # Another change of variable, to theta from xtilde
        theta = [np.arccos(xVals) for xVals in xtilde]

        # Creates T, the array which holds the values of our chebyshev polynomials
        # evaluated at theta (ie T = cos(n(theta)) )
        T = [np.cos(np.outer(t, np.array(range(N)))) for N,t in zip(self['N'],theta)]

        y = [np.zeros((len(self['cf'][0]),len(t)),dtype=np.complex) for t in theta]
        for i,interval in enumerate(self['cf']):
            for j,cf in enumerate(interval):
                y[i][j] = np.dot(T[i],cf)

        # Sets the keys self['y'] and self['x'] so the user can plot them, or get/reduce error
        self['y'] = np.concatenate(y,1)
        self['x'] = np.concatenate(xIntervals)

        return self['y']


    def plot(self, plotName=""):
        if self['y'] == []:
            print("Cannot plot--please call deval first")
            return

        plt.figure(plotName)
        for y in self['y']:
            plt.plot(self['x'],y)

        plt.show()

    def savePlot(self, fileName="chebPlot.png"):
        import matplotlib
        matplotlib.use('Agg')
        if self['y'] == []:
            print("Cannot plot--please call deval first")
            return

        fig = plt.figure(fileName)
        for y in self['y']:
            plt.plot(self['x'],y)

        fig.savefig('./'+str(fileName))
