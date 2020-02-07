import sympy as sy
import numpy as np
import copy
from itertools import combinations

class wedge:
    """
    wedge is a class which can be used to make arrays of objects (usu. sympy
    Symbols or equations) which also have a corresponding basis in terms of
    wedge products, and to then multiply, add, or take the wedge products of
    these wedge arrays.
    """
    class wedgeterm:
        """
        wedgeterm is a simple structure which holds the
        values for a single term in a wedge object. This class is not intended
        for use by the user.
        """
        def __init__(self, coefficient, basisTuple):
            self.coeff = sy.sympify(coefficient) # the coefficients are symbolic
            self.basis = list(basisTuple)
            self.sort_basis()

        def __str__(self):
            return "%s , %s" % (self.coeff, self.basis)

        def __mul__(self, operand):
            self.coeff *= operand
            return self

        def sort_basis(self):
            if self.basis == sorted(self.basis):
                pass
            else:
                # This is a simple sorting algorithm which makes use of the
                #  property:    (e_2 ^ e_1) == -(e_1 ^ e_2)
                i = 0
                while i < len(self.basis)-1:
                    if self.basis[i] > self.basis[i+1]:
                        temp = self.basis[i]
                        self.basis[i] = self.basis[i+1]
                        self.basis[i+1] = temp
                        self.coeff *= -1
                        i = 0
                    else:
                        i += 1

        def wedgeproduct(self, w):
            """  Take the wedge product with another wedgeterm. """
            out = wedge.wedgeterm(self.coeff*w.coeff,[*self.basis,*w.basis])
            return out

    def __init__(self,coefficient=None,basisTuple=None):
        """ The initializer can take 0 inputs (creating a wedge object with
         nothing in it), or 2 inputs (which will create an object with a single
         wedgeterm inside it). """
        if coefficient is None and basisTuple is None:
            self.terms = []
        elif coefficient is None or basisTuple is None:
            raise ValueError("""Instantiating a wedge object requires either
            two parameters or no parameters.""")
        else:
            w = self.wedgeterm(coefficient, basisTuple)
            self.terms = [w]

    def __add__(self, otherWedge):
        """ Add two wedge objects together. This simply puts their list of
        terms together, returning a new wedge object with both terms in one
        list. If two terms have the same basis, these terms' coefficients are
        added together. Otherwise, two new terms result. """
        new = copy.deepcopy(self)
        for otherterm in otherWedge.terms:
            new.addTerm(otherterm.coeff, otherterm.basis)
        return new

    def __getitem__(self, index):
        """
        Returns the wedgeterm at the given index.
        """
        return self.terms[index]

    def __iter__(self):
        """
        This allows for iteration through a wedge object. Iterating through a
        wedge object will return each of the wedgeterms in the list of terms.
        """
        self._i = 0
        return self

    def __next__(self):
        """
        Part of the necessary functionality for iterating through a wedge object.
        """
        self._i += 1
        if self._i > len(self.terms): raise StopIteration
        return self.terms[self._i-1]

    def __len__(self):
        """
        Returns the number of wedgeterms in the current wedge object.
        """
        return len(self.terms)

    def __contains__(self, coeff):
        """
        Returns whether or not the coefficient coeff is contained in a
        wedgeterm which is contained by this object.
        """
        for term in self.terms:
            if term.coeff - coeff == 0:
                return True
        return False

    def __mul__(self, operand):
        """
        Scalar multiplication.
        """
        new = copy.deepcopy(self)
        for term in new.terms:
            term *= operand
        return new

    def __str__(self):
        """
        Print the wedge object as a string.
        """
        returnString = str([str(term) for term in self.terms])
        return returnString

    def addTerm(self, coefficient, basisTuple):
        """
        This method is used to add another term to the current wedge object.
        If the new term has the same basis as an existing term, then the wedge
        instance will have the same number of terms as before calling this
        method. Otherwise, a new term will be added.
        """
        for term in self.terms:
            if basisTuple == term.basis:
                term.coeff += coefficient
                return
        self.terms.append(self.wedgeterm(coefficient,basisTuple))
        self.sort()

    def remove(self, index):
        """
        This method is used to remove the term shown by the given index, which
        may be an integer, or a list of integers.
        """
        try:
            for idx in index:
                self.terms.pop(idx)
        except:
            self.terms.pop(index)

    def __matmul__(self, otherWedge): #NOTE: should I overload matmul or use a separate function instead?
        """
        Take the wedge product with another wedge object.
        """
        out = wedge()
        for term1 in self.terms:
            for term2 in otherWedge.terms:
                newTerm = term1.wedgeproduct(term2)
                out.addTerm(newTerm.coeff, newTerm.basis)
        if len(out) > 0:
            out.sort()
        return out

    def sort(self):
        """
        This method sorts the current wedge according to basis, and removes any
        terms which are equal to zero because they have repeated wedge terms.
        """
        # This portion sorts the terms according to their bases
        listOfBasesLength = [len(term.basis) for term in self.terms]
        maxBasisLength = max(listOfBasesLength)
        currKey = lambda x: tuple([x.basis[k] for k in range(maxBasisLength)
                                                          if k < len(x.basis)])
        self.terms = sorted(self.terms, key=currKey)

        # This portion removes terms which are equal to zero
        toRemove = []
        for i,term in enumerate(self.terms):
            if len(term.basis) > len(set(term.basis)):
                toRemove.append(i)
        self.remove(toRemove)


def makeCompound(inMat,k):
    """
    This function takes a square matrix of type numpy.ndarray or sympy.Matrix,
    and turns it into the corresponding compound matrix of degree k.
    """
    rowLen, colLen = np.shape(inMat)
    if rowLen != colLen:
        raise ValueError("makeCompound can only be used on a square matrix.")
        return

    def makeBasisWedge(n,k=2):
        """
        This function is a helper function provided to make a wedge object with all
        of the bases needed for an n-dimensional matrix raised to the given k. It
        is not intended for direct use by the user.
        """
        out = wedge()
        bases = combinations(range(1,n+1),k)
        for basis in bases:
            out += wedge(0, basis)
        return out

    def liftMatrix(inMat,k,basis):#,bases,size):
        """
        This function takes a matrix and lifts it to a given basis. It is not
        intended for direct use by the user.
        """
        # We take the transpose here because it is easier to iterate through
        #  the rows than it is to iterate through the columns.
        A = np.array(inMat).T
        outMat = np.zeros(size,dtype=object)
        listToBeMultiplied = [[wedge(1,[num,]) for num in basis] for _ in basis]
        pre_out = wedge()
        # These loops fill in the values of listToBeMultiplied so that each of
        #  the lists in listToBeMultiplied contains the wedge objects which will
        #  have their wedge products taken, IE:
        #         [ [(Ae_1),e_2,e_3], [e_1,(Ae_2),e_3], [e_1,e_2,(Ae_3)] ]
        for i,num in enumerate(basis):
            wedges = wedge()
            for j,coeff in enumerate(A[num-1]):
                wedges += wedge(coeff,[j+1,])
            listToBeMultiplied[i][i] = wedges

        # These loops take the wedge products of the wedge objects in
        #  listToBeMultiplied and write their values to pre_out
        for i,toMultiply in enumerate(listToBeMultiplied):
            temp = toMultiply[0]
            for index in range(len(basis)-1):
                temp @= toMultiply[index+1] # @ here is the wedge product
            pre_out += temp

        # adding bases to pre_out ensures that pre_out contains all necessary
        #  bases, even the ones that have a coefficient of zero
        pre_out += bases # this is not a typo. this uses bases from line 245
        outMat = [val.coeff for val in pre_out]
        return outMat

    # This is the beginning of the code for makeCompound
    bases = makeBasisWedge(np.shape(inMat)[0],k)
    size = len(bases)
    outMat = np.zeros((size,size),dtype=object)
    for i,w in enumerate(bases):
        outMat[i] = liftMatrix(inMat,k,w.basis)

    # Since we took the transpose in liftMatrix, we now take the transpose again,
    #  returning the correct matrix.
    return outMat.T
