# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:48:26 2017

@author: Wadlo
"""

from sympy import Matrix
from sympy import Symbol
from sympy import sympify
from sympy import Function
from stablab import finite_difference as finiteDiff
from stablab.finite_difference_code import approximate as fd
from stablab.finite_difference_code.writing_functions import writeFunctionsFile
from stablab.finite_difference_code.writing_functions import writeRunnerFile
import math
import numpy as np

def createJacobianEquations(NUM_OF_EQUATIONS, stencil, n, j, equations):
    jacobianEquations = NUM_OF_EQUATIONS*[0]
    for eq in range(NUM_OF_EQUATIONS):
        jacobianEquations[eq] = equations.jacobian(Matrix(stencil[eq][n+1]))
    return jacobianEquations

def prepareStrings(inputList, derivatives):
    #Replace the derivatives.
    for listIndex in range(len(inputList)):
        inputList[listIndex] = inputList[listIndex].replace("_xx", "_x_x")
        inputList[listIndex] = inputList[listIndex].replace("_tt", "_t_t")
        for derivative in derivatives:
            inputList[listIndex] = inputList[listIndex].replace("_"+derivative, ".diff("+derivative+")")
    return inputList

def defineFunctions(unknownsString, knownEquations):
    derivatives = [Symbol('t'),Symbol('x')]
    unknowns = list(unknownsString)
    myDictionary = {}
    #Define the unknowns as functions of x and t
    for index in range(len(unknowns)):
        currentString = unknowns[index]
        unknowns[index] = Function(unknowns[index])(*derivatives)
        myDictionary[currentString] = unknowns[index]
    #Define known equation as functions of x and t
    for index in range(len(knownEquations)):
        currentString = splitEquals(knownEquations[index])[0]
        myDictionary[currentString] = Function(currentString)(*derivatives)
    #Define K,H,t,x as symbols in the dictionary.
    myDictionary['K'] = Symbol('K')
    myDictionary['H'] = Symbol('H')
    myDictionary['t'] = derivatives[0]
    myDictionary['x'] = derivatives[1]
    return myDictionary

def splitEquals(inputString):
    outputList = []
    equalsIndex = inputString.index('=')
    outputList.append(inputString[0:equalsIndex])
    outputList[0] = outputList[0].replace(' ', '')
    outputList.append(" (" + inputString[equalsIndex+1:len(inputString)] + ") ")
    return outputList
    
def oneSideEquation(inputString):
    sidesList = splitEquals(inputString)
    return sidesList[0] + "-" + sidesList[1]

def substituteKnownEquations(equations, knownEquations, myDictionary):  
    for repeat in range(20):
        for equationIndex in range(len(equations)):
            for knownEquation in knownEquations:
                leftSide, rightSide = splitEquals(knownEquation)
                #print(leftSide, " = ",rightSide)
                leftSide = myDictionary[leftSide]
                rightSide = sympify(rightSide, locals=myDictionary)
                equations[equationIndex] = equations[equationIndex].subs(leftSide, rightSide)

def createStencil(unknowns):
    NUM_OF_EQUATIONS = len(unknowns)
    STENCIL_ROWS = 2
    STENCIL_COLS = 3
    stencil = [[STENCIL_COLS*[0] for row in range(STENCIL_ROWS)] for eq in range(NUM_OF_EQUATIONS)]
    for eq in range(NUM_OF_EQUATIONS):
        for row in range(STENCIL_ROWS):
            for col in range(STENCIL_COLS):
                if row == 0: insideString = 'n,'
                else: insideString = 'n+1,'
                if col == 0: insideString += 'j-1'
                elif col == 1: insideString += 'j'
                else: insideString += 'j+1'
                stencil[eq][row][col] = Symbol(str(unknowns[eq])+'['+insideString+']')
    #print(stencil)
    return stencil
    
def simplifyCalculus(equations):
    for equationIndex in range(len(equations)):
        equations[equationIndex] = equations[equationIndex].doit()

def getParameters(equations, stencil, myDictionary):
    parameters = []
    for equation in equations:
        parameters = set(parameters).union(equation.free_symbols)
    parameters = parameters-set([myDictionary['H'],myDictionary['K']])
    for stencilRow in stencil:
        parameters = parameters-set(stencilRow[0])
        parameters = parameters-set(stencilRow[1])

    return list(parameters)

def initializeSolver():
    #Define p as a dictionary and initialize default values as the heat equation.
    p = {}
    p["pde"] = ["U_t = U_x_x"]
    p["pdeFunctions"] = ['U']
    p["helperEquations"] = []
    p["finiteDifference"] = finiteDiff.crankNicolson
    
    return p

def plot(inMatrix, name):
    True
   
def initMatrixWithFunction(T, L, T_POINTS, X_POINTS, f):
    #Make some helper variables    
    X_STEPS = X_POINTS - 1
    DELTA_X = 2.0*L/X_STEPS
    
    #Create matrix and value of x at t=0.
    a= np.zeros((T_POINTS, X_POINTS))
    for x in range(0,X_POINTS):
        a[0,x] = f(-L + x*DELTA_X)
    
    #Create the value of t at x= {0,X_POINTS-1}
    a[0:T_POINTS,X_POINTS-1] = a[0,X_POINTS-1]    
    a[0:T_POINTS,0] = a[0,0]
    return a

    
def createInitialVector(equationType, p):
    T = p["T"]
    L = p["L"]
    T_POINTS = p["pdeTPoints"]
    X_POINTS = p["pdeXPoints"]
    
    if equationType == "tanh":
       
        f = lambda x:  1.5-np.tanh(x/2.0)
        return initMatrixWithFunction(T, L, T_POINTS, X_POINTS, f)
    else:
        return np.zeros((T_POINTS, X_POINTS))
