# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:49:51 2017

@author: Jalen Morgan, Taylor Paskett
"""

import numpy as np
import sympy
from stablab.finite_difference_code import pde
from sympy import Matrix
from stablab.finite_difference_code import approximate

"""Used for both pdes and odes"""
def newtonSolve(initialGuess, newtonFunction, newtonJacobian, p=[], MAX_ERROR = 1e-8, TIMEOUT = 45, printIterations = True):
    count = 0    
    
    #Make the initial guess for the coefficient List
    inVector = initialGuess
    outVector = newtonFunction(inVector, p)
    
    #print(max(max(map(abs,outVector))))
    #Loop through Newton's method.    
    while max(map(abs,outVector))-MAX_ERROR > 0:
    #while True:
        count += 1
        
        A = newtonJacobian(inVector, p)
        b = (outVector - np.dot(A,inVector))
        inVector = np.linalg.solve(A,-b)
        outVector = newtonFunction(inVector, p)
        
        #Print the progress
        if printIterations == True: print(count, end='')
        if count == TIMEOUT:
            #print("should be zero:", outVector)
            return (inVector)  
    return (inVector)

def stringToSympy(inputString, myDict):
    #Replace _x, _t, etc. with derivative.
    for index in range(len(inputString)):
        inputString[index] = inputString[index].replace("_x", ".diff(x)")
    symbolOde = sympy.sympify(inputString, locals=myDict)
    return symbolOde

def sympyGetParameters(inputSympy, myDict):
    parameterSet = set()
    for index in range(len(inputSympy)):
        symbolSet = inputSympy[index].free_symbols
        parameterSet = parameterSet | symbolSet
    parameterSet -= set(myDict.values())
    return list(parameterSet)
    
def shouldRewriteFile(fileName, stringToCompare):
    try:
        fileToReplace = open(fileName,'r')
        fileString = ''.join(fileToReplace.readlines())
        if fileString == stringToCompare:
            print("************Not rewriting************")
            return False
        else:
            print("**************Rewriting**************")
            return True
    except FileNotFoundError:
        print("****************Writing***************")
        return True

#Sets the default parameters so the user can simply run with defaults.
def init():    
    p = {
        "name": "", 
        "newtonError": 1e-5,
        
        #Ode defaults
        "odeXList": [],
        "odeGuess": [],
        "ode": [],  
        "odeJacobian": ["estimate"],
        "odeUnknowns": [],
        "odeChebPoints": 10,
        "odeGraphPoints": 100,
        "odeBC": [],
        "odeJacobianBC": ["estimate"],
        "L": 1,
        
        #Pde defaults
        "pde": [],
        "pdeInitial": "tanh",
        "pdeSubstitute": [],
        "pdeUnknowns": [],
        "pdeFD": crankNicholson,
        "pdeXPoints": 35,
        "pdeTPoints": 35,
	"pdeInitialValueFiles": [],
        "T": 1
        }
    return p    
    
def generateFiniteDifferenceConservation(f0,f1,g,B,unknowns, **kwargs):
    #Input an equation of type 'f0(u)_t + f1(u)_x + g(u) = (B(u)u_x)_x'
    f0 = toList(f0)
    f1 = toList(f1)
    g = toList(g)
    B = toDoubleList(B)

    #Assure inputs are of the correct form.
    if not (len(B) == len(B[0])):
        raise ValueError("B must be a square matrix")
    if not (len(f0) == len(f1)):
        raise ValueError("f0 and f1 must be the same size.")
    if not (len(f0) == len(g)):
        raise ValueError("f0 and g must be the same size.")
    if not (len(f0) == len(B)):
        raise ValueError("f0 and B[0] must be same size")

    unknowns = toList(unknowns)
    pdeString = []
    for i in range(len(f0)):
        bterm = ''
        for j in range(len(B[0])):
            if not j == 0:
                bterm += " + "
            bterm += str(B[i][j])+'*'+str(unknowns[j])+'_xx + ' + str(unknowns[j])+'_x'+'*'+str(B[i][j])+'_x'
        #bterm = 'U_xx'
        #bterm = 'U_xx'
        pdeString.append('('+str(f0[i])+')_t + ('+str(f1[i])+')_x + '+str(g[i])+' - ('+bterm+')')
    print(pdeString)
    #print('generateFiniteDifference('+str(pdeString)+','+str(unknowns)+','+str(**kwargs)+')')
    return generateFiniteDifference(pdeString, unknowns, **kwargs)

def generateFiniteDifference(pdeString, unknowns, **kwargs):
    #Convert from a list of coefficients to a list of points in X_POINTS so 
    #They can be used as an initial guess for the newton solve of the PDE.
    myDict = {"knownEquations": [], "fd": crankNicholson}
    myDict.update(kwargs)

    unknowns = toList(unknowns)
    equations = toList(pdeString)
    #print(str(unknowns))
    #print(str(equations))
    #print(str(kwargs))

    knownEquations = myDict["knownEquations"]
    fdMethod = myDict["fd"]
    
    #Prepare the strings and define symbol functions of x and t
    pde.prepareStrings(equations, ('t', 'x'))    
    myDictionary = pde.defineFunctions(unknowns,knownEquations)
    
    #Sympify the equations and plug them in to the pde.    
    equations = pde.sympify(equations, locals = myDictionary)
    pde.substituteKnownEquations(equations, knownEquations, myDictionary)

    print(equations)
    pde.simplifyCalculus(equations)
    
    #Plug in finite differences and create jacobian
    stencil = pde.createStencil(unknowns)
    finiteDifference(equations, myDictionary, stencil, unknowns, fdMethod)
    #substituteFiniteDifference(equations, myDictionary, stencil, unknowns)
    parameters = pde.getParameters(equations, stencil, myDictionary)
    jacobianEquations = pde.createJacobianEquations(len(equations), stencil, 0, 1, Matrix(equations))
    
    #Create the folder and fill it.
    import os
    if not os.path.exists("__generated__"):
        os.makedirs("__generated__")
    
    #Write both the runner file and the Functions file.
    fileName = fdMethod.__name__+ "_functions.py"
    #writeRunnerFile(fileName, unknowns, parameters)
    pde.writeFunctionsFile( "__generated__/" + fileName, unknowns, equations, jacobianEquations, parameters)
    import importlib
    functionsFile = importlib.import_module("__generated__."+fileName.replace(".py",""))

    return [functionsFile.f, functionsFile.createJacobian]
   
def getInitialCondition(inVector, inFunction):
    output = np.zeros(len(inVector))
    for i in range(len(output)):
        output[i] = inFunction(inVector[i])
    return output

def toDoubleList(inputList):
    #if isinstance(inputList, list):
    #    if isInstance(inputList[0], list):
    #        return inputList
    #    else:
    #        return [inputList]
    #else:
    #    return [[inputList]]
    if isinstance(inputList, str) or isinstance(inputList, float) or isinstance(inputList, int):
        return [[inputList]]
    elif isinstance(inputList[0], str) or isinstance(inputList[0], float) or isinstance(inputList[0], int):
        return [inputList]
    else:
        return inputList
    
def toTripleList(inputList):     
    if isinstance(inputList, str) or isinstance(inputList, float) or isinstance(inputList, int):
        return [[[inputList]]]
    elif isinstance(inputList[0], str) or isinstance(inputList[0], float) or isinstance(inputList[0], int):
        return [[inputList]]
    elif isinstance(inputList[0][0], str) or isinstance(inputList[0][0], float) or isinstance(inputList[0][0], int):
        return [inputList]
    else:
        return inputList
    
def jacobianWithBoundary(jacobian, leftBound, rightBound, matrices, n, K, H, P):
    output = jacobian(matrices, n, K, H, P)
    leftBound = toDoubleList(leftBound(matrices, n))
    rightBound = toDoubleList(rightBound(matrices, n))
    output = [output]
    for eq in range(len(leftBound)):
        output[eq][ 0:len(leftBound[0]),0] = leftBound[eq]
        output[eq][ -len(rightBound[0]):len(output[0]),-1] = rightBound[eq]
    return output[0]
    
def functionWithBoundary(f, leftBound, rightBound, matrices, P, K, H, n):       
    output = f(matrices, P, K, H, n)
    leftList = leftBound(matrices, n)
    rightList = rightBound(matrices, n)
    numPoints = len(matrices[0])
    for i in range(len(matrices)):
        output[i*numPoints] = leftList[i]
        output[(i+1)*numPoints-1] = rightList[i]

    return output

def evolve(xPoints, tPoints, lBound, rBound, t0, myFunctions, **kwargs):
    myDict = {"p":[], "fd":"crankNicholson", "MAX_ERROR":.01}
    myDict.update(kwargs)
        
    f = lambda matrices, time, K, H, P: functionWithBoundary(myFunctions[0], lBound[0], rBound[0], matrices, time, K, H, P)
    jac = lambda matrices, time, K, H, P: jacobianWithBoundary(myFunctions[1], lBound[1], rBound[1], matrices, time, K, H, P)
    
    t0 = toDoubleList(t0)
        
    numVars = len(t0)
    matrixList = []
    for i in range(numVars):
        if True:
            currArray = np.zeros((len(tPoints),len(xPoints)))
            currArray[0] = t0[i]
            matrixList.append(currArray)
    #print("Len",len(matrixList))
    approximate.solveSystem(matrixList, xPoints, tPoints, myDict["p"], myDict["MAX_ERROR"], f,jac)
    return matrixList

def toList(inputType):
    if isinstance(inputType, list):
        return inputType
    else:
        return [inputType]
#    if isinstance(inputType, str):
#        return [inputType]
#    else:
#        return inputType

def graph(unknown, matrixList):
    approximate.plotMatrix(matrixList, unknown)
        
def getBoundaryFunctions(pdeString, pdeVariables):
    #Write the Function
    outputString = """def lBoundFunction(UIn, n):
    """
    for i in range(len(pdeVariables)):
        outputString += pdeVariables[i]
        outputString += " = UIn[" +str(i) + """]
    """
    outputString += """return """
    pdeStringOutput = toList(pdeString)
    for i in range(len(pdeStringOutput)):
        pdeStringOutput[i] = pdeStringOutput[i].replace("(","[")
        pdeStringOutput[i] = pdeStringOutput[i].replace(")","]")
    
    #Write the Derivative.
    outputString += str(pdeStringOutput).replace("'","")
    outputString += """
    def lBoundDerivative(UIn, n):
"""
    for i in range(len(pdeVariables)):
        outputString += pdeVariables[i]
        outputString += " = UIn[" +str(i) + """]
    """
    
    #print(outputString)

def finiteDifference(eq, myDictionary, stencil, unknowns, fdFunction):
    n = 0 
    j = 1
    t = myDictionary['t']
    x = myDictionary['x']
    h = myDictionary['H']
    k = myDictionary['K']
    
    #Loop through the equations and the unknowns.
    for eqNum in range(len(eq)):
        for i in range(len(unknowns)):
            unknown = myDictionary[unknowns[i]]
            
            (Uxx, Ux, Ut, U) = fdFunction(stencil[i], n, j, k, h)
            eq[eqNum] = eq[eqNum].subs(unknown.diff(x).diff(x),Uxx)            
            eq[eqNum] = eq[eqNum].subs(unknown.diff(x),Ux)
            eq[eqNum] = eq[eqNum].subs(unknown.diff(t),Ut)
            eq[eqNum] = eq[eqNum].subs(unknown,U)
            
def crankNicholson(U, n, j, k, h):
    Uxx = ((U[n+1][j+1] - 2*U[n+1][j] + U[n+1][j-1])/(h**2) + 
          (U[n][j+1] - 2*U[n][j] + U[n][j-1])/(h**2))/2
    Ux = ((U[n+1][j+1] - U[n+1][j-1])/(2*h) + 
          (U[n][j+1] - U[n][j-1])/(2*h))/2
    Ut = (U[n+1][j] - U[n][j])/(k)
    UOut = (U[n+1][j]+U[n+1][j])/2
    return (Uxx, Ux, Ut, UOut)

def explicit(U, n, j, k, h):
    Uxx = (U[n][j+1]-2*U[n][j]+U[n][j-1])/(h**2)
    Ux = (U[n][j+1]-U[n][j-1])/(2*h)
    Ut = (U[n+1][j] - U[n][j])/(k)
    UOut = U[n][j]
    return (Uxx, Ux, Ut, UOut)

def implicit(U, n, j, k, h):
    Uxx = (U[n+1][j+1]-2*U[n+1][j]+U[n+1][j-1])/(h**2)
    Ux = (U[n+1][j+1]-U[n+1][j-1])/(2*h)
    Ut = (U[n+1][j] - U[n][j])/(k)
    UOut = U[n+1][j]
    return (Uxx, Ux, Ut, UOut)

        
    
