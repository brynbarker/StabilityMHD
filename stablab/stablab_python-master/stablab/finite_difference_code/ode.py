# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:44:55 2017

@author: Jalen Morgan, Taylor Paskett
"""

import numpy as np
import sympy
from matplotlib import pyplot as plt
import importlib
import textwrap

"""Files used to solve Odes."""
#getChebyshev returns the CHEB_POLYth chebyshev polynomial evaluated at x.  It
#will be returned as a float.
def getCheb(CHEB_POLY, x, p):
    L = p["L"]
    if CHEB_POLY == 0: return 1.0
    elif CHEB_POLY == 1: return x/L
    else:
        prevCheby = 1.0
        currCheby = x/L
        for i in range(2,CHEB_POLY+1):
            tempCheby = currCheby
            currCheby = currCheby*2.0*x/L - prevCheby
            prevCheby = tempCheby
            
    return float(currCheby)

#getChebyshevDeriv calculates and returns (as a float) the CHEB_POLYth 
#chebyshev polynomial evaluated at x.
def getChebDeriv(CHEB_POLY, x, p):
    L =  p["L"]
    n = CHEB_POLY
    theta = np.arccos(x/L)
    return ((n)*np.sin(n*theta)/np.sin(theta))/L


def getChebPoints(CHEB_POINTS, p):
    L = p["L"]
    outList = []
    for i in range(CHEB_POINTS):
        outList.append(L*np.cos(np.pi*(2*i+1)/(2*CHEB_POINTS)))
    return outList

def evalOde(x, coefficients, p):
    solution = 0
    for i in range(len(coefficients)):
        solution += coefficients[i]*getCheb(i,x,p)
    return solution

def evalOdeDerivative(x, coefficients, p):
    L = p["L"]
    solution = 0
    theta = np.arccos(x/L)
    for n in range(len(coefficients)):
        solution += (coefficients[n]*n*np.sin(n*theta)/np.sin(theta))/L
    return solution

def plugInFunctions(functionString):
    functionString = functionString.replace("x", "xList[row]")
    functionString = functionString.replace("Derivative(U(xList[row], t), xList[row])", "evalOdeDerivative(xList[row], inVector)")
    functionString = functionString.replace("Derivative(U(xList[row], t), t, xList[row])", "getChebDeriv(col, xList[row])")
    functionString = functionString.replace("Derivative(U(xList[row], t), t)", "getCheb(col, xList[row])")
    functionString = functionString.replace("U(xList[row], t)","evalOde(xList[row],inVector)")
    return functionString

def makeFunctionUserFriendly(functionString):
    functionString = functionString.replace("U(x, t)", "y")
    functionString = functionString.replace("Derivative(y, x)", "0")
    functionString = functionString.replace("y*Derivative(y, t)", "y")
    functionString = functionString.replace("Derivative(y, t)", "1") # ?
    return functionString

def getNewtonStrings(odeSympy, bcSympy, p):
    t = p["t"]
    
    #Create the newton Header
    newtonHeaderString = textwrap.dedent("""
        from ode import evalOde
        from ode import evalOdeDerivative
        from ode import getCheb
        from ode import getChebDeriv
        import numpy as np
        """)

    #Create the newton Function
    newtonFunctionString = textwrap.dedent("""
        def newtonFunction(x,y,p):
            ode = """)
    odeString = str(odeSympy[0])
    odeString = makeFunctionUserFriendly(odeString)
    newtonFunctionString += odeString
    newtonFunctionString += """
    return [ode] \n"""
    
    #Create the Jacobian Function
    newtonJacobianString = textwrap.dedent("""
        def newtonJacobian(x, y, p):
            jacobian = """)
    newtonJacobianString += makeFunctionUserFriendly(str(odeSympy[0].diff(t)))
    newtonJacobianString += """
    return [jacobian] \n"""   
    
    #Create the BC Function
    newtonBCString = textwrap.dedent("""
        def newtonBC(y,p):
            BC = """)
    newtonBCString += str(bcSympy[0]).replace("U","y")
    newtonBCString += """
    return BC \n"""
                                      
                                      
    #Create the BC derivative Function
    newtonBCPrimeString = textwrap.dedent("""
        def newtonBCPrime(y,p):
            BC = """)
    newtonBCPrimeString += str(bcSympy[0]).replace("U","y")
    newtonBCPrimeString += """
    return BC \n """                                                   
                                           
    printStrings = newtonHeaderString + newtonFunctionString + newtonJacobianString + newtonBCString + newtonBCPrimeString
    return printStrings

def approximateDerivative(x, y, p):
    newtonFunction = p["ode"]
    DELTA = 1e-5
    approxDerivative = []
    for eq in range(len(p["odeUnknowns"])):
        approxDerivative.append((newtonFunction(x+DELTA,y,p)[eq] - newtonFunction(x-DELTA,y,p)[eq])/(2*DELTA))
    return approxDerivative

def approximateDerivativeBC(y, p):
    newtonBC = p["odeBC"]
    return newtonBC(y,p)

def getNewtonFunction(p):
    if callable(p["ode"]):
        print("Loaded the ode newtonFunction")
    else:
        #Define x, t as symbols so U can be defined as a function of x and t and
        #sympy can later take the derivative of the sympy ode in terms of x and t.
        myDict = {}
        x = sympy.Symbol("x")
        t = sympy.Symbol("t")
        myDict["x"] = x
        myDict["t"] = t
              
        #Define U as a function of x and t so taking the derivative with sympy
        #Doesn't assume it's a constant and set it to zero.
        myDict["U"] = sympy.Function("U")(x, t)
        
        #Convert the ode string to sympy to make a machine generated newton 
        #function and newton jacobian.
        odeString = p["ode"]
        odeSympy = pypde.stringToSympy(odeString, myDict)
        odeBCString = p["odeBC"]
        odeBCSympy = pypde.stringToSympy(odeBCString, myDict)
        
        #Print the values and verify that stringToSympy works properly.
        print(odeSympy)
        print(odeBCSympy)
        
        #Create the newtonFunction and jacobianFunction file that will later be 
        #passed into newtonSolve.  Save them to the file newtonFunctions.
        newtonStrings = getNewtonStrings(odeSympy, odeBCSympy, myDict)
        fileName = p["name"]
        fileName += "NewtonFunctions.py"
        if pypde.shouldRewriteFile(fileName, newtonStrings):
            textFile = open(fileName, "w")
            textFile.write(newtonStrings)
            textFile.close()
        
        #Import the functions that were just written so they can be passed as 
        #parameters to newtonFunction.
        print("Created the ode newtonFunction")
        p["ode"] = importlib.import_module(fileName.replace(".py",""))
        p["ode"] = p["ode"].newtonFunction
        p["odeBC"] = importlib.import_module(fileName.replace(".py",""))
        p["odeBC"] = p["odeBC"].newtonBC

def getNewtonJacobian(p):
    if callable(p["odeJacobian"]):
        print("Loaded the ode newtonJacobian")
    elif p["odeJacobian"] == ["estimate"]:
        p["odeJacobian"] = approximateDerivative
    else:   
        fileName = p["name"]
        fileName += "NewtonFunctions.py"
        p["odeJacobian"] = importlib.import_module(fileName.replace(".py",""))
        p["odeJacobian"] = p["odeJacobian"].newtonJacobian
        p["odeJacobianBC"] = importlib.import_module(fileName.replace(".py",""))
        p["odeJacobianBC"] = p["odeJacobianBC"].newtonBCPrime
        print("Created the ode newtonFunction")
    if p["odeJacobianBC"] == ["estimate"]:
        p["odeJacobianBC"] = approximateDerivativeBC
        
    
def getNewtonGuess(p):
    #Set the default guess that will be plugged into newton's method.
    if len(p["odeGuess"]) == 0 or len(p["odeGuess"]) != p["odeChebPoints"]:
        p["odeGuess"] = np.full(p["odeChebPoints"],0.0)
        print("Defaulted to zero vector")
    return p["odeGuess"]

def newtonJacobianBC(inVector, p):
    newtonJacobian = p["odeJacobian"]
    newtonBCderiv = p["odeJacobianBC"]
    newtonBC = p["odeBC"]
    xList = p["odeXList"]
    SIZE = len(inVector)
    NUM_EQUATIONS = len(p["odeUnknowns"])
    jacobianMatrix = np.zeros((SIZE,SIZE))
    for row in range(SIZE):
        for col in range(SIZE):
            jacobianMatrix[row,col] = newtonJacobian(xList[row], evalOde(xList[row],inVector, p)*getCheb(col,xList[row], p), p) - getChebDeriv(col,xList[row], p)
            #jacobianMatrix[eq][row,col] = newtonJacobian(xList[row], evalOde(xList[row],inVector, p), p)[eq]*getCheb(col,xList[row], p) - getChebDeriv(col,xList[row], p)
    
    #Add the boundary condition.
    bc = lambda xVal: evalOde(xVal, inVector, p)
    findRemainder = lambda xVal: newtonBC(bc, p) - evalOde(xVal, inVector, p)
    for eq in range(NUM_EQUATIONS):    
        for col in range(SIZE):
            bcJac = lambda xVal: getCheb(col, xVal, p) - findRemainder(xVal)
            jacobianMatrix[SIZE-1,col] = newtonBCderiv(bcJac,p)[eq]
    return jacobianMatrix
    
def newtonFunctionBC(inVector, p):
    newtonFunction = p["ode"]
    newtonBC = p["odeBC"]
    xList = p["odeXList"]
    SIZE = len(inVector)
    NUM_EQUATIONS = len(p["odeUnknowns"])
    outVector = np.zeros((SIZE))
    for eq in range(NUM_EQUATIONS):
        for row in range(SIZE):
            outVector[row] = newtonFunction(xList[row], evalOde(xList[row],inVector, p), p)[eq] - evalOdeDerivative(xList[row],inVector, p)
     
    #Adding the boundary condition.
    bc = lambda xVal: evalOde(xVal, inVector, p)
    for eq in range(NUM_EQUATIONS):
        outVector[SIZE-1] = newtonBC(bc,p)[eq]
    return outVector

def solve(p):
    NUM_EQUATIONS = len(p["odeUnknowns"])
    
    #Write and Get the newton function if the user did not define it.  Also 
    # add in the boundary conditions and make the function output a vector.
    getNewtonFunction(p)

    #Get jacobian function if the user did not define it.  Also add the 
    # boundary condition while making the jacobian function return a vector.
    getNewtonJacobian(p)

    #Set the x points that chebyshev will evaluate at.
    p["odeXList"] = getChebPoints(p["odeChebPoints"], p)
    
    #use Newton's to solve for the profile coefficients
    newtonGuess = getNewtonGuess(p)
    odeCoefficients = NUM_EQUATIONS*[0]
    #print(odeCoefficients)
    for eq in range(len(p["odeUnknowns"])):
        odeCoefficients[eq] = (pypde.newtonSolve(newtonGuess, newtonFunctionBC, newtonJacobianBC, p))
    
    #Return the found coefficients as the profile solution to ode.solve.
    return odeCoefficients


"""Plotting functions"""
def plot(odeCoefficients, p):    
    #Use evalODE to find the ODE as a vector of evenly spaced points.
    #Graph it
    L = p["L"]
    graphPoints = p["odeGraphPoints"]
    chebxPoints = np.linspace(-1,1,graphPoints)
    xPoints = np.linspace(-L,L,graphPoints)
    yPoints = []
    for eq in range(len(p["odeUnknowns"])):        
        for i in range(graphPoints):
            yPoints.append(evalOde(L*chebxPoints[i],odeCoefficients[eq], p))
        plt.plot(xPoints, yPoints, label=p["odeUnknowns"][eq])
        plt.legend()
    
