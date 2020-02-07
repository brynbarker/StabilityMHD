# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:01:31 2017

@author: Jalen Morgan
"""

import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.nan) 

"""********************************************
Initializing functions
********************************************"""    
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

"""********************************************************
Loop through time, get and set the next time step.
********************************************************"""
def getTimeVectors(inputMatrices, time):
    #Give a time and it will concatenate the vectors in each matrix.
    xLength = len(inputMatrices[0][0])
    output = []
    for i in range(len(inputMatrices)):
        output = np.append(output, inputMatrices[i][time,0:xLength])
        #output = np.append(output, inputMatrices[i][time,1:xLength-1])
    return output
    
def setTimeVectors(solution, matrices, time):
    #Pass it a large vector, and it will splice it into the matrices
    xLength = len(matrices[0][0])
    for i in range(len(matrices)):
        solutionVector = solution[i*int(len(solution)/len(matrices)):(i+1)*int(len(solution)/len(matrices))]
        matrices[i][time,0:xLength] = solutionVector
        #matrices[i][time,1:xLength-1] = solutionVector

def solveSystem(matrices, xPoints, tPoints, P, MAX_ERROR, f, createJacobian):
    T = tPoints[-1]- tPoints[0]
    L = abs(xPoints[-1]- xPoints[0])

    #make some variables
    T_POINTS = len(matrices[0])
    X_POINTS = len(matrices[0][0])
    K = 1.0*T/(T_POINTS-1)
    H = 2.0*L/(X_POINTS-1)
    printPercent = True
    solution = getTimeVectors(matrices,0)
    
    #Loop through time, solving and assigning next timestep.
    for time in range(len(matrices[0])-1):       
        if printPercent == True: print(int(100*time/(len(matrices[0])-1)), "% .. ", end='')
        setTimeVectors(solution, matrices, time+1)
        solution = newtonSolve(matrices, time, K, H, P, MAX_ERROR, f, createJacobian)
        setTimeVectors(solution, matrices, time+1)
        
    #Print when finished.
    if printPercent == True: print("100 % .. Finished")
    
"""********************************************
Equations and functions for using newton's method
********************************************"""
def graph(unknowns, matrixList):
    for i in range(len(unknowns)):
        approximate.plotMatrix(matrixList[i], unknowns[i])

def newtonSolve(matrices, time, K, H, P, MAX_ERROR, f, jacobian):
    #Initialize some variables.
    printIterations = True
    inVector = getTimeVectors(matrices, time+1)
    outVector = f(matrices, P, K, H, time)
    TIMEOUT = 40
    count = 0
    
    #print("First iteration")
    #Loop until the largest component is less than MAX_ERROR.
    if printIterations == True: print("(", end='')
    while max(map(abs,outVector))-MAX_ERROR > 0:
        count += 1
        #Create a Jacobian matrix A with inVector as the guess for n+1
        A = jacobian(matrices, time+1, K, H, P)
        b = outVector - np.dot(A,inVector)
        inVector = np.linalg.solve(A,-b)
        setTimeVectors(inVector, matrices, time+1)
        outVector = f(matrices, P, K, H, time)
        
        #Print info and Break if a solution isn't found.
        if printIterations == True: print(count, end='')
        if count == TIMEOUT:
            print("+ ...)")
            return inVector
        #print("end of while loop")
    if printIterations == True: print(")")
    return inVector

"""**********************************************
Equations for drawing and plotting information
**********************************************"""
def printRounded(listOne):
    #Determine the Decimal places that will be rounded to.
    DECIMAL_PLACES = 16
    roundedList = list(listOne)
    for item in range(len(roundedList)):
        roundedList[item] = round(roundedList[item], DECIMAL_PLACES)
    print (roundedList)

def plotMatrix(a,text=""):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    #Create the x, y, and z grid from the array a.
    y = range(0,len(a[0]))
    x = range(0,len(a))
    X, Y = np.meshgrid(x, y)
    Z = a[X,Y]
    #print(Z)

    #Graph x,y,z in 3d.  
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y,Z, linewidth=0, antialiased=False)
    plt.title(text)
    plt.show()

"""************************************************
Equations for writing to and loading from text files
************************************************"""
def loadInitialValues(fileName, T_POINTS, X_POINTS):    
    #Create matrix and value of x at t=0.
    a= np.zeros((T_POINTS, X_POINTS))
    
    #Get from the file the condition where t = 0.
    myFile = open(fileName,'r')
    
    a = np.zeros((T_POINTS, X_POINTS))
    myList = str(myFile.read()).split(' ')
    newList = [float(i) for i in myList[0:len(myList)-1]]

    myFile.close()
    a[0] = newList
    
    #Create the value of t at x= {0,X_POINTS-1}
    a[0:T_POINTS,X_POINTS-1] = a[0,X_POINTS-1]    
    a[0:T_POINTS,0] = a[0,0]
    
    return a

if __name__ == "__main__":
    jacobian = lambda x,y,z,h,w: np.zeros((2,5,5))
    leftBound = [[1,1,1,1],[1,1,2,2]]
    rightBound = [[1,1,1,1],[3,3,4,4]]
    matrices = np.zeros((2,5,5))
    time = 0
    K = 1
    H = 1
    P = 1
    print(jacobianWithBoundary(jacobian, leftBound, rightBound, matrices, time, K, H, P))
