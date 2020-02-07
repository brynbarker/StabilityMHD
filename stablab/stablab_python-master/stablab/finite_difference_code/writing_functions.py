
# -*- coding: utf-8 -*-
"""
These are the functions used to write the python code after a finite-difference
method is plugged in and the new equations are found.  There are two functions, 
writeRunnerFile and writeFunctionsFile that create and write to the python
file.  The other functions simply add a string to the solution.
"""

"""********************************************
********************************************"""

import os.path

def writeHeader(filename):
    outputString = """
# -*- coding: utf-8 -*-
\"\"\"
User can define initial conditions and certain variables to see how the system
evolves over time

@author: pypde generated file
\"\"\"

import numpy as np
import approximate as fd
from """ + filename.replace(".py","") + """_functions import f
from """ + filename.replace(".py","") + """_functions import createJacobian

"""
    return (outputString)

def writeFunctionsHeader():
    outputString = """

# -*- coding: utf-8 -*-
\"\"\"
Contains the generated files with finite difference methods interlaced.  f is 
the output of the system of equations given some input.  createJacobian creates
the jacobian for use in Newton's method.

@author: pypde generated file
\"\"\"

import numpy as np

"""
    return (outputString)


def writeInitFunctions(u):
    outputString = """
#Functions to set the initial values of the system
"""
    for function in range(len(u)):
        outputString += "def "
        outputString += str(u[function])
        outputString += "Init(T, L, T_POINTS, X_POINTS):"
        outputString += """
    f = lambda x:  1.5-np.tanh(x/2.0)
    return fd.initMatrixWithFunction(T, L, T_POINTS, X_POINTS, f)
    
"""
    return(outputString)
    
def writeBoundaryFunctions():
    outputString = """
#Write the boundary functions, most basically defined as 0.
def lBoundFunction(U, n):
    return [U[n+1,0]]
def rBoundFunction(U, n):
    return [U[n+1,-1]-1.7]
def lBoundDerivative():
    return [0]
def rBoundDerivative():
    return [0]
"""
    return outputString
    
def writefFunction(u, equations, jacobianEquations, P):
    outputString = """
#loop through equations, creating a vector for all j's
def f(matrices, parameters, K, H, n):
    """
    for i in range(len(u)):
        outputString += str(u[i]) + " = matrices[" + str(i) + """]
    """
    for i in range(len(P)):
        outputString += str(P[i]) + " = parameters[\"" + str(P[i]) + """\"]
    """
    outputString += """outVector = []
    """
    #outputString += """
    #Get the boundary values.
    #lBoundValues = lBoundFunction(U, n)
    #rBoundValues = rBoundFunction(U, n)
    #"""
    for i in range(len(u)):
        outputString += """
    #Add values for equation """
        outputString += str(i+1)
    #    outputString += """
    #outVector.append(lBoundValues[""" + str(i) + """])
        outputString += """
    outVector.append(0)
    for j in range(1,len(matrices[0][0])-1):
        outVector.append("""
        outputString += str(equations[i])
        outputString += """)
    outVector.append(0)"""
    #outVector.append(rBoundValues[""" + str(i) + "])"
    outputString += """
    return outVector

    """  
    return(outputString)

def writeJacobianFunction(u, jacobianEquations, P):
    outputString = """
def createJacobian(matrices, time, K, H, parameters):
    #Get the boundary values
    """
    
    #lBoundDerivativeValues = lBoundDerivative()
    #rBoundDerivativeValues = rBoundDerivative()
    
    for i in range(len(u)):
        outputString += str(u[i]) + " = matrices[" + str(i) + """]
    """
    for i in range(len(P)):
        outputString += str(P[i]) + " = parameters[\"" + str(P[i]) + """\"]
    """
    outputString += """
    #Create the empty matrix.
    jacobianDimensions = len(matrices)*(len(matrices[0][0]))
    jacobian = np.zeros((jacobianDimensions, jacobianDimensions))
    quadrantDimensions = len(matrices[0][0])
    n = time -1"""
    
    for quadrantRow in range(len(u)):
        for quadrantCol in range(len(u)):
            outputString += """
            
    #Loop through quadrant ("""
            outputString += str(quadrantRow+1)+","+str(quadrantCol+1)+")."
    #       outputString += """
    
    #jacobian[quadrantDimensions*""" + str(quadrantRow) + ",quadrantDimensions*"
    #        outputString += str(quadrantRow)+":quadrantDimensions*("+str(quadrantRow) 
    #        outputString += "+1)-1] = lBoundDerivativeValues["+str(quadrantRow)+","
    #        outputString += str(quadrantCol)+"""]
            outputString += """
    jacobian[quadrantDimensions*""" + str(quadrantRow) + ",quadrantDimensions*"
            outputString += str(quadrantRow)+"] = 1"
            outputString += """
    jacobian[quadrantDimensions*(1+"""+str(quadrantRow)+")-1,quadrantDimensions*(1+"
            outputString += str(quadrantRow)+")-1] = 1" + """
    for row in range(quadrantDimensions-2):
        for col in range(quadrantDimensions-2):
            row += 1
            col += 1
            j = col
            if (row - col == 1): jacobian[row+quadrantDimensions*"""
            outputString += str(quadrantRow)
            outputString += ",col+quadrantDimensions*"
            outputString += str(quadrantCol)
            outputString += "] = "
            outputString += str(jacobianEquations[quadrantCol][quadrantRow,0])
            outputString += """
            if (row == col): jacobian[row+quadrantDimensions*"""
            outputString += str(quadrantRow)
            outputString += ",col+quadrantDimensions*"
            outputString += str(quadrantCol)
            outputString += "] = "
            outputString += str(jacobianEquations[quadrantCol][quadrantRow,1])
            outputString += """
            if (row - col == -1): jacobian[row+quadrantDimensions*"""
            outputString += str(quadrantRow)
            outputString += ",col+quadrantDimensions*"
            outputString += str(quadrantCol)
            outputString += "] = "
            outputString += str(jacobianEquations[quadrantCol][quadrantRow,2])

    outputString += """
    return(jacobian)
    """
    return outputString  
   
def writeMainFunction(u, p):
    outputString = """
if __name__ == "__main__":   
    #User defined constants
    T = 10
    L = 20
    T_POINTS = 50
    X_POINTS = 50

    #Create the parameters
    """
    for parameter in range(len(p)):
        outputString += str(p[parameter]) + """ = 1.0
    """
    if len(p) > 0:
        outputString += "parameters = (" + str(p[0])
        for parameter in range(1,len(p)):
            outputString += ", " + str(p[parameter])
        outputString += ")"
    else:
        outputString += "parameters=[]"
    outputString += """
    MAX_ERROR = 1e-8
    
    #Initialize and solve the system."""
    
    for eq in range(len(u)):
        outputString += """
    """+str(u[eq]) + "Matrix = " + str(u[eq])+"Init(T,L,T_POINTS,X_POINTS)"
        outputString += "  #" + str(u[eq]) + 'Matrix = fd.loadInitialValues("'+u[eq]+"Profile.txt\", T_POINTS, X_POINTS)"
    outputString += """
    fd.solveSystem(("""
    for eq in range(len(u)):
            outputString += str(u[eq])+"Matrix, "
    outputString += """), T,L, parameters, MAX_ERROR, f, createJacobian)
    """
    
    for eq in range(len(u)):
        outputString += "fd.plotMatrix(" + str(u[eq]) + "Matrix, \"" + u[eq] + "\")" + """
    """

    return(outputString)
    
def writeRunnerFile(fileName, unknownsStrings, parameters):
    outputString = ""
    outputString += writeHeader(fileName)
    outputString += writeInitFunctions(unknownsStrings)
    outputString += writeMainFunction(unknownsStrings, parameters) 
    if os.path.exists(fileName):
        if input("'" + fileName + "' already exists.  Replace it? (y/n): ") == "y":
            target = open(fileName,"w")
            target.write(outputString)
            target.close()
            print("'" + fileName + "' has been replaced.")
        else:
            print("'" + fileName + "' not changed.")
    else:
        target = open(fileName,"w")
        target.write(outputString)
        target.close()
        print("'" + fileName + "' has been created.")
    print()
        
def writeFunctionsFile(fileName, unknownsStrings, equations, jacobianEquations, parameters):
    outputString = ""
    outputString += writeFunctionsHeader()
    outputString += writeBoundaryFunctions()
    outputString += writefFunction(unknownsStrings, equations, jacobianEquations, parameters)
    outputString += writeJacobianFunction(unknownsStrings,jacobianEquations, parameters)
    target = open(fileName,"w")
    target.write(outputString)
    target.close()
    if os.path.exists(fileName):
        print("'" + fileName + "' has been replaced.")
    else:
        print("'" + fileName + "' has been created.")
