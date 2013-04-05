
from marching2FLib import getMesh
import marching

from math import sqrt, sin, cos, acos, atan2, trunc, pi
import numpy as np

try:
    from enthought.mayavi.mlab import *
except ImportError:
    from mayavi.mlab import *


SPHERE, BINARY_ROCHE, MISALIGNED_BINARY_ROCHE, ROTATE_ROCHE, TORUS = range(5)
potentialType = BINARY_ROCHE
maxNumberTriangles = 100000
# for BINARY_ROCHE
numParams = 6
D = 0.75
q = 0.6
F = 1.5
Omega = 3.8

#sphere
#numParams = 3
#R = 4.0

# Roateta Roche
#numParams = 4
#Omega = 0.9
#RPole = 1.0


# first works well roche, second sphere
delta = 0.02
#delta = 0.2
#delta = 0.05

tableA = getMesh(potentialType, maxNumberTriangles, numParams, D, q, F, Omega, delta)
#tableA = getMesh(SPHERE, maxNumberTriangles, numParams, R, delta)
#tableA = getMesh(ROTATE_ROCHE, maxNumberTriangles, numParams, Omega, RPole, delta)
#print "N = ",len(table)
#print "table = ",tableA
#table = np.zeros((len(Ts), 16))
print "length of tableA = ", len(tableA[0])
print "table[0] = ",tableA[0]
#table1 = np.zeros((10, 2))
#print "table1 = ",table1
table = tableA[0]

table = marching.cdiscretize(delta,maxNumberTriangles,'BinaryRoche',D,q,F,Omega)

N = len(table)
print "the number of triangles = ", N
x = np.hstack([table[:,i+4] for i in range(0,9,3)])
y = np.hstack([table[:,i+4] for i in range(1,9,3)])
z = np.hstack([table[:,i+4] for i in range(2,9,3)])
triangles = [(i,N+i,2*N+i) for i in range(N)]
triangular_mesh(x,y,z,triangles)
show()


#print "q = ", table
