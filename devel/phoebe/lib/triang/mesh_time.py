#
# Timing the meshing in Phoebe beta
#

import time

from phoebe2.algorithms import cmarching
d = 1.0
q = 1.0
F = 1.0
Phi = 10.0
delta = 0.005
maxpoints = int(1e6)
mesh_args = ('BinaryRoche', d, q, F, Phi)

start = time.time()
table = cmarching.discretize(delta, maxpoints, *mesh_args)
end = time.time()


print "time[ms]=", 1000*(end-start)
print "len(table)=", len(table)
