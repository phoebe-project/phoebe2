#  Testing the C++ wrapper for calculation of limb darkening 
#  Author: Martin Horvat, August 2016


from libphoebe import *

mu = 0.1

for i in range(1, 400):
  d = ld_funcD(mu, ("linear", 0.1))
  g = ld_gradparD(mu, ("linear", 0.1))
  

print d, " ", g

for i in range(1, 400):
  d = ld_funcD(mu, ("quadratic", 0.1, 0.2))
  g = ld_gradparD(mu, ("quadratic", 0.1, 0.2))
  

print d, " ", g
