from libtest import *
import numpy as np
import sys

for i in xrange(10):
  #a = fun1(10.)
  a = fun2(10.)
  print sys.getrefcount(a)

