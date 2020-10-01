"""
this is a python implementation for a forward propagation of a pre-trained EBAI
ANN (see https://github.com/aprsa/ebai)
"""

import numpy as np
import os

def _activate(val, theta=0.75, tau=0.35):
	return 1./(1.+np.exp(-(val-theta)/tau))

def _rescale(val, bounds):
    return bounds[:,0] + (val-0.1)/(0.9-0.1)*(bounds[:,1]-bounds[:,0])

_dir = os.path.dirname(__file__)
i2h = np.loadtxt(os.path.join(_dir, 'i2h.weights'))
h2o = np.loadtxt(os.path.join(_dir, 'h2o.weights'))

bounds = np.loadtxt(os.path.join(_dir, 'bounds.data'))

def ebai_forward(fluxes):
	if len(fluxes) != 201:
		raise ValueError("fluxes must have length of 201 (evenly sampled in phase-space)")

	return _rescale(_activate(np.matmul(h2o, _activate(np.matmul(i2h, fluxes)))), bounds=bounds)
