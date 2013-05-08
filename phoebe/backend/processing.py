"""
List of user-defined contraints.
"""
from phoebe.dynamics import keplerorbit
import logging

logger = logging.getLogger('PROC')

def test_constraint(self):
    self.params['star']['incl'] = 2*self.params['mesh']['delta']
    
    
def binary_light_curve_constraints(self,time,eclipse_separation=0.5):
    ecc = self[0].params['orbit']['ecc']
    omega = keplerobit.per0_from_eclipse_separation(eclipse_separation,ecc)
    self[0].params['orbit']['per0'] = omega,'rad'
    
def wd_eclipse_separation(self,time,eclipse_separation=0.5):
    ecc = self.params['root']['ecc']
    omega = keplerorbit.per0_from_eclipse_separation(eclipse_separation,ecc)
    self.params['root']['omega'] = omega,'rad'
    logger.info('Derived omega={} from eclipse separation={} and ecc={}'.format(omega,eclipse_separation,ecc))