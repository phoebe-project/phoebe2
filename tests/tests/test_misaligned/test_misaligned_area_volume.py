"""
  Testing if area and volume of misaligned Roche lobe is correctly calculated
"""

import libphoebe as ph
import math as m

def test_area_volume():
  
  # detacted case (only meaningful)
  choice = 0
  q = 1
  F = 1
  d = 1
  Omega0 = 10
  theta = 0.9
  area0 = 0.1554255703973858
  volume0 = 5.7617852701434746e-03

  av = ph.roche_misaligned_area_volume(q, F, d, theta, Omega0, choice, larea=True, lvolume=True)
  
  assert(m.fabs(av["larea"] - area0) < 1e-8*area0)
  assert(m.fabs(av["lvolume"] - volume0) < 1e-8*volume0)

if __name__ == '__main__':
  
  test_area_volume()
