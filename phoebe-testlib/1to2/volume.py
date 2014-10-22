import phoebeBackend as phb1
import phoebe as phb2
import matplotlib.pyplot as plt
import numpy as np

def count_rectangles(N):
    count = 0
    theta = [np.pi/2*(k-0.5)/N for k in range(1,N+1)]
    for th in theta:
        Mk = 1+int(1.3*N*np.sin(th))
        phi = [np.pi*(l-0.5)/Mk for l in range(1,Mk+1)]
        count += len(phi)
    return 8*count

N = 200
delta = 0.012

eb = phb2.Bundle('default.phoebe')
eb['maxpoints@mesh:marching@primary'] = 1e8
eb['maxpoints@mesh:marching@secondary'] = 1e8
eb['pot@primary'] = 3.8
eb['pot@secondary'] = 3.8

deltas, Ntriangles, V12s, V22s = [], [], [], []
for delta in np.linspace(0.1, delta, 25):
    eb['delta@mesh:marching@primary'] = delta
    eb['delta@mesh:marching@secondary'] = delta
    eb.set_time(0)
    vol1_2 = eb.get_object('primary').volume()
    vol2_2 = eb.get_object('secondary').volume()
    deltas.append(delta)
    Ntriangles.append(len(eb['primary'].get_mesh()['teff']))
    V12s.append(vol1_2)
    V22s.append(vol2_2)
    print delta, vol1_2, vol2_2


phb1.init()
phb1.configure()

phb1.open('default.phoebe')
phb1.setpar('phoebe_lcno', 1)
sma = phb1.getpar('phoebe_sma')
phb1.setpar('phoebe_pot1', 3.8)
phb1.setpar('phoebe_pot2', 3.8)

Ns, Nrectangles, V11s, V21s = [], [], [], []
for N in range(20, 201, 2):
    phb1.setpar('phoebe_grid_finesize1', N)
    phb1.setpar('phoebe_grid_finesize2', N)

    dummy = phb1.lc((0.0,), 0)
    vol1_1 = sma**3*phb1.getpar('phoebe_vol1')
    vol2_1 = sma**3*phb1.getpar('phoebe_vol2')

    Ns.append(N)
    Nrectangles.append(count_rectangles(N))
    V11s.append(vol1_1)
    V21s.append(vol2_1)
    print N, vol1_1, vol2_1

phb1.quit()

pdV1s = 100*(np.array(V11s)-V12s[-1])/V12s[-1]
pdV2s = 100*(np.array(V21s)-V22s[-1])/V22s[-1]
plt.xlabel('Number of WD rectangles')
plt.ylabel('Percent difference in volume')
#~ plt.plot(Ns, pdV1s, 'b-')
plt.plot(Nrectangles, pdV2s, 'g-')
plt.show()

pdV1s = 100*(np.array(V12s)-V11s[-1])/V11s[-1]
pdV2s = 100*(np.array(V22s)-V21s[-1])/V11s[-1]
plt.xlabel('Number of PHOEBE triangles')
plt.ylabel('Percent difference in volume')
#~ plt.plot(deltas, pdV1s, 'b-')
plt.plot(Ntriangles, pdV2s, 'g-')
plt.show()

plt.xlabel('Number of surface elements')
plt.ylabel('Percent difference in volume')
plt.plot(Nrectangles, 100*(np.array(V21s)-V22s[-1])/V22s[-1], 'k-')
plt.plot(Ntriangles, 100*(np.array(V22s)-V21s[-1])/V11s[-1], 'k-')
plt.show()
