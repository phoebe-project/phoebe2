#!/usr/bin/env python3

"""
Create fake data.

"""

from math import pi,cos

ANG = 1.e-10

ucoord = 0.0
vcoord = 0.0
vis2data = 0.2
vis2err = 0.01
eff_wave = 6625*ANG
eff_band = 100*ANG
dataset = 1

f = open("Vis.dat", "w")
f.write("# hjd ucoord vcoord eff_wave eff_band vis2data vis2err\n")

t1 = 2456451.3721
t1 = 0.0
t1 = 0.25
dt = 1.0e-8
dt = 0.0

u1 = 0.0
u2 = 330.0
du = 1.0

t = t1
u = u1
while u < u2:
    t += dt
    u += du

    vis2data = 0.0

    f.write("%.8f  %.8e  %.8e  %.8e  %.8e  %.8f  %.8f  %d\n" % (t, u, vcoord, eff_wave, eff_band, vis2data, vis2err, dataset))

#f.close()
#f = open("Vis2.dat", "w")
#f.write("# hjd ucoord vcoord eff_wave eff_band vis2data vis2err\n")

t1 = t1+1
v1 = u1
v2 = u2
dv = du
dataset = 2

t = t1
v = v1
while v < v2:
    t += dt
    v += dv

    vis2data = 0.0

    f.write("%.8f  %.8e  %.8e  %.8e  %.8e  %.8f  %.8f  %d\n" % (t, ucoord, v, eff_wave, eff_band, vis2data, vis2err, dataset))

f.close()

