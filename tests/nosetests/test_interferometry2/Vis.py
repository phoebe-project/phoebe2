#!/usr/bin/env python3

ANG = 1.e-10

ucoord = 0.0
vcoord = 0.0
vis2data = 0.2
vis2err = 0.01
eff_wave = 6625*ANG
eff_band = 100*ANG

f = open("Vis.dat", "w")
f.write("# jd ucoord vcoord eff_wave eff_band vis2data vis2err\n")

t1 = 2456451.372100
dt = 1.0e-8
dt = 0.0

u1 = 0.0
u2 = 330.0
du = 5.0
v1 = u1
v2 = u2
dv = du

t = t1
u = u1
while u < u2:
    t += dt
    u += du
    v = v1
    while v < v2:
        t += dt
        v += dv
        f.write("%.8f  %.8e  %.8e  %.8e  %.8e  %.8f  %.8f\n" % (t, u, v, eff_wave, eff_band, vis2data, vis2err))

f.close()


