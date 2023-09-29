#!/usr/bin/gnuplot

set xl "u [m]"
set yl "v [m]"

tmp=350.
set xr [-tmp:tmp]
set yr [-tmp:tmp]
set size ratio -1
set zeroaxis
set cbr [0:1]
set palette rgbformulae 33,13,10

p \
  "Vis.dat" u 2:3:6 w p lc palette z,\

pa -1

set term png small
set out "Vis.png"
rep

