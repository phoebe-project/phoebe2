#!/usr/bin/gnuplot

set xl "u [m]"
set yl "v [m]"
set cbl "|V|^2 [1]"

tmp = 350
set xr [-tmp:tmp]
set yr [-tmp:tmp]
set cbr [0.001:1]
set logscale cb
set palette rgbformulae 33,13,10
set zeroaxis
set size ratio -1

p \
  "test_interferometry2.out" u (+$2):(+$3):5 w p pt 5 lc palette z not,\
  "test_interferometry2.out" u (+$2):(-$3):5 w p pt 5 lc palette z not,\
  "test_interferometry2.out" u (-$2):(+$3):5 w p pt 5 lc palette z not,\
  "test_interferometry2.out" u (-$2):(-$3):5 w p pt 5 lc palette z not,\

pa -1

set term png small
set out "uv.png"
rep


