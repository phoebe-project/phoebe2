#!/usr/bin/gnuplot

set xl "u [m]"
set yl "v [m]"
set cbl "|V|^2 [1]"

tmp = 350
set xr [-tmp:tmp]
set yr [-tmp:tmp]
set cbr [0:1]
set palette rgbformulae 33,13,10
set zeroaxis
set size ratio -1

p "test_interferometry.out" u 2:3:5 w p lc palette z not,\
  "<awk '($2>0.0)' test_interferometry.out" u 2:($5*100) w l lc 'gray' not,\
  "<awk '($3>0.0)' test_interferometry.out" u (-$5*100):3 w l lc 'gray' not

pa -1

set term png small
set out "uv.png"
rep


