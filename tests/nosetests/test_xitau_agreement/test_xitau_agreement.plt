#!/usr/bin/gnuplot

set xl "u [m]"
set yl "|V|^2 [1]"

p "test_xitau_agreement.out" u 2:5,\
  "xitau/visibility.dat" u 2:5

pa -1

set term png small
set out "test_xitau_agreement.png"
rep


