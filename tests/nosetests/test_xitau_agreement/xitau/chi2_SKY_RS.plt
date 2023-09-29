#!/usr/bin/gnuplot

deg = pi/180.
arcsec = deg/3600.
au = 1.49597870700e11  # m, from IAU 2012
pc = 648000.0/pi*au  # m, from IAU 2015
R_S = 6.957e8  # m; from IAU 2015

set colors classic
set term x11

set tit "orbit misaligned w. observations..."
set xl "x [R_S]"
set yl "y [R_S]"
set size ratio -1

tmp=2.e-5
tmp=0.05*au/R_S
set xr [-tmp:tmp]
set yr [-tmp:tmp]
set zeroaxis
set ang rad

load "T0.plt"

d_pc = x_param22
f(x) = x*arcsec*(d_pc*pc)/au
g(x) = x/au

set cbl "ibod"
set cbr [2:3]
set cbtics 1
set palette defined (\
  0.0 '#ff0000',\
  1.0 '#ff8000' \
  )

p \
  "<awk '($2==-1)' out_JDATE_heliocentric.dat" u ($3*au/R_S):($4*au/R_S) t "1" w l lt 1,\
  "<awk '($2==-2)' out_JDATE_heliocentric.dat" u ($3*au/R_S):($4*au/R_S) t "2" w l lt 2,\
  "<awk '($2==-3)' out_JDATE_heliocentric.dat" u ($3*au/R_S):($4*au/R_S) t "3" w l lt 3,\
  "<awk '($2==-4)' out_JDATE_heliocentric.dat" u ($3*au/R_S):($4*au/R_S) t "4" w l lt 4,\
  sprintf("<awk '($2==-1) && ($1==%.10f)' out_JDATE_heliocentric.dat", T0) u ($3*au/R_S):($4*au/R_S) t "T_0" w p lc 0 pt 1 ps 2,\
  sprintf("<awk '($2==-2) && ($1==%.10f)' out_JDATE_heliocentric.dat", T0) u ($3*au/R_S):($4*au/R_S) not     w p lc 0 pt 1 ps 2,\
  sprintf("<awk '($2==-3) && ($1==%.10f)' out_JDATE_heliocentric.dat", T0) u ($3*au/R_S):($4*au/R_S) not     w p lc 0 pt 1 ps 2,\
  sprintf("<awk '($2==-4) && ($1==%.10f)' out_JDATE_heliocentric.dat", T0) u ($3*au/R_S):($4*au/R_S) not     w p lc 0 pt 1 ps 2,\
  "arcsec_AU.dat" u 2:3 t "observ." w p lt 7 pt 1 ps 0.5,\
  "<awk '($1==\"2458462.6132540000\")' arcsec_AU.dat" u 2:3 not w p lt 7 pt 1 ps 3.5 lw 2,\
  "<./ellipses.awk arcsec_AU.dat" u 1:2 not w l lt 7,\
  "chi2_SKY.dat" u (-f($2)*sin($3*deg)):(f($2)*cos($3*deg)):8 t "residua" w l lc palette z lw 3,\
  "<awk '(FNR>1){ print $0,ARGIND; }' nodes005.tmp" u (g($2)):(g($3)) t "shape" w d lc 'gray'
pa -1

set term png small
set out "chi2_SKY.png"
rep

q


