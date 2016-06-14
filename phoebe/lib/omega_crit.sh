#!/bin/bash

#
# Script to generate the map of critical potential from already 
# generated data.
#
# Author: Martin Horvat, March 2016

# compile
make omega_crit

# run
./omega_crit


echo "Available configuration:"

a=(`awk '{print $3}' omega_crit.dat  | sort | uniq` )
c=("green" "blue"  "yellow"  "red")

n=${#a[@]}
i=0
fun=-1

while [ "$i" -lt "$n" ]
do
  echo "${a[$i]}"
  
  cb="${cb}\"${a[$i]}\" $i"
  pal="${pal}$i \"${c[$i]}\""
  fun="(x == ${a[$i]} ? $i : ${fun})"
  
  i=$((i+1))
  
  if [ "$i" -ne "$n" ]
  then
    cb="${cb}, "
    pal="${pal}, "
  fi
  
done

echo "${cb}"
echo "${pal}"
echo "${fun}"

gnuplot << EOF
set term pdf enh

set xlabel 'q'
set ylabel 'F'

set cbtics (${cb})

set palette defined (${pal})

f(x)=${fun}

set title "Order of the potentials in Lagrange points {/Symbol W}(L_i): L_{i_1} < L_{i_2} < L_{i_3}"

set view map

set palette maxcolors $n

set out 'omega_crit.pdf'

unset key

splot 'omega_crit.dat' u 1:2:(f(\$3)) w image 
EOF

