#!/bin/bash

rm -r pi-path.dat temp
touch temp

while read p; do
  #echo $p
  num=$(grep $p Pi*.txt | sed 's/:/ /g' | awk '{print $1}' | sort -n | uniq | wc -l)
  echo $p $num >> temp
done < pi-path-unique-resids-nowat-notype.dat

cat temp | sort -rn -k 2 > pi-path.dat
