#!/bin/bash

if [ -x cuspotrf ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo cuspotrf ${u} ${n}
      ./cuspotrf ${u} ${n} 1
    done | tee cuspotrf_${u}.txt
  done
fi

if [ -x cuspotri ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo cuspotri ${u} ${n}
      ./cuspotri ${u} ${n} 1
    done | tee cuspotri_${u}.txt
  done
fi

if [ -x custrtri ]
  then
  for u in u l
    do
    for d in u n
      do
      for n in {64..4096..64}
        do
        echo custrtri ${u} ${d} ${n}
        ./custrtri ${u} ${d} ${n} 1
      done
    done | tee custrtri_${u}_${d}.txt
  done
fi

if [ -x cuslauum ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo cuslauum ${u} ${n}
      ./cuslauum ${u} ${n} 1
    done | tee cuslauum_${u}.txt
  done
fi

if [ -x cuslogdet ]
  then
  for n in {1024..1048576..1024}
    do
    echo cuslogdet ${n}
    ./cuslogdet ${n} 1
  done | tee cuslogdet.txt
fi
