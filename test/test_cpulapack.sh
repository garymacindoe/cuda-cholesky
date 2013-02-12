#!/bin/bash

if [ -x spotrf ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo spotrf ${u} ${n}
      ./spotrf ${u} ${n}
    done | tee spotrf_${u}.txt
  done
fi

if [ -x spotri ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo spotri ${u} ${n}
      ./spotri ${u} ${n}
    done | tee spotri_${u}.txt
  done
fi

if [ -x strtri ]
  then
  for u in u l
    do
    for d in u n
      do
      for n in {64..4096..64}
        do
        echo strtri ${u} ${d} ${n}
        ./strtri ${u} ${d} ${n}
      done
    done | tee strtri_${u}_${d}.txt
  done
fi

if [ -x slauum ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo slauum ${u} ${n}
      ./slauum ${u} ${n}
    done | tee slauum_${u}.txt
  done
fi

if [ -x slogdet ]
  then
  for n in {1024..1048576..1024}
    do
    echo slogdet ${n}
    ./slogdet ${n}
  done | tee slogdet.txt
fi
