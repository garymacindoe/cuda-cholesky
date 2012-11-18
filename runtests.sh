#!/bin/bash

GPU=1

# CPU SGEMM
if [ -e sgemm ]
  then
  for tA in n t
    do
    for tB in n t
      do
      echo sgemm ${tA} ${tB} 128 128 128
      ./sgemm ${tA} ${tB} 128 128 128
    done
  done
fi

# CPU DGEMM
if [ -e dgemm ]
  then
  for tA in n t
    do
    for tB in n t
      do
      echo dgemm ${tA} ${tB} 128 128 128
      ./dgemm ${tA} ${tB} 128 128 128
    done
  done
fi

# CPU CGEMM
if [ -e cgemm ]
  then
  for tA in n t c
    do
    for tB in n t c
      do
        echo cgemm ${tA} ${tB} 128 128 128
      ./cgemm ${tA} ${tB} 128 128 128
    done
  done
fi

# CPU ZGEMM
if [ -e zgemm ]
  then
  for tA in n t c
    do
    for tB in n t c
      do
        echo zgemm ${tA} ${tB} 128 128 128
      ./zgemm ${tA} ${tB} 128 128 128
    done
  done
fi
