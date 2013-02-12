#!/bin/bash

if [ -x sgemm ]
  then
  for k in {16..1024..16}
    do
    echo sgemm n n 512 480 ${k}
    ./sgemm n n 512 480 ${k}
  done | tee sgemm_n_n_512_480.txt

  for k in {16..1024..16}
    do
    echo sgemm n t 512 480 ${k}
    ./sgemm n t 512 480 ${k}
  done | tee sgemm_n_t_512_480.txt

  for k in {8..1024..8}
    do
    echo sgemm t n 384 480 ${k}
    ./sgemm t n 384 480 ${k}
  done | tee sgemm_t_n_384_480.txt

  for k in {8..1024..8}
    do
    echo sgemm t t 384 480 ${k}
    ./sgemm t t 384 480 ${k}
  done | tee sgemm_t_n_384_480.txt
fi

if [ -x ssyrk ]
  then
  for k in {16..1024..16}
    do
    echo ssyrk u n 512 ${k}
    ./ssyrk u n 512 ${k}
  done | tee ssyrk_u_n_512.txt

  for k in {8..1024..8}
    do
    echo ssyrk u t 384 ${k}
    ./ssyrk u t 384 ${k}
  done | tee ssyrk_u_t_384.txt

  for k in {16..1024..16}
    do
    echo ssyrk l n 512 ${k}
    ./ssyrk l n 512 ${k}
  done | tee ssyrk_l_n_512.txt

  for k in {8..1024..8};
    do
    echo ssyrk l t 384 ${k}
    ./ssyrk l t 384 ${k}
  done | tee ssyrk_l_t_384.txt
fi

if [ -x strsm ]
  then
  for u in u l
    do
    for t in n t
      do
      for d in u n
        do
        for m in {8..512..8}
          do
          echo strsm l ${u} ${t} ${d} ${m} 11520
          ./strsm l ${u} ${t} ${d} ${m} 11520
        done | tee strsm_l_${u}_${t}_${d}_11520.txt
      done
    done
  done

  for u in u l
    do
    for t in n t
      do
      for d in u n
        do
        for n in {8..512..8}
          do
          echo strsm r ${u} ${t} ${d} 15360 ${n}
          ./strsm r ${u} ${t} ${d} 15360 ${n}
        done | tee strsm_r_${u}_${t}_${d}_15360.txt
      done
    done
  done
fi

if [ -x strmm ]
  then
  for u in u l
    do
    for d in u n
      do
      for m in {16..1024..16}
        do
        echo strmm l ${u} n ${d} ${m} 480
        ./strmm l ${u} n ${d} ${m} 480
      done | tee strmm_l_${u}_n_${d}_480.txt
    done
    for d in u n
      do
      for m in {8..1024..8}
        do
        echo strmm l ${u} t ${d} ${m} 480
        ./strmm l ${u} t ${d} ${m} 480
      done | tee strmm_l_${u}_t_${d}_480.txt
    done
  done

  for u in u l
    do
    for d in u n
      do
      for n in {16..1024..16}
        do
        echo strmm r ${u} n ${d} 512 ${n}
        ./strmm r ${u} n ${d} 512 ${n}
      done | tee strmm_r_${u}_n_${d}_512.txt
    done
    for d in u n
      do
      for n in {8..1024..8}
        do
        echo strmm r ${u} t ${d} 384 ${n}
        ./strmm r ${u} t ${d} 384 ${n}
      done | tee strmm_r_${u}_t_${d}_384.txt
    done
  done
fi
