#!/bin/bash

if [ -x cusgemm ]
  then
  for k in {16..1024..16}
    do
    echo cusgemm n n 512 480 ${k}
    ./cusgemm n n 512 480 ${k} 1
  done | tee cusgemm_n_n_512_480.txt

  for k in {16..1024..16}
    do
    echo cusgemm n t 512 480 ${k}
    ./cusgemm n t 512 480 ${k} 1
  done | tee cusgemm_n_t_512_480.txt

  for k in {8..1024..8}
    do
    echo cusgemm t n 384 480 ${k}
    ./cusgemm t n 384 480 ${k} 1
  done | tee cusgemm_t_n_384_480.txt

  for k in {8..1024..8}
    do
    echo cusgemm t t 384 480 ${k}
    ./cusgemm t t 384 480 ${k} 1
  done | tee cusgemm_t_n_384_480.txt
fi

if [ -x cussyrk ]
  then
  for k in {16..1024..16}
    do
    echo cussyrk u n 512 ${k}
    ./cussyrk u n 512 ${k} 1
  done | tee cussyrk_u_n_512.txt

  for k in {8..1024..8}
    do
    echo cussyrk u t 384 ${k}
    ./cussyrk u t 384 ${k} 1
  done | tee cussyrk_u_t_384.txt

  for k in {16..1024..16}
    do
    echo cussyrk l n 512 ${k}
    ./cussyrk l n 512 ${k} 1
  done | tee cussyrk_l_n_512.txt

  for k in {8..1024..8};
    do
    echo cussyrk l t 384 ${k}
    ./cussyrk l t 384 ${k} 1
  done | tee cussyrk_l_t_384.txt
fi

if [ -x custrsm ]
  then
  for u in u l
    do
    for t in n t
      do
      for d in u n
        do
        for m in {8..512..8}
          do
          echo custrsm l ${u} ${t} ${d} ${m} 11520
          ./custrsm l ${u} ${t} ${d} ${m} 11520 1
        done | tee custrsm_l_${u}_${t}_${d}_11520.txt
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
          echo custrsm r ${u} ${t} ${d} 15360 ${n}
          ./custrsm r ${u} ${t} ${d} 15360 ${n} 1
        done | tee custrsm_r_${u}_${t}_${d}_15360.txt
      done
    done
  done
fi

if [ -x custrmm ]
  then
  for u in u l
    do
    for d in u n
      do
      for m in {16..1024..16}
        do
        echo custrmm l ${u} n ${d} ${m} 480
        ./custrmm l ${u} n ${d} ${m} 480 1
      done | tee custrmm_l_${u}_n_${d}_480.txt
    done
    for d in u n
      do
      for m in {8..1024..8}
        do
        echo custrmm l ${u} t ${d} ${m} 480
        ./custrmm l ${u} t ${d} ${m} 480 1
      done | tee custrmm_l_${u}_t_${d}_480.txt
    done
  done

  for u in u l
    do
    for d in u n
      do
      for n in {16..1024..16}
        do
        echo custrmm r ${u} n ${d} 512 ${n}
        ./custrmm r ${u} n ${d} 512 ${n} 1
      done | tee custrmm_r_${u}_n_${d}_512.txt
    done
    for d in u n
      do
      for n in {8..1024..8}
        do
        echo custrmm r ${u} t ${d} 384 ${n}
        ./custrmm r ${u} t ${d} 384 ${n} 1
      done | tee custrmm_r_${u}_t_${d}_384.txt
    done
  done
fi
