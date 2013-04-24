#!/bin/bash

if [ -x sgemm ]
  then
  for k in {16..1024..16}
    do
    echo sgemm n n 512 480 ${k}
    ./sgemm n n 512 480 ${k} 1
  done | tee sgemm_n_n_512_480.txt

  for k in {16..1024..16}
    do
    echo sgemm n t 512 480 ${k}
    ./sgemm n t 512 480 ${k} 1
  done | tee sgemm_n_t_512_480.txt

  for k in {8..1024..8}
    do
    echo sgemm t n 384 480 ${k}
    ./sgemm t n 384 480 ${k} 1
  done | tee sgemm_t_n_384_480.txt

  for k in {8..1024..8}
    do
    echo sgemm t t 384 480 ${k}
    ./sgemm t t 384 480 ${k} 1
  done | tee sgemm_t_t_384_480.txt
fi

if [ -x ssyrk ]
  then
  for k in {16..1024..16}
    do
    echo ssyrk u n 512 ${k}
    ./ssyrk u n 512 ${k} 1
  done | tee ssyrk_u_n_512.txt

  for k in {8..1024..8}
    do
    echo ssyrk u t 384 ${k}
    ./ssyrk u t 384 ${k} 1
  done | tee ssyrk_u_t_384.txt

  for k in {16..1024..16}
    do
    echo ssyrk l n 512 ${k}
    ./ssyrk l n 512 ${k} 1
  done | tee ssyrk_l_n_512.txt

  for k in {8..1024..8}
    do
    echo ssyrk l t 384 ${k}
    ./ssyrk l t 384 ${k} 1
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
          ./strsm l ${u} ${t} ${d} ${m} 11520 1
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
          ./strsm r ${u} ${t} ${d} 15360 ${n} 1
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
        ./strmm l ${u} n ${d} ${m} 480 1
      done | tee strmm_l_${u}_n_${d}_480.txt
    done
    for d in u n
      do
      for m in {8..1024..8}
        do
        echo strmm l ${u} t ${d} ${m} 480
        ./strmm l ${u} t ${d} ${m} 480 1
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
        ./strmm r ${u} n ${d} 512 ${n} 1
      done | tee strmm_r_${u}_n_${d}_512.txt
    done
    for d in u n
      do
      for n in {16..1024..16}
        do
        echo strmm r ${u} t ${d} 512 ${n}
        ./strmm r ${u} t ${d} 512 ${n} 1
      done | tee strmm_r_${u}_t_${d}_512.txt
    done
  done
fi

if [ -x dgemm ]
  then
  for k in {16..1024..16}
    do
    echo dgemm n n 320 384 ${k}
    ./dgemm n n 320 384 ${k} 1
  done | tee dgemm_n_n_320_384.txt

  for k in {16..1024..16}
    do
    echo dgemm n t 320 384 ${k}
    ./dgemm n t 320 384 ${k} 1
  done | tee dgemm_n_t_320_384.txt

  for k in {8..1024..8}
    do
    echo dgemm t n 256 240 ${k}
    ./dgemm t n 256 240 ${k} 1
  done | tee dgemm_t_n_256_240.txt

  for k in {8..1024..8}
    do
    echo dgemm t t 256 240 ${k}
    ./dgemm t t 256 240 ${k} 1
  done | tee dgemm_t_t_256_240.txt
fi

if [ -x dsyrk ]
  then
  for k in {16..1024..16}
    do
    echo dsyrk u n 320 ${k}
    ./dsyrk u n 320 ${k} 1
  done | tee dsyrk_u_n_320.txt

  for k in {8..1024..8}
    do
    echo dsyrk u t 256 ${k}
    ./dsyrk u t 256 ${k} 1
  done | tee dsyrk_u_t_256.txt

  for k in {16..1024..16}
    do
    echo dsyrk l n 320 ${k}
    ./dsyrk l n 320 ${k} 1
  done | tee dsyrk_l_n_320.txt

  for k in {8..1024..8}
    do
    echo dsyrk l t 256 ${k}
    ./dsyrk l t 256 ${k} 1
  done | tee dsyrk_l_t_256.txt
fi

if [ -x dtrsm ]
  then
  for u in u l
    do
    for t in n t
      do
      for d in u n
        do
        for m in {4..512..4}
          do
          echo dtrsm l ${u} ${t} ${d} ${m} 3840
          ./dtrsm l ${u} ${t} ${d} ${m} 3840 1
        done | tee dtrsm_l_${u}_${t}_${d}_3840.txt
      done
    done
  done

  for u in u l
    do
    for t in n t
      do
      for d in u n
        do
        for n in {4..512..4}
          do
          echo dtrsm r ${u} ${t} ${d} 3840 ${n}
          ./dtrsm r ${u} ${t} ${d} 3840 ${n} 1
        done | tee dtrsm_r_${u}_${t}_${d}_3840.txt
      done
    done
  done
fi

if [ -x dtrmm ]
  then
  for u in u l
    do
    for d in u n
      do
      for m in {16..1024..16}
        do
        echo dtrmm l ${u} n ${d} ${m} 384
        ./dtrmm l ${u} n ${d} ${m} 384 1
      done | tee dtrmm_l_${u}_n_${d}_384.txt
    done
    for d in u n
      do
      for m in {8..1024..8}
        do
        echo dtrmm l ${u} t ${d} ${m} 240
        ./dtrmm l ${u} t ${d} ${m} 240 1
      done | tee dtrmm_l_${u}_t_${d}_240.txt
    done
  done

  for u in u l
    do
    for d in u n
      do
      for n in {16..1024..16}
        do
        echo dtrmm r ${u} n ${d} 320 ${n}
        ./dtrmm r ${u} n ${d} 320 ${n} 1
      done | tee dtrmm_r_${u}_n_${d}_320.txt
    done
    for d in u n
      do
      for n in {16..1024..16}
        do
        echo dtrmm r ${u} t ${d} 320 ${n}
        ./dtrmm r ${u} t ${d} 320 ${n} 1
      done | tee dtrmm_r_${u}_t_${d}_320.txt
    done
  done
fi

if [ -x cgemm ]
  then
  for k in {16..1024..16}
    do
    echo cgemm n n 320 384 ${k}
    ./cgemm n n 320 384 ${k} 1
  done | tee cgemm_n_n_320_384.txt

  for k in {16..1024..16}
    do
    echo cgemm n t 320 384 ${k}
    ./cgemm n t 320 384 ${k} 1
  done | tee cgemm_n_t_320_384.txt

  for k in {16..1024..16}
    do
    echo cgemm n c 320 384 ${k}
    ./cgemm n c 320 384 ${k} 1
  done | tee cgemm_n_c_320_384.txt

  for k in {8..1024..8}
    do
    echo cgemm t n 256 240 ${k}
    ./cgemm t n 256 240 ${k} 1
  done | tee cgemm_t_n_256_240.txt

  for k in {8..1024..8}
    do
    echo cgemm t t 256 240 ${k}
    ./cgemm t t 256 240 ${k} 1
  done | tee cgemm_t_t_256_240.txt

  for k in {8..1024..8}
    do
    echo cgemm t c 256 240 ${k}
    ./cgemm t c 256 240 ${k} 1
  done | tee cgemm_t_c_256_240.txt

  for k in {8..1024..8}
    do
    echo cgemm c n 256 240 ${k}
    ./cgemm c n 256 240 ${k} 1
  done | tee cgemm_c_n_256_240.txt

  for k in {8..1024..8}
    do
    echo cgemm c t 256 240 ${k}
    ./cgemm c t 256 240 ${k} 1
  done | tee cgemm_c_t_256_240.txt

  for k in {8..1024..8}
    do
    echo cgemm c c 256 240 ${k}
    ./cgemm c c 256 240 ${k} 1
  done | tee cgemm_c_c_256_240.txt
fi

if [ -x cherk ]
  then
  for k in {16..1024..16}
    do
    echo cherk u n 320 ${k}
    ./cherk u n 320 ${k} 1
  done | tee cherk_u_n_320.txt

  for k in {8..1024..8}
    do
    echo cherk u t 256 ${k}
    ./cherk u t 256 ${k} 1
  done | tee cherk_u_t_256.txt

  for k in {8..1024..8}
    do
    echo cherk u c 256 ${k}
    ./cherk u c 256 ${k} 1
  done | tee cherk_u_c_256.txt

  for k in {16..1024..16}
    do
    echo cherk l n 320 ${k}
    ./cherk l n 320 ${k} 1
  done | tee cherk_l_n_320.txt

  for k in {8..1024..8}
    do
    echo cherk l t 256 ${k}
    ./cherk l t 256 ${k} 1
  done | tee cherk_l_t_256.txt

  for k in {8..1024..8}
    do
    echo cherk l c 256 ${k}
    ./cherk l c 256 ${k} 1
  done | tee cherk_l_c_256.txt
fi

if [ -x ctrsm ]
  then
  for u in u l
    do
    for t in n t c
      do
      for d in u n
        do
        for m in {4..512..4}
          do
          echo ctrsm l ${u} ${t} ${d} ${m} 3840
          ./ctrsm l ${u} ${t} ${d} ${m} 3840 1
        done | tee ctrsm_l_${u}_${t}_${d}_3840.txt
      done
    done
  done

  for u in u l
    do
    for t in n t c
      do
      for d in u n
        do
        for n in {4..512..4}
          do
          echo ctrsm r ${u} ${t} ${d} 3840 ${n}
          ./ctrsm r ${u} ${t} ${d} 3840 ${n} 1
        done | tee ctrsm_r_${u}_${t}_${d}_3840.txt
      done
    done
  done
fi

if [ -x ctrmm ]
  then
  for u in u l
    do
    for d in u n
      do
      for m in {16..1024..16}
        do
        echo ctrmm l ${u} n ${d} ${m} 384
        ./ctrmm l ${u} n ${d} ${m} 384 1
      done | tee ctrmm_l_${u}_n_${d}_384.txt
    done
    for d in u n
      do
      for m in {8..1024..8}
        do
        echo ctrmm l ${u} t ${d} ${m} 240
        ./ctrmm l ${u} t ${d} ${m} 240 1
      done | tee ctrmm_l_${u}_t_${d}_240.txt
    done
    for d in u n
      do
      for m in {8..1024..8}
        do
        echo ctrmm l ${u} c ${d} ${m} 240
        ./ctrmm l ${u} c ${d} ${m} 240 1
      done | tee ctrmm_l_${u}_c_${d}_240.txt
    done
  done

  for u in u l
    do
    for d in u n
      do
      for n in {16..1024..16}
        do
        echo ctrmm r ${u} n ${d} 320 ${n}
        ./ctrmm r ${u} n ${d} 320 ${n} 1
      done | tee ctrmm_r_${u}_n_${d}_320.txt
    done
    for d in u n
      do
      for n in {16..1024..16}
        do
        echo ctrmm r ${u} t ${d} 320 ${n}
        ./ctrmm r ${u} t ${d} 320 ${n} 1
      done | tee ctrmm_r_${u}_t_${d}_320.txt
    done
    for d in u n
      do
      for n in {16..1024..16}
        do
        echo ctrmm r ${u} c ${d} 320 ${n}
        ./ctrmm r ${u} c ${d} 320 ${n} 1
      done | tee ctrmm_r_${u}_c_${d}_320.txt
    done
  done
fi

if [ -x zgemm ]
  then
  for k in {16..1024..16}
    do
    echo zgemm n n 256 240 ${k}
    ./zgemm n n 256 240 ${k} 1
  done | tee zgemm_n_n_256_240.txt

  for k in {16..1024..16}
    do
    echo zgemm n t 256 240 ${k}
    ./zgemm n t 256 240 ${k} 1
  done | tee zgemm_n_t_256_240.txt

  for k in {16..1024..16}
    do
    echo zgemm n c 256 240 ${k}
    ./zgemm n c 256 240 ${k} 1
  done | tee zgemm_n_c_256_240.txt

  for k in {4..1024..4}
    do
    echo zgemm t n 120 160 ${k}
    ./zgemm t n 120 160 ${k} 1
  done | tee zgemm_t_n_120_160.txt

  for k in {8..1024..8}
    do
    echo zgemm t t 120 128 ${k}
    ./zgemm t t 120 128 ${k} 1
  done | tee zgemm_t_t_120_128.txt

  for k in {8..1024..8}
    do
    echo zgemm t c 120 128 ${k}
    ./zgemm t c 120 128 ${k} 1
  done | tee zgemm_t_c_120_128.txt

  for k in {4..1024..4}
    do
    echo zgemm c n 120 160 ${k}
    ./zgemm c n 120 160 ${k} 1
  done | tee zgemm_c_n_120_160.txt

  for k in {8..1024..8}
    do
    echo zgemm c t 120 128 ${k}
    ./zgemm c t 120 128 ${k} 1
  done | tee zgemm_c_t_120_128.txt

  for k in {8..1024..8}
    do
    echo zgemm c c 120 128 ${k}
    ./zgemm c c 120 128 ${k} 1
  done | tee zgemm_c_c_120_128.txt
fi

if [ -x zherk ]
  then
  for k in {16..1024..16}
    do
    echo zherk u n 256 ${k}
    ./zherk u n 256 ${k} 1
  done | tee zherk_u_n_256.txt

  for k in {4..1024..4}
    do
    echo zherk u t 120 ${k}
    ./zherk u t 120 ${k} 1
  done | tee zherk_u_t_120.txt

  for k in {4..1024..4}
    do
    echo zherk u c 120 ${k}
    ./zherk u c 120 ${k} 1
  done | tee zherk_u_c_120.txt

  for k in {16..1024..16}
    do
    echo zherk l n 256 ${k}
    ./zherk l n 256 ${k} 1
  done | tee zherk_l_n_256.txt

  for k in {4..1024..4}
    do
    echo zherk l t 120 ${k}
    ./zherk l t 120 ${k} 1
  done | tee zherk_l_t_120.txt

  for k in {4..1024..4}
    do
    echo zherk l c 120 ${k}
    ./zherk l c 120 ${k} 1
  done | tee zherk_l_c_120.txt
fi

if [ -x ztrsm ]
  then
  for u in u l
    do
    for t in n t c
      do
      for d in u n
        do
        for m in {2..512..2}
          do
          echo ztrsm l ${u} ${t} ${d} ${m} 1920
          ./ztrsm l ${u} ${t} ${d} ${m} 1920 1
        done | tee ztrsm_l_${u}_${t}_${d}_1920.txt
      done
    done
  done

  for u in u l
    do
    for t in n t c
      do
      for d in u n
        do
        for n in {2..512..2}
          do
          echo ztrsm r ${u} ${t} ${d} 1920 ${n}
          ./ztrsm r ${u} ${t} ${d} 1920 ${n} 1
        done | tee ztrsm_r_${u}_${t}_${d}_1920.txt
      done
    done
  done
fi

if [ -x ztrmm ]
  then
  for u in u l
    do
    for d in u n
      do
      for m in {16..1024..16}
        do
        echo ztrmm l ${u} n ${d} ${m} 240
        ./ztrmm l ${u} n ${d} ${m} 240 1
      done | tee ztrmm_l_${u}_n_${d}_240.txt
    done
    for d in u n
      do
      for m in {4..1024..4}
        do
        echo ztrmm l ${u} t ${d} ${m} 160
        ./ztrmm l ${u} t ${d} ${m} 160 1
      done | tee ztrmm_l_${u}_t_${d}_160.txt
    done
    for d in u n
      do
      for m in {4..1024..4}
        do
        echo ztrmm l ${u} c ${d} ${m} 160
        ./ztrmm l ${u} c ${d} ${m} 160 1
      done | tee ztrmm_l_${u}_c_${d}_160.txt
    done
  done

  for u in u l
    do
    for d in u n
      do
      for n in {16..1024..16}
        do
        echo ztrmm r ${u} n ${d} 256 ${n}
        ./ztrmm r ${u} n ${d} 256 ${n} 1
      done | tee ztrmm_r_${u}_n_${d}_256.txt
    done
    for d in u n
      do
      for n in {8..1024..8}
        do
        echo ztrmm r ${u} t ${d} 256 ${n}
        ./ztrmm r ${u} t ${d} 256 ${n} 1
      done | tee ztrmm_r_${u}_t_${d}_256.txt
    done
    for d in u n
      do
      for n in {8..1024..8}
        do
        echo ztrmm r ${u} c ${d} 256 ${n}
        ./ztrmm r ${u} c ${d} 256 ${n} 1
      done | tee ztrmm_r_${u}_c_${d}_256.txt
    done
  done
fi
