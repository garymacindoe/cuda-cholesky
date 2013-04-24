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
  done | tee cusgemm_t_t_384_480.txt
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

  for k in {8..1024..8}
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
      for n in {16..1024..16}
        do
        echo custrmm r ${u} t ${d} 512 ${n}
        ./custrmm r ${u} t ${d} 512 ${n} 1
      done | tee custrmm_r_${u}_t_${d}_512.txt
    done
  done
fi

if [ -x cudgemm ]
  then
  for k in {16..1024..16}
    do
    echo cudgemm n n 320 384 ${k}
    ./cudgemm n n 320 384 ${k} 1
  done | tee cudgemm_n_n_320_384.txt

  for k in {16..1024..16}
    do
    echo cudgemm n t 320 384 ${k}
    ./cudgemm n t 320 384 ${k} 1
  done | tee cudgemm_n_t_320_384.txt

  for k in {8..1024..8}
    do
    echo cudgemm t n 256 240 ${k}
    ./cudgemm t n 256 240 ${k} 1
  done | tee cudgemm_t_n_256_240.txt

  for k in {8..1024..8}
    do
    echo cudgemm t t 256 240 ${k}
    ./cudgemm t t 256 240 ${k} 1
  done | tee cudgemm_t_t_256_240.txt
fi

if [ -x cudsyrk ]
  then
  for k in {16..1024..16}
    do
    echo cudsyrk u n 320 ${k}
    ./cudsyrk u n 320 ${k} 1
  done | tee cudsyrk_u_n_320.txt

  for k in {8..1024..8}
    do
    echo cudsyrk u t 256 ${k}
    ./cudsyrk u t 256 ${k} 1
  done | tee cudsyrk_u_t_256.txt

  for k in {16..1024..16}
    do
    echo cudsyrk l n 320 ${k}
    ./cudsyrk l n 320 ${k} 1
  done | tee cudsyrk_l_n_320.txt

  for k in {8..1024..8}
    do
    echo cudsyrk l t 256 ${k}
    ./cudsyrk l t 256 ${k} 1
  done | tee cudsyrk_l_t_256.txt
fi

if [ -x cudtrsm ]
  then
  for u in u l
    do
    for t in n t
      do
      for d in u n
        do
        for m in {4..512..4}
          do
          echo cudtrsm l ${u} ${t} ${d} ${m} 3840
          ./cudtrsm l ${u} ${t} ${d} ${m} 3840 1
        done | tee cudtrsm_l_${u}_${t}_${d}_3840.txt
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
          echo cudtrsm r ${u} ${t} ${d} 3840 ${n}
          ./cudtrsm r ${u} ${t} ${d} 3840 ${n} 1
        done | tee cudtrsm_r_${u}_${t}_${d}_3840.txt
      done
    done
  done
fi

if [ -x cudtrmm ]
  then
  for u in u l
    do
    for d in u n
      do
      for m in {16..1024..16}
        do
        echo cudtrmm l ${u} n ${d} ${m} 384
        ./cudtrmm l ${u} n ${d} ${m} 384 1
      done | tee cudtrmm_l_${u}_n_${d}_384.txt
    done
    for d in u n
      do
      for m in {8..1024..8}
        do
        echo cudtrmm l ${u} t ${d} ${m} 240
        ./cudtrmm l ${u} t ${d} ${m} 240 1
      done | tee cudtrmm_l_${u}_t_${d}_240.txt
    done
  done

  for u in u l
    do
    for d in u n
      do
      for n in {16..1024..16}
        do
        echo cudtrmm r ${u} n ${d} 320 ${n}
        ./cudtrmm r ${u} n ${d} 320 ${n} 1
      done | tee cudtrmm_r_${u}_n_${d}_320.txt
    done
    for d in u n
      do
      for n in {16..1024..16}
        do
        echo cudtrmm r ${u} t ${d} 320 ${n}
        ./cudtrmm r ${u} t ${d} 320 ${n} 1
      done | tee cudtrmm_r_${u}_t_${d}_320.txt
    done
  done
fi

if [ -x cucgemm ]
  then
  for k in {16..1024..16}
    do
    echo cucgemm n n 320 384 ${k}
    ./cucgemm n n 320 384 ${k} 1
  done | tee cucgemm_n_n_320_384.txt

  for k in {16..1024..16}
    do
    echo cucgemm n t 320 384 ${k}
    ./cucgemm n t 320 384 ${k} 1
  done | tee cucgemm_n_t_320_384.txt

  for k in {16..1024..16}
    do
    echo cucgemm n c 320 384 ${k}
    ./cucgemm n c 320 384 ${k} 1
  done | tee cucgemm_n_c_320_384.txt

  for k in {8..1024..8}
    do
    echo cucgemm t n 256 240 ${k}
    ./cucgemm t n 256 240 ${k} 1
  done | tee cucgemm_t_n_256_240.txt

  for k in {8..1024..8}
    do
    echo cucgemm t t 256 240 ${k}
    ./cucgemm t t 256 240 ${k} 1
  done | tee cucgemm_t_t_256_240.txt

  for k in {8..1024..8}
    do
    echo cucgemm t c 256 240 ${k}
    ./cucgemm t c 256 240 ${k} 1
  done | tee cucgemm_t_c_256_240.txt

  for k in {8..1024..8}
    do
    echo cucgemm c n 256 240 ${k}
    ./cucgemm c n 256 240 ${k} 1
  done | tee cucgemm_c_n_256_240.txt

  for k in {8..1024..8}
    do
    echo cucgemm c t 256 240 ${k}
    ./cucgemm c t 256 240 ${k} 1
  done | tee cucgemm_c_t_256_240.txt

  for k in {8..1024..8}
    do
    echo cucgemm c c 256 240 ${k}
    ./cucgemm c c 256 240 ${k} 1
  done | tee cucgemm_c_c_256_240.txt
fi

if [ -x cucherk ]
  then
  for k in {16..1024..16}
    do
    echo cucherk u n 320 ${k}
    ./cucherk u n 320 ${k} 1
  done | tee cucherk_u_n_320.txt

  for k in {8..1024..8}
    do
    echo cucherk u t 256 ${k}
    ./cucherk u t 256 ${k} 1
  done | tee cucherk_u_t_256.txt

  for k in {8..1024..8}
    do
    echo cucherk u c 256 ${k}
    ./cucherk u c 256 ${k} 1
  done | tee cucherk_u_c_256.txt

  for k in {16..1024..16}
    do
    echo cucherk l n 320 ${k}
    ./cucherk l n 320 ${k} 1
  done | tee cucherk_l_n_320.txt

  for k in {8..1024..8}
    do
    echo cucherk l t 256 ${k}
    ./cucherk l t 256 ${k} 1
  done | tee cucherk_l_t_256.txt

  for k in {8..1024..8}
    do
    echo cucherk l c 256 ${k}
    ./cucherk l c 256 ${k} 1
  done | tee cucherk_l_c_256.txt
fi

if [ -x cuctrsm ]
  then
  for u in u l
    do
    for t in n t c
      do
      for d in u n
        do
        for m in {4..512..4}
          do
          echo cuctrsm l ${u} ${t} ${d} ${m} 3840
          ./cuctrsm l ${u} ${t} ${d} ${m} 3840 1
        done | tee cuctrsm_l_${u}_${t}_${d}_3840.txt
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
          echo cuctrsm r ${u} ${t} ${d} 3840 ${n}
          ./cuctrsm r ${u} ${t} ${d} 3840 ${n} 1
        done | tee cuctrsm_r_${u}_${t}_${d}_3840.txt
      done
    done
  done
fi

if [ -x cuctrmm ]
  then
  for u in u l
    do
    for d in u n
      do
      for m in {16..1024..16}
        do
        echo cuctrmm l ${u} n ${d} ${m} 384
        ./cuctrmm l ${u} n ${d} ${m} 384 1
      done | tee cuctrmm_l_${u}_n_${d}_384.txt
    done
    for d in u n
      do
      for m in {8..1024..8}
        do
        echo cuctrmm l ${u} t ${d} ${m} 240
        ./cuctrmm l ${u} t ${d} ${m} 240 1
      done | tee cuctrmm_l_${u}_t_${d}_240.txt
    done
    for d in u n
      do
      for m in {8..1024..8}
        do
        echo cuctrmm l ${u} c ${d} ${m} 240
        ./cuctrmm l ${u} c ${d} ${m} 240 1
      done | tee cuctrmm_l_${u}_c_${d}_240.txt
    done
  done

  for u in u l
    do
    for d in u n
      do
      for n in {16..1024..16}
        do
        echo cuctrmm r ${u} n ${d} 320 ${n}
        ./cuctrmm r ${u} n ${d} 320 ${n} 1
      done | tee cuctrmm_r_${u}_n_${d}_320.txt
    done
    for d in u n
      do
      for n in {16..1024..16}
        do
        echo cuctrmm r ${u} t ${d} 320 ${n}
        ./cuctrmm r ${u} t ${d} 320 ${n} 1
      done | tee cuctrmm_r_${u}_t_${d}_320.txt
    done
    for d in u n
      do
      for n in {16..1024..16}
        do
        echo cuctrmm r ${u} c ${d} 320 ${n}
        ./cuctrmm r ${u} c ${d} 320 ${n} 1
      done | tee cuctrmm_r_${u}_c_${d}_320.txt
    done
  done
fi

if [ -x cuzgemm ]
  then
  for k in {16..1024..16}
    do
    echo cuzgemm n n 256 240 ${k}
    ./cuzgemm n n 256 240 ${k} 1
  done | tee cuzgemm_n_n_256_240.txt

  for k in {16..1024..16}
    do
    echo cuzgemm n t 256 240 ${k}
    ./cuzgemm n t 256 240 ${k} 1
  done | tee cuzgemm_n_t_256_240.txt

  for k in {16..1024..16}
    do
    echo cuzgemm n c 256 240 ${k}
    ./cuzgemm n c 256 240 ${k} 1
  done | tee cuzgemm_n_c_256_240.txt

  for k in {4..1024..4}
    do
    echo cuzgemm t n 120 160 ${k}
    ./cuzgemm t n 120 160 ${k} 1
  done | tee cuzgemm_t_n_120_160.txt

  for k in {8..1024..8}
    do
    echo cuzgemm t t 120 128 ${k}
    ./cuzgemm t t 120 128 ${k} 1
  done | tee cuzgemm_t_t_120_128.txt

  for k in {8..1024..8}
    do
    echo cuzgemm t c 120 128 ${k}
    ./cuzgemm t c 120 128 ${k} 1
  done | tee cuzgemm_t_c_120_128.txt

  for k in {4..1024..4}
    do
    echo cuzgemm c n 120 160 ${k}
    ./cuzgemm c n 120 160 ${k} 1
  done | tee cuzgemm_c_n_120_160.txt

  for k in {8..1024..8}
    do
    echo cuzgemm c t 120 128 ${k}
    ./cuzgemm c t 120 128 ${k} 1
  done | tee cuzgemm_c_t_120_128.txt

  for k in {8..1024..8}
    do
    echo cuzgemm c c 120 128 ${k}
    ./cuzgemm c c 120 128 ${k} 1
  done | tee cuzgemm_c_c_120_128.txt
fi

if [ -x cuzherk ]
  then
  for k in {16..1024..16}
    do
    echo cuzherk u n 256 ${k}
    ./cuzherk u n 256 ${k} 1
  done | tee cuzherk_u_n_256.txt

  for k in {4..1024..4}
    do
    echo cuzherk u t 120 ${k}
    ./cuzherk u t 120 ${k} 1
  done | tee cuzherk_u_t_120.txt

  for k in {4..1024..4}
    do
    echo cuzherk u c 120 ${k}
    ./cuzherk u c 120 ${k} 1
  done | tee cuzherk_u_c_120.txt

  for k in {16..1024..16}
    do
    echo cuzherk l n 256 ${k}
    ./cuzherk l n 256 ${k} 1
  done | tee cuzherk_l_n_256.txt

  for k in {4..1024..4}
    do
    echo cuzherk l t 120 ${k}
    ./cuzherk l t 120 ${k} 1
  done | tee cuzherk_l_t_120.txt

  for k in {4..1024..4}
    do
    echo cuzherk l c 120 ${k}
    ./cuzherk l c 120 ${k} 1
  done | tee cuzherk_l_c_120.txt
fi

if [ -x cuztrsm ]
  then
  for u in u l
    do
    for t in n t c
      do
      for d in u n
        do
        for m in {2..512..2}
          do
          echo cuztrsm l ${u} ${t} ${d} ${m} 1920
          ./cuztrsm l ${u} ${t} ${d} ${m} 1920 1
        done | tee cuztrsm_l_${u}_${t}_${d}_1920.txt
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
          echo cuztrsm r ${u} ${t} ${d} 1920 ${n}
          ./cuztrsm r ${u} ${t} ${d} 1920 ${n} 1
        done | tee cuztrsm_r_${u}_${t}_${d}_1920.txt
      done
    done
  done
fi

if [ -x cuztrmm ]
  then
  for u in u l
    do
    for d in u n
      do
      for m in {16..1024..16}
        do
        echo cuztrmm l ${u} n ${d} ${m} 240
        ./cuztrmm l ${u} n ${d} ${m} 240 1
      done | tee cuztrmm_l_${u}_n_${d}_240.txt
    done
    for d in u n
      do
      for m in {4..1024..4}
        do
        echo cuztrmm l ${u} t ${d} ${m} 160
        ./cuztrmm l ${u} t ${d} ${m} 160 1
      done | tee cuztrmm_l_${u}_t_${d}_160.txt
    done
    for d in u n
      do
      for m in {4..1024..4}
        do
        echo cuztrmm l ${u} c ${d} ${m} 160
        ./cuztrmm l ${u} c ${d} ${m} 160 1
      done | tee cuztrmm_l_${u}_c_${d}_160.txt
    done
  done

  for u in u l
    do
    for d in u n
      do
      for n in {16..1024..16}
        do
        echo cuztrmm r ${u} n ${d} 256 ${n}
        ./cuztrmm r ${u} n ${d} 256 ${n} 1
      done | tee cuztrmm_r_${u}_n_${d}_256.txt
    done
    for d in u n
      do
      for n in {8..1024..8}
        do
        echo cuztrmm r ${u} t ${d} 256 ${n}
        ./cuztrmm r ${u} t ${d} 256 ${n} 1
      done | tee cuztrmm_r_${u}_t_${d}_256.txt
    done
    for d in u n
      do
      for n in {8..1024..8}
        do
        echo cuztrmm r ${u} c ${d} 256 ${n}
        ./cuztrmm r ${u} c ${d} 256 ${n} 1
      done | tee cuztrmm_r_${u}_c_${d}_256.txt
    done
  done
fi
