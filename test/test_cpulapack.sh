#!/bin/bash

if [ -x spotrf ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo spotrf ${u} ${n}
      ./spotrf ${u} ${n} 1
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
      ./spotri ${u} ${n} 1
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
        ./strtri ${u} ${d} ${n} 1
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
      ./slauum ${u} ${n} 1
    done | tee slauum_${u}.txt
  done
fi

if [ -x slogdet ]
  then
  for n in {1024..1048576..1024}
    do
    echo slogdet ${n}
    ./slogdet ${n} 1
  done | tee slogdet.txt
fi

if [ -x dpotrf ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo dpotrf ${u} ${n}
      ./dpotrf ${u} ${n} 1
    done | tee dpotrf_${u}.txt
  done
fi

if [ -x dpotri ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo dpotri ${u} ${n}
      ./dpotri ${u} ${n} 1
    done | tee dpotri_${u}.txt
  done
fi

if [ -x dtrtri ]
  then
  for u in u l
    do
    for d in u n
      do
      for n in {64..4096..64}
        do
        echo dtrtri ${u} ${d} ${n}
        ./dtrtri ${u} ${d} ${n} 1
      done
    done | tee dtrtri_${u}_${d}.txt
  done
fi

if [ -x dlauum ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo dlauum ${u} ${n}
      ./dlauum ${u} ${n} 1
    done | tee dlauum_${u}.txt
  done
fi

if [ -x dlogdet ]
  then
  for n in {1024..1048576..1024}
    do
    echo dlogdet ${n}
    ./dlogdet ${n} 1
  done | tee dlogdet.txt
fi

if [ -x cpotrf ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo cpotrf ${u} ${n}
      ./cpotrf ${u} ${n} 1
    done | tee cpotrf_${u}.txt
  done
fi

if [ -x cpotri ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo cpotri ${u} ${n}
      ./cpotri ${u} ${n} 1
    done | tee cpotri_${u}.txt
  done
fi

if [ -x ctrtri ]
  then
  for u in u l
    do
    for d in u n
      do
      for n in {64..4096..64}
        do
        echo ctrtri ${u} ${d} ${n}
        ./ctrtri ${u} ${d} ${n} 1
      done
    done | tee ctrtri_${u}_${d}.txt
  done
fi

if [ -x clauum ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo clauum ${u} ${n}
      ./clauum ${u} ${n} 1
    done | tee clauum_${u}.txt
  done
fi

if [ -x clogdet ]
  then
  for n in {1024..1048576..1024}
    do
    echo clogdet ${n}
    ./clogdet ${n} 1
  done | tee clogdet.txt
fi

if [ -x zpotrf ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo zpotrf ${u} ${n}
      ./zpotrf ${u} ${n} 1
    done | tee zpotrf_${u}.txt
  done
fi

if [ -x zpotri ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo zpotri ${u} ${n}
      ./zpotri ${u} ${n} 1
    done | tee zpotri_${u}.txt
  done
fi

if [ -x ztrtri ]
  then
  for u in u l
    do
    for d in u n
      do
      for n in {64..4096..64}
        do
        echo ztrtri ${u} ${d} ${n}
        ./ztrtri ${u} ${d} ${n} 1
      done
    done | tee ztrtri_${u}_${d}.txt
  done
fi

if [ -x zlauum ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo zlauum ${u} ${n}
      ./zlauum ${u} ${n} 1
    done | tee zlauum_${u}.txt
  done
fi

if [ -x zlogdet ]
  then
  for n in {1024..1048576..1024}
    do
    echo zlogdet ${n}
    ./zlogdet ${n} 1
  done | tee zlogdet.txt
fi
