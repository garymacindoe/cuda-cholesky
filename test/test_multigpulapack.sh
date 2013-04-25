#!/bin/bash

if [ -x cumultigpuspotrf ]
  then
  for u in u l
    do
    for n in {512..8192..512}
      do
      echo cumultigpuspotrf ${u} ${n}
      ./cumultigpuspotrf ${u} ${n} 1
    done | tee cumultigpuspotrf_${u}.txt
  done
fi

if [ -x cumultigpuspotri ]
  then
  for u in u l
    do
    for n in {512..8192..512}
      do
      echo cumultigpuspotri ${u} ${n}
      ./cumultigpuspotri ${u} ${n} 1
    done | tee cumultigpuspotri_${u}.txt
  done
fi

if [ -x cumultigpustrtri ]
  then
  for u in u l
    do
    for d in u n
      do
      for n in {512..8192..512}
        do
        echo cumultigpustrtri ${u} ${d} ${n}
        ./cumultigpustrtri ${u} ${d} ${n} 1
      done
    done | tee cumultigpustrtri_${u}_${d}.txt
  done
fi

if [ -x cumultigpuslauum ]
  then
  for u in u l
    do
    for n in {512..8192..512}
      do
      echo cumultigpuslauum ${u} ${n}
      ./cumultigpuslauum ${u} ${n} 1
    done | tee cumultigpuslauum_${u}.txt
  done
fi


if [ -x cumultigpudpotrf ]
  then
  for u in u l
    do
    for n in {512..8192..512}
      do
      echo cumultigpudpotrf ${u} ${n}
      ./cumultigpudpotrf ${u} ${n} 1
    done | tee cumultigpudpotrf_${u}.txt
  done
fi

if [ -x cumultigpudpotri ]
  then
  for u in u l
    do
    for n in {512..8192..512}
      do
      echo cumultigpudpotri ${u} ${n}
      ./cumultigpudpotri ${u} ${n} 1
    done | tee cumultigpudpotri_${u}.txt
  done
fi

if [ -x cumultigpudtrtri ]
  then
  for u in u l
    do
    for d in u n
      do
      for n in {512..8192..512}
        do
        echo cumultigpudtrtri ${u} ${d} ${n}
        ./cumultigpudtrtri ${u} ${d} ${n} 1
      done
    done | tee cumultigpudtrtri_${u}_${d}.txt
  done
fi

if [ -x cumultigpudlauum ]
  then
  for u in u l
    do
    for n in {512..8192..512}
      do
      echo cumultigpudlauum ${u} ${n}
      ./cumultigpudlauum ${u} ${n} 1
    done | tee cumultigpudlauum_${u}.txt
  done
fi


if [ -x cumultigpucpotrf ]
  then
  for u in u l
    do
    for n in {512..8192..512}
      do
      echo cumultigpucpotrf ${u} ${n}
      ./cumultigpucpotrf ${u} ${n} 1
    done | tee cumultigpucpotrf_${u}.txt
  done
fi

if [ -x cumultigpucpotri ]
  then
  for u in u l
    do
    for n in {512..8192..512}
      do
      echo cumultigpucpotri ${u} ${n}
      ./cumultigpucpotri ${u} ${n} 1
    done | tee cumultigpucpotri_${u}.txt
  done
fi

if [ -x cumultigpuctrtri ]
  then
  for u in u l
    do
    for d in u n
      do
      for n in {512..8192..512}
        do
        echo cumultigpuctrtri ${u} ${d} ${n}
        ./cumultigpuctrtri ${u} ${d} ${n} 1
      done
    done | tee cumultigpuctrtri_${u}_${d}.txt
  done
fi

if [ -x cumultigpuclauum ]
  then
  for u in u l
    do
    for n in {512..8192..512}
      do
      echo cumultigpuclauum ${u} ${n}
      ./cumultigpuclauum ${u} ${n} 1
    done | tee cumultigpuclauum_${u}.txt
  done
fi

if [ -x cumultigpuclogdet ]
  then
  for n in {1024..1048576..1024}
    do
    echo cumultigpuclogdet ${n}
    ./cumultigpuclogdet ${n} 1
  done | tee cumultigpuclogdet.txt
fi

if [ -x cumultigpuzpotrf ]
  then
  for u in u l
    do
    for n in {512..8192..512}
      do
      echo cumultigpuzpotrf ${u} ${n}
      ./cumultigpuzpotrf ${u} ${n} 1
    done | tee cumultigpuzpotrf_${u}.txt
  done
fi

if [ -x cumultigpuzpotri ]
  then
  for u in u l
    do
    for n in {512..8192..512}
      do
      echo cumultigpuzpotri ${u} ${n}
      ./cumultigpuzpotri ${u} ${n} 1
    done | tee cumultigpuzpotri_${u}.txt
  done
fi

if [ -x cumultigpuztrtri ]
  then
  for u in u l
    do
    for d in u n
      do
      for n in {512..8192..512}
        do
        echo cumultigpuztrtri ${u} ${d} ${n}
        ./cumultigpuztrtri ${u} ${d} ${n} 1
      done
    done | tee cumultigpuztrtri_${u}_${d}.txt
  done
fi

if [ -x cumultigpuzlauum ]
  then
  for u in u l
    do
    for n in {512..8192..512}
      do
      echo cumultigpuzlauum ${u} ${n}
      ./cumultigpuzlauum ${u} ${n} 1
    done | tee cumultigpuzlauum_${u}.txt
  done
fi

if [ -x cumultigpuzlogdet ]
  then
  for n in {1024..1048576..1024}
    do
    echo cumultigpuzlogdet ${n}
    ./cumultigpuzlogdet ${n} 1
  done | tee cumultigpuzlogdet.txt
fi
