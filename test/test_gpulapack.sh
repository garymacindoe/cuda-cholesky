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

if [ -x cudpotrf ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo cudpotrf ${u} ${n}
      ./cudpotrf ${u} ${n} 1
    done | tee cudpotrf_${u}.txt
  done
fi

if [ -x cudpotri ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo cudpotri ${u} ${n}
      ./cudpotri ${u} ${n} 1
    done | tee cudpotri_${u}.txt
  done
fi

if [ -x cudtrtri ]
  then
  for u in u l
    do
    for d in u n
      do
      for n in {64..4096..64}
        do
        echo cudtrtri ${u} ${d} ${n}
        ./cudtrtri ${u} ${d} ${n} 1
      done
    done | tee cudtrtri_${u}_${d}.txt
  done
fi

if [ -x cudlauum ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo cudlauum ${u} ${n}
      ./cudlauum ${u} ${n} 1
    done | tee cudlauum_${u}.txt
  done
fi

if [ -x cudlogdet ]
  then
  for n in {1024..1048576..1024}
    do
    echo cudlogdet ${n}
    ./cudlogdet ${n} 1
  done | tee cudlogdet.txt
fi

if [ -x cucpotrf ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo cucpotrf ${u} ${n}
      ./cucpotrf ${u} ${n} 1
    done | tee cucpotrf_${u}.txt
  done
fi

if [ -x cucpotri ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo cucpotri ${u} ${n}
      ./cucpotri ${u} ${n} 1
    done | tee cucpotri_${u}.txt
  done
fi

if [ -x cuctrtri ]
  then
  for u in u l
    do
    for d in u n
      do
      for n in {64..4096..64}
        do
        echo cuctrtri ${u} ${d} ${n}
        ./cuctrtri ${u} ${d} ${n} 1
      done
    done | tee cuctrtri_${u}_${d}.txt
  done
fi

if [ -x cuclauum ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo cuclauum ${u} ${n}
      ./cuclauum ${u} ${n} 1
    done | tee cuclauum_${u}.txt
  done
fi

if [ -x cuclogdet ]
  then
  for n in {1024..1048576..1024}
    do
    echo cuclogdet ${n}
    ./cuclogdet ${n} 1
  done | tee cuclogdet.txt
fi

if [ -x cuzpotrf ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo cuzpotrf ${u} ${n}
      ./cuzpotrf ${u} ${n} 1
    done | tee cuzpotrf_${u}.txt
  done
fi

if [ -x cuzpotri ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo cuzpotri ${u} ${n}
      ./cuzpotri ${u} ${n} 1
    done | tee cuzpotri_${u}.txt
  done
fi

if [ -x cuztrtri ]
  then
  for u in u l
    do
    for d in u n
      do
      for n in {64..4096..64}
        do
        echo cuztrtri ${u} ${d} ${n}
        ./cuztrtri ${u} ${d} ${n} 1
      done
    done | tee cuztrtri_${u}_${d}.txt
  done
fi

if [ -x cuzlauum ]
  then
  for u in u l
    do
    for n in {64..4096..64}
      do
      echo cuzlauum ${u} ${n}
      ./cuzlauum ${u} ${n} 1
    done | tee cuzlauum_${u}.txt
  done
fi

if [ -x cuzlogdet ]
  then
  for n in {1024..1048576..1024}
    do
    echo cuzlogdet ${n}
    ./cuzlogdet ${n} 1
  done | tee cuzlogdet.txt
fi
