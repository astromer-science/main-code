#!/bin/bash
 

phycores=$(echo $sudoPW|cat - /proc/cpuinfo|grep -m 1 "cpu cores"|awk '{print $ 4;}')

echo $phycores  

for ((i=1;i<=phycores;++i))
do
   echo "Core $i"
   python3 mem_time.py $i
done