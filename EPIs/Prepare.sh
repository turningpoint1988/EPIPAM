#!/usr/bin/bash

command=${1}
threadnum=3
tmp="/tmp/$$.fifo"
mkfifo ${tmp}
exec 6<> ${tmp}
rm ${tmp}
for((i=0; i<${threadnum}; i++))
do
    echo ""
done >&6

for experiment in $(ls ./ | grep -E "GM12878|HeLa-S3|HUVEC|IMR90|K562|NHEK")
do
  read -u6
  {
    echo "working on $experiment."
    if [ ! -d ./EPIs/$experiment/data ]; then
        mkdir ./EPIs/$experiment/data

    fi
    
    python DataPrepare_${command}.py -c `pwd`/EPIs/$experiment \
                                     -n $experiment 
    
    echo "" >&6
  }&
done
wait
exec 6>&-
