#!/usr/bin/bash

command=${1}
threadnum=4
tmp="/tmp/$$.fifo"
mkfifo ${tmp}
exec 6<> ${tmp}
rm ${tmp}
for((i=0; i<${threadnum}; i++))
do
    echo ""
done >&6

for experiment in $(ls ./ChIA-PET/ | grep -E "K562Ctcf|K562Pol2|Mcf7Ctcf|Helas3Pol2")
do
  read -u6
  {
    echo "working on $experiment."
    if [ ! -d ./$experiment/data ]; then
        mkdir ./$experiment/data

    fi
    
    python ./ChIA-PET/DataPrepare_${command}.py -c `pwd`/ChIA-PET/$experiment \
                                                -n $experiment 
    
    echo "" >&6
  }&
done
wait
exec 6>&-

