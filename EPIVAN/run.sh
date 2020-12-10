#!/usr/bin/bash

Datadir=${1}
for experiment in $(ls ./${Datadir}/)
do
    echo "working on $experiment."
    
    python train.py -c $(pwd)/${Datadir}/${experiment} -n ${experiment}

done
