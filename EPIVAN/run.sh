#!/usr/bin/bash

Datadir=${1}
for experiment in $(ls ./${Datadir}/ | grep -E "K562_CTCF|K562_POLR2A|MCF7_CTCF|HeLaS3_POLR2A")
do
    echo "working on $experiment."
    
    python train.py -c $(pwd)/${Datadir}/${experiment} -n ${experiment}

done
