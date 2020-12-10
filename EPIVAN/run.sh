#!/usr/bin/bash

Datadir=${1}
for experiment in $(ls ./${Datadir}/ | grep -E "GM12878|HeLa-S3|HUVEC|IMR90|K562|NHEK")
do
    echo "working on $experiment."
    
    python train.py -c $(pwd)/${Datadir}/${experiment} -n ${experiment}

done
