#!/usr/bin/bash

# EPIs or ChIA-PET
Data=${1}
for experiment in $(ls ./${Data}/ | grep -E "GM12878|HeLa-S3|HUVEC|IMR90|K562|NHEK")
do
    echo "working on ${experiment}."
    if [ ! -d ./models/${experiment} ]; then
        mkdir ./models/${experiment}
    else
        continue
    fi
    
    python train_epiattention.py -d `pwd`/${Data}/${experiment} \
                               -n ${experiment} \
                               -g 0 \
                               -b 100 \
                               -lr 0.001 \
                               -e 15 \
                               -w 0.0005 \
                               -c `pwd`/models/${experiment}
done
