#!/bin/bash

cnt=1
cntmax=2

while [ $cnt -le $cntmax ]
do
        echo Run $cnt
        python openmm_run.py $cnt > openmm_run_$cnt.out
        cnt=$(($cnt+1))
done
