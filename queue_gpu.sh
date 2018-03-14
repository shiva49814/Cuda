#!/bin/sh

qsub -q mamba -l mem=5gb,nodes=1:ppn=7:gpus=1 -d $(pwd) bench_gpu.sh

