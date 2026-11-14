#!/bin/bash
gpu=$1; name=$2; limit=$3; poslr=$4; shift 4
out=/home/ubuntu/d-work/gpt/bench/lab-review/runs; mkdir -p $out; cd /home/ubuntu/gpt-pp
CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=/home/ubuntu/gpt-pp timeout -s TERM $limit /home/ubuntu/d-work/envs/gpt/bin/python /home/ubuntu/d-work/gpt/bench/lab-review/pos_rate.py $poslr --run $name --out $out --results none --no-compile --overwrite "$@" > $out/$name.out 2>&1
echo "$name exit=$? $(grep -E '^eval|^done' $out/$name.out | tr '\n' ';' | cut -c1-900)"
