#!/bin/bash
# usage: run.sh <gpu> <name> <timeout-seconds> <trainer args...>   (scratch runs of the review; never writes into the repository)
gpu=$1; name=$2; limit=$3; shift 3
out=/home/ubuntu/d-work/gpt/bench/lab-review/runs
mkdir -p $out
cd /home/ubuntu/gpt-pp
CUDA_VISIBLE_DEVICES=$gpu timeout -s TERM $limit /home/ubuntu/d-work/envs/gpt/bin/python -m gptpp.train --run $name --out $out --results none --no-compile --overwrite "$@" > $out/$name.out 2>&1
echo "$name exit=$? $(grep -E '^eval|^done' $out/$name.out | tr '\n' ';' | cut -c1-900)"
