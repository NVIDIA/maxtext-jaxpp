#!/bin/bash

maxtext_dir="$(realpath $(dirname $0)/../)"

export CONTAINER_IMAGE=${CONTAINER_IMAGE:-"gitlab-master.nvidia.com/cml/jaxpp_dev/maxtext:latest"}
export XLA_FLAGS="--xla_gpu_enable_command_buffer=" # needed for PGLE
export USE_PGLE=True
export USE_PROFILER=False
export EXTRA_SLURM_FLAGS="--network=sharp"
export MODEL=gpt3-175b

for i in $(seq 0 4); do
  # `sleep 1` just to give each run a unique time stamp.
  sleep 1
  NUM_NODES=$(((2 ** ${i}) * 8)) TP=8 PP=8 GA=64 MBS=$((2 ** (${i} + 1))) DP=$((2 ** ${i})) SEQ_LEN=2048 ${maxtext_dir}/slurm/maxtext.sh
done
