#!/bin/bash

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"

SCRIPT_NAME="${0##*/}"
echo "Running \`${SCRIPT_NAME}\`"

# Stop execution if any command exits with error
set -ex

# Randomly Generated Vars
OUTPUT_PATH="/dev/shm/${RANDOM}"
RUN_NAME="llama2-$(date +%Y-%m-%d-%H-%M)"

# MaxText Runtime
MAXTEXT_MODEL=""

MAX_TARGET_LENGTH=256
BS_PER_DEVICE=1

# Default Argument Values
BYPASS_ARGUMENTS=""

# Loop through arguments and process them
for arg in "$@"
do
    case $arg in
        --model=*)
        MAXTEXT_MODEL="${arg#*=}"
        shift # Remove --model= from processing
        ;;
        --max_target_length=*)
        MAX_TARGET_LENGTH="${arg#*=}"
        shift # Remove --max_target_length= from processing
        ;;
        --batch_size_per_device=*)
        BS_PER_DEVICE="${arg#*=}"
        shift # Remove --batch_size_per_device= from processing
        ;;
        *)
        BYPASS_ARGUMENTS="${BYPASS_ARGUMENTS} ${arg}"
        ;;
    esac
done

if [[ -z ${MAXTEXT_MODEL} ]]; then
    echo "ERROR: \`--model=<model_name>\` has not been provided."
    exit 1
fi

# ------------- DEBUG FLAGS -------------
# ulimit -c unlimited
# export CUDA_LAUNCH_BLOCKING=1
# export CUDA_COREDUMP_SHOW_PROGRESS=1
# export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
# export CUDA_ENABLE_CPU_COREDUMP_ON_EXCEPTION=1
# export CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1
# export CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1
# export CUDA_COREDUMP_FILE="/workdir/maxtext/core.dump"
# export PYTHONFAULTHANDLER=1
# export JAX_TRACEBACK_FILTERING=off

# export NCCL_IB_SL=1
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NVTE_FUSED_ATTN=1
# export NCCL_DEBUG=VERSION

# Note:
# Communication Threshold:
# - 1 Gib (gibibits)   = 134217728 bytes
# - 0.5 Gib (gibibits) =  67108864 bytes

# Debug XLA Flags
# --xla_disable_hlo_passes=rematerialization
# --xla_disable_all_hlo_passes
# --xla_enable_hlo_passes_only=...,...,...
# --xla_allow_excess_precision

# export XLA_FLAGS="
#     --xla_disable_hlo_passes=rematerialization
#     --xla_dump_to=$OUTPUT_PATH/$RUN_NAME/HLO_dumps/
#     --xla_gpu_all_gather_combine_threshold_bytes=134217728
#     --xla_gpu_all_reduce_combine_threshold_bytes=134217728
#     --xla_gpu_reduce_scatter_combine_threshold_bytes=67108864
#     --xla_gpu_enable_highest_priority_async_stream=true
#     --xla_gpu_enable_latency_hiding_scheduler=true
#     --xla_gpu_enable_pipelined_all_gather=true
#     --xla_gpu_enable_pipelined_all_reduce=true
#     --xla_gpu_enable_pipelined_reduce_scatter=true
#     --xla_gpu_enable_reduce_scatter_combine_by_dim=false
#     --xla_gpu_enable_triton_softmax_fusion=false
#     --xla_allow_excess_precision"

# ------------- DISABLED_XLA_OPTIONS -------------
#    --xla_gpu_enable_triton_gemm=false
#    --xla_gpu_enable_while_loop_double_buffering=true
#    --xla_gpu_graph_level=0: Disable CUDA Graph

# Deprecated: https://github.com/openxla/xla/commit/d7a758b212fe9f9d2ad6fa58ec0e4ad5a1c53ead
#     --xla_gpu_simplify_all_fp_conversions"  # Replace with --xla_allow_excess_precision

# Removed: https://github.com/openxla/xla/commit/4298a065f7b8381d690a95fe6d437cff037b3ff4
#     --xla_gpu_enable_all_gather_combine_by_dim=false
#     --xla_gpu_enable_async_all_gather=true
#     --xla_gpu_enable_async_all_reduce=true
#     --xla_gpu_enable_async_reduce_scatter=true"

CUDA_VISIBLE_DEVICES=0 python ${BASE_DIR}/MaxText/train.py \
	MaxText/configs/base.yml \
	attention=dot_product \
    base_output_directory=${OUTPUT_PATH} \
    dataset_type=synthetic \
    dcn_data_parallelism=1 \
    ici_tensor_parallelism=1 \
    enable_checkpointing=false \
    hardware=gpu \
    max_target_length=${MAX_TARGET_LENGTH} \
    model_name=${MAXTEXT_MODEL} \
    per_device_batch_size=${BS_PER_DEVICE} \
    remat_policy=minimal_flash \
    run_name=${RUN_NAME} \
    steps=30 \
    dtype=float16 \
    ${BYPASS_ARGUMENTS}
