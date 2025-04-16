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

n_procs=8
n_gpus=1
command="python -u MaxText/train.py MaxText/configs/base.yml                                       \
        run_name=runner_jaxpp_$(date +%Y-%m-%d-%H-%M) base_output_directory=/tmp/log        \
        hardware=gpu dataset_type=synthetic model_name=gpt3-52k steps=20 dtype=bfloat16     \
        max_target_length=2048 per_device_batch_size=16                                      \
        dcn_data_parallelism=1 ici_data_parallelism=2 ici_tensor_parallelism=2              \
        ici_pipeline_parallelism=2 num_pipeline_repeats=2 num_pipeline_microbatches=32       \
        enable_checkpointing=false use_jaxpp=True jaxpp_remote=False schedule=interleaved_1f1b    \
        distributed_initialization=True use_pgle=False profiler=xplane"

seq 0 $((n_procs - 1)) | xargs -P $n_procs -I {} bash -c ' \
n_gpus=$2; \
start=$(({} * n_gpus)); \
end=$((start + n_gpus - 1)); \
JAX_COORDINATOR_IP="localhost" JAX_COORDINATOR_PORT=1234 NNODES=$1 NODE_RANK={} \
CUDA_VISIBLE_DEVICES=$(seq -s, $start $end) $3 1> {}.out 2> {}.err' _ $n_procs $n_gpus "$command"
