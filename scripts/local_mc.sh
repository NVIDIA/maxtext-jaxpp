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

if [ -z "$N_PROCS" ] || [ -z "$N_GPUS" ] || [ -z "$COMMAND" ]; then
  echo "N_PROCS, N_GPUS, and COMMAND must be set"
  exit 1
fi

seq 0 $(($N_PROCS - 1)) | xargs -P $N_PROCS -I {} bash -c ' \
n_gpus=$2; \
start=$(({} * n_gpus)); \
end=$((start + n_gpus - 1)); \
JAX_COORDINATOR_IP="localhost" JAX_COORDINATOR_PORT=1234 NNODES=$1 NODE_RANK={} \
JAXPP_ENABLE_LICM=1 \
NVTE_FUSED_ATTN=1 \
CUDA_VISIBLE_DEVICES=$(seq -s, $start $end) $3' _ $N_PROCS $N_GPUS "$COMMAND"
