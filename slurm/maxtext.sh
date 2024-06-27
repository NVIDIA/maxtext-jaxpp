#!/usr/bin/env bash

if [ -z ${MODEL} ]; then
  echo "Expect a model name: gpt3-175b or llama2-70b"
  exit 1
fi

set -x

# Overwritable vars
NUM_NODES=${NUM_NODES:-8}
PP=${PP:-8}
DP=${DP:-1}
TP=${TP:-8}
VP=${VP:-3}
SEQ_LEN=${SEQ_LEN:-2048}
DTYPE=${DTYPE:-"bfloat16"}
STEPS=${STEPS:-10}
MBS=${MBS:-2}
GA=${GA:-64}
USE_PGLE=${USE_PGLE:-"False"}
SCHEDULE=${SCHEDULE:-"interleaved_1f1b"}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-coreai_dlalgo_llm}
SLURM_JOB=${SLURM_JOB:-${SLURM_ACCOUNT}-jaxpp:${USER}-train-maxtext}
SLURM_TIME=${SLURM_TIME:-"01:00:00"}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-"."}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-"gitlab-master.nvidia.com/cml/jaxpp_dev/maxtext:7b7eecce"}

# Non-overwritable vars
container_logs_dir=/tmp/logs
timestamp=$(date +%Y%m%d-%H%M%S)
jaxpp_dir="./third_party/jaxpp"
log_dir="/workdir/maxtext/third_party/jaxpp/logs/"
# NOTE: output_dir should not contain `:` or `,` since that breaks
#   pyxis' `--container-mounts`
output_dir=$(mktemp -d -p ${SLURM_LOG_DIR} "maxtext-${SLURM_LOG_TAG:+${SLURM_LOG_TAG}_}$timestamp""_XXXX")
hostname=$(hostname)
if [[ "${hostname}" == cs-oci-ord-login-* ]]; then
partition=polar
gpus_per_node=8
elif [[ "${hostname}" == login-eos* ]]; then
partition=batch
fi

command="python /workdir/maxtext/MaxText/train.py /workdir/maxtext/MaxText/configs/base.yml \
        run_name=runner_jaxpp_${timestamp} base_output_directory=$container_logs_dir        \
        model_name=${MODEL} dtype=${DTYPE} steps=${STEPS}                                   \
        ici_tensor_parallelism=${TP} dcn_data_parallelism=${DP}                             \
        hardware=gpu dataset_type=synthetic enable_checkpointing=False                      \
        per_device_batch_size=$(( ($MBS * $GA) / ($PP * $TP) ))                             \
        num_microbatches=${GA} max_target_length=${SEQ_LEN}                                 \
        num_workers=${PP} num_stages=$((${VP} * ${PP}))                                     \
        use_jaxpp=True schedule=${SCHEDULE}                                                 \
        use_pgle=${USE_PGLE} distributed_initialization=True"

sbatch_flags="--chdir=${output_dir}                                                         \
        -A ${SLURM_ACCOUNT} -J ${SLURM_JOB} -p ${partition}                                 \
        -N ${NUM_NODES} ${gpus_per_node:+--gpus-per-node=${gpus_per_node}}                  \
        --time=${SLURM_TIME} -o slurm_out.log -e slurm_err.log"

common_srun_flags="--label --container-image=${CONTAINER_IMAGE} --container-mounts=$(realpath $output_dir):${log_dir},$(realpath .):/workdir/maxtext"

sbatch ${sbatch_flags} "${jaxpp_dir}/script/slurm/ray-on-slurm.sh" "${command}" "${common_srun_flags}"
