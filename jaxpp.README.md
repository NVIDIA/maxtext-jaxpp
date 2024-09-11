# Overview

This repository is a fork of [MaxText](https://github.com/google/maxtext) created for training with [JaxPP](https://gitlab-master.nvidia.com/CML/jaxpp).

# Notable changes

The changes between this repo and the upstream MaxText is kept minimal in general.
Some of the notable changes are listed below.

* The `__call__` method of the `Decoder` class in [MaxText/layers/models.py](MaxText/layers/models.py)
  calls `jaxpp.pipeline_enter_stage` to mark stage boundaries for pipeline parallelism.
* The `maybe_initialize_jax_distributed_system` function in [MaxText/max_utils.py](MaxText/max_utils.py)
  creates `JaxWorkerMesh` to be used by JaxPP.
* [MaxText/train.py](MaxText/train.py) contains changes to
** Enable pipeline parallelism for the train step, and
** Mark the pipeline loop in the train step with `jaxpp.accumulate_grads`.

# Docker image

For ease of use, we provide a docker image with the latest revision of this fork under `/workdir/maxtext` and the latest revision of JaxPP under `/workdir/maxtext/third_party/jaxpp`.
The docker image has all the dependencies that are needed to use MaxText with JaxPP installed.
The docker image is available at [`gitlab-master.nvidia.com:5005/cml/jaxpp_dev/maxtext:latest`](gitlab-master.nvidia.com:5005/cml/jaxpp_dev/maxtext:latest).

# Benchmarks

## GPT3
MaxText GPT3 has been tested with JaxPP.

[`MaxText/train.py`](MaxText/train.py) is used to run the benchmark.
The following example command line runs the script on a single node with 2 DP (data parallelism) groups.
In this example, each DP group runs the computation, using 2-way pipeline parallelism and 2-way tensor parallelism.

```bash
RAY_ADDRESS=local \
  python /workspaces/maxtext/MaxText/train.py \
  MaxText/configs/base.yml \
  run_name=gpt3-52k \
  base_output_directory=/tmp/log hardware=gpu dataset_type=synthetic \
  model_name=gpt3-52k steps=20 dtype=bfloat16 max_target_length=1024 per_device_batch_size=4 \
  dcn_data_parallelism=2 ici_tensor_parallelism=2 enable_checkpointing=false use_jaxpp=True \
  distributed_initialization=True ici_pipeline_parallelism=2 num_pipeline_microbatches=2 use_pgle=False
```

To train on cs-oci-ord or eos, [`slurm/maxtext.sh](slurm/maxtext.sh) can be run on the login node.
The script dispatches the docker image to the slurm nodes, mounts the MaxText directory from the local file system to `/workdir/maxtext` in the container, and runs the training loop.
The following table shows the environtment variables that can be set to control the behavior of this script.

| Environment variable   | Description                                           | Default |
| ---                    | ---                                                   | ---     |
| `NUM_NODES`            | number of slurm nodes to be used for execution        | `8` |
| `PP`                   | degree of pipeline parallelism                        | `8` |
| `DP`                   | degree of data parallelism within a stage execution   | `1` |
| `TP`                   | degree of tensor parallelism within a stage execution | `8` |
| `VP`                   | degree of interleaving for interleaved 1F1B schedule  | `3` |
| `SEQ_LEN`              | sequence length                                       | `2048` |
| `DTYPE`                | data type                                             | `bfloat16` |
| `STEPS`                | number of train steps                                 | `20` |
| `MBS`                  | microbatch size                                       | `2` |
| `GA`                   | number of microbatches                                | `64` |
| `USE_PGLE`             | Enable Profile Guided Latency Estimator (PGLE)        | `False` |
| `Schedule`             | Pipeline parallelism schedule                         | `interleaved_1f1b` |
| `SLURM_ACCOUNT`        | slurm account                                         | `coreai_dlalgo_llm` |
| `SLURM_JOB`            | slurm job name                                        | `${SLURM_ACCOUNT}-jaxpp:${USER}-train-maxtext}` |
| `SLURM_TIME`           | slurm time limit                                      | `01:00:00` |
| `SLURM_LOG_DIR`        | log root directory where the log directory for the execution is created | `.` |
| `CONTAINER_IMAGE`      | test container                                        | `gitlab-master.nvidia.com/cml/jaxpp_dev/maxtext:latest` |

Our latest performance numbers with the 175B configuration are provided below.

| Model size | # nodes | DP | TP | PP | VP | MBS | GBS | GA | seq len | # layers | # heads | modeldim | hidden dim | vocab size |
| ---        | ---     | -- | -- | -- | -- | --- | --- | -- | ---     | ---      | ---     | ---      | ---        | ---        |
| 175B       | 8       | 1  | 8  | 8  | 3  | 2   | 128 | 64 | 2048    | 96       | 96      | 12288    | 49152      | 51200      |

| Cluster    | tokens/s | Step time (s) | TFLOPs |          |
| ---        | ---      | ---           | ---    | ---      |
| cs-oci-ord | 10027.15 | 26.14         | 168.7  |          |
| eos        | 23871.10 | 10.98         | 401.5  |          |
| eos        | 26391.74 |  9.93         | 443.9  | FP8 GEMM |

# Profiling

Profiling is enabled by default in the 6th step, and the first 7 steps are ignored in the performance statistics.
It allows the performance statstics to be collected without the profiling overhead while producing the profiling data while running the benchmarks.