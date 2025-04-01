# Overview

This repository is a fork of [MaxText](https://github.com/google/maxtext) created for training with [JaxPP](https://github.com/NVIDIA/jaxpp).

# Notable changes

The changes between this repo and the upstream MaxText is kept minimal in general.
Some of the notable changes are listed below.

* The `__call__` method of the `Decoder` class in [MaxText/layers/models.py](MaxText/layers/models.py)
  calls `jaxpp.pipeline_enter_stage` to mark stage boundaries for pipeline parallelism.
* The `maybe_initialize_jax_distributed_system` function in [MaxText/max_utils.py](MaxText/max_utils.py)
  creates `RemoteMpmdMesh` to be used by JaxPP.
* [MaxText/train.py](MaxText/train.py) contains changes to
** Enable pipeline parallelism for the train step, and
** Mark the pipeline loop in the train step with `jaxpp.accumulate_grads`.

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

# Profiling

Profiling is enabled by default in the 6th step, and the first 7 steps are ignored in the performance statistics.
It allows the performance statstics to be collected without the profiling overhead while producing the profiling data while running the benchmarks.