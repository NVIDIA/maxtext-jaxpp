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
 * Enable pipeline parallelism for the train step, and
 * Mark the pipeline loop in the train step with `jaxpp.accumulate_grads`.

# Docker image

For ease of use, we provide a docker image with this fork under `/workdir/maxtext`.
The docker image has all the dependencies that are needed to use MaxText with JaxPP installed.

## Building and Testing Docker Container

The build process uses the JaxPP base image as a starting point. Follow the instructions at [JaxPP's Building the Base Image](https://github.com/NVIDIA/jaxpp#building-the-base-image) to build the `jaxpp-base` image first.

### Prerequisites
- Docker installed and configured
- NVIDIA Container Toolkit installed
- JaxPP base image built and available locally

### Building the Main Image

After building the base image, you can build the main image:

```bash
docker build --force-rm=true \
  -f jaxpp.Dockerfile \
  --build-arg BASE_IMAGE=jaxpp-base \
  -t maxtext-jaxpp .
```

### Running Tests

The container includes several test suites for different models:

1. **Tiny Llama2 Model Tests**:
```bash
docker run --gpus=all --shm-size=10.24gb --ulimit memlock=-1 --ulimit stack=67108864 \
  -e CUDA_VISIBLE_DEVICES=0 --rm --workdir /workdir/maxtext maxtext-jaxpp \
  "nvidia-smi && bash /workdir/maxtext/scripts/multigpu_jaxpp.sh --max_target_length=64 --model=default \
   --dp=1 --pp=1 --mp=1 --batch_size_per_device=1 --num_pipeline_microbatches=1 \
   jaxpp_remote=false base_emb_dim=1024 base_num_query_heads=8 base_num_kv_heads=8 base_mlp_dim=11008 \
   base_num_decoder_layers=1 head_dim=128 vocab_size=32000 enable_dropout=false \
   logits_via_embedding=false normalization_layer_epsilon=1.0e-5 decoder_block=llama2"
```

2. **Tiny Mixtral Model Tests**:
```bash
docker run --gpus=all --shm-size=10.24gb --ulimit memlock=-1 --ulimit stack=67108864 \
  -e CUDA_VISIBLE_DEVICES=0 --rm --workdir /workdir/maxtext maxtext-jaxpp \
  "nvidia-smi && bash /workdir/maxtext/scripts/multigpu_jaxpp.sh --max_target_length=64 --model=default \
   --dp=1 --pp=1 --mp=1 --batch_size_per_device=1 --num_pipeline_microbatches=1 \
   jaxpp_remote=false base_emb_dim=1024 base_num_query_heads=2 base_num_kv_heads=2 base_mlp_dim=896 \
   base_num_decoder_layers=2 head_dim=16 vocab_size=32000 enable_dropout=false \
   logits_via_embedding=false normalization_layer_epsilon=1.0e-5 num_experts=8 \
   num_experts_per_tok=2 decoder_block=mistral"
```

3. **Tiny Mistral Model Tests**:
```bash
docker run --gpus=all --shm-size=10.24gb --ulimit memlock=-1 --ulimit stack=67108864 \
  -e CUDA_VISIBLE_DEVICES=0 --rm --workdir /workdir/maxtext maxtext-jaxpp \
  "nvidia-smi && bash /workdir/maxtext/scripts/multigpu_jaxpp.sh --max_target_length=64 --model=default \
   --dp=1 --pp=1 --mp=1 --batch_size_per_device=1 --num_pipeline_microbatches=1 \
   jaxpp_remote=false base_emb_dim=2048 base_num_query_heads=1 base_num_kv_heads=1 base_mlp_dim=896 \
   base_num_decoder_layers=1 head_dim=8 vocab_size=32000 enable_dropout=false \
   logits_via_embedding=false normalization_layer_epsilon=1.0e-5 decoder_block=mistral"
```

Note: The tests require GPU access and sufficient GPU memory.

# Profiling

Profiling is enabled by default in the 6th step, and the first 7 steps are ignored in the performance statistics.
It allows the performance statstics to be collected without the profiling overhead while producing the profiling data while running the benchmarks.