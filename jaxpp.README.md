# Overview

This repository is a fork of [MaxText](https://github.com/AI-Hypercomputer/maxtext) created for training with [JaxPP](https://github.com/NVIDIA/jaxpp).


# Notable changes

The changes between this repo and the upstream MaxText is kept minimal in general.
Some of the notable changes are listed below.

* The `__call__` method of the `Decoder` class in [src/maxtext/layers/decoders.py](src/maxtext/layers/decoders.py)
  calls `jaxpp.pipeline_enter_stage` to mark stage boundaries for pipeline parallelism.
* The `maybe_initialize_jax_distributed_system` function in [src/maxtext/utils/max_utils.py](src/maxtext/utils/max_utils.py)
  creates `RemoteMpmdMesh` to be used by JaxPP.
* [src/maxtext/trainers/pre_train/train.py](src/maxtext/trainers/pre_train/train.py) contains changes to
 * Enable pipeline parallelism for the train step, and
 * Mark the pipeline loop in the train step with `jaxpp.treduce`.

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
# Check if jaxpp-base image exists
if [ -z "$(docker images -q jaxpp-base)" ]; then
  echo "Error: jaxpp-base image not found. Please build it first using the instructions at https://github.com/NVIDIA/jaxpp#building-the-base-image."
else
  docker build --force-rm=true \
    -f jaxpp.Dockerfile \
    --build-arg BASE_IMAGE=jaxpp-base \
    -t maxtext-jaxpp .
fi
```

### Running Tests

The container includes several test suites for different models:

1. **Tiny Llama4 Model Tests**:
```bash
docker run --gpus=all --shm-size=10.24gb --ulimit memlock=-1 --ulimit stack=67108864 \
  -e CUDA_VISIBLE_DEVICES=0 --rm --workdir /workdir/maxtext maxtext-jaxpp \
  "nvidia-smi && CONFIG_FILE=./scripts/llama4_proxy_config.sh bash scripts/test_1gpu_config.sh"
```

2. **Tiny Mixtral Model Tests**:
```bash
docker run --gpus=all --shm-size=10.24gb --ulimit memlock=-1 --ulimit stack=67108864 \
  -e CUDA_VISIBLE_DEVICES=0 --rm --workdir /workdir/maxtext maxtext-jaxpp \
  "nvidia-smi && MODEL_CONFIG='model_name=mixtral-8x7b override_model_config=True base_num_decoder_layers=2 base_emb_dim=512 base_mlp_dim=1792' bash scripts/test_1gpu_config.sh"
```

3. **Tiny Mistral Model Tests**:
```bash
docker run --gpus=all --shm-size=10.24gb --ulimit memlock=-1 --ulimit stack=67108864 \
  -e CUDA_VISIBLE_DEVICES=0 --rm --workdir /workdir/maxtext maxtext-jaxpp \
  "nvidia-smi && bash MODEL_CONFIG='model_name=mistral-7b override_model_config=True base_num_decoder_layers=2' bash scripts/test_1gpu_config.sh"
```

Note: The tests require GPU access and sufficient GPU memory.

# Profiling

Profiling is enabled by default in the 6th step, and the first 7 steps are ignored in the performance statistics.
It allows the performance statstics to be collected without the profiling overhead while producing the profiling data while running the benchmarks.
