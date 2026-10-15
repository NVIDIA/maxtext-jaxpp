# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BASE_IMAGE
FROM $BASE_IMAGE AS base
ARG JAX_INSTALL_URL

COPY --chown=$USER_UID:$USER_GID . maxtext

ARG JAXPP_PATH=/workdir/jaxpp
# Build TE against the installed CUDA13 JAX, not its isolated CUDA12 build environment.
RUN JAXPP_JAX_VERSION=$(python -c \
        'from importlib.metadata import version; print(version("jax"))') && \
    if [ -z "${VIRTUAL_ENV}" ]; then \
        echo "[failed] VIRTUAL_ENV variable is not set."; \
        exit 1; \
    fi && \
    uv pip install --no-cache-dir --resolution=lowest \
        "nvidia-cudnn-frontend>=1.29.0" "pybind11[global]>=2.13.6" && \
    "${VIRTUAL_ENV}/bin/pip" install --no-build-isolation \
        'transformer-engine[core-cu13,jax]==2.19.0' && \
    uv pip install --no-cache-dir \
        -e '/workdir/maxtext[cuda13]' \
        -e "${JAXPP_PATH}" \
        "jax[cuda13]==${JAXPP_JAX_VERSION}" \
        --no-build-isolation-package transformer-engine-jax \
        --resolution=lowest && \
    python -c 'from pathwaysutils.elastic import elastic, manager' && \
    uv pip install triton==3.7.1 && \
    if [[ -n "$JAX_INSTALL_URL" ]]; then uv pip install $JAX_INSTALL_URL; fi
