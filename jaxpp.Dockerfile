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

COPY requirements.txt /tmp/requirements.txt
RUN pip install -U pip && pip install --no-cache-dir -U -r /tmp/requirements.txt

COPY --chown=$USER_UID:$USER_GID . maxtext

RUN pip install --no-cache-dir -e '/workdir/jaxpp[dev]'
