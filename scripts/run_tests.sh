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

bash scripts/local_mc.sh

RAY_ADDRESS=local python MaxText/train.py MaxText/configs/base.yml run_name=runner_jaxpp_$(date +%Y-%m-%d-%H-%M) base_output_directory=/tmp/log hardware=gpu dataset_type=synthetic model_name=gpt3-52k steps=20 dtype=bfloat16 max_target_length=2048 per_device_batch_size=4 dcn_data_parallelism=1 ici_data_parallelism=2 ici_tensor_parallelism=2 ici_pipeline_parallelism=2 num_pipeline_repeats=1 num_pipeline_microbatches=8 enable_checkpointing=false use_jaxpp=True schedule=interleaved_1f1b use_pgle=False

cd MaxText/

tests=$(python3 -m pytest --co -q -W ignore::DeprecationWarning tests/train_compile_jaxpp_test.py | awk '/^[[:space:]]*$/{exit} {print}')

for t in $tests; do
    python3 -m pytest --log-cli-level=INFO -s "$t"
done
