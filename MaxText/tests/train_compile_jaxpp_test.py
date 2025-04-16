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

import unittest

from train_compile import main as train_compile_main


def get_args(
    model_name,
    num_nodes,
    ici_dp,
    ici_tp,
    dcn_pp,
    vp,
    ga,
    quantization=None,
    disable_cache: bool = False,
):
    res = (
        None,
        "configs/base.yml",
        f"model_name={model_name}",
        "attention=dot_product",
        "remat_policy=minimal",
        "dtype=bfloat16",
        "max_target_length=2048",
        "per_device_batch_size=2",
        "hardware=gpu",
        # SPMD Parallelism
        f"ici_data_parallelism={ici_dp}",
        f"ici_tensor_parallelism={ici_tp}",
        # Pipeline
        f"dcn_pipeline_parallelism={dcn_pp}",
        f"num_pipeline_microbatches={ga}",
        f"num_pipeline_repeats={vp}",
        # JaxPP
        "use_jaxpp=True",
        "jaxpp_remote=False",
        "schedule=interleaved_1f1b",
        "compile_topology=a3",
        f"compile_topology_num_slices={num_nodes}",
    )
    if quantization is not None:
        res = res + (f"quantization={quantization}",)
    if disable_cache:
        res = res + ("jax_cache_dir=",)
    return res


class TrainCompile(unittest.TestCase):
    def test_compile_gpt3(self):
        train_compile_main(
            get_args(
                model_name="gpt3-175b",
                num_nodes=16,
                ici_dp=2,
                ici_tp=4,
                dcn_pp=8,
                vp=6,
                ga=32,
            )
        )

    def test_compile_gpt3_fp8(self):
        train_compile_main(
            get_args(
                model_name="gpt3-175b",
                num_nodes=16,
                ici_dp=2,
                ici_tp=4,
                dcn_pp=8,
                vp=6,
                ga=32,
                quantization="fp8",
            )
        )

    def test_compile_llama2(self):
        train_compile_main(
            get_args(
                model_name="llama2-70b",
                num_nodes=8,
                ici_dp=2,
                ici_tp=4,
                dcn_pp=4,
                vp=10,
                ga=16,
            )
        )
