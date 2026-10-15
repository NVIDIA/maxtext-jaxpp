# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JAXPP stage boundaries and state updates in the upstream NNX decoder."""

from types import SimpleNamespace
from unittest.mock import patch

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.common.common_types import DecoderBlockType
from maxtext.layers.nnx_decoders import NNXDecoder


class _StatefulLayer(nnx.Module):

  def __init__(self):
    self.weight = nnx.Param(jnp.ones(()))
    self.calls = nnx.Variable(jnp.zeros((), dtype=jnp.int32))

  def __call__(self, inputs, *args, kv_cache=None, **kwargs):
    del args, kwargs
    self.calls[...] += 1
    return inputs + self.weight[...], kv_cache


class _TinyDecoder(NNXDecoder):
  """Use real decoder execution with lightweight NNX layers and output head."""

  def __init__(self, num_layers, stages, repeats=1, deepseek=False, use_jaxpp=True):
    self.config = SimpleNamespace(
        use_jaxpp=use_jaxpp,
        using_pipeline_parallelism=use_jaxpp,
        scan_layers=False,
        num_decoder_layers=num_layers,
        dcn_pipeline_parallelism=1,
        ici_pipeline_parallelism=stages,
        num_pipeline_repeats=repeats,
        remat_policy="full",
        parameter_memory_host_offload=False,
        gradient_accumulation_steps=2,
        hardware="gpu",
        decoder_block=DecoderBlockType.DEEPSEEK if deepseek else DecoderBlockType.DEFAULT,
        first_num_dense_layers=1,
        engram_layers=[],
        attention="dot_product",
        num_vocab_tiling=1,
    )
    self.is_gemma4_small = False
    self.is_deepseek = deepseek
    for index in range(num_layers):
      name = f"layers_{index}"
      if deepseek:
        name = "dense_layers_0" if index == 0 else f"moe_layers_{index - 1}"
      setattr(self, name, _StatefulLayer())

  def get_remat_policy(self):
    return jax.checkpoint_policies.nothing_saveable

  def _apply_embedding(self, shared_embedding, tokens, *args, **kwargs):
    del shared_embedding, args, kwargs
    return tokens.astype(jnp.float32)[..., None]

  def apply_output_head(self, shared_embedding, hidden_state, deterministic, model_mode):
    del shared_embedding, deterministic, model_mode
    return hidden_state * 2


@pytest.mark.parametrize(
    "num_layers, stages, repeats, expected",
    [(5, 3, 1, [2, 4, 10]), (9, 3, 2, [2, 4, 6, 7, 8, 18])],
)
@pytest.mark.parametrize("deepseek", [False, True])
def test_stage_boundaries_include_final_output_head(num_layers, stages, repeats, expected, deepseek):
  decoder = _TinyDecoder(num_layers, stages, repeats, deepseek)
  tokens = jnp.zeros((1, 1), dtype=jnp.int32)
  with patch("maxtext.layers.nnx_decoders.jaxpp.pipeline_enter_stage", side_effect=lambda value, name: value) as marker:
    logits, _, _ = decoder(None, tokens, tokens)
  assert [call.args[1] for call in marker.call_args_list] == [f"stage_{i}" for i in range(stages * repeats)]
  np.testing.assert_array_equal([np.asarray(call.args[0]).item() for call in marker.call_args_list], expected)
  np.testing.assert_array_equal(logits, expected[-1])


@pytest.mark.parametrize("use_jaxpp", [False, True])
def test_nnx_gradient_and_mutable_state_survive_remat(use_jaxpp):
  decoder = _TinyDecoder(5, 3, use_jaxpp=use_jaxpp)
  tokens = jnp.zeros((1, 1), dtype=jnp.int32)
  with patch("maxtext.layers.nnx_decoders.jaxpp.pipeline_enter_stage", side_effect=lambda value, name: value) as marker:
    loss, grads = nnx.value_and_grad(lambda module: jnp.sum(module(None, tokens, tokens)[0]))(decoder)
  np.testing.assert_array_equal(loss, 10)
  for index in range(5):
    np.testing.assert_array_equal(grads[f"layers_{index}"]["weight"][...], 2)
    np.testing.assert_array_equal(getattr(decoder, f"layers_{index}").calls[...], 1)
  if not use_jaxpp:
    marker.assert_not_called()
