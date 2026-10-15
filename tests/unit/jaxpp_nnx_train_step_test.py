# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the shared NNX training step with JAXPP accumulation."""

import functools
from types import SimpleNamespace
from unittest.mock import patch

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from maxtext.common.common_types import ShardMode
from maxtext.common.train_state_nnx import TrainStateNNX
from maxtext.trainers.pre_train import train


class GateLogit(nnx.Module):

  def __init__(self):
    self.bias = nnx.Variable(jnp.zeros(2))


class _Model(nnx.Module):

  def __init__(self):
    self.weight = nnx.Param(jnp.ones((2, 1)))
    self.decoder = nnx.Module()
    self.decoder.layers_0 = GateLogit()
    self.decoder.layers_1 = GateLogit()


def _config(use_jaxpp, clipping=0.0, bias=False):
  return SimpleNamespace(
      use_jaxpp=use_jaxpp,
      gradient_accumulation_steps=1,
      num_pipeline_microbatches=2,
      shard_optimizer_over_data=False,
      shard_mode=ShardMode.AUTO,
      debug_sharding=False,
      global_batch_size_to_train_on=4,
      max_target_length=1,
      ici_pipeline_parallelism=1,
      dcn_pipeline_parallelism=1,
      num_pipeline_repeats=1,
      schedule="1f1b",
      grad_dtype=jnp.float32,
      parameter_memory_host_offload=False,
      optimizer_memory_host_offload=False,
      gradient_clipping_threshold=clipping,
      skip_step_on_spikes=False,
      moe_dropless_fallback="none",
      routed_bias=bias,
      routed_bias_update_rate=0.1,
      num_experts=2,
      model_name="tiny",
      use_qk_clip=False,
  )


def _loss(model, config, data, *unused, **kwargs):
  del unused, kwargs
  value = jnp.sum((data["inputs"] @ model.weight.get_value()) ** 2)
  weight = jnp.asarray(data["inputs"].shape[0], jnp.float32)
  intermediates = {}
  if config.routed_bias:
    intermediates = {
        "decoder": {
            "layers_0": {"moe_bias_updates": (jnp.array([3, 1], jnp.int32),)},
            "layers_1": {"moe_bias_updates": (jnp.array([1, 3], jnp.int32),)},
        }
    }
  return (value if config.use_jaxpp else value / weight), {
      "xent_sum": value,
      "total_weights": weight,
      "intermediate_outputs": intermediates,
      "moe_lb_loss": jnp.asarray(0.0),
      "indexer_loss": jnp.asarray(0.0),
      "mtp_loss": jnp.asarray(0.0),
      "moe_bias_updates": None,
  }


def _step(config):
  model = _Model()
  graphdef, state = nnx.split(TrainStateNNX(model, nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)))
  mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("data", "expert"))
  shardings = jax.tree.map(lambda _: jax.NamedSharding(mesh, jax.P()), state)
  return state, jax.jit(
      functools.partial(train.train_step, graphdef, config, shardings, shardings.model.filter(nnx.Param))
  )


@pytest.mark.parametrize("clipping", [0.0, 0.1])
def test_pipeline_microbatches_use_shared_train_step_with_ga_one(clipping):
  actual, step = _step(_config(True, clipping))
  expected, reference = _step(_config(False, clipping))
  data = {"inputs": jnp.arange(8.0).reshape(4, 2) / 4}
  with patch.object(train, "loss_fn", _loss):
    for _ in range(2):
      actual, metrics = step(actual, data)
      expected, expected_metrics = reference(expected, data)
      assert jax.tree.structure(actual) == jax.tree.structure(expected)
      for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, e, rtol=2e-5, atol=1e-6)
      np.testing.assert_allclose(metrics["scalar"]["learning/loss"], expected_metrics["scalar"]["learning/loss"])


def test_bias_updates_reach_every_unscanned_layer_once_per_global_batch():
  state, step = _step(_config(True, bias=True))
  with patch.object(train, "loss_fn", _loss):
    state, _ = step(state, {"inputs": jnp.ones((4, 2))})
  np.testing.assert_allclose(state.model.decoder.layers_0.bias.get_value(), [-0.1, 0.1])
  np.testing.assert_allclose(state.model.decoder.layers_1.bias.get_value(), [0.1, -0.1])
