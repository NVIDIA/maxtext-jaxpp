# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Functional NNX state and numerical contracts for JAXPP accumulation."""

from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from flax.nnx import variablelib
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from maxtext.common.common_types import ShardMode
from maxtext.common.train_state_nnx import TrainStateNNX
from maxtext.utils import gradient_accumulation as ga


class _TinyModel(nnx.Module):

  def __init__(self, dropout=False):
    self.rngs = nnx.Rngs(0)
    self.linear = nnx.Linear(2, 1, rngs=self.rngs)
    self.dropout = dropout

  def __call__(self, inputs):
    output = self.linear(inputs)
    if self.dropout:
      output = output + jax.random.uniform(self.rngs(), output.shape)
    return output


class _PipelineModel(nnx.Module):

  def __init__(self, dropout=False):
    rngs = nnx.Rngs(0)
    self.first = nnx.Linear(2, 3, rngs=rngs)
    self.second = nnx.Linear(3, 1, rngs=rngs)
    self.dropout = dropout
    if dropout:
      self.first_rngs = nnx.Rngs(1)
      self.second_rngs = nnx.Rngs(2)

  def __call__(self, inputs):
    hidden = jnp.tanh(self.first(inputs))
    if self.dropout:
      hidden += jax.random.uniform(self.first_rngs(), hidden.shape)
    hidden = ga.jaxpp.mark_stage_end(hidden, "first")
    output = self.second(hidden)
    if self.dropout:
      output += jax.random.uniform(self.second_rngs(), output.shape)
    return ga.jaxpp.mark_stage_end(output, "second")


def _loss(model, config, data, dropout_rng, params, is_train=True):
  del config, dropout_rng, params, is_train
  predictions = model(data["inputs"])
  weights = data["weights"]
  xent_sum = jnp.sum(jnp.square(predictions - data["targets"]) * weights)
  return xent_sum, {
      "xent_sum": xent_sum,
      "total_weights": jnp.sum(weights),
      "moe_lb_loss": jnp.asarray(0.0),
      "indexer_loss": jnp.asarray(0.0),
      "mtp_loss": jnp.asarray(0.0),
  }


class GradientAccumulationJaxppNNXTest(unittest.TestCase):

  def setUp(self):
    self.config = SimpleNamespace(
        use_jaxpp=True,
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
    )
    self.data = {
        "inputs": jnp.arange(8.0).reshape(4, 2) / 4,
        "targets": jnp.zeros((4, 1)),
        "weights": jnp.ones((4, 1)),
    }
    self.mesh = Mesh(np.asarray(jax.devices()[:1]).reshape((1, 1)), ("data", "expert"))

  def _run(self, model, loss_fn=_loss):
    shardings = jax.tree.map(lambda _: NamedSharding(self.mesh, PartitionSpec()), nnx.state(model, nnx.Param))
    graphdef, state = nnx.split(model)

    @jax.jit
    def run(state, data):
      local_model = nnx.merge(graphdef, state)
      output = ga.gradient_accumulation_loss_and_grad(loss_fn, self.config, local_model, None, shardings, data, None)
      return nnx.state(local_model), output

    with jax.set_mesh(self.mesh):
      next_state, output = run(state, self.data)
    nnx.update(model, next_state)
    return output

  def test_matches_direct_full_batch_gradients_with_masked_tokens(self):
    model = _TinyModel()
    self.data["weights"] = jnp.array([[1.0], [0.0], [1.0], [1.0]])
    kernel, bias = model.linear.kernel.get_value(), model.linear.bias.get_value()

    def full_batch_loss(kernel, bias):
      predictions = self.data["inputs"] @ kernel + bias
      return jnp.sum(jnp.square(predictions) * self.data["weights"]) / jnp.sum(self.data["weights"])

    expected_loss, expected_grads = jax.value_and_grad(full_batch_loss, argnums=(0, 1))(kernel, bias)
    loss, aux, grads = self._run(model)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-6)
    np.testing.assert_allclose(grads["linear"]["kernel"].get_value(), expected_grads[0], rtol=1e-6)
    np.testing.assert_allclose(grads["linear"]["bias"].get_value(), expected_grads[1], rtol=1e-6)
    self.assertEqual(float(aux["total_weights"]), 3.0)
    np.testing.assert_array_equal(model.linear.kernel.get_value(), kernel)

  def test_empty_batch_returns_zero_loss_and_gradients(self):
    self.data["weights"] = jnp.zeros((4, 1))
    loss, _, grads = self._run(_TinyModel())
    self.assertEqual(float(loss), 0.0)
    for gradient in jax.tree.leaves(grads):
      np.testing.assert_array_equal(gradient, jnp.zeros_like(gradient))

  def test_bfloat16_auxiliary_loss_matches_full_batch_loss_and_gradients(self):
    model = _TinyModel()
    model.linear.kernel.set_value(jnp.array([[1.0], [0.0]]))
    model.linear.bias.set_value(jnp.zeros((1,)))

    def loss_with_bfloat16_aux(model, config, data, dropout_rng, params, is_train=True):
      xent_sum, aux = _loss(model, config, data, dropout_rng, params, is_train=is_train)
      aux["moe_lb_loss"] = (xent_sum / aux["total_weights"]).astype(jnp.bfloat16)
      return xent_sum + aux["moe_lb_loss"] * aux["total_weights"], aux

    def full_batch_loss(model):
      loss, aux = loss_with_bfloat16_aux(model, self.config, self.data, None, None)
      return loss / aux["total_weights"]

    expected_loss, expected_grads = nnx.value_and_grad(full_batch_loss)(model)
    loss, aux, grads = self._run(model, loss_with_bfloat16_aux)
    self.assertEqual(loss.dtype, jnp.float32)
    self.assertEqual(aux["moe_lb_loss"].dtype, jnp.bfloat16)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-6)
    self.assertGreater(float(aux["moe_lb_loss"]), 0.0)
    for actual, expected in zip(jax.tree.leaves(grads), jax.tree.leaves(expected_grads), strict=True):
      np.testing.assert_allclose(actual, expected, rtol=1e-6)

  def test_batch_statistics_are_rejected(self):
    model = _TinyModel()
    model.running_mean = nnx.BatchStat(jnp.asarray(0.0))
    with self.assertRaisesRegex(ValueError, "sequential BatchStat updates"):
      self._run(model)

  def test_data_parallel_microbatches_match_full_batch(self):
    if len(jax.devices()) < 2:
      self.skipTest("Needs two devices; use JAX_NUM_CPU_DEVICES=2 for a CPU run")
    self.mesh = Mesh(np.asarray(jax.devices()[:2]).reshape((2, 1)), ("data", "expert"))
    self.test_matches_direct_full_batch_gradients_with_masked_tokens()

  def test_two_optimizer_steps_match_full_batch_training(self):
    model, reference_model = _TinyModel(), _TinyModel()
    optimizer = nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param)
    reference_optimizer = nnx.Optimizer(reference_model, optax.sgd(0.1), wrt=nnx.Param)

    def direct_loss(model):
      loss, aux = _loss(model, self.config, self.data, None, None)
      return loss / aux["total_weights"]

    for _ in range(2):
      loss, _, grads = self._run(model)
      reference_loss, reference_grads = nnx.value_and_grad(direct_loss)(reference_model)
      np.testing.assert_allclose(loss, reference_loss, rtol=1e-6)
      optimizer.update(model, grads)
      reference_optimizer.update(reference_model, reference_grads)
      for actual, expected in zip(
          jax.tree.leaves(nnx.state(model, nnx.Param)), jax.tree.leaves(nnx.state(reference_model, nnx.Param))
      ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    self.assertEqual(int(optimizer.step.get_value()), 2)

  def test_dropout_rng_advances_without_replacing_original_key(self):
    model = _TinyModel(dropout=True)
    original_key = jax.random.key_data(model.rngs.default.key.get_value())
    original_count = int(model.rngs.default.count.get_value())
    first_loss, _, _ = self._run(model)
    self.assertEqual(int(model.rngs.default.count.get_value()), original_count + 2)
    np.testing.assert_array_equal(jax.random.key_data(model.rngs.default.key.get_value()), original_key)
    second_loss, _, _ = self._run(model)
    self.assertEqual(int(model.rngs.default.count.get_value()), original_count + 4)
    self.assertNotEqual(float(first_loss), float(second_loss))

  def test_fp8_overwrite_gradients_use_max_and_are_not_normalized(self):
    owg_type = variablelib.variable_type_from_name("_overwrite_with_gradient", allow_register=True)
    model = _TinyModel()
    model.linear.amax = owg_type(jnp.asarray(0.0))

    @jax.custom_vjp
    def record_amax(value, stat):
      del stat
      return value

    def forward(value, stat):
      return value, (jnp.max(jnp.abs(value)), stat)

    def backward(residual, cotangent):
      amax, _ = residual
      return cotangent, amax

    record_amax.defvjp(forward, backward)

    def fp8_loss(model, config, data, dropout_rng, params, is_train=True):
      del config, dropout_rng, params, is_train
      predictions = record_amax(model(data["inputs"]), model.linear.amax.get_value())
      xent_sum = jnp.sum(predictions**2)
      return xent_sum, {
          "xent_sum": xent_sum,
          "total_weights": jnp.asarray(predictions.shape[0], dtype=jnp.float32),
          "moe_lb_loss": jnp.asarray(0.0),
          "indexer_loss": jnp.asarray(0.0),
          "mtp_loss": jnp.asarray(0.0),
      }

    expected_amax = jnp.max(jnp.abs(model(self.data["inputs"])))
    _, _, grads = self._run(model, fp8_loss)
    np.testing.assert_allclose(model.linear.amax.get_value(), expected_amax, rtol=1e-6)
    self.assertNotIn("amax", grads["linear"])

  def test_two_stage_mpmd_train_state_matches_full_batch(self):
    self._assert_mpmd_training(dropout=False)

  def test_two_stage_mpmd_dropout_matches_functional_accumulation(self):
    self._assert_mpmd_training(dropout=True)

  def _assert_mpmd_training(self, dropout):
    if len(jax.devices()) < 2:
      self.skipTest("Needs two devices; use JAX_NUM_CPU_DEVICES=2 for a CPU run")
    mesh = Mesh(np.asarray(jax.devices()[:2]).reshape((2, 1, 1)), ("stage", "data", "expert"))
    mpmd_mesh = ga.jaxpp.MpmdMesh(mesh, "stage")
    self.config.ici_pipeline_parallelism = 2
    model, reference_model = _PipelineModel(dropout), _PipelineModel(dropout)
    state = TrainStateNNX(model, nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param))
    reference = TrainStateNNX(reference_model, nnx.Optimizer(reference_model, optax.adam(0.01), wrt=nnx.Param))
    graphdef, state_arrays = nnx.split(state)
    shardings = jax.tree.map(
        lambda _: NamedSharding(mpmd_mesh.lowering_mesh(), PartitionSpec()), nnx.state(model, nnx.Param)
    )

    def step(state_arrays, data):
      state = nnx.merge(graphdef, state_arrays)
      loss, _, grads = ga.gradient_accumulation_loss_and_grad(
          _loss, self.config, state.model, None, shardings, data, None
      )
      state.apply_gradients(grads)
      return nnx.state(state), loss

    def reference_loss(model):
      loss, aux = _loss(model, self.config, self.data, None, None)
      return loss / aux["total_weights"]

    compiled = ga.jaxpp.mpmd_jit_with_loop(step, mpmd_mesh).compile(state_arrays, self.data)
    (mpmd_state, mpmd_data), _ = ga.jaxpp.spmd_to_mpmd_reshard(
        mpmd_mesh, ((state_arrays, jax.tree.map(jnp.copy, self.data)), {}), compiled.in_shardings, threshold=1 << 20
    )
    reference_arrays = nnx.state(reference)
    for _ in range(2):
      mpmd_state, loss = compiled(mpmd_state, mpmd_data)
      if dropout:
        reference_arrays, expected_loss = jax.jit(step)(reference_arrays, self.data)
      else:
        expected_loss, expected_grads = nnx.value_and_grad(reference_loss)(reference.model)
        reference.apply_gradients(expected_grads)
        reference_arrays = nnx.state(reference)
      np.testing.assert_allclose(loss.first_mpmd_replica, expected_loss, rtol=1e-5)
      actual_state = jax.tree.map(lambda value: value.first_mpmd_replica, mpmd_state)
      for actual, expected in zip(jax.tree.leaves(actual_state), jax.tree.leaves(reference_arrays)):
        if jax.dtypes.issubdtype(expected.dtype, jax.dtypes.prng_key):
          actual, expected = jax.random.key_data(actual), jax.random.key_data(expected)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
  unittest.main()
