# Copyright 2025-2026 Google LLC
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Functions for gradient accumulation (GA)"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental.xla_metadata import set_xla_metadata
from jax.sharding import NamedSharding
from flax import nnx
from flax.nnx import variablelib

from maxtext.common.common_types import ShardMode
from maxtext.layers.moe import load_balance_updates_from_counts
from maxtext.utils.sharding import maybe_shard_with_name

import jaxpp.api as jaxpp


def load_schedule(config):
  """Load jaxpp pipeline schedule based on config."""
  from jaxpp import __version__ as jaxpp_version
  from packaging.version import Version

  pipeline_parallel_dim = config.dcn_pipeline_parallelism * config.ici_pipeline_parallelism
  num_logical_stages = config.num_pipeline_repeats * pipeline_parallel_dim
  schedule = None
  if config.schedule == "1f1b":
    assert num_logical_stages <= pipeline_parallel_dim
    schedule = jaxpp.Std1F1B(num_logical_stages)
  elif config.schedule == "eager_1f1b":
    assert num_logical_stages <= pipeline_parallel_dim
    schedule = jaxpp.Eager1F1B(num_logical_stages)
  elif config.schedule == "interleaved_1f1b":
    if Version(jaxpp_version) > Version("0.6.1"):
      schedule = jaxpp.Interleaved1F1B(num_logical_stages, pipeline_parallel_dim, config.fuse_steady_state)
    else:
      schedule = jaxpp.Interleaved1F1B(num_logical_stages, pipeline_parallel_dim)
  elif config.schedule == "kimik2":
    schedule = jaxpp.KimiK2(num_logical_stages, pipeline_parallel_dim, config.fuse_steady_state)
  elif config.schedule == "zero_bubble":
    assert num_logical_stages <= pipeline_parallel_dim
    schedule = jaxpp.ZeroBubble(num_logical_stages)
  elif config.schedule == "dualpipev":
    schedule = jaxpp.DualPipeV(num_logical_stages, pipeline_parallel_dim)
  else:
    raise NotImplementedError(f"Unknown schedule {config.schedule}")
  return schedule


def add_leading_axis(axis_name: str, path: jax.tree_util.KeyPath, s: jax.sharding.NamedSharding):
  """Add a leading axis to a NamedSharding for vmap."""
  assert isinstance(s, jax.sharding.NamedSharding)
  used = {n for ns in s.spec for n in (ns if isinstance(ns, tuple) else (ns,))}
  if axis_name in used:
    raise ValueError(
        f"mesh axis name {axis_name} cannot appear in "
        f"out_shardings. Found out_shardings{jax.tree_util.keystr(path)}={s.spec}"
    )
  return jax.sharding.NamedSharding(s.mesh, jax.sharding.PartitionSpec(axis_name, *s.spec), memory_kind=s.memory_kind)


def gradient_accumulation_loss_and_grad(
    _loss_fn,
    config,
    model,
    params,
    params_shardings,
    data,
    dropout_rng,
    extra_dpo_args=(),
):
  """
  Calculates gradients using gradient accumulation.

  This function computes the gradient of `_loss_fn` over multiple microbatches
  and accumulates them before returning a single, averaged gradient. It supports
  two modes:

  1. Standard mode: Uses `jax.lax.scan` for efficient accumulation on device.
  2. JaxPP mode: Uses `jaxpp.treduce` with pipeline parallelism schedules when
     `config.use_jaxpp` is True.

  It also supports a `shard_optimizer_over_data` mode (e.g., ZeRO-1) where
  parameters are cast to bf16 and sharded *before* the accumulation loop
  to perform the all-gather in lower precision.

  Args:
      _loss_fn: The loss function to differentiate. Its signature is expected
          to be: `(model, config, data, dropout_rng, params, *extra_args, is_train=True)`.
      config: Model and training configuration object. Must contain
          `gradient_accumulation_steps`, `shard_optimizer_over_data`, and optionally
          `use_jaxpp`, `num_pipeline_microbatches`, `schedule`, etc.
      model: The model module.
      params: The model parameters (PyTree). This is only used for Linen. For NNX,
          we can get the params from the model.
      params_shardings: The sharding constraints for the parameters (PyTree).
      data: A PyTree of batched data. The leading dimension is assumed
          to be the total batch size (microbatch_size * num_accumulations).
      dropout_rng: JAX PRNGKey for dropout.

  Returns:
      A tuple containing:
      - total_loss (Array): The mean loss, averaged over all microbatches.
      - final_aux (PyTree): Auxiliary outputs, summed across microbatches.
      - raw_grads (PyTree): The accumulated and averaged gradients.
  """

  def _maybe_shard_with_name(inputs, sharding_names):
    """Wrapper of maybe_shard_with_name with fixed shard_mode"""
    return maybe_shard_with_name(inputs, sharding_names, config.shard_mode, debug_sharding=config.debug_sharding)

  use_jaxpp = getattr(config, "use_jaxpp", False)
  is_nnx = isinstance(model, nnx.Module)
  if use_jaxpp and not is_nnx:
    raise TypeError("JAXPP gradient accumulation requires an NNX model")

  # For ZeRO-1 + GA, read the resolved "data" axis size from the mesh rather than
  # config.ici_data_parallelism, which may be -1 (auto-fill) and resolves to 1 when
  # FSDP already consumes every device — in which case data parallelism is not active.
  param_mesh = jax.tree.leaves(params_shardings)[0].mesh
  data_parallel_active = config.shard_mode == ShardMode.EXPLICIT and param_mesh.shape.get("data", 1) > 1
  if data_parallel_active:
    # reduced/unreduced PartitionSpecs are rejected inside a jax.lax.scan carry: scan
    # traces its body against an AbstractMesh whose axis types are all Auto, and the
    # annotations require Explicit axes. Keep plain params_shardings in the carry and
    # apply the data-parallel all-reduce to the gradients after the scan instead.
    ga_params_shardings = params_shardings
    grad_shardings = params_shardings
  else:
    ga_params_shardings = grad_shardings = params_shardings

  if is_nnx:
    if use_jaxpp:
      owg_type = variablelib.variable_type_from_name("_overwrite_with_gradient", allow_register=True)
      custom_param_filter = nnx.Any(owg_type)
      nnx.pop(model, nnx.Intermediate)
      if nnx.state(model, nnx.BatchStat):
        raise ValueError("JAXPP microbatches do not support sequential BatchStat updates")
      graphdef, params, custom_params, rest = nnx.split(model, nnx.Param, custom_param_filter, ...)
    else:
      graphdef, params, rest = nnx.split(model, nnx.Param, ...)

  # When using Zero-1 optimizer sharding, cast params to lower precision and apply sharding constraints
  # so that all-gather is done once in the lower precision before the gradient accumulation loop
  if config.shard_optimizer_over_data:

    def convert_to_bf16(param):
      if param.dtype == jnp.float32:
        return param.astype(jnp.bfloat16)
      return param

    ga_params = jax.tree_util.tree_map(convert_to_bf16, params)
  else:
    ga_params = params

  ga_params = jax.tree.map(_maybe_shard_with_name, ga_params, ga_params_shardings)
  if use_jaxpp and data_parallel_active:
    ga_params = jax.tree.map(
        _maybe_shard_with_name, ga_params, jax.tree.map(update_sharding_for_reduced, params_shardings)
    )

  # Under manual GA, _loss_fn returns the token-summed xent (plus aux terms scaled by the microbatch token count), so
  # differentiating it directly seeds the backward pass with 1 where the GA=1 loss (xent_sum / W) seeds it with 1/W.
  # Every backward cotangent is then about W times its GA=1 value. A static (fixed-range) fp8 calibration of the
  # backward cotangents (e.g. bwd_quantization_calibration_method=fixed,0.01) is sized for the GA=1 magnitude and
  # saturates, which distorts the gradient. Scale the differentiated loss by 1/S, with S the static global-batch
  # token capacity, so the cotangents carry the GA=1 magnitude, and fold S back into the final division below.
  # The accumulated gradient is mathematically unchanged: sum_i grad(L_i) / W.
  use_tunix_ga = getattr(config, "use_tunix_gradient_accumulation", False)
  token_capacity = getattr(config, "global_batch_size_to_train_on", 0) * getattr(config, "max_target_length", 0)
  loss_scale = 1.0 / float(token_capacity) if (token_capacity > 0 and not use_tunix_ga) else 1.0

  # functools.wraps keeps _loss_fn's signature visible, which nnx.value_and_grad needs to resolve is_train=.
  @functools.wraps(_loss_fn)
  def _scaled_loss_fn(*args, **kwargs):
    loss, aux = _loss_fn(*args, **kwargs)
    return loss * loss_scale, aux

  if is_nnx:
    grad_func = nnx.value_and_grad(_scaled_loss_fn, argnums=0, has_aux=True)
  else:
    grad_func = jax.value_and_grad(_scaled_loss_fn, argnums=4, has_aux=True)

  def accumulate_gradient(acc_grad_and_loss, data):
    ga_params = acc_grad_and_loss["ga_params"]
    # Scope double-buffering to model scans, not the enclosing gradient accumulation scan.
    # This metadata only affects XLA:GPU; see https://github.com/openxla/xla/pull/43468.
    # Enable it with JAX_OPTIMIZATION_LEVEL=O1 and
    # XLA_FLAGS="--xla_gpu_enable_while_loop_unrolling=WHILE_LOOP_UNROLLING_MANUAL_UNROLL".
    if is_nnx:
      # Reconstruct the model using the fixed parameters (ga_params)
      # and the advancing non-parameter state (RNGs) from the carry.
      local_model = nnx.merge(graphdef, ga_params, acc_grad_and_loss["rest_state"], copy=True)
      with set_xla_metadata(_xla_loop_unroll_strategy="double-buffer"):
        (_, aux), cur_batch_gradient = grad_func(local_model, config, data, None, None, is_train=True)
      _, _, next_rest_state = nnx.split(local_model, nnx.Param, ...)
      acc_grad_and_loss["rest_state"] = next_rest_state
    else:
      rng = (
          jax.random.fold_in(dropout_rng, acc_grad_and_loss["total_weights"].astype(jnp.int32))
          if dropout_rng is not None
          else None
      )
      with set_xla_metadata(_xla_loop_unroll_strategy="double-buffer"):
        (_, aux), cur_batch_gradient = grad_func(model, config, data, rng, ga_params, is_train=True)
    acc_grad_and_loss["loss"] += aux["xent_sum"] + aux.get("dpo_loss", 0.0)
    acc_grad_and_loss["moe_lb_loss"] += aux["moe_lb_loss"]
    acc_grad_and_loss["indexer_loss"] += aux["indexer_loss"]
    acc_grad_and_loss["mtp_loss"] += aux["mtp_loss"]
    acc_grad_and_loss["grad"] = jax.tree_util.tree_map(lambda x, y: x + y, cur_batch_gradient, acc_grad_and_loss["grad"])
    acc_grad_and_loss["total_weights"] += aux["total_weights"]
    return acc_grad_and_loss, aux

  def reshape_to_microbatch_accumulations(batch_arr):
    """Reshape global batch to microbatches, assuming batch axis is leading."""
    num_microbatches = config.gradient_accumulation_steps
    microbatch_shape = (batch_arr.shape[0] // num_microbatches, num_microbatches) + batch_arr.shape[1:]
    reshaped_batch_arr = jnp.reshape(batch_arr, microbatch_shape)
    return jnp.swapaxes(reshaped_batch_arr, 0, 1)

  accumulation_steps = config.gradient_accumulation_steps
  if use_jaxpp:
    accumulation_steps = config.num_pipeline_microbatches * param_mesh.shape.get("data", 1)
    grad_and_loss, aux = _jaxpp_gradient_accumulation_loss_and_grad(
        _scaled_loss_fn,
        config,
        graphdef,
        ga_params,
        custom_params,
        rest,
        params_shardings,
        data,
        custom_param_filter,
        extra_dpo_args,
    )
  else:
    data = jax.tree_util.tree_map(reshape_to_microbatch_accumulations, data)
    init_grad = jax.tree_util.tree_map(jnp.zeros_like, ga_params)
    init_grad = jax.tree.map(_maybe_shard_with_name, init_grad, grad_shardings)
    init_grad_and_loss = {
        "loss": 0.0,  # accumulates xent_sum across microbatches
        "grad": init_grad,
        "total_weights": 0,
        "moe_lb_loss": 0.0,
        "indexer_loss": 0.0,
        "mtp_loss": 0.0,
        "ga_params": ga_params,
    }
    if is_nnx:
      init_grad_and_loss["rest_state"] = rest  # pyrefly: ignore[unbound-name]

    grad_and_loss, aux = jax.lax.scan(
        accumulate_gradient, init_grad_and_loss, data, length=config.gradient_accumulation_steps
    )
  if use_jaxpp:
    # Token normalization is shared by parameters on every pipeline stage.
    # Anchor zero contributions to their gradients so JAXPP replicates the
    # scalar without counting the token weights more than once.
    total_weights = grad_and_loss["total_weights"]
    grad_and_loss["total_weights"] = jaxpp.cross_mpmd_all_reduce(
        total_weights,
        *(jaxpp.place_with(jnp.zeros_like(total_weights), g) for g in jax.tree.leaves(grad_and_loss["grad"])),
    )
  has_weights = grad_and_loss["total_weights"] > 0
  denominator = jnp.maximum(grad_and_loss["total_weights"], 1)
  loss_terms = (
      grad_and_loss["loss"] / denominator,
      grad_and_loss["moe_lb_loss"] / accumulation_steps,
      grad_and_loss["indexer_loss"] / accumulation_steps,
      grad_and_loss["mtp_loss"] / accumulation_steps,
  )
  if use_jaxpp:
    # Auxiliary losses (e.g. TE MoE load balancing) may use the activation
    # dtype, but cross-stage reduction requires every contribution to match.
    loss_terms = tuple(term.astype(jnp.float32) for term in loss_terms)
    loss = jaxpp.cross_mpmd_all_reduce(*loss_terms)
  else:
    loss = sum(loss_terms)
  loss = jnp.where(has_weights, loss, 0.0)
  raw_grads = grad_and_loss["grad"]
  if data_parallel_active and not use_jaxpp:
    # Mark the gradients unreduced over the "data" axis now that we're outside the
    # scan; this triggers the cross-replica all-reduce. The annotation can't live in
    # the scan carry (see above), so it's applied here instead.
    unreduced_shardings = jax.tree.map(update_sharding_for_unreduced, params_shardings)
    raw_grads = jax.tree.map(_maybe_shard_with_name, raw_grads, unreduced_shardings)
  raw_grads = jax.tree.map(_maybe_shard_with_name, raw_grads, params_shardings)
  divisor = accumulation_steps if use_tunix_ga else denominator * loss_scale
  raw_grads = jax.tree_util.tree_map(
      lambda arr: jnp.where(has_weights, (arr / divisor).astype(arr.dtype), jnp.zeros_like(arr)),
      raw_grads,
  )
  scanned_aux = aux
  aux = jax.tree.map(lambda x: jnp.sum(x, axis=0), aux)  # pytype: disable=module-attr
  if use_jaxpp:
    aux["total_weights"] = grad_and_loss["total_weights"]
  # The auxiliary losses are per-microbatch means; the step's value is their mean over microbatches (as in `loss`
  # above), not their sum, so the logged learning/{moe_lb,indexer,mtp}_loss match GA=1.
  for key in ("moe_lb_loss", "indexer_loss", "mtp_loss"):
    if key in aux:
      aux[key] = aux[key] / accumulation_steps
  if getattr(config, "routed_bias", False) and getattr(config, "routed_bias_update_rate", 0.0) > 0.0:
    # RoutedMoE emits raw int32 counts for manual GA and JAXPP microbatches;
    # convert the counts summed across microbatches into a single full-batch sign() update.
    def finalize_bias(counts):
      if hasattr(counts, "dtype") and jnp.issubdtype(counts.dtype, jnp.integer):
        return load_balance_updates_from_counts(counts, config.num_experts, config.routed_bias_update_rate)
      return counts

    for key in ("moe_bias_updates", "mtp_moe_bias_updates"):
      if aux.get(key) is not None:
        aux[key] = jax.tree.map(finalize_bias, aux[key])
    if "intermediate_outputs" in aux:
      aux["intermediate_outputs"] = jax.tree_util.tree_map_with_path(
          lambda path, val: (
              finalize_bias(val) if any(getattr(key, "key", None) == "moe_bias_updates" for key in path) else val
          ),
          aux["intermediate_outputs"],
      )
  if "te_moe_capacity_overflow" in scanned_aux:
    aux["te_moe_capacity_overflow"] = jnp.any(scanned_aux["te_moe_capacity_overflow"], axis=0)
    aux["te_moe_max_total_recv_tokens"] = jnp.max(scanned_aux["te_moe_max_total_recv_tokens"], axis=0)
    aux["te_moe_recv_capacity_per_rank"] = jnp.min(scanned_aux["te_moe_recv_capacity_per_rank"], axis=0)
  if "moe_required_rbf" in scanned_aux:
    # log_required_ragged_buffer_factor probe: the step needs the largest factor over microbatches.
    aux["moe_required_rbf"] = jnp.max(scanned_aux["moe_required_rbf"], axis=0)
  if "moe_max_load_ratio" in scanned_aux:
    aux["moe_max_load_ratio"] = jnp.max(scanned_aux["moe_max_load_ratio"], axis=0)
    aux["moe_max_load_ratio_mean"] = jnp.mean(scanned_aux["moe_max_load_ratio_mean"], axis=0)

  if is_nnx:
    nnx.update(model, grad_and_loss["rest_state"])

  return loss, aux, raw_grads


def _jaxpp_gradient_accumulation_loss_and_grad(
    loss_fn, config, graphdef, params, custom_params, rest, params_shardings, data, custom_param_filter, extra_dpo_args
):
  """Differentiate pure NNX state for independently scheduled microbatches.

  Trainable gradients are summed. FP8 overwrite gradients are maxima and are
  applied to the model separately, without token normalization or clipping.
  """
  mesh = jax.tree.leaves(params_shardings)[0].mesh
  data_replicas = mesh.shape.get("data", 1)
  microbatches = config.num_pipeline_microbatches
  axis = 1 if data_replicas > 1 else 0

  def microbatched(a):
    shape = (data_replicas, microbatches, -1) + a.shape[1:]
    spec = ("data", None, "expert") + (None,) * (a.ndim - 1)
    if data_replicas == 1:
      shape, spec = shape[1:], spec[1:]
    reshaped = a.reshape(shape)
    return maybe_shard_with_name(
        reshaped,
        NamedSharding(mesh, jax.sharding.PartitionSpec(*spec)),
        config.shard_mode,
        debug_sharding=config.debug_sharding,
    )

  data = jax.tree.map(microbatched, data)

  # NNX custom-gradient variables can live beneath a Linen bridge without
  # parameter sharding metadata. Place them with the closest parameter subtree.
  def place_custom(path, variable):
    subtree = params
    for key in path:
      key = getattr(key, "key", getattr(key, "idx", None))
      if key not in subtree:
        break
      subtree = subtree[key]
    return variable.replace(value=jaxpp.place_with(variable.get_value(), jax.tree.leaves(subtree)[0]))

  custom_params = jax.tree_util.tree_map_with_path(
      place_custom, custom_params, is_leaf=lambda x: isinstance(x, nnx.Variable)
  )
  rest_filter = nnx.Not(nnx.Any(nnx.Param, custom_param_filter, nnx.Intermediate, nnx.RngKey))
  original_rest = dict(nnx.to_flat_state(rest))

  def diff_wrapper(params, custom_params, rest, microbatch, index):
    def fold_rng(variable):
      if issubclass(variable.type, nnx.RngKey):
        fold = functools.partial(jax.random.fold_in, data=index)
        for _ in range(variable.get_value().ndim):
          fold = jax.vmap(fold)
        return variable.replace(value=fold(variable.get_value()))
      return variable

    local_rest = jax.tree.map(fold_rng, rest, is_leaf=lambda x: isinstance(x, nnx.Variable))
    local_model = nnx.merge(graphdef, params, custom_params, local_rest, copy=True)
    loss, aux = loss_fn(local_model, config, microbatch, None, None, *extra_dpo_args, is_train=True)
    next_rest = nnx.state(local_model, rest_filter)
    # Initialization-only RNG streams (e.g. params) are not consumed by the
    # forward pass. Leave their state untouched, including its stage ownership.
    next_rest = nnx.from_flat_state(
        [
            (path, variable)
            for path, variable in nnx.to_flat_state(next_rest)
            if not (issubclass(variable.type, nnx.RngCount) and variable.get_value() is original_rest[path].get_value())
        ]
    )
    return loss, (aux, next_rest)

  grad_func = jax.value_and_grad(diff_wrapper, argnums=(0, 1), has_aux=True)

  def compute_grads(microbatch, index):
    if data_replicas > 1:
      index = index * data_replicas + jax.lax.axis_index("jaxpp_data")
    (_, (aux, next_rest)), grads = grad_func(params, custom_params, rest, microbatch, index)
    return aux, grads, next_rest

  if data_replicas > 1:
    compute_grads = jax.vmap(compute_grads, in_axes=(0, None), axis_name="jaxpp_data", spmd_axis_name="data")

  def compute_at(index):
    microbatch = jax.tree.map(lambda x: jnp.take(x, index, axis=axis), data)
    return compute_grads(microbatch, index)

  aux, (raw_grads, custom_grads), next_rest = jaxpp.treduce_i(
      compute_at,
      microbatches,
      schedule=load_schedule(config),
      operation=(jaxpp.Concat(axis=axis), (jaxpp.Add, jaxpp.Max), jaxpp.Concat(axis=axis)),
  )
  if data_replicas > 1:
    grad_shardings = jax.tree_util.tree_map_with_path(functools.partial(add_leading_axis, "data"), params_shardings)
    raw_grads = jax.lax.with_sharding_constraint(raw_grads, grad_shardings)
    raw_grads = jax.tree.map(lambda x: jnp.sum(x, axis=0), raw_grads)
    custom_grads = jax.tree.map(lambda x: jnp.max(x, axis=0), custom_grads)
    aux = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), aux)
    next_rest = jax.tree.map(lambda x: x[-1], next_rest)
  next_rest = jax.tree.map(lambda x: x[-1], next_rest)

  # Keys remain fixed across calls; counters account for every microbatch, so
  # the next optimizer step gets new dropout randomness as with upstream GA.
  def advance_count(path, variable):
    if issubclass(variable.type, nnx.RngCount):
      keys = tuple(getattr(key, "key", getattr(key, "idx", None)) for key in path)
      original = original_rest[keys].get_value()
      count = original + (variable.get_value() - original) * microbatches * data_replicas
      # A shared dropout stream is consumed by several stages but its final
      # counter is produced only by the last one. Replicate that update so the
      # next optimizer step can read it on all stages that own the stream.
      subtree = raw_grads
      matched = False
      for key in keys:
        if not hasattr(subtree, "keys") or key not in subtree:
          break
        subtree, matched = subtree[key], True
      if matched:
        count = jaxpp.cross_mpmd_all_reduce(
            count,
            *(jaxpp.place_with(jnp.zeros_like(count), g) for g in jax.tree.leaves(subtree)),
        )
      return variable.replace(value=count)
    return variable

  next_rest = jax.tree_util.tree_map_with_path(advance_count, next_rest, is_leaf=lambda x: isinstance(x, nnx.Variable))
  grad_and_loss = {
      "loss": jnp.sum(aux["xent_sum"]) + jnp.sum(aux.get("dpo_loss", 0.0)),
      "grad": raw_grads,
      "total_weights": jnp.sum(aux["total_weights"]),
      "moe_lb_loss": jnp.sum(aux["moe_lb_loss"]),
      "indexer_loss": jnp.sum(aux["indexer_loss"]),
      "mtp_loss": jnp.sum(aux["mtp_loss"]),
      "rest_state": nnx.State.merge(custom_grads, next_rest),
  }
  return grad_and_loss, aux


# GA helper functions
def update_sharding_for_reduced(sharding: NamedSharding) -> NamedSharding:
  """
  Add reduced on data axis of given NamedSharding
  """
  return sharding.update(spec=sharding.spec.update(reduced={"data"}))


def update_sharding_for_unreduced(sharding: NamedSharding) -> NamedSharding:
  """
  Add unreduced on data axis of given NamedSharding
  """
  return sharding.update(spec=sharding.spec.update(unreduced={"data"}))
