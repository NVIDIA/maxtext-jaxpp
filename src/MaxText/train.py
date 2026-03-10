# Copyright 2023–2025 Google LLC
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# pylint: disable=g-bad-todo, abstract-method, consider-using-with
"""Training loop and Decoding of the model."""

# Calling jax.device_count here prevents a "TPU platform already registered" error.
# See github.com/google/maxtext/issues/20 for more

from typing import Any, Sequence
import contextlib
import datetime
import functools
import os

from absl import app

import numpy as np

import pathwaysutils  # pylint: disable=unused-import

import tensorflow as tf

import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.linen import partitioning as nn_partitioning

from MaxText import pyconfig
from MaxText import sharding
from MaxText.layers.multi_token_prediction import calculate_mtp_acceptance_rate, calculate_mtp_loss
from MaxText.common_types import ShardMode
from MaxText.globals import EPS
# Placeholder: internal

from MaxText.gradient_accumulation import gradient_accumulation_loss_and_grad
from MaxText.vocabulary_tiling import vocab_tiling_linen_loss
# pylint: disable=too-many-positional-arguments

from maxtext.common import checkpointing, profiler
from maxtext.common.goodput import (
    GoodputEvent,
    create_goodput_recorder,
    maybe_monitor_goodput,
    maybe_record_goodput,
)
from maxtext.common.gcloud_stub import cloud_diagnostics as _cloud_diag, is_decoupled
from maxtext.common.gcloud_stub import vertex_tensorboard_modules
from maxtext.common.metric_logger import MetricLogger, record_activation_metrics
from maxtext.trainers.post_train.dpo.dpo_utils import _merge_dpo_state, _split_dpo_state, dpo_loss_fn
from maxtext.utils import exceptions
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import train_utils

"""
JaxPP related imports
"""
# system
import subprocess

from statistics import mean

# jaxpp
from jaxpp import __version__ as jaxpp_version
from packaging.version import Version
import jaxpp.api as jaxpp

_diag_modules = _cloud_diag()
diagnostic, debug_configuration, diagnostic_configuration, stack_trace_configuration = _diag_modules
VertexTensorboardManager, _vertex_tb_is_stub = vertex_tensorboard_modules()



def get_first_step(state):
  return int(max_utils.maybe_unwrap(state.step))


# -----------------------------------------------------------------------------
# Top-level Functions
# -----------------------------------------------------------------------------


def loss_fn(model, config, data, dropout_rng, params, is_train=True):
  """loss_fn for both train and eval.

  Args:
    model: A nn.Module
    config: Config of parameters
    data: Batch of data to apply to the model
    dropout_rng: A key to use to generate rng for dropout
    params: Model params
    is_train: True for train_step and False for eval_step

  Returns:
    loss: average loss
    aux: a dictionary including intermediate_outputs, total_loss, and total_weights
  """
  # decimate proportion of data when per_device_batch_size<1
  if is_train and not config.use_jaxpp:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_train_on, :]
  else:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_eval_on, :]
  mutable_collections = ["intermediates"]
  if config.mtp_num_layers > 0 and is_train:
    # The single model.apply call now triggers the entire chain if MTP is enabled:
    # Decoder runs -> returns hidden_state -> MTPBlock uses it -> MTPBlock sows losses -> we reap them here.
    mutable_collections.append("mtp_losses")

  # During evaluation, if the acceptance rate test is enabled, we must
  # make its specific collection mutable so the MTPBlock can sow into it.
  if config.mtp_eval_target_module > 0 and not is_train:
    mutable_collections.append("mtp_acceptance")

  if isinstance(model, nn.Module):
    # inputs, targets, segments, positions = apply_args
    rng1, aqt_rng = jax.random.split(dropout_rng)

    # Flax Linen model
    logits, intermediate_outputs = model.apply(
        params,
        data["inputs"],
        data["inputs_position"],
        decoder_segment_ids=data["inputs_segmentation"],
        encoder_images=data["images"] if config.use_multimodal else None,
        encoder_image_masks=data["image_masks"] if config.use_multimodal and "image_masks" in data else None,
        enable_dropout=config.enable_dropout if is_train else False,
        rngs={"dropout": rng1, "params": aqt_rng},
        mutable=mutable_collections,
        decoder_target_tokens=data["targets"],
        decoder_target_mask=data["targets_segmentation"],
    )

    if config.num_vocab_tiling > 1:
      hidden_state_key = ("intermediates", "decoder", "hidden_states")
      hidden_states = maxtext_utils.get_nested_value(intermediate_outputs, hidden_state_key)[0]
      total_loss = vocab_tiling_linen_loss(hidden_states, data, config, model, params, is_train)
    else:
      one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
      xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets)
      xent = sharding.maybe_shard_with_logical(
          xent,
          ("activation_embed_and_logits_batch", "activation_length"),
          model.mesh,
          config.shard_mode,
          debug_sharding=config.debug_sharding,
      )
      # Mask out paddings at the end of each example.
      xent = xent * (data["targets_segmentation"] != 0)
      total_loss = jnp.sum(xent)
  else:
    # Flax NNX model
    logits = model(
        decoder_input_tokens=data["inputs"],
        decoder_positions=data["inputs_position"],
        decoder_segment_ids=data["inputs_segmentation"],
        encoder_images=data["images"] if config.use_multimodal else None,
        encoder_image_masks=data["image_masks"] if config.use_multimodal and "image_masks" in data else None,
        enable_dropout=config.enable_dropout if is_train else False,
        decoder_target_tokens=data["targets"],
        decoder_target_mask=data["targets_segmentation"],
    )
    intermediate_outputs = {}
    one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
    xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets)
    xent = nn.with_logical_constraint(xent, ("activation_embed_and_logits_batch", "activation_length"))
    # Mask out paddings at the end of each example.
    xent = xent * (data["targets_segmentation"] != 0)
    total_loss = jnp.sum(xent)

  total_weights = jnp.sum(data["targets_segmentation"] != 0)
  # If gradient accumulation is enabled, we don't need to divide total_loss
  # by total_weights and then multiply the computed gradient by total_weights,
  # since it's equivalent to computing the gradient from total_loss.
  # This simplification reduces the number of operations and makes it easier
  # for XLA to move all-reduce out of the gradient accumulation loop when use
  # Zero1+GA to reduce communication overhead.
  # EPS was used to avoid division by zero, but it's not needed when gradient
  # accumulation is enabled since there's no division.
  if config.gradient_accumulation_steps > 1 and not config.use_tunix_gradient_accumulation:
    loss = total_loss
  else:
    # When using Tunix gradient accumulation, we revert to standard normalization.
    # Unlike the manual accumulation path above, Tunix (via optax.MultiSteps) expects
    # a normalized loss for each step. It handles the accumulation state
    # updates and scaling internally.
    loss = total_loss / (total_weights + EPS)

  # Calculate and Add MTP Loss
  mtp_loss = 0.0
  if config.mtp_num_layers > 0 and is_train:
    mtp_loss = calculate_mtp_loss(intermediate_outputs, config)
    loss += mtp_loss

  # get MoE load balance loss
  moe_lb_loss = 0.0
  if config.num_experts > 1:
    nested_key = ("intermediates", "decoder", "layers", "moe_lb_loss")
    total_moe_lb_loss = maxtext_utils.get_nested_value(intermediate_outputs, nested_key, 0.0)
    moe_lb_loss = jnp.mean(jnp.array(total_moe_lb_loss))
    loss += moe_lb_loss

  # get MoE routed bias term updates
  moe_bias_updates = None
  if config.routed_bias and config.routed_bias_update_rate > 0.0:
    nested_key = ("intermediates", "decoder", "moe_layers", "moe_bias_updates")
    moe_bias_updates = maxtext_utils.get_nested_value(intermediate_outputs, nested_key, None)

  # Add the model's primary output to the intermediates dict so it can be used
  # by the acceptance rate calculation in eval_step.
  intermediate_outputs["logits"] = logits

  aux = {
      "intermediate_outputs": intermediate_outputs,
      "total_loss": total_loss,
      "total_weights": total_weights,
      "moe_lb_loss": moe_lb_loss,
      "moe_bias_updates": moe_bias_updates,
      "mtp_loss": mtp_loss,
  }
  return loss, aux


def load_schedule(config):
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
  elif config.schedule == "zero_bubble":
      assert num_logical_stages <= pipeline_parallel_dim
      schedule = jaxpp.ZeroBubble(num_logical_stages)
  elif config.schedule == "dualpipev":
      schedule = jaxpp.DualPipeV(num_logical_stages, pipeline_parallel_dim)
  else:
      raise NotImplementedError(f"Unknown schedule {config.schedule}")
  return schedule


def add_leading_axis(
    axis_name: str, path: jax.tree_util.KeyPath, s: jax.sharding.NamedSharding
):
    assert isinstance(s, jax.sharding.NamedSharding)
    used = {n for ns in s.spec for n in (ns if isinstance(ns, tuple) else (ns,))}
    if axis_name in used:
        raise ValueError(
            f"mesh axis name {axis_name} cannot appear in "
            f"out_shardings. Found out_shardings{jax.tree_util.keystr(path)}={s.spec}"
        )
    return jax.sharding.NamedSharding(s.mesh, jax.sharding.PartitionSpec(axis_name, *s.spec), memory_kind=s.memory_kind)


def train_step(model, config, state_mesh_shardings, params_shardings, state, data, dropout_rng):
  """

  Args:
    model: A nn.Module
    state: A pytree of the current state of the model
    data: Batch of data to apply to the model
    dropout_rng: A key to use to generate rng for dropout

  Returns:
    new_state: Same format as state.
    metrics: Dictionary of model metrics such as loss, training rate, etc.
    rng2: A new rng key that can be used in future calls.

  """
  reference_params, reference_params_sharding, extra_dpo_args, _loss_fn = (
      [],
      [],
      [],
      loss_fn,
  )
  if config.use_dpo:
    state, reference_params = _split_dpo_state(state)
    state_mesh_shardings, reference_params_sharding = _split_dpo_state(state_mesh_shardings)
    extra_dpo_args = [reference_params]
    _loss_fn = dpo_loss_fn

  params = state.params

  if config.gradient_accumulation_steps > 1 or config.use_jaxpp:
    loss, aux, raw_grads = gradient_accumulation_loss_and_grad(
        _loss_fn,
        config,
        model,
        params,
        params_shardings,
        data,
        dropout_rng,
        extra_dpo_args,
    )
  else:
    if config.optimizer_memory_host_offload:
      if config.use_dpo:
        reference_params = jax.device_put(
            reference_params,
            max_utils.with_memory_kind(reference_params_sharding, "device"),
        )
        extra_dpo_args = [reference_params]

    if config.shard_optimizer_over_data:
      params = jax.tree.map(
          functools.partial(sharding.maybe_shard_with_name, shard_mode=config.shard_mode),
          params,
          params_shardings,
      )
    grad_func = jax.value_and_grad(_loss_fn, argnums=4, has_aux=True)
    (loss, aux), raw_grads = grad_func(model, config, data, dropout_rng, params, *extra_dpo_args, is_train=True)

  raw_grads = jax.lax.with_sharding_constraint(raw_grads, state_mesh_shardings.params)
  raw_grads = jax.tree_util.tree_map(
      lambda x: x.astype(config.grad_dtype) if x.dtype == jnp.float32 else x,
      raw_grads,
  )
  # Compute raw gradient norm for clipping and metrics (before extracting aux values)
  owg = raw_grads.pop(nn.fp8_ops.OVERWRITE_WITH_GRADIENT, None)
  raw_grad_norm = max_utils.l2norm_pytree(raw_grads)
  
  intermediate_outputs = aux["intermediate_outputs"]
  total_weights = aux["total_weights"]
  moe_lb_loss = aux["moe_lb_loss"]
  moe_bias_updates = aux["moe_bias_updates"]
  mtp_loss = aux["mtp_loss"]
  
  if config.gradient_clipping_threshold > 0:
    grads = maxtext_utils.apply_gradient_clipping(raw_grads, state, raw_grad_norm, config.gradient_clipping_threshold)
  else:
    grads = raw_grads
  if owg is not None:
    grads[nn.fp8_ops.OVERWRITE_WITH_GRADIENT] = owg
  if config.optimizer_memory_host_offload:
    state = state.replace(
        opt_state=jax.device_put(
            state.opt_state,
            jax.tree_util.tree_map(
                lambda x: x.with_memory_kind(kind="device"),
                state_mesh_shardings.opt_state,
            ),
        )
    )
  # Move all parameters to device before optimizer update
  if config.parameter_memory_host_offload:
    max_logging.log("\nMoving all parameters to device before optimizer update")

    def move(path, value):
      max_logging.log(f"train.py: Moving f{path} to device")
      return value.with_memory_kind(kind="device")

    state = state.replace(
        params=jax.device_put(
            state.params,
            jax.tree_util.tree_map_with_path(move, state_mesh_shardings.params),
        )
    )
  new_state = state.apply_gradients(grads=grads)

  # Apply updates for Auxiliary-Loss-Free load balancing for DeepSeek family
  if config.routed_bias and config.routed_bias_update_rate > 0.0 and moe_bias_updates is not None:
    target_path = ("params", "decoder", "moe_layers", "DeepSeekMoeBlock_0", "MoeBlock_0", "gate", "bias")
    # Flax 'sow' returns a tuple, so we take the first element [0].
    # Updates the shape to be aligned with state.
    moe_bias_updates = jnp.array(moe_bias_updates[0]).transpose()
    new_state = maxtext_utils.update_state_param(new_state, target_path, moe_bias_updates)

  if config.use_jaxpp:
    # TODO: refine logic to match the one in MaxText's gradient accumulation
    #  or use that altogether (add support for scan instead of
    #  treduce in JaxPP)
    scalar_metrics = {
        "learning/loss": loss.sum() / total_weights.sum(),
        "learning/moe_lb_loss": moe_lb_loss.sum() if hasattr(moe_lb_loss, 'sum') else moe_lb_loss,
        "learning/mtp_loss": mtp_loss.sum() if hasattr(mtp_loss, 'sum') else mtp_loss,
        "learning/total_weights": total_weights.sum(),
    }
  else:
    scalar_metrics = {
        "learning/loss": loss,
        "learning/moe_lb_loss": moe_lb_loss,
        "learning/mtp_loss": mtp_loss,
        "learning/total_weights": total_weights,
    }
  if not config.optimizer_memory_host_offload:
    owg = grads.pop(nn.fp8_ops.OVERWRITE_WITH_GRADIENT, None)
    scalar_metrics["learning/grad_norm"] = max_utils.l2norm_pytree(grads["params"])
    if owg is not None:
      grads[nn.fp8_ops.OVERWRITE_WITH_GRADIENT] = owg
    scalar_metrics["learning/raw_grad_norm"] = raw_grad_norm

    new_params = new_state.params
    owg = new_params.pop(nn.fp8_ops.OVERWRITE_WITH_GRADIENT, None)
    scalar_metrics["learning/param_norm"] = max_utils.l2norm_pytree(new_params)
    if owg is not None:
      new_params[nn.fp8_ops.OVERWRITE_WITH_GRADIENT] = owg
    new_state = new_state.replace(params=new_params)
  if config.use_dpo:
    scalar_metrics["learning/dpo_reward_accuracy"] = aux["reward_accuracy"]
  metrics = {
      "scalar": scalar_metrics,
      "scalars": {},
  }

  if config.record_internal_nn_metrics:
    record_activation_metrics(metrics, intermediate_outputs, config)

  if config.use_dpo:
    new_state = _merge_dpo_state(new_state, reference_params)

  return new_state, metrics


def eval_step(model, config, state, data, dropout_rng):
  """eval_step no backprop and new state compared with train_step."""

  reference_params, extra_dpo_args, _loss_fn = [], [], loss_fn
  if config.use_dpo:
    state, reference_params = _split_dpo_state(state)
    extra_dpo_args = [reference_params]
    _loss_fn = dpo_loss_fn

  eval_loss_fn = functools.partial(_loss_fn, model, config, data, dropout_rng, is_train=False)
  loss, aux = eval_loss_fn(state.params, *extra_dpo_args)

  mtp_acceptance_rate = 0.0
  if config.mtp_eval_target_module > 0:
    mtp_acceptance_rate = calculate_mtp_acceptance_rate(aux["intermediate_outputs"], config)

  total_loss = aux["total_loss"]
  total_weights = aux["total_weights"]
  moe_lb_loss = aux["moe_lb_loss"]
  mtp_loss = aux["mtp_loss"]
  metrics = {
      "scalar": {
          "evaluation/loss": loss,
          "evaluation/total_loss": total_loss,
          "evaluation/total_weights": total_weights,
          "evaluation/moe_lb_loss": moe_lb_loss,
          "evaluation/mtp_loss": mtp_loss,
          "evaluation/mtp_acceptance_rate_percent": mtp_acceptance_rate,
      },
  }
  if config.use_dpo:
    metrics["scalar"]["evaluation/dpo_reward_accuracy"] = aux["reward_accuracy"]

  return jax.tree.map(jax._src.numpy.lax_numpy._array_copy, metrics)


def train_loop(config, recorder, state=None):
  """Main Training loop."""
  (
      init_rng,
      checkpoint_manager,
      state_mesh_shardings,
      model,
      mesh,
      learning_rate_schedule,
      data_iterator,
      data_loader,
      rampup_manager,
      eval_data_iterator,
      state,
  ) = train_utils.setup_train_loop(config, recorder)

  if config.use_dpo:
    if "reference_params" not in state.params:
      reference_params = jax.tree.map(jnp.copy, state.params["params"])
      state = _merge_dpo_state(state, reference_params)
    state_mesh_shardings = _merge_dpo_state(state_mesh_shardings, state_mesh_shardings.params["params"])

  params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)

  mpmd_mesh = jaxpp.MpmdMesh(mesh, "stage") if config.use_jaxpp else None
  if config.use_jaxpp:
    mesh = mpmd_mesh.lowering_mesh()

  p_train_step, p_eval_step = train_utils.jit_train_and_eval_step(
      config,
      model,
      mpmd_mesh.jax_mesh if config.use_jaxpp else mesh,
      state,
      state_mesh_shardings,
      train_step,
      eval_step,
      eval_data_iterator,
      jax.tree.map(lambda s: s.update(mesh=mpmd_mesh.lowering_mesh()), params_shardings) if config.use_jaxpp else params_shardings,
  )

  if config.use_jaxpp:
    shaped_batch = maxtext_utils.get_shaped_batch(config)
    p_train_step = p_train_step.compile(state, shaped_batch, init_rng)
    args_mpmd_shardings, kwargs_mpmd_shardings = p_train_step.in_shardings
    state = jaxpp.spmd_to_mpmd_reshard(mpmd_mesh, state, args_mpmd_shardings[0])
  else:
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
      shaped_batch = maxtext_utils.get_shaped_batch(config)
      if config.shard_optimizer_over_data:
        state = sharding.maybe_shard_with_name(state, state_mesh_shardings, config.shard_mode)
      maxtext_utils.maybe_dump_jaxpr(config, p_train_step, (state, shaped_batch, init_rng))
      if config.compiled_trainstep_file == "":  # compile only when there is no pre-compiled file loaded
        compiled = p_train_step.lower(state, shaped_batch, init_rng).compile()
        compiled_stats = compiled.memory_analysis()
        max_utils.print_compiled_memory_stats(compiled_stats)

  start_step = get_first_step(state)  # this is the start_step for training
  prof = profiler.Profiler(config, offset_step=start_step)
  metric_logger = MetricLogger(config=config, learning_rate_schedule=learning_rate_schedule)

  # Write train config params, num model params, and XLA flags to tensorboard
  metric_logger.write_setup_info_to_tensorboard(state.params)

  task_times = None
  try:
    step_time = []
    step_tflops = []
    # NOTE: The dict values are unused when use_jaxpp is False.
    profiling_process_ids = {pid: "" for pid in jax.process_indices()}
    if config.use_jaxpp:
      idx = tuple(slice(None) if i == mpmd_mesh.mpmd_axis else 0 for i in range(len(mpmd_mesh.jax_mesh.shape)))
      first_device_per_mpmd_rank = mpmd_mesh.jax_mesh.devices[idx]
      profiling_process_ids = {d.process_index: d for d in first_device_per_mpmd_rank}

    last_step_completion = datetime.datetime.now()
    for step in np.arange(start_step, config.steps):
      prof.maybe_activate_profiler(step, state, maybe_mpmd_mesh=mpmd_mesh, profiling_process_ids=profiling_process_ids)

      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        example_batch = data_loader.load_next_batch(rampup_manager=rampup_manager)
        # pylint: disable=not-callable
        nextrng = jax.jit(jax.random.fold_in)(init_rng, step)

        if config.use_jaxpp:
          example_batch, nextrng = jaxpp.spmd_to_mpmd_reshard(mpmd_mesh, (example_batch, nextrng), p_train_step.in_shardings[0][1:])

        # We don't want to collect task times at the same time we're profiling
        enable_task_times = config.use_jaxpp and step == start_step + config.skip_first_n_steps_for_profiler + 1
        with maybe_record_goodput(recorder, GoodputEvent.STEP, step), jaxpp.collect_task_times_ms(enable_task_times) as step_task_times:
          with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
            if config.shard_optimizer_over_data and not config.use_jaxpp:
              state = sharding.maybe_shard_with_name(state, state_mesh_shardings, config.shard_mode)
            state, metrics = p_train_step(state, example_batch, nextrng)

          if step_task_times is not None:
            task_times = step_task_times

      step_time_delta = datetime.datetime.now() - last_step_completion
      last_step_completion = datetime.datetime.now()

      state_to_save = state if not config.use_dpo else _split_dpo_state(state)[0]
      checkpointing.maybe_save_checkpoint(checkpoint_manager, state_to_save, config, data_iterator, step)

      if config.dump_hlo and step == (config.dump_step if config.dump_step >= 0 else start_step):
        jax.block_until_ready(state)  # Ensure compilation has finished.
        gcs_utils.upload_dump(
            config.dump_hlo_local_dir,
            config.dump_hlo_gcs_dir,
            module_name=config.dump_hlo_module_name,
            delete_local_after=config.dump_hlo_delete_local_after,
            all_host_upload=config.dump_hlo_upload_all,
        )

      if config.eval_interval > 0 and step > start_step and (step + 1) % config.eval_interval == 0:
        assert eval_data_iterator
        # Explicitly reset the eval iterator and counters before starting the eval loop
        eval_data_iterator.reset()
        metric_logger.reset_eval_metrics()

        eval_step_count = 0
        # pylint: disable=not-callable
        for eval_batch in eval_data_iterator:
          if config.eval_steps > 0 and eval_step_count >= config.eval_steps:
            break
          with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
            eval_metrics = p_eval_step(state, eval_batch, nextrng)
          metric_logger.record_eval_metrics(step, metrics=eval_metrics)
          max_logging.log(f"Completed eval step {eval_step_count}")
          eval_step_count += 1
        metric_logger.record_eval_metrics(step, eval_step_count=eval_step_count)
        if metric_logger.cumulative_eval_metrics["scalar"]["eval/avg_loss"] <= config.target_eval_loss:
          prof.deactivate()
          raise exceptions.StopTraining(f"Target loss {config.target_eval_loss=} is achieved.")

      prof.maybe_deactivate_profiler(step, state, profiling_process_ids=profiling_process_ids)

      if step == start_step:
        max_utils.print_mem_stats("After params initialized")

      metric_logger.buffer_and_write_train_metrics(metrics, step, step_time_delta)
      step_time.append(metrics['scalar']['perf/step_time_seconds'])
      step_tflops.append(metrics['scalar']['perf/per_device_tflops_per_sec'])

    if config.use_jaxpp:
      assert mpmd_mesh is not None
      state = jaxpp.mpmd_to_spmd_reshard(mpmd_mesh, state, state_mesh_shardings)

      if prof.mode != "":
        command = """find . -wholename '*proc_*_mpmd*/*.xplane.pb' | sort | awk '{line=$0; sub(/.*mpmd_/, "", line); sub(/_.*/, "", line); printf "%d:%s:0 ", line, $0}'"""
        subprocess.run(
          [f"merge_multihost_xplanes $({command})"],
          shell=True, cwd=config.tensorboard_dir, check=True
        )

    if config.save_checkpoint_on_completion:
      state_to_save = state if not config.use_dpo else _split_dpo_state(state)[0]
      checkpointing.maybe_save_checkpoint(checkpoint_manager, state_to_save, config, data_iterator)
    if checkpoint_manager is not None:
      # in case the last checkpoint_period checkpoint is still in progress
      checkpoint_manager.wait_until_finished()
  except exceptions.StopTraining as e:
    max_logging.log(f"Training stopped: {str(e)}")
  finally:
    metric_logger.flush_metrics_and_cleanup()

  # last_profiling_step + 2 as (1) we count steps from 0, and (2) the execution time for merge_multihost_xplanes is
  # counted toward the execution time for the step right after the last profiling step.
  num_warmup_steps = (prof.finished_initial_profile_step + 2) if prof.mode != "" else 6
  max_logging.log(
      f"excluding the first {num_warmup_steps} steps: avg time per step {mean(step_time[num_warmup_steps:])}, avg tflops per step {mean(step_tflops[num_warmup_steps:])}"
  )

  if task_times is not None:
    for task_name, times in task_times.items():
      max_logging.log(f"task {task_name}: {mean(times)} ms")

  return state


def initialize(argv: Sequence[str]) -> tuple[pyconfig.HyperParameters, Any, Any]:
  """Initialization of hyperparameters and utilities"""
  pathwaysutils.initialize()
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  # TF allocates extraneous GPU memory when using TFDS data
  # this leads to CUDA OOMs. WAR for now is to hide GPUs from TF
  tf.config.set_visible_devices([], "GPU")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )
  # TODO: mazumdera@ : ensure missing mandatory fields in base.yml are filled in in argv,
  # or fill in here
  config = pyconfig.initialize(argv)
  max_utils.print_system_information()
  train_utils.validate_train_config(config)
  jax.config.update("jax_use_shardy_partitioner", config.shardy)
  # update explicit sharding-supported config
  if config.shard_mode == ShardMode.EXPLICIT:
    jax.config.update("jax_remove_size_one_mesh_axis_from_type", True)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path or ""
  vertex_tensorboard_manager = VertexTensorboardManager()
  if config.use_vertex_tensorboard or os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"):
    vertex_tensorboard_manager.configure_vertex_tensorboard(config)

  # Create the Goodput recorder
  recorder = create_goodput_recorder(config)

  # Stack traces configurations
  debug_config = debug_configuration.DebugConfig(
      stack_trace_config=stack_trace_configuration.StackTraceConfig(
          collect_stack_trace=config.collect_stack_trace,
          stack_trace_to_cloud=config.stack_trace_to_cloud,
          stack_trace_interval_seconds=config.stack_trace_interval_seconds,
      )
  )
  diagnostic_config = diagnostic_configuration.DiagnosticConfig(debug_config)
  return config, recorder, diagnostic_config


def run(config, recorder, diagnostic_config):
  """Run the job given hyperparameters and utilities.

  In decoupled mode (DECOUPLE_GCLOUD=TRUE) cloud diagnostics may be stubbed; if so, skip wrapping.
  """
  # Use nullcontext when diagnostics are stubbed or in decoupled mode
  diagnostics_context = (
      contextlib.nullcontext()
      if is_decoupled() or getattr(diagnostic, "__class__", None).__name__ == "_StubDiag"
      else diagnostic.diagnose(diagnostic_config)
  )

  if is_decoupled() or getattr(diagnostic, "__class__", None).__name__ == "_StubDiag":
    max_logging.log("[DECOUPLED NO-OP] skipping cloud diagnostics wrapper.")

  with (
      diagnostics_context,
      maybe_record_goodput(recorder, GoodputEvent.JOB),
      max_utils.maybe_get_transformer_engine_context(config),
      maybe_monitor_goodput(config),
  ):
    train_loop(config, recorder)


def main(argv: Sequence[str]) -> None:
  config, recorder, diagnostic_config = initialize(argv)
  run(config, recorder, diagnostic_config)


if __name__ == "__main__":
  app.run(main)
