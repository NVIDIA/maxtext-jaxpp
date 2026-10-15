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

"""CPU smoke coverage of real NNX model initialization and JAXPP training."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from flax import nnx
import jax
import jax.numpy as jnp
import jaxpp.array
from jaxpp import api as jaxpp_api
import numpy as np
import pytest

from maxtext.configs import pyconfig
from maxtext.models.models import Transformer
from maxtext.trainers.pre_train import train
from maxtext.utils import train_utils


@pytest.mark.cpu_only
class JaxppTrainingTelemetryTest(unittest.TestCase):
  """Regression coverage for the NNX loop's JAXPP performance and profile output."""

  def test_fifty_step_summary_discards_warmup_and_trims_paired_samples(self):
    summary = train._performance_summary_after_warmup(list(range(50)), list(range(1000, 1050)), 6)
    self.assertEqual(summary, (27.5, 1027.5, 8, 28))
    # Sort by time while keeping each step's TFLOP/s paired with its time.
    summary = train._trimmed_performance_summary([101, 0, 2, 100, 1], [800, 100, 300, 200, 900])
    self.assertEqual(summary, (103 / 3, 1400 / 3, 1, 3))

  def test_short_run_has_no_post_warmup_summary(self):
    self.assertIsNone(train._performance_summary_after_warmup([1.0] * 6, [2.0] * 6, 6))
    self.assertEqual(train._performance_summary_after_warmup(list(range(10)), list(range(10)), 6), (7.5, 7.5, 0, 4))

  def test_sample_mismatch_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "sample counts do not match"):
      train._performance_summary_after_warmup(list(range(7)), list(range(6)), 6)

  def test_warmup_is_relative_to_resumed_run(self):
    self.assertEqual(train._jaxpp_performance_warmup_steps(True, "xplane", 105, 100), 7)
    self.assertEqual(train._jaxpp_performance_warmup_steps(True, "", 105, 100), 6)
    self.assertIsNone(train._jaxpp_performance_warmup_steps(False, "", 0, 0))

  def _mesh(self, multi_process=True):
    return SimpleNamespace(jax_mesh=SimpleNamespace(is_multi_process=multi_process), mpmd_dim=2)

  def _trace(self, root, stage, step=None):
    suffix = "" if step is None else f"step_{step}"
    path = Path(root) / f"proc_{stage:06}_mpmd_{stage:02}_gpu_{stage:06}_{suffix}" / "plugins/profile/session/host.xplane.pb"
    path.parent.mkdir(parents=True)
    path.touch()
    return path.resolve()

  def test_xplanes_merge_once_after_all_hosts_finish_in_numeric_stage_order(self):
    events = []
    with TemporaryDirectory(prefix="jaxpp profile ") as root:
      stage_one = self._trace(root, 1)
      stage_zero = self._trace(root, 0)
      with (
          patch.object(train.jax, "process_index", return_value=0),
          patch.object(train.multihost_utils, "sync_global_devices", side_effect=lambda _: events.append("ready")),
          patch.object(train.subprocess, "run", side_effect=lambda *args, **kwargs: events.append("merge")) as merge,
      ):
        train._merge_jaxpp_xplanes(root, self._mesh(), "xplane")
      self.assertEqual(events, ["ready", "merge"])
      merge.assert_called_once_with(
          ["merge_multihost_xplanes", f"0:{stage_zero}:0", f"1:{stage_one}:0"], cwd=Path(root), check=True
      )

  def test_periodic_profile_windows_have_separate_inputs_and_destinations(self):
    with TemporaryDirectory() as root:
      traces = {step: [self._trace(root, stage, step) for stage in (0, 1)] for step in (5, 10)}
      with (
          patch.object(train.jax, "process_index", return_value=0),
          patch.object(train.multihost_utils, "sync_global_devices"),
          patch.object(train.subprocess, "run") as merge,
      ):
        train._merge_jaxpp_xplanes(root, self._mesh(), "xplane")
      self.assertEqual(merge.call_count, 2)
      for call, (step, paths) in zip(merge.call_args_list, traces.items(), strict=True):
        self.assertEqual(call.args[0], ["merge_multihost_xplanes", f"0:{paths[0]}:0", f"1:{paths[1]}:0"])
        self.assertEqual(call.kwargs["cwd"], Path(root) / f"jaxpp_merged_step_{step}")

  def test_other_ranks_only_wait_for_profile_files(self):
    with (
        patch.object(train.jax, "process_index", return_value=1),
        patch.object(train.multihost_utils, "sync_global_devices") as ready,
        patch.object(train.subprocess, "run") as merge,
    ):
      train._merge_jaxpp_xplanes("unused", self._mesh(), "xplane")
    ready.assert_called_once()
    merge.assert_not_called()

  def test_missing_stage_files_warn_and_do_not_make_a_partial_profile(self):
    with TemporaryDirectory() as root:
      self._trace(root, 0)
      with (
          patch.object(train.jax, "process_index", return_value=0),
          patch.object(train.multihost_utils, "sync_global_devices"),
          patch.object(train.subprocess, "run") as merge,
          patch.object(train.max_logging, "log") as log,
      ):
        train._merge_jaxpp_xplanes(root, self._mesh(), "xplane")
      merge.assert_not_called()
      self.assertIn("stages [1] missing", log.call_args.args[0])

  def test_nsys_and_single_process_profiles_do_not_use_the_xplane_merger(self):
    with (
        patch.object(train.multihost_utils, "sync_global_devices") as ready,
        patch.object(train.subprocess, "run") as merge,
    ):
      train._merge_jaxpp_xplanes("unused", self._mesh(), "nsys")
      train._merge_jaxpp_xplanes("unused", self._mesh(multi_process=False), "xplane")
    ready.assert_not_called()
    merge.assert_not_called()


@pytest.mark.cpu_only
class JaxppTEMoeOverflowTest(unittest.TestCase):
  """Host decisions must use scalar values, not MPMD wrappers or missing zeros."""

  def setUp(self):
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape(1), ("stage",))
    self.mpmd_mesh = jaxpp_api.MpmdMesh(mesh, "stage")
    self.sharding = jaxpp_api.MpmdSharding(self.mpmd_mesh, {0}, jax.sharding.PartitionSpec())

  def _scalar(self, value, owned=True):
    arrays = []
    if owned:
      sharding = jax.sharding.NamedSharding(self.mpmd_mesh.unstack[0], jax.sharding.PartitionSpec())
      arrays = [jax.device_put(jnp.asarray(value, jnp.int32), sharding)]
    return jaxpp_api.MpmdArray(arrays, self.sharding, shape=(), dtype=jnp.int32)

  def test_wrapped_zero_does_not_overflow_and_true_flag_preserves_counts(self):
    zero = self._scalar(0)
    window = [(4, zero, self._scalar(8), self._scalar(16)), (5, self._scalar(1), self._scalar(17), self._scalar(16))]
    self.assertEqual(train._te_moe_overflowing_steps(window, True), [(5, 1, 17, 16)])
    self.assertFalse(zero.is_deleted())

  def test_non_owner_gets_the_same_decision_as_the_remote_owner(self):
    for overflow in (0, 1):
      with self.subTest(overflow=overflow):
        window = [(7, self._scalar(0, False), self._scalar(0, False), self._scalar(0, False))]
        remote = np.asarray([[[overflow, 1], [17, 1], [16, 1]]], dtype=np.int32)
        with patch.object(
            train.multihost_utils, "process_allgather", side_effect=lambda local, **kwargs: np.stack([local, remote])
        ) as gather:
          result = train._te_moe_overflowing_steps(window, True)
        np.testing.assert_array_equal(gather.call_args.args[0], np.zeros((1, 3, 2), np.int32))
        self.assertEqual(result, [(7, 1, 17, 16)] if overflow else [])

  def test_flag_count_and_capacity_can_have_different_owners(self):
    window = [(8, self._scalar(0, False), self._scalar(0, False), self._scalar(16))]
    flag_owner = np.asarray([[[1, 1], [0, 0], [0, 0]]], dtype=np.int32)
    count_owner = np.asarray([[[0, 0], [17, 1], [0, 0]]], dtype=np.int32)
    with patch.object(
        train.multihost_utils,
        "process_allgather",
        side_effect=lambda local, **kwargs: np.stack([local, flag_owner, count_owner]),
    ):
      self.assertEqual(train._te_moe_overflowing_steps(window, True), [(8, 1, 17, 16)])

  def test_missing_owner_is_rejected_after_all_hosts_participate(self):
    window = [(9, self._scalar(0, False), self._scalar(0, False), self._scalar(0, False))]
    with patch.object(
        train.multihost_utils, "process_allgather", side_effect=lambda local, **kwargs: local[None]
    ) as gather:
      with self.assertRaisesRegex(RuntimeError, "no owning process"):
        train._te_moe_overflowing_steps(window, True)
    gather.assert_called_once()

  def test_non_jaxpp_keeps_the_original_check_without_a_collective(self):
    window = [(4, jnp.int32(0), jnp.int32(8), jnp.int32(16)), (5, jnp.int32(1), jnp.int32(17), jnp.int32(16))]
    with patch.object(train.multihost_utils, "process_allgather") as gather:
      self.assertEqual(train._te_moe_overflowing_steps(window, False), [(5, 1, 17, 16)])
      self.assertEqual(train._te_moe_overflowing_steps([], True), [])
    gather.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "decoder_block,repeats,microbatches,schedule,dropout_rate",
    [
        pytest.param("default", 1, 2, "1f1b", 0.0, id="dense"),
        pytest.param("default", 2, 4, "interleaved_1f1b", 0.1, id="dense-interleaved-dropout"),
        pytest.param("qwen3_moe", 2, 4, "interleaved_1f1b", 0.0, id="qwen3-moe-interleaved"),
    ],
)
def test_real_nnx_transformer_runs_two_jaxpp_steps(
    tmp_path, decoder_block, repeats, microbatches, schedule, dropout_rate
):
  if jax.default_backend() != "cpu" or jax.device_count() != 2:
    pytest.skip("Run with JAX_PLATFORMS=cpu and XLA_FLAGS=--xla_force_host_platform_device_count=2")
  metrics_path = tmp_path / "metrics.jsonl"
  config_path = Path(__file__).resolve().parents[2] / "src/maxtext/configs/decoupled_base_test.yml"
  config = pyconfig.initialize(
      ["jaxpp_nnx_train_loop_test", str(config_path)],
      run_name="jaxpp_nnx_smoke",
      base_output_directory=str(tmp_path),
      tensorboard_dir=str(tmp_path / "tensorboard"),
      metrics_file=str(metrics_path),
      enable_tensorboard=False,
      enable_checkpointing=False,
      save_checkpoint_on_start=False,
      save_checkpoint_on_completion=False,
      dataset_type="synthetic",
      dataset_path="",
      packing=False,
      model_name="default",
      decoder_block=decoder_block,
      base_emb_dim=8,
      base_num_query_heads=2,
      base_num_kv_heads=2,
      head_dim=4,
      base_mlp_dim=16,
      base_num_decoder_layers=4,
      vocab_size=32,
      max_target_length=8,
      per_device_batch_size=2.0,
      dtype="float32",
      weight_dtype="float32",
      grad_dtype="float32",
      steps=2,
      learning_rate_schedule_steps=4,
      log_period=1,
      hardware="cpu",
      attention="dot_product",
      shard_mode="auto",
      scan_layers=False,
      use_jaxpp=True,
      schedule=schedule,
      num_pipeline_repeats=repeats,
      num_pipeline_microbatches=microbatches,
      dropout_rate=dropout_rate,
      enable_dropout=dropout_rate > 0,
      ici_pipeline_parallelism=2,
      ici_fsdp_parallelism=1,
      ici_data_parallelism=1,
      ici_expert_parallelism=1,
      remat_policy="minimal",
      enable_goodput_recording=False,
      enable_checkpoint_cloud_logger=False,
      monitor_goodput=False,
      profiler="",
      enable_jax_profiler=False,
      **(
          {
              "num_experts": 4,
              "num_experts_per_tok": 2,
              "base_moe_mlp_dim": 16,
              "mlp_activations": ["silu", "linear"],
              "use_qk_norm": True,
              "norm_topk_prob": True,
              "load_balance_loss_weight": 0.001,
              "sparse_matmul": False,
              "megablox": False,
          }
          if decoder_block == "qwen3_moe"
          else {}
      ),
  )
  *_, initial_state = train_utils.setup_train_loop(config, recorder=None)
  assert isinstance(initial_state.model, Transformer)
  initial_params = jax.tree.map(np.asarray, nnx.state(initial_state.model, nnx.Param))
  initial_rng_counts = jax.tree.map(np.asarray, nnx.state(initial_state.model, nnx.RngCount))

  # CPU devices do not expose GPU allocator statistics used to size reshard chunks.
  with patch.object(jaxpp.array, "_get_working_memory_threshold", return_value=1 << 20):
    final_state = train.train_loop(config, recorder=None)
  jax.block_until_ready(final_state)
  assert isinstance(final_state, nnx.State)
  assert int(final_state.optimizer.step.get_value()) == 2
  final_params = nnx.filter_state(final_state.model, nnx.Param)
  assert jax.tree.structure(initial_params) == jax.tree.structure(final_params)
  assert any(
      not np.array_equal(before, np.asarray(after))
      for before, after in zip(jax.tree.leaves(initial_params), jax.tree.leaves(final_params), strict=True)
  )
  if dropout_rate > 0:
    final_rng_counts = nnx.filter_state(final_state.model, nnx.RngCount)
    assert any(
        np.any(np.asarray(after) > before)
        for before, after in zip(jax.tree.leaves(initial_rng_counts), jax.tree.leaves(final_rng_counts), strict=True)
    )
  metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
  assert len(metrics) == 2
  assert all(np.isfinite(step["learning/loss"]) for step in metrics)
