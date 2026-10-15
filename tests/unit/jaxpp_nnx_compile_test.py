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

"""NNX graph and sharding contracts at the JAXPP compilation boundary."""

import functools
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from flax import nnx
import jax
import jax.numpy as jnp
import jaxpp.api as jaxpp
import numpy as np
import optax
import pytest

from maxtext.common.train_state_nnx import TrainStateNNX
from maxtext.configs import pyconfig
from maxtext.trainers.pre_train import train_compile
from maxtext.utils import train_utils


class _NestedModel(nnx.Module):

  def __init__(self, mesh):
    self.mesh = mesh
    self.weight = nnx.Param(jnp.ones((2, 2)), out_sharding=(None, None))
    self.constrain = functools.partial(jax.lax.with_sharding_constraint, shardings=jax.NamedSharding(mesh, jax.P()))


def _mesh():
  count = min(2, len(jax.devices()))
  return jax.sharding.Mesh(np.asarray(jax.devices()[:count]).reshape(count, 1), ("stage", "data"))


def _config(**overrides):
  settings = dict(
      use_jaxpp=True,
      enable_diloco=False,
      compiled_trainstep_file="",
      logical_axis_rules=(),
      input_data_sharding_logical_axes=(None, None),
      debug_sharding=False,
      lora=None,
      compile_topology="test",
      compile_topology_num_slices=1,
      compile_xla_flags="",
  )
  settings.update(overrides)
  return SimpleNamespace(**settings)


def test_shaped_inputs_build_nested_static_graph_on_execution_mesh():
  mesh = _mesh()
  execution_mesh = jaxpp.MpmdMesh(mesh, "stage").lowering_mesh()

  def create_model(config, model_mesh):
    del config
    factory = functools.partial(_NestedModel, model_mesh)
    return factory, nnx.eval_shape(factory)

  def abstract_state(config, physical_mesh, init_fn, is_training):
    del config, is_training
    state = nnx.state(nnx.eval_shape(init_fn))
    state_shardings = jax.tree.map(lambda _: jax.NamedSharding(physical_mesh, jax.P()), state)
    return state, None, state_shardings

  with (
      patch.object(train_compile.model_creation_utils, "create_nnx_abstract_model", side_effect=create_model) as factory,
      patch.object(train_compile.maxtext_utils, "get_abstract_state", side_effect=abstract_state) as get_state,
      patch.object(train_compile.maxtext_utils, "create_learning_rate_schedule", return_value=lambda step: 0.1),
      patch.object(train_compile.optimizers, "get_optimizer", return_value=optax.adam(0.1)) as optimizer,
      patch.object(
          train_compile.maxtext_utils,
          "get_shaped_batch",
          return_value={"inputs": jax.ShapeDtypeStruct((2, 2), jnp.float32)},
      ),
  ):
    args, kwargs, shardings, _, graphdef = train_compile.get_shaped_inputs(mesh, _config())

  assert factory.call_args.args[1] == execution_mesh
  assert optimizer.call_args.kwargs["mesh"] == execution_mesh
  assert get_state.call_args.args[1] == mesh
  assert not kwargs and len(args) == 2
  state = nnx.merge(graphdef, args[0])
  assert state.model.mesh == execution_mesh
  assert state.model.constrain.keywords["shardings"].mesh == execution_mesh
  assert all(s.mesh == execution_mesh for s in jax.tree.leaves(shardings))
  assert jax.tree.structure(args[0]) == jax.tree.structure(shardings)


def test_runtime_wrapper_preserves_nnx_variable_metadata_in_specs():
  mesh = _mesh()
  model = _NestedModel(jaxpp.MpmdMesh(mesh, "stage").lowering_mesh())
  state = TrainStateNNX(model, nnx.Optimizer(model, optax.adam(0.1), wrt=nnx.Param))
  graphdef, values = nnx.split(state)
  shardings = jax.tree.map(lambda _: jax.NamedSharding(mesh, jax.P()), values)
  params_shardings = shardings.model

  def step(model, config, state_shardings, params_shardings, state, data):
    del model, config, state_shardings, params_shardings, data
    return state, {}

  with patch.object(train_utils.jaxpp, "mpmd_jit_with_loop") as wrapper:
    result = train_utils.jit_train_step(
        _config(),
        graphdef,
        values,
        shardings,
        jax.NamedSharding(mesh, jax.P()),
        step,
        params_shardings,
        mesh,
    )
  assert result is wrapper.return_value
  call = wrapper.call_args
  assert call.kwargs["mpmd_mesh"].jax_mesh == mesh
  assert jax.tree.structure(call.kwargs["in_shardings"][0]) == jax.tree.structure(values)
  assert all(isinstance(s, jax.P) for s in jax.tree.leaves(call.kwargs["in_shardings"][0]))
  assert call.kwargs["in_shardings"][0].model.weight.get_metadata() == values.model.weight.get_metadata()
  bound = call.args[0]
  assert all(s.mesh == model.mesh for s in jax.tree.leaves(bound.args[2]))
  assert all(s.mesh == model.mesh for s in jax.tree.leaves(bound.args[3]))


def test_aot_wrapper_compiles_nnx_args_on_execution_mesh():
  mesh = _mesh()
  args = (jax.ShapeDtypeStruct((2,), jnp.float32),)
  in_shardings = (jax.NamedSharding(mesh, jax.P()),)
  with patch.object(train_compile.jaxpp, "mpmd_jit_with_loop") as wrapper:
    result = train_compile.jit_and_compile(
        lambda value: value,
        args,
        {},
        mesh,
        in_shardings,
        None,
        (),
        (),
        _config(),
        train_compile.axis_rules(()),
    )
  assert result is wrapper.return_value.compile.return_value
  wrapper.return_value.compile.assert_called_once_with(*args)
  assert wrapper.call_args.kwargs["mpmd_mesh"].jax_mesh == mesh
  assert wrapper.call_args.kwargs["in_shardings"] == (jax.P(),)


def test_jaxpp_serialized_executables_fail_before_compilation():
  config = _config(compiled_trainstep_file="serialized_step")
  with pytest.raises(NotImplementedError, match="Serializing"):
    train_compile.validate_config(config)
  with pytest.raises(NotImplementedError, match="Loading"):
    train_utils.jit_train_step(config, None, None, None, None, None, None, mesh=_mesh())


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"scan_layers": True}, "scan_layers=False"),
        ({"enable_checkpointing": True}, "does not support enable_checkpointing"),
        ({"enable_multi_tier_checkpointing": True}, "does not support enable_multi_tier_checkpointing"),
        ({"enable_emergency_checkpoint": True}, "does not support enable_emergency_checkpoint"),
        ({"num_vocab_tiling": 2}, "does not support num_vocab_tiling"),
        ({"lora": {"enable_lora": True, "lora_rank": 8}}, "does not support lora.enable_lora"),
        ({"elastic_enabled": True}, "does not support elastic_enabled"),
    ],
)
def test_unsupported_jaxpp_options_fail_during_config_validation(tmp_path, overrides, message):
  if jax.device_count() != 2:
    pytest.skip("Run with two CPU devices")
  config_path = Path(__file__).resolve().parents[2] / "src/maxtext/configs/decoupled_base_test.yml"
  options = dict(
      run_name="jaxpp_nnx_config_test",
      base_output_directory=str(tmp_path),
      enable_checkpointing=False,
      dataset_type="synthetic",
      hardware="cpu",
      use_jaxpp=True,
      scan_layers=False,
      base_num_decoder_layers=4,
      ici_pipeline_parallelism=2,
      ici_fsdp_parallelism=1,
      num_pipeline_repeats=1,
      num_pipeline_microbatches=2,
  )
  options.update(overrides)
  with pytest.raises(ValueError, match=message):
    pyconfig.initialize(["jaxpp_nnx_config_test", str(config_path)], **options)
