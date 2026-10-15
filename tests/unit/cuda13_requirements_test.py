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

"""Offline checks for the CUDA12/CUDA13 dependency relationship."""

from pathlib import Path
import re

from packaging.requirements import Requirement
import pytest

from dependencies.scripts import generate_cuda13_requirements as generator


def _requirements(text):
  return {
      requirement.name: requirement
      for line in text.splitlines()
      if line and not line.startswith("#")
      for requirement in [Requirement(line)]
  }


def test_generated_cuda13_snapshot_is_current_and_shared_dependencies_match():
  source = generator.SOURCE_FILE.read_text()
  actual = generator.OUTPUT_FILE.read_text()
  assert actual == generator.render_cuda13_requirements(source)
  cuda12, cuda13 = _requirements(source), _requirements(actual)
  for name, requirement in cuda12.items():
    if name not in generator.RENAMED_PACKAGES and name not in generator.REMOVED_PACKAGES:
      assert str(cuda13[name]) == str(requirement), name
  assert str(cuda13["pathwaysutils"].specifier) == ">=0.1.11"
  assert str(cuda13["flax"].specifier) == ">=0.12.9"
  assert str(cuda13["tokamax"].specifier) == ">=0.0.14"
  assert "tensorflow" not in cuda13 and "seqio" not in cuda13


def test_cuda13_packages_keep_correct_names_versions_and_platform_markers():
  requirements = _requirements(generator.render_cuda13_requirements(generator.SOURCE_FILE.read_text()))
  assert str(requirements["jax-cuda13-plugin"].specifier) == ">=0.11.1"
  assert str(requirements["transformer-engine-cu13"].specifier) == ">=2.19.0"
  assert str(requirements["nvidia-cudnn-cu13"].specifier) == ">=9.24.0.43"
  assert str(requirements["nvidia-cufft"].specifier) == ">=12.0.0.15"
  for name in (*generator.RENAMED_PACKAGES.values(), "nvidia-cuda-crt", "nvidia-nvvm"):
    if not name.startswith("transformer-engine"):
      assert str(requirements[name].marker) == 'sys_platform == "linux"'
  assert "nvidia-cutlass-dsl-libs-cu12" in requirements
  assert "nvidia-cutlass-dsl-libs-cu13" in requirements
  assert "jax-cuda12-plugin" not in requirements
  assert "transformer-engine-cu12" not in requirements


def test_docker_te_bootstrap_does_not_downgrade_aligned_requirements():
  dockerfile = (Path(__file__).resolve().parents[2] / "jaxpp.Dockerfile").read_text()
  pinned = re.search(r"transformer-engine\[core-cu13,jax\]==([0-9.]+)", dockerfile)
  assert pinned is not None
  requirements = _requirements(generator.OUTPUT_FILE.read_text())
  for name in ("transformer-engine", "transformer-engine-cu13", "transformer-engine-jax"):
    assert requirements[name].specifier.contains(pinned.group(1)), name
  assert "--no-build-isolation-package transformer-engine-jax" in dockerfile
  assert "from pathwaysutils.elastic import elastic, manager" in dockerfile


@pytest.mark.parametrize("source", ["-r requirements.txt", "flax", "nvidia-new-library-cu12>=1.0", "flax>=1\nflax>=2"])
def test_invalid_or_unmapped_requirements_fail_explicitly(source):
  with pytest.raises(ValueError):
    generator.render_cuda13_requirements(source)
