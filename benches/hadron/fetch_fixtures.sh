#!/usr/bin/env bash
# Copyright 2026 Algorithmiq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fetches the external fixtures for benches/hadron into .cache/ (gitignored). Re-run any time;
# it skips a clone whose target directory already exists.
set -euo pipefail

cache_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.cache"
mkdir -p "$cache_dir"
cd "$cache_dir"

if [ ! -d qat ]; then
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io.git qat
  (cd qat && git sparse-checkout set \
    data/observable-estimations/circuit-models/su2_hadron_dynamics_lsh)
else
  echo "qat/ already present, skipping"
fi

if [ ! -d lsh_data ]; then
  git clone --depth 1 https://github.com/mathew0036/lsh_data.git
else
  echo "lsh_data/ already present, skipping"
fi

echo "Fixtures ready under $cache_dir"
