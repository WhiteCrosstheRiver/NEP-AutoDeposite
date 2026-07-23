#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
python3 setup.py build_ext --inplace
python3 -c "import fast_xyz; print('fast_xyz OK:', fast_xyz.write_model_xyz, fast_xyz.read_dump_xyz)"
