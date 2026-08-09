#!/usr/bin/env bash
# Run the approved Sprint 1.5 one-episode baseline smoke procedure.

set -uo pipefail

fail() {
  echo "smoke_error: $*" >&2
  echo "BASELINE_SMOKE_FAIL"
  exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXPECTED_CONDA_ENV="alfworld-self-improve"
BASELINE_CONFIG="${PROJECT_ROOT}/configs/collection/baseline.yaml"

cd "${PROJECT_ROOT}" || fail "cannot enter repository root"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${EXPECTED_CONDA_ENV}" ]]; then
  fail "activate Conda environment ${EXPECTED_CONDA_ENV} (current: ${CONDA_DEFAULT_ENV:-none})"
fi

PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)" \
  || fail "cannot determine Python version"
if [[ "${PYTHON_VERSION}" != "3.10" ]]; then
  fail "Python 3.10 is required (current: ${PYTHON_VERSION})"
fi
echo "python: $(python --version 2>&1)"

python - <<'PY' || fail "CUDA or BF16 preflight failed"
import torch

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("CUDA device does not report BF16 support")
print(f"cuda_device: {torch.cuda.get_device_name(0)}")
print(f"torch: {torch.__version__} (CUDA {torch.version.cuda})")
PY

if [[ -z "${ALFWORLD_CONFIG_PATH:-}" || ! -f "${ALFWORLD_CONFIG_PATH}" ]]; then
  fail "ALFWORLD_CONFIG_PATH must name an existing file"
fi
if [[ -z "${ALFWORLD_DATA:-}" || ! -d "${ALFWORLD_DATA}" ]]; then
  fail "ALFWORLD_DATA must name an existing directory"
fi
echo "alfworld_config: ${ALFWORLD_CONFIG_PATH}"
echo "alfworld_data: ${ALFWORLD_DATA}"

CONFIG_OUTPUT="$(python - "${BASELINE_CONFIG}" <<'PY'
import sys
from pathlib import Path

import yaml

project_root = Path.cwd()
collection_path = Path(sys.argv[1])
collection = yaml.safe_load(collection_path.read_text(encoding="utf-8"))
model_path = Path(collection["model_config"])
if not model_path.is_absolute():
    model_path = project_root / model_path
model = yaml.safe_load(model_path.read_text(encoding="utf-8"))
generation = model.get("generation", {})
if model.get("enable_thinking") is not False:
    raise SystemExit("enable_thinking must be false")
if generation.get("max_new_tokens") != 32:
    raise SystemExit("max_new_tokens must be 32")
output_dir = Path(collection["output_dir"])
if not output_dir.is_absolute():
    output_dir = project_root / output_dir
print(output_dir.resolve())
PY
)" || fail "portable baseline configuration preflight failed"
OUTPUT_DIR="${CONFIG_OUTPUT}"
echo "enable_thinking: false"
echo "max_new_tokens: 32"

GIT_REVISION="$(git rev-parse HEAD 2>/dev/null)" || fail "cannot resolve Git revision"
GIT_STATUS="$(git status --porcelain --untracked-files=all 2>/dev/null)" \
  || fail "cannot read Git status"
RELEVANT_STATUS="$(printf '%s\n' "${GIT_STATUS}" | grep -Ev '^\?\? results/baseline/sprint1-5-[^/]+/' || true)"
if [[ -n "${RELEVANT_STATUS}" ]]; then
  printf '%s\n' "${RELEVANT_STATUS}" >&2
  fail "research worktree has changes outside prior smoke outputs"
fi
if [[ -n "${GIT_STATUS}" ]]; then
  echo "git_status: clean except prior untracked smoke outputs"
else
  echo "git_status: clean"
fi
echo "git_revision: ${GIT_REVISION}"

HOST_LABEL="$(hostname -s 2>/dev/null | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:].-')"
if [[ -z "${HOST_LABEL}" ]]; then
  fail "cannot derive hostname for run name"
fi
UTC_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)" || fail "cannot create UTC timestamp"
RUN_NAME="sprint1-5-${HOST_LABEL}-${UTC_TIMESTAMP}"
RUN_DIRECTORY="${OUTPUT_DIR}/${RUN_NAME}"
if [[ -e "${RUN_DIRECTORY}" ]]; then
  fail "run directory already exists: ${RUN_DIRECTORY}"
fi

echo "run_name: ${RUN_NAME}"
if ! python scripts/collect_baseline.py \
  --config "${BASELINE_CONFIG}" \
  --episodes 1 \
  --run-name "${RUN_NAME}"; then
  fail "baseline runner failed; partial artifacts may remain in ${RUN_DIRECTORY}"
fi

if ! python scripts/validate_baseline_artifacts.py \
  "${RUN_DIRECTORY}" \
  --expected-git-revision "${GIT_REVISION}"; then
  exit 1
fi
