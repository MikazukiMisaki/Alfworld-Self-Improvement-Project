# University A40 Server Setup

## Scope

This guide migrates the Sprint 1.5 runtime to the university A40 server. It
does not change the baseline prompt, parser, collector, recovery behavior, or
selector logic. Commands in this document are instructions only and have not
been executed by the repository migration work.

Server facts were captured in `Server_Information.md` on 2026-08-09:

- home: `/home/lizening`;
- GPU: NVIDIA A40, 46,068 MiB;
- driver: 550.163.01;
- reported CUDA compatibility: 12.4;
- Conda: 26.3.2;
- base Python: 3.13.13;
- persistent `/home` free space at capture time: approximately 80 GiB.

The recorded GPU was already at 100% utilization with about 27 GiB allocated.
Do not start the Qwen smoke test until the allocated GPU has sufficient free
memory.

## Configuration strategy

Portable experiment configuration remains in:

```text
configs/environment/alfworld_text.yaml
configs/model/qwen3_8b.yaml
configs/collection/baseline.yaml
```

Host-specific paths and runtime preferences belong only in ignored files under
`configs/local/`. The university host inventory is:

```text
configs/local/university_server.yaml
```

The current collection entry point does not merge that inventory YAML into the
portable experiment configuration. Until configuration composition is added in
a separately reviewed Sprint 1.5 change, export the corresponding environment
variables in the shell. This avoids embedding server paths in research code or
portable YAML.

The upstream ALFWorld `base_config.yaml` is also machine-local because it
resolves data through `ALFWORLD_DATA`. Store its university copy under
`configs/local/` and point `ALFWORLD_CONFIG_PATH` to it.

## Recommended Python and dependencies

Use a dedicated Conda environment with Python 3.10. Do not use the server's
base Python 3.13 environment: ALFWorld officially recommends Python 3.9 and
supports Python 3.9+, while older TextWorld dependencies are more reliably
supported by Python 3.10 than Python 3.13.

The initial reproducible stack is:

- Python 3.10;
- PyTorch 2.6.0 CUDA 12.4 wheel;
- ALFWorld 0.4.2, text-only;
- Transformers 4.56.2 (`>=4.51.0` is required for Qwen3);
- Accelerate 1.10.1 for device placement;
- NumPy 1.26.4, PyYAML 6.0.2, and pytest 8.4.2.

After the first successful environment and model smoke tests, freeze the exact
resolved environment before collecting baseline evidence.

## Exact setup commands

### 1. Clone or update the project

For a new checkout:

```bash
cd /home/lizening
git clone https://github.com/MikazukiMisaki/Alfworld-Self-Improvement-Project.git
cd /home/lizening/Alfworld-Self-Improvement-Project
git status --short --branch
```

For an existing clean checkout:

```bash
cd /home/lizening/Alfworld-Self-Improvement-Project
git status --short --branch
git pull --ff-only
```

Do not pull over an uncommitted server worktree.

### 2. Create the isolated Python environment

```bash
conda create --name alfworld-sprint15 python=3.10 pip=25.1 -y
conda activate alfworld-sprint15
python --version
python -m pip install --upgrade setuptools wheel
```

### 3. Install the CUDA and Python dependencies

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install alfworld==0.4.2 transformers==4.56.2 accelerate==1.10.1 numpy==1.26.4 PyYAML==6.0.2 pytest==8.4.2
python -m pip install --editable '.[baseline,dev]'
```

This project uses ALFWorld's text environment, so the visual/THOR extras and an
X server are not required.

### 4. Create cache and dataset directories

```bash
mkdir -p /home/lizening/hf_cache
mkdir -p /home/lizening/alfworld_data
mkdir -p /home/lizening/Alfworld-Self-Improvement-Project/configs/local
```

### 5. Export the host-local runtime paths

Run these exports in every experiment shell, or place them in a private shell
activation script outside Git:

```bash
export HF_HOME=/home/lizening/hf_cache
export HF_HUB_CACHE=/home/lizening/hf_cache/hub
export ALFWORLD_DATA=/home/lizening/alfworld_data
export ALFWORLD_CONFIG_PATH=/home/lizening/Alfworld-Self-Improvement-Project/configs/local/alfworld_base_university.yaml
export TOKENIZERS_PARALLELISM=false
```

Do not store access tokens in YAML or committed shell scripts. If Hugging Face
authentication is ever required, use the Hugging Face CLI credential store or
an injected environment variable.

### 6. Download the pinned ALFWorld configuration and data

```bash
curl --fail --location https://raw.githubusercontent.com/alfworld/alfworld/0.4.2/configs/base_config.yaml --output /home/lizening/Alfworld-Self-Improvement-Project/configs/local/alfworld_base_university.yaml
ALFWORLD_DATA=/home/lizening/alfworld_data alfworld-download
```

The pinned upstream config uses `$ALFWORLD_DATA` rather than old `/home_lab/`
paths. Verify the required text-environment files:

```bash
test -f /home/lizening/alfworld_data/logic/alfred.pddl
test -f /home/lizening/alfworld_data/logic/alfred.twl2
test -d /home/lizening/alfworld_data/json_2.1.1/train
test -d /home/lizening/alfworld_data/json_2.1.1/valid_seen
test -d /home/lizening/alfworld_data/json_2.1.1/valid_unseen
```

### 7. Verify software and GPU visibility

```bash
conda activate alfworld-sprint15
cd /home/lizening/Alfworld-Self-Improvement-Project
nvidia-smi
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0)); print('bf16', torch.cuda.is_bf16_supported())"
python -c "from importlib.metadata import version; print('alfworld', version('alfworld')); print('transformers', version('transformers')); print('accelerate', version('accelerate')); print('PyYAML', version('PyYAML'))"
```

All CUDA checks, including BF16 support, must succeed before loading Qwen3-8B.

### 8. Run repository tests

```bash
conda activate alfworld-sprint15
cd /home/lizening/Alfworld-Self-Improvement-Project
PYTHONPATH=src python -m unittest discover -s tests -v
```

### 9. Cache the Qwen3-8B checkpoint

```bash
conda activate alfworld-sprint15
export HF_HOME=/home/lizening/hf_cache
export HF_HUB_CACHE=/home/lizening/hf_cache/hub
python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='Qwen/Qwen3-8B'))"
```

Record the resolved snapshot commit printed in the cache path. The portable
model configuration keeps `enable_thinking: false`, greedy decoding, BF16, and
a 32-token output cap unchanged.

### 10. Freeze the resolved environment

```bash
conda activate alfworld-sprint15
cd /home/lizening/Alfworld-Self-Improvement-Project
conda env export --no-builds > configs/local/university_server.conda.yaml
python -m pip freeze > configs/local/university_server.requirements.txt
```

These files remain machine-local and ignored. Copy their contents into run
provenance only through a reviewed, portable manifest mechanism.

### 11. Baseline smoke command — run only after validation review

```bash
conda activate alfworld-sprint15
cd /home/lizening/Alfworld-Self-Improvement-Project
export HF_HOME=/home/lizening/hf_cache
export HF_HUB_CACHE=/home/lizening/hf_cache/hub
export ALFWORLD_DATA=/home/lizening/alfworld_data
export ALFWORLD_CONFIG_PATH=/home/lizening/Alfworld-Self-Improvement-Project/configs/local/alfworld_base_university.yaml
python scripts/collect_baseline.py --episodes 1 --run-name sprint1-5-university-a40-smoke
```

This command is documented but should not be run until the Sprint 1.5 artifact
and termination-semantics review is complete.

## Machine assumptions that changed

| Assumption | Previous server | University server action |
|---|---|---|
| Home root | `/home_lab/lizening` | `/home/lizening` |
| Project path | several `/home_lab/.../GitHub/...` variants | `/home/lizening/Alfworld-Self-Improvement-Project` |
| Hugging Face cache | old default or `/home_lab` cache | `HF_HOME=/home/lizening/hf_cache` |
| ALFWorld data | `/home_lab/lizening/alfworld_data` or `.cache/alfworld` | `/home/lizening/alfworld_data` |
| ALFWorld config | old A40-local absolute-path YAML | pinned 0.4.2 config using `$ALFWORLD_DATA` |
| Python | historical Python 3.9 artifacts | isolated Python 3.10; never base Python 3.13 |
| CUDA/GPU | old lab A40 installation | A40, driver 550.163.01, CUDA 12.4 wheel |
| Runtime selection | `device: auto` | local inventory requests `cuda`; current portable loader still uses `auto` and must verify CUDA resolution |
| Storage | old lab filesystem | persistent `/home` has an 80 GiB free-space snapshot; monitor model/data usage |

## Repository path audit

Old `/home_lab/` paths remain in historical `results/baseline` manifests and
trajectories. Those artifacts are immutable evidence and must not be rewritten;
they are not valid configuration inputs for new runs.

The old `configs/local/alfworld_base_a40.yaml` is already tracked in Git from an
earlier commit. Adding `configs/local/` to `.gitignore` does not untrack an
already tracked file. After preserving any needed historical provenance, use
this one-time command in a dedicated cleanup change:

```bash
git rm --cached configs/local/alfworld_base_a40.yaml
```

Do not execute that cleanup as part of server installation, and do not delete
historical result artifacts. There were no old Conda environment names or old
Hugging Face cache variables elsewhere in active runtime code; the dependency
environment was simply undocumented and unpinned.

## Git hygiene check

Before any commit, run:

```bash
git check-ignore -v configs/local/university_server.yaml
git check-ignore -v configs/local/alfworld_base_university.yaml
git status --short
git grep -n '/home/lizening' -- ':!Server_Information.md' ':!SERVER_SETUP.md'
git grep -n '/home_lab/'
```

New university paths may appear in ignored local files and this setup guide,
but never in research code, portable experiment configs, or committed runtime
artifacts outside explicit run provenance.

## Sprint 1.5 B0/B1 controlled pilot

The two collection configs differ only in action-selection mode. Both use the
same `valid_seen` environment, seeds 42--44, maximum horizon, model, decoding
settings, and environment-provided action order. This is a small diagnostic,
not a Sprint 2 experiment.

```bash
conda activate alfworld-self-improve
cd /home/lizening/Alfworld-Self-Improvement-Project
source configs/local/activate_university_server.sh

PAIR_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

python scripts/collect_baseline.py \
  --config configs/collection/baseline.yaml \
  --episodes 3 \
  --run-name "sprint1-5-b0-${PAIR_TIMESTAMP}"

python scripts/collect_baseline.py \
  --config configs/collection/baseline_indexed.yaml \
  --episodes 3 \
  --run-name "sprint1-5-b1-${PAIR_TIMESTAMP}"
```

Run the default one-episode B1 smoke test with:

```bash
bash scripts/smoke_baseline.sh
```

Run the same wrapper in B0 mode when a free-form diagnostic is needed:

```bash
BASELINE_CONFIG=configs/collection/baseline.yaml bash scripts/smoke_baseline.sh
```
