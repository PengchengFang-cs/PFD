# Privileged Foresight Distillation: Zero-Cost Future Correction for World Action Models

Code release for **Privileged Foresight Distillation: Zero-Cost Future Correction for World Action Models**.

[![arXiv](https://img.shields.io/badge/arXiv-2604.25859-b31b1b.svg)](https://arxiv.org/abs/2604.25859)

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[![Chinese](https://img.shields.io/badge/README-Chinese-d14836.svg)](./README_zh.md)

Pengcheng Fang, Hongli Chen, Xiaohao Cai

Paper: [arXiv:2604.25859](https://arxiv.org/abs/2604.25859) | [PDF](https://arxiv.org/pdf/2604.25859)

PFD improves world-action models by distilling privileged future-conditioned training signals into a future-free action correction module. During deployment, the policy only uses the current observation history and does not require test-time future video generation.

This repository contains the training and evaluation code for PFD on LIBERO and RoboTwin. Checkpoints, datasets, logs, run outputs, and internal experiment notes are intentionally not included in this GitHub repository.

## Contents

- [Repository Layout](#repository-layout)
- [Environment](#environment)
- [Model Preparation](#model-preparation)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Checkpoints](#checkpoints)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Repository Layout

```text
PFD-public/
├── configs/
│   ├── data/                 # Dataset configs for LIBERO and RoboTwin
│   ├── model/                # FastWAM and PFD model configs
│   └── task/                 # Hydra task configs
├── scripts/
│   ├── train.py
│   ├── train_zero1.sh        # DeepSpeed ZeRO-1 training launcher
│   ├── preprocess_action_dit_backbone.py
│   └── precompute_text_embeds.py
├── experiments/
│   ├── libero/               # LIBERO evaluation manager and utilities
│   └── robotwin/             # RoboTwin evaluation manager and policy wrapper
├── src/fastwam/              # Core model, dataset, runtime, and trainer code
└── third_party/RoboTwin/     # Adapted RoboTwin evaluation code
```

The Python package is still named `fastwam` for compatibility with the original code paths and configs.

## Environment

```bash
conda create -n pfd python=3.10 -y
conda activate pfd
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

LIBERO and RoboTwin require their own simulator dependencies. Install those environments from their official repositories before running benchmark evaluation.

## Model Preparation

Set the Wan/DiffSynth model root. The default configs expect external model files under `./checkpoints`.

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

Preprocess the ActionDiT backbone before training:

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam_pfd_action512_partial.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

The generated backbone is a local artifact and is ignored by Git.

## Datasets

Place datasets under `./data` using the paths expected by the configs:

```text
data/
├── libero_mujoco3.3.2/
│   ├── libero_10_no_noops_lerobot/
│   ├── libero_goal_no_noops_lerobot/
│   ├── libero_object_no_noops_lerobot/
│   └── libero_spatial_no_noops_lerobot/
└── robotwin2.0/
    ├── dataset_stats.json
    └── robotwin2.0/
        ├── data/
        ├── meta/
        └── videos/
```

Dataset files are not tracked by this repository.

## Training

Precompute T5 text embeddings:

```bash
python scripts/precompute_text_embeds.py \
  task=libero_uncond_2cam224_1e-4 \
  model=fastwam_pfd_action512_partial
```

Train PFD on LIBERO:

```bash
bash scripts/train_zero1.sh 8 \
  task=libero_uncond_2cam224_1e-4 \
  model=fastwam_pfd_action512_partial
```

Train PFD on RoboTwin:

```bash
bash scripts/train_zero1.sh 8 \
  task=robotwin_uncond_3cam_384_1e-4 \
  model=fastwam_pfd_action512_partial
```

To initialize from a base world-action-model checkpoint, pass:

```bash
init_checkpoint=/path/to/base_checkpoint.pt
```

PFD lightweight training-state checkpoints are saved under the run directory when PFD is enabled. These outputs are ignored by Git.

## Evaluation

For LIBERO, install the official LIBERO environment first, then run:

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  model=fastwam_pfd_action512_partial \
  ckpt=/path/to/pfd_checkpoint.pt \
  MULTIRUN.num_gpus=8
```

For RoboTwin, follow the official RoboTwin setup instructions, download required assets, and create the policy symlink:

```bash
ln -sfn "$(pwd)/experiments/robotwin/fastwam_policy" "$(pwd)/third_party/RoboTwin/policy/fastwam_policy"
```

Then run:

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  model=fastwam_pfd_action512_partial \
  ckpt=/path/to/pfd_checkpoint.pt \
  MULTIRUN.num_gpus=8
```

## Checkpoints

This code-only GitHub repository does not include model checkpoints. Put local or released weights under `./checkpoints` or pass absolute checkpoint paths through the Hydra command line.

Ignored artifact classes include:

- `checkpoints/`
- `data/`
- `runs/`
- `logs/`
- `evaluate_results/`
- `archive/`
- `idea-stage/`, `refine-logs/`, `review-stage/`
- `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.bin`

## Acknowledgements

This codebase builds on the FastWAM training and evaluation stack and includes adapted RoboTwin evaluation code. We thank the Wan, LIBERO, RoboTwin, LeRobot, and DiffSynth communities for their open-source infrastructure.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{fang2026pfd,
  title={Privileged Foresight Distillation: Zero-Cost Future Correction for World Action Models},
  author={Fang, Pengcheng and Chen, Hongli and Cai, Xiaohao},
  journal={arXiv preprint arXiv:2604.25859},
  year={2026}
}
```
