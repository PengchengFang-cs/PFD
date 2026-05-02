# PFD

**Privileged Foresight Distillation: Future-Free Correction for World Action Models** 的代码发布仓库。

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[![中文](https://img.shields.io/badge/README-Chinese-d14836.svg)](./README_zh.md)

PFD 的目标是在训练阶段利用带有未来信息的 privileged signal，蒸馏出一个部署时不需要未来视频生成的 action correction 模块。推理时，策略只使用当前观测历史，不需要 test-time future imagination。

本仓库只包含 LIBERO / RoboTwin 上的训练与推理评测代码。checkpoint、数据集、日志、运行输出和内部实验记录不会放入 GitHub。

## 目录结构

```text
PFD-public/
├── configs/
│   ├── data/                 # LIBERO / RoboTwin 数据配置
│   ├── model/                # FastWAM 与 PFD 模型配置
│   └── task/                 # Hydra task 配置
├── scripts/
│   ├── train.py
│   ├── train_zero1.sh
│   ├── preprocess_action_dit_backbone.py
│   └── precompute_text_embeds.py
├── experiments/
│   ├── libero/               # LIBERO 评测入口
│   └── robotwin/             # RoboTwin 评测入口和 policy wrapper
├── src/fastwam/              # 核心模型、数据、runtime 和 trainer
└── third_party/RoboTwin/     # 适配后的 RoboTwin 评测代码
```

为了兼容已有配置和导入路径，Python package 仍然叫 `fastwam`。

## 环境安装

```bash
conda create -n pfd python=3.10 -y
conda activate pfd
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

LIBERO 和 RoboTwin 还需要各自的 simulator 环境。跑 benchmark 前请先按官方仓库完成安装。

## 模型准备

默认配置会从 `./checkpoints` 读取外部 Wan/DiffSynth 模型文件：

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

训练前预处理 ActionDiT backbone：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam_pfd_action512_partial.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

生成的 backbone 属于本地 artifact，不会进入 Git。

## 数据集

配置默认使用下面的相对路径：

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

数据文件不会放进本仓库。

## 训练

预计算文本 embedding：

```bash
python scripts/precompute_text_embeds.py \
  task=libero_uncond_2cam224_1e-4 \
  model=fastwam_pfd_action512_partial
```

LIBERO 上训练 PFD：

```bash
bash scripts/train_zero1.sh 8 \
  task=libero_uncond_2cam224_1e-4 \
  model=fastwam_pfd_action512_partial
```

RoboTwin 上训练 PFD：

```bash
bash scripts/train_zero1.sh 8 \
  task=robotwin_uncond_3cam_384_1e-4 \
  model=fastwam_pfd_action512_partial
```

如果需要从基础 world-action-model checkpoint 初始化：

```bash
init_checkpoint=/path/to/base_checkpoint.pt
```

PFD 开启时会在 run 目录下保存 lightweight training-state checkpoint。这些输出默认被 `.gitignore` 排除。

## 推理评测

LIBERO：

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  model=fastwam_pfd_action512_partial \
  ckpt=/path/to/pfd_checkpoint.pt \
  MULTIRUN.num_gpus=8
```

RoboTwin 需要先按官方说明完成环境和 assets 安装，然后创建 policy 软链接：

```bash
ln -sfn "$(pwd)/experiments/robotwin/fastwam_policy" "$(pwd)/third_party/RoboTwin/policy/fastwam_policy"
```

再运行：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  model=fastwam_pfd_action512_partial \
  ckpt=/path/to/pfd_checkpoint.pt \
  MULTIRUN.num_gpus=8
```

## Checkpoint 说明

本 GitHub 仓库暂时不包含 checkpoint。请把本地或后续发布的权重放到 `./checkpoints`，或通过命令行传入绝对路径。

以下内容不会进入 Git：

- `checkpoints/`
- `data/`
- `runs/`
- `logs/`
- `evaluate_results/`
- `archive/`
- `idea-stage/`, `refine-logs/`, `review-stage/`
- `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.bin`

## 致谢

本代码基于 FastWAM 的训练和评测框架，并包含适配后的 RoboTwin 评测代码。感谢 Wan、LIBERO、RoboTwin、LeRobot 和 DiffSynth 等开源项目。

## 引用

论文发布后会补充引用信息。
