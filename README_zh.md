# Privileged Foresight Distillation: Zero-Cost Future Correction for World Action Models

**Privileged Foresight Distillation: Zero-Cost Future Correction for World Action Models** 的代码发布仓库。

[![arXiv](https://img.shields.io/badge/arXiv-2604.25859-b31b1b.svg)](https://arxiv.org/abs/2604.25859)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-f7c843)](https://huggingface.co/AmberJar/PFD)
[![Hugging Face Dataset - LIBERO](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset%20LIBERO-f7c843)](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)
[![Hugging Face Dataset - RoboTwin](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset%20RoboTwin-f7c843)](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam)

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[![中文](https://img.shields.io/badge/README-Chinese-d14836.svg)](./README_zh.md)

Pengcheng Fang, Hongli Chen, Xiaohao Cai

论文：[arXiv:2604.25859](https://arxiv.org/abs/2604.25859) | [PDF](https://arxiv.org/pdf/2604.25859)

PFD 的目标是在训练阶段利用带有未来信息的 privileged signal，蒸馏出一个部署时不需要未来视频生成的 action correction 模块。推理时，策略只使用当前观测历史，不需要 test-time future imagination。

本仓库包含 LIBERO / RoboTwin 上的训练与推理评测代码。已发布的 LIBERO checkpoint 放在 [Hugging Face](https://huggingface.co/AmberJar/PFD)，benchmark 数据沿用上方链接中的 FastWAM 预处理 LIBERO / RoboTwin 数据集。大文件 artifact、日志、运行输出和内部实验记录不会放入 GitHub。

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

推荐按下面的已验证环境安装：

```bash
conda create -n pfd python=3.10 -y
conda activate pfd
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

如果本地 CUDA 版本不同，请先安装匹配的 PyTorch / torchvision wheel，再安装本仓库。LIBERO 和 RoboTwin 还需要各自的 simulator 环境。LIBERO 评测请先按官方 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 仓库完成安装，并保持 MuJoCo 与数据版本一致：

```bash
pip install mujoco==3.3.2
```

RoboTwin 评测请按官方 [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) 仓库完成环境和 assets 安装。

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

## 数据集下载

PFD 使用 FastWAM 预处理后的 benchmark 数据集。

### LIBERO

从 Hugging Face 下载 LIBERO 压缩包：

- https://huggingface.co/datasets/yuanty/LIBERO-fastwam

然后解压到 `data/libero_mujoco3.3.2`：

```bash
mkdir -p data/libero_mujoco3.3.2
cd data/libero_mujoco3.3.2

for f in *.tar.gz; do
  tar -xzf "$f"
done
```

### RoboTwin

从 Hugging Face 下载 RoboTwin 分片压缩包：

- https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam

然后拼接并解压：

```bash
mkdir -p data/robotwin2.0
cd data/robotwin2.0

cat robotwin2.0.tar.gz.part-* | tar -xzf -
```

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

## 使用已发布 Checkpoint 推理

PFD 的 LIBERO checkpoint 已发布在 [Hugging Face](https://huggingface.co/AmberJar/PFD)。该 checkpoint 对应 `fastwam_pfd_action512_partial` 配置，并训练了最后 12 层 action layers 与最后 12 层 video layers。

```bash
pip install -U huggingface_hub

huggingface-cli download AmberJar/PFD \
  libero_pfd_action512_partial_12x12_step62000.pt \
  dataset_stats.json \
  config.yaml \
  manifest.json \
  --local-dir ./checkpoints/pfd_release
```

下载后的本地结构应为：

```text
checkpoints/pfd_release/
├── libero_pfd_action512_partial_12x12_step62000.pt
├── dataset_stats.json
├── config.yaml
└── manifest.json
```

评测已发布的 LIBERO checkpoint：

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  model=fastwam_pfd_action512_partial \
  ckpt=./checkpoints/pfd_release/libero_pfd_action512_partial_12x12_step62000.pt \
  EVALUATION.dataset_stats_path=./checkpoints/pfd_release/dataset_stats.json \
  MULTIRUN.num_gpus=8
```

该 checkpoint 的 LIBERO 全套评测结果为 `1962/2000 = 98.10%`。

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
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
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
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
  MULTIRUN.num_gpus=8
```

## Checkpoint 说明

本 GitHub 仓库不直接存放大 checkpoint 文件。已发布权重托管在 [Hugging Face](https://huggingface.co/AmberJar/PFD)。本地下载或新训练的权重可以放到 `./checkpoints`，也可以通过 Hydra 命令行传入绝对路径。评测时请通过 `EVALUATION.dataset_stats_path` 传入匹配的 `dataset_stats.json`，或把它放在 checkpoint 同级目录。

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

如果这个仓库对你的研究有帮助，请引用：

```bibtex
@article{fang2026pfd,
  title={Privileged Foresight Distillation: Zero-Cost Future Correction for World Action Models},
  author={Fang, Pengcheng and Chen, Hongli and Cai, Xiaohao},
  journal={arXiv preprint arXiv:2604.25859},
  year={2026}
}
```

本实现基于 FastWAM 的训练和评测框架。如果使用了这部分代码栈，也请引用 FastWAM：

```bibtex
@article{yuan2026fastwam,
  title={Fast-WAM: Do World Action Models Need Test-time Future Imagination?},
  author={Tianyuan Yuan and Zibin Dong and Yicheng Liu and Hang Zhao},
  journal={arXiv preprint arXiv:2603.16666},
  year={2026},
  url={https://arxiv.org/abs/2603.16666}
}
```
