<div align="center">

# 🎬 OSP-Next

**Open-Sora-Plan · Next Generation**

*A scalable text-to-video diffusion trainer & runner built on PyTorch 2.8, FSDP2,
Ulysses + Skiparse sequence parallelism — supporting both CUDA GPUs and Ascend NPUs,
with optional HIF8 quantization and a GRPO + LoRA reinforcement-learning post-training pipeline.*

<p>
  <a href="<ARXIV_URL>">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-OSP--Next-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white">
  </a>
  <a href="<HUGGINGFACE_URL>">
    <img alt="HuggingFace" src="https://img.shields.io/badge/🤗_HuggingFace-OSP--Next-FFD21E?style=for-the-badge">
  </a>
  <a href="<MODELSCOPE_URL>">
    <img alt="ModelScope" src="https://img.shields.io/badge/ModelScope-OSP--Next-624AFF?style=for-the-badge">
  </a>
  <a href="LICENSE.txt">
    <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-22B14C?style=for-the-badge">
  </a>
  <a href="https://github.com/<OWNER>/<REPO>/stargazers">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/<OWNER>/<REPO>?style=for-the-badge&logo=github&color=FFD43B">
  </a>
</p>

</div>

---

## 🎞️ Demo Gallery

A side-by-side comparison of the same prompt across three models. Click a cell to
play the video.

<!-- TODO: replace the placeholder paths once the demo mp4 / gif assets land. -->

<table>
  <thead>
    <tr>
      <th align="center">Prompt</th>
      <th align="center">Wan 2.1</th>
      <th align="center">OSP-Next</th>
      <th align="center">OSP-Next-HiF8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><sub><i>"A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage..."</i></sub></td>
      <td align="center"><a href="<DEMO_WAN21_1>"><img src="<DEMO_WAN21_1_THUMB>" width="240"></a></td>
      <td align="center"><a href="<DEMO_OSP_1>"><img src="<DEMO_OSP_1_THUMB>" width="240"></a></td>
      <td align="center"><a href="<DEMO_OSP_HIF8_1>"><img src="<DEMO_OSP_HIF8_1_THUMB>" width="240"></a></td>
    </tr>
    <tr>
      <td><sub><i>"Several giant wooly mammoths approach treading through a snowy meadow..."</i></sub></td>
      <td align="center"><a href="<DEMO_WAN21_2>"><img src="<DEMO_WAN21_2_THUMB>" width="240"></a></td>
      <td align="center"><a href="<DEMO_OSP_2>"><img src="<DEMO_OSP_2_THUMB>" width="240"></a></td>
      <td align="center"><a href="<DEMO_OSP_HIF8_2>"><img src="<DEMO_OSP_HIF8_2_THUMB>" width="240"></a></td>
    </tr>
    <tr>
      <td><sub><i>"Drone view of waves crashing against the rugged cliffs along Big Sur's garay point beach..."</i></sub></td>
      <td align="center"><a href="<DEMO_WAN21_3>"><img src="<DEMO_WAN21_3_THUMB>" width="240"></a></td>
      <td align="center"><a href="<DEMO_OSP_3>"><img src="<DEMO_OSP_3_THUMB>" width="240"></a></td>
      <td align="center"><a href="<DEMO_OSP_HIF8_3>"><img src="<DEMO_OSP_HIF8_3_THUMB>" width="240"></a></td>
    </tr>
  </tbody>
</table>

> 💡 **HiF8 takeaway** — OSP-Next-HiF8 matches the visual quality of the bf16
> baseline while running roughly **1.6× faster** and using **~50% less HBM** on
> Ascend NPUs.

---

## 📦 Model Downloads

| Model            | Params | 🤗 HuggingFace                       | <img src="https://modelscope.cn/favicon.ico" height="14"> ModelScope |
|------------------|-------:|--------------------------------------|----------------------------------------------------------------------|
| OSP-Next 1.3B    | 1.3B   | [`<HF_OSP_NEXT_1_3B>`](<HF_URL_1_3B>)     | [`<MS_OSP_NEXT_1_3B>`](<MS_URL_1_3B>)         |
| OSP-Next 14B     |  14B   | [`<HF_OSP_NEXT_14B>`](<HF_URL_14B>)       | [`<MS_OSP_NEXT_14B>`](<MS_URL_14B>)           |
| OSP-Next-HiF8 1.3B | 1.3B | [`<HF_HIF8_1_3B>`](<HF_URL_HIF8_1_3B>)    | [`<MS_HIF8_1_3B>`](<MS_URL_HIF8_1_3B>)        |
| OSP-Next-HiF8 14B  |  14B | [`<HF_HIF8_14B>`](<HF_URL_HIF8_14B>)      | [`<MS_HIF8_14B>`](<MS_URL_HIF8_14B>)          |
| T5 Text Encoder  |  ~5B   | [`<HF_T5>`](<HF_URL_T5>)                  | [`<MS_T5>`](<MS_URL_T5>)                      |
| WAN VAE          |  ~0.4B | [`<HF_VAE>`](<HF_URL_VAE>)                | [`<MS_VAE>`](<MS_URL_VAE>)                    |

After downloading, point the config fields below to your local copies:

```yaml
model_config:
  pretrained_model_dir_or_checkpoint: "/path/to/model"     # ← OSP-Next ckpt
vae_config:
  vae_path: "/path/to/vae"                                  # ← WAN VAE
text_encoder_config:
  checkpoint_path: "/path/to/text_encoder"                  # ← T5 weights
  text_tokenizer_path: "/path/to/text_tokenizer"            # ← T5 tokenizer
```

---

## 📑 Table of Contents

- [🎞️ Demo Gallery](#️-demo-gallery)
- [📦 Model Downloads](#-model-downloads)
- [🧱 Project Layout](#-project-layout)
- [⚙️ Environment Setup](#️-environment-setup)
  - [GPU (CUDA)](#-gpu-cuda)
  - [NPU (Ascend)](#-npu-ascend)
  - [⚡ Flash-Attention (optional, GPU only)](#-flash-attention-optional-gpu-only)
  - [🔬 HIF8 NPU Quant Kernel (optional, NPU only)](#-hif8-npu-quant-kernel-optional-npu-only)
- [🎥 Inference Pipeline](#-inference-pipeline)
- [🏋️ Training Pipeline](#️-training-pipeline)
  - [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
  - [Reinforcement Learning (GRPO + LoRA)](#reinforcement-learning-grpo--lora)
- [🛠️ Tips & Troubleshooting](#️-tips--troubleshooting)
- [📝 Citation](#-citation)
- [📄 License](#-license)
- [⭐ Star History](#-star-history)

---

## 🧱 Project Layout

```
OSP-Next/
├── configs/                     # All YAML configs
│   ├── infer/{gpu,npu}/         # Inference configs (per backend)
│   └── train/{gpu,npu}/         # Training configs (per backend)
├── scripts/                     # Launch scripts (torchrun)
│   ├── infer/{gpu,npu}/         # Inference launchers
│   └── train/{gpu,npu}/         # Training launchers
├── ospnext/                     # Core library
│   ├── modules/                 # Diffusion / VAE / T5 / attention / HIF8
│   ├── distributed/             # FSDP2 + sequence-parallel state & comm
│   ├── data/                    # Datasets, samplers, collators
│   ├── pipelines/               # End-to-end inference pipelines
│   ├── rewards/                 # VideoAlign reward (for RL)
│   ├── schedulers/              # Flow matching scheduler
│   └── quant_cy_npu/            # HIF8 quant op (NPU custom kernel)
├── train/
│   ├── train_osp.py             # Entry: SFT training
│   └── train_osp_RL.py          # Entry: GRPO + LoRA RL training
├── infer/
│   └── infer_osp.py             # Entry: text-to-video inference
├── assets/t2v/                  # Sample prompt files
├── requirements.txt             # GPU pip requirements
├── requirements_npu.txt         # NPU pip requirements
└── pyproject.toml               # Editable install metadata
```

---

## ⚙️ Environment Setup

We strongly recommend using **conda + editable install** so that every entry
point (`train/`, `infer/`, custom scripts) sees the `ospnext` package
automatically.

### 🟢 GPU (CUDA)

```bash
# 1. Create the conda env
conda create -n ospnext python=3.10 -y
conda activate ospnext

# 2. Install all dependencies in editable mode
cd /path/to/OSP-Next
pip install -e .
```

What this installs:

- `torch==2.8.0`, `torchvision==0.23.0` (CUDA build, picked by pip wheel)
- `diffusers>=0.31`, `transformers>=4.55`, `accelerate>=1.4`, `peft>=0.10`, `trl>=0.11`
- All data / IO / logging utilities listed in `pyproject.toml`

> ⚠️  `flash_attn` is **not** in `pyproject.toml` because building it via plain
> `pip` is fragile. Build it manually only if you want FA2 / FA3 acceleration —
> see the [next section](#-flash-attention-optional-gpu-only). Without it, the
> code falls back to PyTorch SDPA automatically.

Equivalent `pip -r` form:

```bash
pip install -r requirements.txt
```

### 🟣 NPU (Ascend)

```bash
# 1. Source the Ascend toolkit (do this in EVERY new shell)
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. Create the conda env
conda create -n ospnext-npu python=3.10 -y
conda activate ospnext-npu

# 3. Install with the NPU extra
cd /path/to/OSP-Next
pip install -e .[npu]
```

The `[npu]` extra adds `torch_npu==2.8.0.post2` on top of the pinned
`torch==2.8.0 + torchvision==0.23.0` from the core dependencies. It does **not**
install `flash_attn` — on NPU we use SDPA / custom kernels.

Equivalent `pip -r` form:

```bash
pip install -r requirements_npu.txt
```

### ⚡ Flash-Attention (optional, GPU only)

The attention layer in `ospnext/modules/attention.py` will try to import
`flash_attn_interface` (FA3) first, fall back to `flash_attn` (FA2), and finally
to PyTorch SDPA. If you want FA acceleration, build from source:

```bash
# Flash-Attention v2 (CUDA 11.8+, Ampere / Hopper)
pip install ninja packaging
pip install flash-attn --no-build-isolation

# OR Flash-Attention v3 (Hopper-only, faster)
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention/hopper
python setup.py install
```

> 💡  If the build keeps OOM-ing the host, lower the parallel job count:
> `MAX_JOBS=4 pip install flash-attn --no-build-isolation`.

### 🔬 HIF8 NPU Quant Kernel (optional, NPU only)

Required only for the `osp_hif8_*` configs in `configs/infer/npu/`. The kernel
lives at `ospnext/quant_cy_npu/base/cusrc/` and produces an in-place `.so`:

```bash
cd ospnext/quant_cy_npu/base/cusrc
# Make sure ASCEND_TOOLKIT_HOME is set (the conda activation does this).
python setup.py build_ext --inplace
```

After a successful build, `ospnext/modules/hif8_linear.py` will import the op
silently. Skip this step entirely if you don't use HIF8 configs.

---

## 🎥 Inference Pipeline

Inference is launched through `infer/infer_osp.py` with a YAML config. The
typical flow is:

1. Copy / edit a config under `configs/infer/{gpu,npu}/`.
2. Update the `/path/to/...` placeholders to point to your local weights /
   prompts / output dir.
3. Run the matching shell script under `scripts/infer/{gpu,npu}/`.

### 📝 Step 1 — Edit the config

Example: `configs/infer/gpu/osp_14b.yaml`. The fields you almost always need
to change are highlighted below:

```yaml
model_name: "osp_next"
pipeline_name: "t2v"
seed: 1024

prompt_txt: "assets/t2v/simple_prompts.txt"   # 🔧 one prompt per line
output_dir: "/path/to/output"                  # 🔧 where to save *.mp4

num_frames: 81                                 # video length
height: 720                                    # spatial resolution
width: 1280
save_fps: 16                                   # output mp4 fps
batch_size: 1                                  # per-rank batch size

fsdp_size: 8                                   # FSDP world size
sp_size: 4                                     # Ulysses SP size
skiparse_sp_size: 4                            # Skiparse SP size
use_sequence_parallel: False                   # toggle Ulysses SP
use_skiparse_sequence_parallel: True           # toggle Skiparse SP
reshard_after_forward: Null                    # FSDP2 setting, leave Null
explicit_prefetching_num_blocks: 2             # NPU memory tradeoff
weight_dtype: "bf16"                           # bf16 / fp16 / fp32
save_with_dcp_api: False                       # match training format

model_config:
  dim: 5120                                    # 14B = 5120; 1.3B = 1536
  ffn_dim: 13824
  num_heads: 40
  num_layers: 40
  skiparse_model_type: "dual_end"              # 'full' disables skiparse
  sparse_ratio: 2
  num_full_blocks: 8
  pretrained_model_dir_or_checkpoint: "/path/to/model"   # 🔧 your weights

scheduler_config:
  scheduler_name: "flow_matching"
  num_inference_steps: 50                      # quality vs speed
  shift: 7.0                                   # flow-matching shift
  guidance_scale: 5.0                          # CFG guidance scale

vae_config:
  vae_path: "/path/to/vae"                     # 🔧 VAE checkpoint
  dtype: "fp32"

text_encoder_config:
  text_len: 512
  checkpoint_path: "/path/to/text_encoder"     # 🔧 T5 checkpoint
  text_tokenizer_path: "/path/to/text_tokenizer"  # 🔧 T5 tokenizer
  use_fsdp: True                               # FSDP-shard the T5 encoder
```

🔧 marked fields **must** be filled in for the run to succeed. Everything else
has reasonable defaults inside the code.

### 🚀 Step 2 — Pick a launch script

| Backend | Model        | Script                                          | Config                                       |
|---------|--------------|-------------------------------------------------|----------------------------------------------|
| GPU     | OSP-Next 1.3B | `scripts/infer/gpu/infer_osp_1_3b.sh`           | `configs/infer/gpu/osp_1_3b.yaml`           |
| GPU     | OSP-Next 14B  | `scripts/infer/gpu/infer_osp_14b.sh`            | `configs/infer/gpu/osp_14b.yaml`            |
| NPU     | OSP-Next 1.3B | `scripts/infer/npu/infer_osp_1_3b.sh`           | `configs/infer/npu/osp_1_3b.yaml`           |
| NPU     | OSP-Next 14B  | `scripts/infer/npu/infer_osp_14b.sh`            | `configs/infer/npu/osp_14b.yaml`            |
| NPU     | HIF8 1.3B     | `scripts/infer/npu/infer_osp_hif8_1_3b.sh`      | `configs/infer/npu/osp_hif8_1_3b.yaml`      |
| NPU     | HIF8 14B      | `scripts/infer/npu/infer_osp_hif8_14b.sh`       | `configs/infer/npu/osp_hif8_14b.yaml`       |

### ▶️ Step 3 — Run

```bash
# Single node (8 GPUs / 16 NPUs is the default in the scripts)
bash scripts/infer/gpu/infer_osp_14b.sh

# Multi-node — override env vars
NNODES=4 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
    bash scripts/infer/gpu/infer_osp_14b.sh
```

Outputs:

```
${output_dir}/
├── 0000_xxxxxxxx_prompt-here.mp4
├── 0001_...mp4
└── video_grid.mp4         # combined preview grid
```

---

## 🏋️ Training Pipeline

Two entry points are provided:

| Entry                   | Purpose                                | Optimizer target            |
|-------------------------|----------------------------------------|-----------------------------|
| `train/train_osp.py`    | Supervised fine-tuning (SFT)           | Full-parameter / FSDP2      |
| `train/train_osp_RL.py` | GRPO RL post-training w/ LoRA          | LoRA adapters only          |

### Supervised Fine-Tuning (SFT)

#### 📝 Step 1 — Edit the training config

Example: `configs/train/gpu/osp_14b.yaml`:

```yaml
model_name: "osp_next"
seed: 1024

output_dir: "/path/to/output"            # 🔧 checkpoint root
training_iteration: 1000000              # total steps
fsdp_size: 8                             # FSDP world size
sp_size: 4
skiparse_sp_size: 4
use_sequence_parallel: False             # Ulysses SP
use_skiparse_sequence_parallel: True     # Skiparse SP (recommended)
gradient_checkpointing: True             # memory ↘️ , compute ↗️
gradient_accumulation_steps: 1
init_max_grad_norm: 1.0
log_interval: 1
save_interval: 1000                      # save every N steps
weight_dtype: "bf16"
ema_decay: 0.9999                        # 0.999  for 14B, 0.9993 for 1.3B
ema_update_interval: 1
save_with_dcp_api: True

wandb_config:
  project_name: "osp_next"               # 🔧 your wandb project
  exp_name:     "osp_next"               # 🔧 run name

model_config:
  dim: 5120
  ffn_dim: 13824
  num_heads: 40
  num_layers: 40
  skiparse_model_type: "dual_end"
  sparse_ratio: 2
  num_full_blocks: 8
  pretrained_model_dir_or_checkpoint: "/path/to/model"   # 🔧 init weights

scheduler_config:
  scheduler_name: "flow_matching"
  use_dynamic_shifting: True
  use_logitnorm_time_sampling: True

vae_config:
  vae_path: "/path/to/vae"               # 🔧 frozen VAE
  dtype: "fp32"

text_encoder_config:
  text_len: 512
  checkpoint_path: "/path/to/text_encoder"  # 🔧 frozen T5
  use_fsdp: True

data_config:
  batch_size: 1                          # per-rank batch size
  num_workers: 16
  shuffle: True
  dataset_name: "t2v_random"             # see ospnext/data/datasets/
  dataset_config:
    text_tokenizer_path: "/path/to/text_tokenizer"   # 🔧
    sample_height: 720
    sample_width: 1280
    sample_num_frames: 81
    tokenizer_max_length: 512
    return_prompt_mask: True
  sampler_name: "stateful_distributed"
  collator_name: "wan_t2v"

optimizer_config:
  lr: 0.00002
  weight_decay: 0
```

#### 🚀 Step 2 — Launch

| Backend | Model         | Script                                       | Config                                  |
|---------|---------------|----------------------------------------------|-----------------------------------------|
| GPU     | 1.3B          | `scripts/train/gpu/train_osp_1_3b.sh`        | `configs/train/gpu/osp_1_3b.yaml`       |
| GPU     | 14B           | `scripts/train/gpu/train_osp_14b.sh`         | `configs/train/gpu/osp_14b.yaml`        |
| NPU     | 1.3B          | `scripts/train/npu/train_osp_1_3b.sh`        | `configs/train/npu/osp_1_3b.yaml`       |
| NPU     | 14B           | `scripts/train/npu/train_osp_14b.sh`         | `configs/train/npu/osp_14b.yaml`        |
| NPU     | HIF8 1.3B     | *(use the inference HIF8 build first)*       | `configs/train/npu/osp_hif8_1_3b.yaml`  |
| NPU     | HIF8 14B      | *(use the inference HIF8 build first)*       | `configs/train/npu/osp_hif8_14b.yaml`   |

```bash
bash scripts/train/gpu/train_osp_14b.sh
```

Resuming is automatic — `Checkpointer.last_training_iteration` picks the most
recent checkpoint folder under `output_dir`.

### Reinforcement Learning (GRPO + LoRA)

The RL post-training uses the same FSDP2 backbone but trains a **LoRA** adapter
on top of frozen base weights, sampled with **SDE → ODE hybrid** denoising,
optimized with **GRPO** against a **VideoAlign reward**.

#### 📝 Step 1 — Edit the RL config

Example: `configs/train/npu/osp_14b_RL.yaml`. The RL-specific blocks
(`lora_config`, `rl_config`) are what you tune most:

```yaml
model_name: "osp_next"
seed: 42
output_dir: "/path/to/output"            # 🔧 RL checkpoint root

num_epochs: 1000                         # RL epochs (not SFT steps)
fsdp_size: 16
sp_size: 4
skiparse_sp_size: 4
use_sequence_parallel: False
use_skiparse_sequence_parallel: True
gradient_checkpointing: True
gradient_accumulation_steps: 1
init_max_grad_norm: 1.0
log_interval: 1
save_interval: 500
weight_dtype: "bf16"
ema_decay: 0.999                         # use 0.999 for 14B
ema_update_interval: 1
save_with_dcp_api: True
model_cpu_offload: False
encoder_cpu_offload: False

wandb_config:
  project_name: "osp_next_RL"
  exp_name: "osp_next_RL"

model_config:
  # ... same as SFT config ...
  pretrained_model_dir_or_checkpoint: "/path/to/model"   # 🔧 base ckpt

scheduler_config:
  scheduler_name: "flow_matching"
  use_dynamic_shifting: True
  use_logitnorm_time_sampling: True

vae_config:
  vae_path: "/path/to/vae"               # 🔧
  dtype: "fp16"

text_encoder_config:
  text_len: 512
  checkpoint_path: "/path/to/text_encoder"           # 🔧
  text_tokenizer_path: "/path/to/tokenizer"          # 🔧
  use_fsdp: True

# RL training uses a text-only prompt dataset; only the tokenizer is needed.
data_config:
  dataset_config:
    text_tokenizer_path: "/path/to/tokenizer"        # 🔧
    tokenizer_max_length: 512

optimizer_config:
  lr: 0.00002                            # 2e-5 for the LoRA optimizer
  weight_decay: 0.001

lora_config:
  rank: 32                               # LoRA rank
  alpha: 64                              # LoRA alpha
  target_modules:                        # which projections get LoRA
    - "self_attn.q"
    - "self_attn.k"
    - "self_attn.v"
    - "self_attn.o"
    - "cross_attn.q"
    - "cross_attn.k"
    - "cross_attn.v"
    - "cross_attn.o"
  # lora_path: "/path/to/existing/lora"  # uncomment to resume from a LoRA ckpt

rl_config:
  prompt_file:      "/path/to/prompt_file"           # 🔧 train prompts (txt)
  eval_prompt_file: "/path/to/eval_prompt_file"      # 🔧 eval prompts (txt)
  height: 720
  width:  1280
  num_frames: 81
  sde_steps: 10                          # # of steps trained with SDE noise
  num_inference_steps: 25                # total denoising steps in sampling
  kl_beta: 0.004                         # KL penalty weight (set 0 to disable)
  num_batches_per_epoch: 4               # batches per RL epoch
  num_image_per_prompt: 4                # k repeats (GRPO group size)
  sample_batch_size: 2                   # batch size during rollout
  train_batch_size:  2                   # batch size during policy update
  reward_fn:
    videoalign: 1.0                      # 🔧 VideoAlign weight; configure
                                         #    the model path inside
                                         #    ospnext/rewards/rewards.py
```

#### 🚀 Step 2 — Launch

| Backend | Model | Script                                    | Config                                |
|---------|-------|-------------------------------------------|----------------------------------------|
| GPU     | 14B   | `scripts/train/gpu/train_osp_14b_RL.sh`   | `configs/train/gpu/osp_14b_RL.yaml`   |
| NPU     | 14B   | `scripts/train/npu/train_osp_14b_RL.sh`   | `configs/train/npu/osp_14b_RL.yaml`   |

```bash
bash scripts/train/npu/train_osp_14b_RL.sh
```

#### 📦 What gets saved during RL

```
${output_dir}/
├── iteration_000000010/        # FSDP checkpoint (full state)
├── iteration_000000010_ema/    # EMA model weights
├── lora-checkpoint-10/         # LoRA-only weights (small, portable)
│   ├── adapter_model.bin
│   └── adapter_config.json
└── rl_training_state.json      # epoch / global_step bookkeeping
```

The `lora-checkpoint-*/` directories are the artifact you typically ship — load
them back with `peft.PeftModel.from_pretrained(base_model, lora_path)`.

---

## 🛠️ Tips & Troubleshooting

### 🔧 Sequence-parallel sizing

Inside any config, `fsdp_size × ddp_size = world_size`. The two SP groups
multiply inside the FSDP group:

```
sp_size  × skiparse_sp_size  ≤  fsdp_size
```

For the 14B model with `sparse_ratio=2`, valid pairs (per-rank shard count
must evenly divide `sparse_ratio² = 4`) are:

| `sp_size` | `skiparse_sp_size` | total SP factor |
|----------:|-------------------:|----------------:|
| 1         | 4                  | 4               |
| 2         | 2                  | 4               |
| 4         | 1                  | 4               |
| 1         | 1                  | 1 (no SP)       |

### 💾 Checkpoint format

`save_with_dcp_api: True` uses PyTorch's distributed-checkpoint API (a folder
per iteration). To consume those weights in plain inference (`save_with_dcp_api:
False`) you can run `convert_to_diffusers_weights.py` to flatten the shards
into a single diffusers-style folder.

### 🐛 Common failures

| Symptom                                             | Fix                                                                                                  |
|-----------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `ImportError: cannot import name 'flash_attn'`      | Either install flash-attn manually, or ignore — code already falls back to SDPA.                     |
| `RuntimeError: NPU error ... aclrtSetDevice`        | Forgot `source /usr/local/Ascend/ascend-toolkit/set_env.sh` before activating the conda env.         |
| `wandb` prompts for login                            | Either run `wandb login` once, or set `WANDB_MODE=offline` (every training script already does this). |
| `from ospnext.quant_cy_npu import ...` fails        | Build the HIF8 kernel (see [HIF8 NPU Quant Kernel](#-hif8-npu-quant-kernel-optional-npu-only)).      |
| Reward init silently returns 1.0                    | Fill in the VideoAlign checkpoint path inside `ospnext/rewards/rewards.py::videoalign_score`.        |

### 📚 Environment variables worth knowing

| Variable                      | Default            | Purpose                                            |
|-------------------------------|--------------------|----------------------------------------------------|
| `MASTER_ADDR` / `MASTER_PORT` | `127.0.0.1:29505`  | torchrun rendezvous                                |
| `NPRC_PER_NODE`               | `8` (GPU) / `16` (NPU) | Processes per node                              |
| `NNODES`                      | `1`                | Total nodes                                        |
| `WANDB_MODE`                  | `online`           | Set to `offline` to disable WandB network usage    |
| `PYTORCH_NPU_ALLOC_CONF`      | `expandable_segments:True` | NPU memory allocator (set in scripts)      |

---

## 📝 Citation

If you find OSP-Next useful in your research, please consider citing:

```bibtex
<!-- TODO: replace with the final citation block once the paper is on arXiv. -->
@article{ospnext2026,
  title        = {<PAPER_TITLE>},
  author       = {<AUTHORS>},
  journal      = {<JOURNAL_OR_ARXIV>},
  year         = {2026},
  eprint       = {<ARXIV_ID>},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV},
  url          = {<ARXIV_URL>},
}
```

Related work this project builds on:

```bibtex
@article{wan2025,
  title  = {Wan: Open and Advanced Large-Scale Video Generative Models},
  author = {Wan Team},
  year   = {2025},
}

@article{opensora_plan,
  title  = {Open-Sora-Plan: A Community-Driven Open Source Project for Sora-Style Text-to-Video Generation},
  author = {PKU-YuanGroup},
  year   = {2024},
}
```

---

## 📄 License

See [`LICENSE.txt`](LICENSE.txt).

---

## ⭐ Star History

<a href="https://star-history.com/#<OWNER>/<REPO>&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)"  srcset="https://api.star-history.com/svg?repos=<OWNER>/<REPO>&type=Date&theme=dark">
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=<OWNER>/<REPO>&type=Date">
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=<OWNER>/<REPO>&type=Date" width="720">
  </picture>
</a>

<sub>If this project helped you, a ⭐ goes a long way 🙌</sub>

