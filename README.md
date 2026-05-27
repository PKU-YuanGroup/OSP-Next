<div align="center">

<img src="assets/logo.png" alt="OSP-Next" width="220">

### Efficient High-Quality Video Generation with Sparse Sequence Parallelism, HiF8 Quantization, and Reinforcement Learning

**Open-Sora Plan · Next Generation**

A scalable **sparse** text-to-video diffusion model, introducing **Skiparse-2D Attention**,
**Sparse Sequence Parallelism (SSP)**, **HiF8 quantization**, and
**Mix-GRPO + LoRA** RL post-training.

</div>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-OSP--Next-b31b1b.svg?logo=arXiv)](<ARXIV_URL>)
[![arXiv](https://img.shields.io/badge/Arxiv-Open--Sora%20Plan-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.00131)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-FFD21E.svg)](https://huggingface.co/yunyangge/OSP-Next)
[![ModelScope](https://img.shields.io/badge/ModelScope-OSP--Next-624AFF.svg?logo=alibabacloud)](https://modelscope.cn/models/beihai123/OSP-Next)
[![GitHub repo stars](https://img.shields.io/github/stars/PKU-YuanGroup/OSP-Next?style=flat&logo=github&logoColor=whitesmoke&label=Stars)](https://github.com/PKU-YuanGroup/OSP-Next/stargazers)

</h5>

---

## 📣 News

- **[2026.05.22]** 🎉🎉🎉 We have open-sourced the **complete training & inference code** for OSP-Next together with the **model weights**. Welcome to give it a try!

---

## ✨ Highlights

OSP-Next is a **sparse video diffusion** framework with four tightly co-designed
contributions — see the [paper](<ARXIV_URL>) for the full technical report.

<table>
<tr>
<td width="50%" valign="top">
<h3>🧩 &nbsp;Skiparse-2D Attention</h3>

A **fixed-rule sparse attention pattern** purpose-built for image / video
modalities, applied independently along height and width. Better aligns with
spatial locality than Skiparse-1D, **approaches the quality of 3D Full
Attention**, and stays **natively compatible with FlashAttention kernels** —
no custom triton or CUDA needed.

</td>
<td width="50%" valign="top">
<h3>🔗 &nbsp;Sparse Sequence Parallelism (SSP)</h3>

A **parallel strategy natively co-designed with Skiparse-2D Attention**.
Compared to Ulysses SP, SSP cuts **inter-rank communication volume by 75%** and
drops the per-block communication steps **from 4 down to 1** — removing the SP
bottleneck for long-video, long-context training.

</td>
</tr>
<tr>
<td width="50%" valign="top">
<h3>🪶 &nbsp;HiF8 Quantization &nbsp;<sub>(NPU only)</sub></h3>

A **dynamic-precision HiF8** scheme (per-tensor exponent / mantissa allocation)
applied on top of the sparse model. The **first work to show that 8-bit
quantization and sparse-model fine-tuning can be done jointly** — the VBench gap stays within ~0.5% with baseline, and inference reaches up to **2.27× speed-up** on a single Ascend 950PR.

</td>
<td width="50%" valign="top">
<h3>🎯 &nbsp;Mix-GRPO RL on Sparse Models</h3>

The **first attempt to apply reinforcement learning to a sparse video
generation model**. Our **Mix-GRPO + LoRA** pipeline shows that RL keeps
pushing the quality / preference frontier of sparse models — and the entire
sparse-model training pipeline is open-sourced for the community.

</td>
</tr>
</table>

### 📊 Performance at a glance

End-to-end speed-ups vs. the **Wan2.1** full-attention baseline, measured on
**5-second · 81-frame** videos at two resolution settings (Tab. 2 in the paper):

<table>
<tr>
<th width="33%" align="center">⚡ NVIDIA H200<br/><sub>OSP-Next · BF16 · FA3 + torch.compile</sub></th>
<th width="33%" align="center">🟣 Ascend 950PR<br/><sub>OSP-Next · BF16 · SDPA</sub></th>
<th width="33%" align="center">🪶 Ascend 950PR<br/><sub>OSP-Next-HiF8 · 8-bit · SDPA</sub></th>
</tr>
<tr>
<td valign="top">

- 720P (padded)
  - **1.53×** single-GPU
  - **1.42×** on 8× GPU
- 768P (native)
  - **1.64×** single-GPU
  - **1.52×** on 8× GPU

</td>
<td valign="top">

- 720P (padded)
  - **1.27×** single-NPU
- 768P (native)
  - **1.76×** single-NPU

</td>
<td valign="top">

- 720P (padded)
  - **1.69×** single-NPU
- 768P (native)
  - **2.27×** single-NPU
- Quality cost
  - **only −0.4 pt** VBench vs BF16

</td>
</tr>
</table>

> 🏆 &nbsp;OSP-Next hits a **VBench total of 83.73%** (Wan2.1 baseline: 83.69%);
> OSP-Next-HiF8 keeps **83.29%** with only a 0.4-pt drop. Full benchmark tables,
> ablations and qualitative comparisons are in the [paper](<ARXIV_URL>).

> ℹ️ &nbsp;Multi-NPU 950PR numbers are not reported yet — Ascend 950PR
> resources are currently in limited supply, so the results for this hardware
> are restricted to a single NPU.

> 🟦 &nbsp;**Bonus** — one codebase, two backends: the same training & inference
> scripts run on **NVIDIA CUDA** *and* **Ascend NPU** — just swap
> `pip install -e .` for `pip install -e .[npu]`.

---

## 🚀 Quick Start

Generate your first OSP-Next video in four commands (GPU example):

```bash
# 1. Clone & install
git clone https://github.com/PKU-YuanGroup/OSP-Next.git && cd OSP-Next
conda create -n ospnext python=3.10 -y && conda activate ospnext
pip install -e .

# 2a. Download the OSP-Next 14B diffusion weights from our repo.
huggingface-cli download yunyangge/OSP-Next --local-dir ./checkpoints/osp_next_14b

# 2b. OSP-Next reuses Wan 2.1's T5 text encoder and VAE — we do NOT re-host
#     them. Grab them from the upstream Wan-AI repo (HuggingFace or ModelScope):
huggingface-cli download Wan-AI/Wan2.1-T2V-14B \
    models_t5_umt5-xxl-enc-bf16.pth \
    Wan2.1_VAE.pth \
    --include "google/umt5-xxl/*" \
    --local-dir ./checkpoints/Wan2.1-T2V-14B

# 3. Edit one config file — point `pretrained_model_dir_or_checkpoint`,
#    `vae_path`, `checkpoint_path` and `text_tokenizer_path` to the directories
#    you just downloaded:
$EDITOR configs/infer/gpu/osp_14b.yaml

# 4. Generate!
bash scripts/infer/gpu/infer_osp_14b.sh
```

> ⏱️  First run takes a few minutes to warm up FSDP2 + compile the kernels;
> subsequent prompts in the same process are much faster.

> 🟣 **On Ascend NPU?** Skip step 1 and follow the
> [🟣 NPU (Ascend)](#-npu-ascend) setup first (CANN 8.5.0, `pip install -e .[npu]`,
> source-build `decord`), then come back to steps 2–4 and swap the GPU script
> in step 4 for its NPU equivalent under `scripts/infer/npu/`.

---

## 🎞️ Demo Gallery

A side-by-side comparison of the same prompt across three models. Hit ▶ on any
cell to play the video right inside the page.

<!--
  TODO: replace each <DEMO_*> placeholder with a GitHub-hosted mp4 URL, e.g.
        https://github.com/user-attachments/assets/<uuid>
  GitHub renders <video src="..."> inline as long as the URL lives on a
  github.com / user-attachments / githubusercontent.com domain.
-->

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
      <td><sub><i>"A handheld 35mm camera holds an extreme close-up on a gray-haired, bearded man in his sixties..."</i></sub></td>
      <td align="center"><video src="https://github.com/user-attachments/assets/284d227d-b0c7-4f54-9930-b466095725a5"     width="240" controls muted loop playsinline preload="metadata"></video></td>
      <td align="center"><video src="https://github.com/user-attachments/assets/8d9a85f9-75f0-4eaa-a01d-a81ccdffc659"       width="240" controls muted loop playsinline preload="metadata"></video></td>
      <td align="center"><video src="https://github.com/user-attachments/assets/e4388ced-20c5-4836-9c66-cdba5b9eeb7f"  width="240" controls muted loop playsinline preload="metadata"></video></td>
    </tr>
    <tr>
      <td><sub><i>"A cream and sable corgi, sporting sleek jet-black sunglasses, trots confidently along a pristine tropical beach..."</i></sub></td>
      <td align="center"><video src="https://github.com/user-attachments/assets/85a4fd1c-d098-4b25-a28d-15a46d64bfb4"     width="240" controls muted loop playsinline preload="metadata"></video></td>
      <td align="center"><video src="https://github.com/user-attachments/assets/86f30f13-09ee-40d3-bbae-0d0627a939ae"       width="240" controls muted loop playsinline preload="metadata"></video></td>
      <td align="center"><video src="https://github.com/user-attachments/assets/e05827a9-f93b-4fcb-8f65-29749668b60d"  width="240" controls muted loop playsinline preload="metadata"></video></td>
    </tr>
    <tr>
      <td><sub><i>"A lone 30-year-old space man strides across an endless salt desert under a vast, electric-blue sky..."</i></sub></td>
      <td align="center"><video src="https://github.com/user-attachments/assets/51026d0d-075a-45d4-b668-598614cfadcd"     width="240" controls muted loop playsinline preload="metadata"></video></td>
      <td align="center"><video src="https://github.com/user-attachments/assets/def9c76b-829e-4cc4-9781-e6e252a34087"       width="240" controls muted loop playsinline preload="metadata"></video></td>
      <td align="center"><video src="https://github.com/user-attachments/assets/16e6ef65-cdc9-4811-822e-0655f72ab300"  width="240" controls muted loop playsinline preload="metadata"></video></td>
    </tr>
  </tbody>
</table>

> 💡 **HiF8 takeaway** — On a single **Ascend 950PR**, OSP-Next-HiF8 reaches
> **1.69× / 2.27× speed-up** over the BF16 baseline under the 5s 720P / 5s 768P
> settings, with only a **0.4 point** drop on the VBench total score.

---

## 📦 Model Downloads

### 🧠 OSP-Next diffusion weights (hosted by us)

| Model             | Params | 🤗 HuggingFace                       | <img src="https://github.com/modelscope.png?size=48" height="14" alt="ModelScope" valign="middle"> ModelScope |
|-------------------|-------:|--------------------------------------|----------------------------------------------------------------------|
| OSP-Next 14B      |  14B   | [`yunyangge/OSP-Next`](https://huggingface.co/yunyangge/OSP-Next) | [`beihai123/OSP-Next`](https://modelscope.cn/models/beihai123/OSP-Next) |
| OSP-Next-HiF8 14B |  14B   | [`yunyangge/OSP-Next`](https://huggingface.co/yunyangge/OSP-Next)  | [`beihai123/OSP-Next`](https://modelscope.cn/models/beihai123/OSP-Next)  |

> ℹ️  The `*_1_3b.yaml` configs and `*_1_3b.sh` launch scripts are kept in the
> repository as ready-to-use templates if you want to train your own 1.3B
> variant, but **no official 1.3B checkpoint is released** at this time.

### 🔡 T5 text encoder & 🎞️ WAN VAE (hosted by Wan-AI)

OSP-Next reuses the **T5 text encoder** and **WAN VAE** released with Wan 2.1
verbatim — we do **not** re-host these weights. Please grab them from the
official Wan-AI repository:

| Component                 | File                                       | 🤗 HuggingFace                                                                                                                              | <img src="https://github.com/modelscope.png?size=48" height="14" alt="ModelScope" valign="middle"> ModelScope                                                     |
|---------------------------|--------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| T5 (UMT5-XXL) weights     | `models_t5_umt5-xxl-enc-bf16.pth`          | [`Wan-AI/Wan2.1-T2V-14B`](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/blob/main/models_t5_umt5-xxl-enc-bf16.pth)                           | [`Wan-AI/Wan2.1-T2V-14B`](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B/file/view/master/models_t5_umt5-xxl-enc-bf16.pth)                                    |
| T5 tokenizer              | `google/umt5-xxl/`                         | [`Wan-AI/Wan2.1-T2V-14B`](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/tree/main/google/umt5-xxl)                                           | [`Wan-AI/Wan2.1-T2V-14B`](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B/files)                                                                               |
| WAN VAE                   | `Wan2.1_VAE.pth`                           | [`Wan-AI/Wan2.1-T2V-14B`](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/blob/main/Wan2.1_VAE.pth)                                            | [`Wan-AI/Wan2.1-T2V-14B`](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B/file/view/master/Wan2.1_VAE.pth)                                                     |

> 💡 All three components live inside the same `Wan-AI/Wan2.1-T2V-14B` repo, so
> a one-shot download is enough — see the snippet in the [🚀 Quick Start](#-quick-start)
> for an example. The same files also work for the 1.3B configs (they share the
> identical T5 / VAE backbone with the 14B model).

### 📌 After downloading

Update the corresponding paths in your config (both inference and training):

```yaml
model_config:
  pretrained_model_dir_or_checkpoint: "/path/to/osp_next_14b"                       # ← OSP-Next ckpt
vae_config:
  vae_path: "/path/to/Wan2.1-T2V-14B/Wan2.1_VAE.pth"                                # ← WAN VAE
text_encoder_config:
  checkpoint_path: "/path/to/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth"        # ← T5 weights
  text_tokenizer_path: "/path/to/Wan2.1-T2V-14B/google/umt5-xxl/"                   # ← T5 tokenizer
```

---

## 📑 Table of Contents

- [📣 News](#-news)
- [✨ Highlights](#-highlights)
  - [📊 Performance at a glance](#-performance-at-a-glance)
- [🚀 Quick Start](#-quick-start)
- [🎞️ Demo Gallery](#️-demo-gallery)
- [📦 Model Downloads](#-model-downloads)
- [🧱 Project Layout](#-project-layout)
- [⚙️ Environment Setup](#️-environment-setup)
  - [🟢 GPU (NVIDIA CUDA)](#-gpu-nvidia-cuda)
    - [1. Install OSP-Next](#1-install-osp-next)
    - [2. *Optional* · Build Flash-Attention](#2-optional--build-flash-attention-hopper--ampere)
  - [🟣 NPU (Ascend)](#-npu-ascend)
    - [1. Install Ascend CANN 8.5.0](#1-install-ascend-cann-850-one-time-before-any-conda-step)
    - [2. Set up the conda env and install OSP-Next](#2-set-up-the-conda-env-and-install-osp-next)
    - [3. Build `decord` from source](#3-build-decord-from-source-aarch64-only)
    - [4. *Optional* · Rebuild the HiF8 NPU Quant Kernel](#4-optional--rebuild-the-hif8-npu-quant-kernel-only-on-import-error)
- [🎥 Inference Pipeline](#-inference-pipeline)
- [🏋️ Training Pipeline](#️-training-pipeline)
  - [📚 Data Preparation](#-data-preparation)
  - [🎓 Supervised Fine-Tuning (SFT)](#-supervised-fine-tuning-sft)
  - [🎯 Reinforcement Learning (Mix-GRPO + LoRA)](#-reinforcement-learning-mix-grpo--lora)
- [🛠️ Tips & Troubleshooting](#️-tips--troubleshooting)
- [🙏 Acknowledgements](#-acknowledgements)
- [📝 Citation](#-citation)
- [📄 License](#-license)
- [⭐ Star History](#-star-history)

---

## 🧱 Project Layout

```
OSP-Next/
├── configs/                     # All YAML configs
│   ├── infer/{gpu,npu}/         # Inference configs (per backend)
│   ├── train/{gpu,npu}/         # Training configs (per backend)
│   ├── filter_config.yaml       # Data-filter / LMDB-build settings
│   └── all_videos.txt           # Example ann_txt index for filter_data.py
├── scripts/                     # Launch scripts (torchrun)
│   ├── infer/{gpu,npu}/         # Inference launchers
│   ├── train/{gpu,npu}/         # Training launchers
│   └── filter_data.sh           # Wrapper: runs filter_data.py
├── ospnext/                     # Core library
│   ├── modules/                 # Diffusion / VAE / T5 / attention / HiF8
│   ├── distributed/             # FSDP2 + sequence-parallel state & comm
│   ├── data/                    # Datasets, samplers, collators
│   ├── pipelines/               # End-to-end inference pipelines
│   ├── rewards/                 # VideoAlign reward (for RL)
│   ├── schedulers/              # Flow matching scheduler
│   ├── utils/                   # Logging, EMA, checkpointing, encoder cache
│   └── quant_cy_npu/            # HiF8 quant op (NPU custom kernel)
├── train/
│   ├── train_osp.py             # Entry: SFT training
│   └── train_osp_RL.py          # Entry: Mix-GRPO + LoRA RL training
├── infer/
│   └── infer_osp.py             # Entry: text-to-video inference
├── merge_lora_weights.py        # Merge RL LoRA into base for deployment
├── filter_data.py               # Entry: build LMDB from annotated video corpus
├── assets/
│   ├── logo.png                 # README logo
│   └── t2v/                     # Sample prompt files
├── requirements.txt             # GPU pip requirements
├── requirements_npu.txt         # NPU pip requirements
├── pyproject.toml               # Editable install metadata
└── LICENSE.txt                  # Project license
```

---

## ⚙️ Environment Setup

We strongly recommend using **conda + editable install** so that every entry
point (`train/`, `infer/`, custom scripts) sees the `ospnext` package
automatically.

> 📦 The setup is split by backend — **pick one** and follow it top-to-bottom.
> GPU users do not need anything from the NPU section and vice versa.

---

### 🟢 GPU (NVIDIA CUDA)

#### 1. Install OSP-Next

```bash
# 1a. Create the conda env
conda create -n ospnext python=3.10 -y
conda activate ospnext

# 1b. Install all dependencies in editable mode
cd /path/to/OSP-Next
pip install -e .
```

What this installs:

- `torch==2.8.0`, `torchvision==0.23.0` (CUDA build, picked by pip wheel)
- `diffusers>=0.31`, `transformers>=4.55`, `accelerate>=1.4`, `peft>=0.10`, `trl>=0.11`
- All data / IO / logging utilities listed in `pyproject.toml`

Equivalent `pip -r` form:

```bash
pip install -r requirements.txt
```

> ⚠️  `flash_attn` is **not** in `pyproject.toml` because building it via plain
> `pip` is fragile. Build it manually only if you want FA2 / FA3 acceleration —
> see [step 2](#2-optional--build-flash-attention-hopper--ampere) right below.
> Without it, the code falls back to PyTorch SDPA automatically.

#### 2. *Optional* · Build Flash-Attention (Hopper / Ampere)

The attention layer in `ospnext/modules/attention.py` tries to import
`flash_attn_interface` (FA3) first, falls back to `flash_attn` (FA2), and
finally to PyTorch SDPA — so this step is **strictly optional**.

```bash
# Flash-Attention v2 (CUDA 11.8+, Ampere / Hopper)
pip install ninja packaging
pip install flash-attn --no-build-isolation

# OR Flash-Attention v3 (Hopper-only, faster)
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention/hopper
python setup.py install
```

> ⏳  **Heads up — this build is slow.** Compiling Flash-Attention from source
> typically takes **30 min – 2 h** depending on CPU / RAM (each CUDA kernel is
> instantiated for many head-dim × dtype × causal combinations). Run it inside
> `tmux` / `screen` so an SSH disconnect doesn't kill it, and don't be alarmed
> if `pip` looks "stuck" — it's just `nvcc` working its way through hundreds of
> translation units. The wheel is cached afterwards, so subsequent reinstalls
> in the same environment are instant.

> 💡  If the build keeps OOM-ing the host, lower the parallel job count:
> `MAX_JOBS=4 pip install flash-attn --no-build-isolation`. The same flag
> applies to the FA3 `setup.py` build.

---

### 🟣 NPU (Ascend)

#### 1. Install Ascend CANN 8.5.0 (one-time, before any conda step)

OSP-Next is pinned to **CANN 8.5.0** — older toolkits (e.g. 8.0.x) are missing
several operators we rely on, and newer pre-release branches have not been
validated. Grab the matching installer for your hardware (Atlas 800T A2 /
Ascend 950PR / …) from the **official Ascend portal**:

> 🔗 **Download:** <https://www.hiascend.com/cann/download>
>
> Pick the **`8.5.0`** release that matches your OS and architecture
> (e.g. *Ubuntu 22.04 aarch64* or *openEuler 22.03 aarch64*), then follow the
> on-page installation guide. After install, the toolkit normally lives at
> `/usr/local/Ascend/ascend-toolkit/`, and step 2 below assumes that path —
> adjust accordingly if you installed elsewhere.

#### 2. Set up the conda env and install OSP-Next

```bash
# 2a. Source the Ascend toolkit (do this in EVERY new shell)
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2b. Create the conda env
conda create -n ospnext-npu python=3.10 -y
conda activate ospnext-npu

# 2c. Install with the NPU extra
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

#### 3. Build `decord` from source (aarch64 only)

`decord` is used by the data pipeline to read training / reward videos. It does
**not** publish pre-built wheels for `aarch64 + Python 3.10`, which is exactly
the configuration most Ascend hosts (Kunpeng / HiSilicon ARM CPUs) run on — a
plain `pip install decord` will therefore fail or pull in an incompatible
binary. Build it once from source:

```bash
# 3a. System deps (Ubuntu / openEuler / OpenAnolis — pick your package manager)
sudo apt-get install -y build-essential cmake ffmpeg \
                        libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev
# (openEuler / CentOS users: dnf install -y gcc-c++ cmake ffmpeg-devel)

# 3b. Clone and build (CPU-only — NPU has no CUDA decode path)
git clone --recursive https://github.com/dmlc/decord
cd decord
mkdir -p build && cd build
cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 3c. Install into the current conda env
cd ../python
python setup.py install
```

> ⚠️  If `python -c "import decord"` still raises `ImportError: libdecord.so:
> cannot open shared object file` after install, the compiled native lib is not
> on the loader path. Add the `build/` directory once and persist it in your
> conda env activate hook:
>
> ```bash
> export LD_LIBRARY_PATH=$(pwd)/../build:$LD_LIBRARY_PATH
> ```

#### 4. *Optional* · Rebuild the HiF8 NPU Quant Kernel (only on import error)

Used **only** for the `osp_hif8_*` configs in `configs/infer/npu/` /
`configs/train/npu/`. **You normally do not need to do anything here** —
`ospnext/quant_cy_npu/` is shipped with a pre-compiled CANN 8.5.0 kernel
(`libnpu_quant_op.so` + `npu_quant.cpython-3??-aarch64-linux-gnu.so`), so a
plain `python -c "from ospnext.quant_cy_npu import *"` should already work.

You only have to rebuild when the import fails — typical symptoms are:

- `ImportError: undefined symbol: ...` (mismatch between our shipped `.so` and your local CANN / Python ABI)
- The shipped `.so` is tagged for a different Python version (e.g. our wheel is built for `cpython-311`, but you installed `python=3.10` per [step 2](#2-set-up-the-conda-env-and-install-osp-next))
- You're running on a CANN release we haven't validated against

The fix is to **re-build the kernel from the upstream HiFloat8 repository**
([global-computing-consortium/HiFloat8](https://github.com/global-computing-consortium/HiFloat8))
and swap the resulting package into our tree:

```bash
# 4a. Clone upstream HiFloat8 anywhere outside this repo.
git clone https://github.com/global-computing-consortium/HiFloat8.git
cd HiFloat8/hif8_npu

# 4b. Re-build against your local CANN + Python (re-source set_env.sh first!).
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build_npu_ops.sh

# 4c. Sanity-check the rebuild inside the upstream tree.
python hif8_bf16.py     # expected: "ABS diff max (zero values): 0"

# 4d. Replace OSP-Next's bundled package with the freshly-built one.
cd /path/to/OSP-Next
rm -rf ospnext/quant_cy_npu
cp -r /path/to/HiFloat8/hif8_npu/quant_cy_npu ospnext/

# 4e. Final check inside the OSP-Next env.
python -c "from ospnext.quant_cy_npu import *; print('HiF8 kernel OK')"
```

> ✅  After the swap, `python -c "from ospnext.quant_cy_npu import *"` should
> succeed. If it still doesn't, re-source `set_env.sh` and re-run `bash
> build_npu_ops.sh` inside the **same** shell — the build is sensitive to
> environment leaks between sessions.

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
explicit_prefetching_num_blocks: 2             
weight_dtype: "bf16"                           # bf16 / fp16 / fp32
save_with_dcp_api: False                       # MUST match the flag used when the checkpoint was saved

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

| Backend | Model            | Script                                          | Config                                       |
|---------|------------------|-------------------------------------------------|----------------------------------------------|
| GPU     | OSP-Next 14B     | `scripts/infer/gpu/infer_osp_14b.sh`            | `configs/infer/gpu/osp_14b.yaml`            |
| GPU     | OSP-Next 1.3B †  | `scripts/infer/gpu/infer_osp_1_3b.sh`           | `configs/infer/gpu/osp_1_3b.yaml`           |
| NPU     | OSP-Next 14B     | `scripts/infer/npu/infer_osp_14b.sh`            | `configs/infer/npu/osp_14b.yaml`            |
| NPU     | OSP-Next 1.3B †  | `scripts/infer/npu/infer_osp_1_3b.sh`           | `configs/infer/npu/osp_1_3b.yaml`           |
| NPU     | HiF8 14B ‡       | `scripts/infer/npu/infer_osp_hif8_14b.sh`       | `configs/infer/npu/osp_hif8_14b.yaml`       |
| NPU     | HiF8 1.3B † ‡    | `scripts/infer/npu/infer_osp_hif8_1_3b.sh`      | `configs/infer/npu/osp_hif8_1_3b.yaml`      |

> † &nbsp;No official **1.3B** checkpoint is released — the 1.3B scripts and
> configs are kept as ready-to-go templates if you choose to train your own
> 1.3B variant from scratch.<br/>
> ‡ &nbsp;The HiF8 scripts require the **HiF8 NPU quant kernel** to import
> successfully — verify with `python -c "from ospnext.quant_cy_npu import *"`,
> and re-build via [NPU setup step 4](#4-optional--rebuild-the-hif8-npu-quant-kernel-only-on-import-error) if needed.

### ▶️ Step 3 — Run

```bash
# Single node — defaults to NPRC_PER_NODE=8.
# For an 8×NPU node, just run as-is. For a 16×NPU node, override:
#   NPRC_PER_NODE=16 bash scripts/infer/npu/infer_osp_14b.sh
bash scripts/infer/gpu/infer_osp_14b.sh

# Multi-node — override env vars (inference uses NNODES; see Tips & Troubleshooting).
NNODES=4 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
    bash scripts/infer/gpu/infer_osp_14b.sh
```

Outputs:

```
${output_dir}/
├── config.yaml            # a snapshot of the launch config (rank 0)
├── video_0.mp4            # one mp4 per prompt, named after its prompt index
├── video_1.mp4
├── ...
└── video_grid.mp4         # NxN tiled preview of all generated clips
```

---

## 🏋️ Training Pipeline

Two entry points are provided:

| Entry                   | Purpose                                | Optimizer target            |
|-------------------------|----------------------------------------|-----------------------------|
| `train/train_osp.py`    | Supervised fine-tuning (SFT)           | Full-parameter / FSDP2      |
| `train/train_osp_RL.py` | Mix-GRPO RL post-training w/ LoRA      | LoRA adapters only          |

### 📚 Data Preparation

> ℹ️  **SFT only.** The RL pipeline (`train/train_osp_RL.py`) consumes a plain
> text prompt file (one prompt per line) and **does not need this step**. See
> the [RL section](#-reinforcement-learning-mix-grpo--lora) for that format.

SFT reads training videos from an **LMDB-backed meta store**. Building it is a
three-step pipeline:

#### Step 1 — Write a meta JSON for every video corpus

For each batch of training videos, produce a JSON file describing each clip:

```jsonc
[
  {
    "path": "path/to/a/video.mp4",             // 🔧 required — video file path
    "cap":  "A stylish woman walks down ...",  // 🔧 required — caption
    "resolution": {"height": 1080, "width": 1920},  // optional, auto-probed if absent
    "fps": 24,                                  // optional, auto-probed if absent
    "num_frames": 81,                           // optional, auto-probed if absent
    "cut": [0, 81]                              // optional — [start, end) frame
                                                // indices when the JSON points
                                                // to a sub-clip of a long video
  },
  {
    "path": "...",
    "cap":  "..."
  }
]
```

You can have many such JSON files — one per corpus / per source.

#### Step 2 — Write the annotation index (`ann_txt`)

Create a `.txt` index that tells the filter where each meta JSON lives and what
its videos' root directory is. **One line per JSON, format:**

```text
<videos_root_dir>,<absolute_path_to_meta_json>
```

For example, `all_videos.txt`:

```text
/data/video_corpus_A,/data/video_corpus_A/meta.json
/data/video_corpus_B,/data/video_corpus_B/meta.json
```

The filter will prepend `<videos_root_dir>` to each relative `path` field
inside the corresponding JSON.

#### Step 3 — Configure the filter and build the LMDB

Edit `configs/filter_config.yaml`:

```yaml
ann_txt_path: "all_videos.txt"           # 🔧 the index from Step 2
save_path:    "/path/to/train/dataset"   # 🔧 destination LMDB folder
sample_height:     720                   # videos will be filtered to fit this
sample_width:      1280
sample_num_frames: 81
train_fps:         16                    # target training fps
min_hxw:           921600                # min H×W; reject anything smaller
                                         #   1080×1920 → 2_073_600 (use 2_000_000)
                                         #    864×1536 → 1_327_104
                                         #    720×1280 →   921_600
                                         #    576×1024 →   589_824
                                         #    480×832  →   399_360
max_h_div_w_ratio: 1.2                   # reject overly portrait videos
min_h_div_w_ratio: 0.4                   # reject overly landscape videos
max_motion_value:  0.02                  # reject overly static / overly shaky
```

Then run:

```bash
bash scripts/filter_data.sh
# equivalent to:
#   python filter_data.py --filter_config configs/filter_config.yaml
```

This produces an LMDB at `save_path`, which becomes the actual dataset
consumed by `train/train_osp.py`. **Two things** must be flipped in your
training config to use it (the shipped configs default to the random-tensor
debug dataset — see the callout below):

```yaml
data_config:
  dataset_name: "wan_t2v"                              # ← real LMDB dataset
  dataset_config:
    metafile_or_dir_path: "/path/to/train/dataset"    # ← Step-3 save_path
    ...
```

> 💡  **Why LMDB?** LMDB keeps memory usage flat during training and avoids the
> memory leaks that pile up when `decord` opens / closes thousands of video
> readers across DataLoader workers.

> 🧪 **`t2v_random` vs `wan_t2v` — pick the right one**
>
> All shipped `configs/train/**.yaml` files set `dataset_name: "t2v_random"`,
> which is a **synthetic random-tensor dataset** (`T2VRandomDataset`) used to
> smoke-test the training loop *without any real data on disk* — convenient
> for verifying that FSDP2 / SP / the optimizer step are all wired up
> correctly. For an actual training run you **must**:
>
> 1. Build the LMDB through this Data Preparation pipeline (Steps 1-3 above).
> 2. Set `data_config.dataset_name: "wan_t2v"` (this picks `WanT2VDataset`).
> 3. Set `data_config.dataset_config.metafile_or_dir_path` to the LMDB folder.
>
> Forgetting any of these silently trains the model on random noise — loss
> will look "fine" but the model learns nothing.

### 🎓 Supervised Fine-Tuning (SFT)

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
ema_decay: 0.9999                        # GPU default; NPU 14B recipe uses 0.999, NPU 1.3B uses 0.9993
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
  # ⚠️  "t2v_random" is a synthetic random-tensor dataset — only use it to
  #    smoke-test the loop. For real training, switch to "wan_t2v" and set
  #    metafile_or_dir_path to the LMDB built in Data Preparation Step 3.
  dataset_name: "t2v_random"             # 🔧 change to "wan_t2v" for real training
  dataset_config:
    text_tokenizer_path: "/path/to/text_tokenizer"   # 🔧
    # metafile_or_dir_path: "/path/to/train/dataset" # 🔧 REQUIRED for wan_t2v
    text_drop_ratio: 0.1
    sample_height: 720
    sample_width: 1280
    sample_num_frames: 81
    tokenizer_max_length: 512
    return_prompt_mask: True
  sampler_name: "stateful_distributed"
  collator_name: "wan_t2v"               # collator stays "wan_t2v" for both modes

optimizer_config:
  lr: 0.00002
  weight_decay: 0
```

#### 🚀 Step 2 — Launch

| Backend | Model           | Script                                       | Config                                  |
|---------|-----------------|----------------------------------------------|-----------------------------------------|
| GPU     | 14B             | `scripts/train/gpu/train_osp_14b.sh`         | `configs/train/gpu/osp_14b.yaml`        |
| GPU     | 1.3B            | `scripts/train/gpu/train_osp_1_3b.sh`        | `configs/train/gpu/osp_1_3b.yaml`       |
| NPU     | 14B             | `scripts/train/npu/train_osp_14b.sh`         | `configs/train/npu/osp_14b.yaml`        |
| NPU     | 1.3B            | `scripts/train/npu/train_osp_1_3b.sh`        | `configs/train/npu/osp_1_3b.yaml`       |
| NPU     | HiF8 14B ‡      | *(copy `train_osp_14b.sh` ↓)*                | `configs/train/npu/osp_hif8_14b.yaml`   |
| NPU     | HiF8 1.3B ‡     | *(copy `train_osp_1_3b.sh` ↓)*               | `configs/train/npu/osp_hif8_1_3b.yaml`  |

> ‡ &nbsp;HiF8 SFT does **not** ship its own launch script — copy
> `scripts/train/npu/train_osp_14b.sh` to `train_osp_hif8_14b.sh` and only
> change the `--config` flag to the HiF8 yaml (e.g.
> `--config configs/train/npu/osp_hif8_14b.yaml`). All other env vars / FSDP
> settings carry over. Make sure the HiF8 NPU kernel imports cleanly first
> (see [NPU setup step 4](#4-optional--rebuild-the-hif8-npu-quant-kernel-only-on-import-error)).

```bash
# Single node (default NPRC_PER_NODE=8)
bash scripts/train/gpu/train_osp_14b.sh

# Multi-node — training scripts read PET_NNODES + RANK (NOT NNODES / NODE_RANK)
PET_NNODES=4 RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29501 \
    bash scripts/train/gpu/train_osp_14b.sh
# … on every other node, bump RANK accordingly: RANK=1, RANK=2, RANK=3
```

Resuming is automatic — `Checkpointer.last_training_iteration` picks the most
recent checkpoint folder under `output_dir`.

### 🎯 Reinforcement Learning (Mix-GRPO + LoRA)

> 🥇 &nbsp;**First RL pipeline for sparse video diffusion.** To the best of our
> knowledge, OSP-Next is the first project to apply RL post-training directly to
> a *sparse* video diffusion model — see the [paper](<ARXIV_URL>) for the
> design rationale.

The RL post-training uses the same FSDP2 backbone but trains a **LoRA** adapter
on top of frozen base weights, sampled with **SDE → ODE hybrid** denoising,
optimized with **Mix-GRPO** against a **VideoAlign reward**.

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
reshard_after_forward: Null
explicit_prefetching_num_blocks: 0
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
profiling: False

wandb_config:
  project_name: "osp_next_RL"
  exp_name: "osp_next_RL"

model_config:
  # ↓ keep dim / num_heads / num_layers / skiparse_* identical to your SFT
  #   config — only the LoRA adapter is being trained, the base must match.
  dim: 5120
  ffn_dim: 13824
  freq_dim: 256
  in_dim: 16
  num_heads: 40
  num_layers: 40
  out_dim: 16
  text_len: 512
  skiparse_model_type: "dual_end"
  sparse_ratio: 2
  num_full_blocks: 8
  pretrained_model_dir_or_checkpoint: "/path/to/model"   # 🔧 base ckpt

scheduler_config:
  scheduler_name: "flow_matching"
  use_dynamic_shifting: True
  use_logitnorm_time_sampling: True

vae_config:
  vae_path: "/path/to/vae"               # 🔧
  dtype: "fp16"                          # RL uses fp16 (rollout-only VAE saves VRAM); SFT/infer use fp32

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
  num_inference_steps: 25                # total denoising steps in rollout sampling
  guidance_scale: 5.0                    # CFG scale used during rollout sampling
  kl_beta: 0.004                         # KL penalty weight (set 0 to disable)
  num_batches_per_epoch: 4               # batches per RL epoch
  num_image_per_prompt: 4                # k repeats (Mix-GRPO group size)
  sample_time_per_prompt: 1              # how many times each prompt is rolled out per epoch
  sample_batch_size: 2                   # batch size during rollout
  train_batch_size:  2                   # batch size during policy update
  eval_num_steps: 50                     # denoising steps used in the eval pass
  eval_freq: 20                          # run eval every N RL epochs
  use_cfg_in_train: True                 # apply CFG in the policy update too
  adv_clip_max: 5.0                      # max abs value for advantage clipping
  clip_range: 1e-4                       # PPO-style ratio clip range
  reward_fn:
    videoalign: 1.0                      # 🔧 VideoAlign weight (1.0 → only reward used);
                                         #    set the actual checkpoint path through the
                                         #    `load_from_pretrained` kwarg in
                                         #    ospnext/rewards/rewards.py :: multi_score(),
                                         #    otherwise scorer init will fail at startup.
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

The RL trainer **only persists the LoRA adapter** — the base model is frozen,
so we deliberately skip saving its weights to keep checkpoints small and
deployable. Each save (`save_interval` epochs + a final save) produces:

```
${output_dir}/
├── lora-checkpoint-10/                   # 🎯 current LoRA (deployable)
│   ├── adapter_model.bin                 # LoRA matrices only
│   ├── adapter_config.json               # PEFT LoRA config
│   ├── adaptive_grad_clipper.pt          # grad-clipper EMA state (resume helper)
│   └── rl_training_state.json            # epoch / global_step bookkeeping
└── lora-checkpoint-10-ema/               # 🎯 EMA-averaged LoRA (recommended for inference)
    ├── adapter_model.bin
    └── adapter_config.json
```

> 💡 Use `lora-checkpoint-{step}-ema/` for inference (matches what's used during
> in-training eval). Use the plain `lora-checkpoint-{step}/` if you want to
> resume RL training — point `lora_config.lora_path` at it to pick up the LoRA
> weights and the sidecar `rl_training_state.json` / grad-clipper state.

#### 🔗 Step 3 — Merge LoRA back into the base model

OSP-Next inference (`infer/infer_osp.py`) loads a **plain (merged) base model**,
not a `PeftModel`. After RL training finishes, run `merge_lora_weights.py` to
fold the LoRA delta into the frozen base weights and save a single
deployment-ready checkpoint:

```python
# merge_lora_weights.py — edit the four paths at the bottom and run once.
from ospnext.modules.osp_next import OSPNextModel
from merge_lora_weights import load_lora_and_merge

model_path = "/path/to/osp_next_base"                         # 🔧 same base used during RL
lora_path  = "/path/to/output_dir/lora-checkpoint-1000-ema/adapter_model.bin"  # 🔧 prefer the -ema variant
save_path  = "/path/to/merged_osp_next_rl"                    # 🔧 destination

model = OSPNextModel.from_pretrained(model_path)
model = load_lora_and_merge(
    model=model,
    lora_path=lora_path,
    lora_rank=32,                  # must match lora_config.rank used in RL
    lora_alpha=64,                 # must match lora_config.alpha used in RL
    lora_target_modules=[
        "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
        "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
    ],
)
model.save_pretrained(save_path)
```

Or just edit the four paths at the bottom of `merge_lora_weights.py` directly
and run:

```bash
python merge_lora_weights.py
```

The script will (1) wrap the base with the same PEFT `LoraConfig`, (2) load
the trained LoRA weights, (3) call `peft.merge_and_unload()` to fold LoRA into
the base, and (4) `save_pretrained()` the merged model.

#### 🎬 Step 4 — Run inference with the merged model

Point your inference config's `pretrained_model_dir_or_checkpoint` at the
merged directory and launch as usual:

```yaml
# configs/infer/{gpu,npu}/osp_14b.yaml
model_config:
  pretrained_model_dir_or_checkpoint: "/path/to/merged_osp_next_rl"   # 🔧
```

```bash
bash scripts/infer/gpu/infer_osp_14b.sh           # or the NPU variant
```

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



### 🐛 Common failures

| Symptom                                             | Fix                                                                                                  |
|-----------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `ImportError: cannot import name 'flash_attn'`      | Either install flash-attn manually, or ignore — code already falls back to SDPA.                     |
| `RuntimeError: NPU error ... aclrtSetDevice`        | Forgot `source /usr/local/Ascend/ascend-toolkit/set_env.sh` before activating the conda env.         |
| NPU op missing / `aclnnXxx not found` at runtime    | Your CANN toolkit is older than the required 8.5.0 — re-install from <https://www.hiascend.com/cann/download>. |
| `pip install decord` fails on Ascend / aarch64       | No prebuilt wheel exists for aarch64 + Py3.10 — build from source (see [step 3 of the NPU setup](#3-build-decord-from-source-aarch64-only)). |
| `ImportError: libdecord.so: cannot open shared object` | Add the decord `build/` directory to `LD_LIBRARY_PATH` (see the callout right under the decord build steps). |
| `wandb` prompts for login                            | Either run `wandb login` once, or set `WANDB_MODE=offline` (every training script already does this). |
| `from ospnext.quant_cy_npu import ...` fails        | Bundled kernel ABI mismatch — rebuild from upstream HiFloat8 and swap the package in (see [step 4 of the NPU setup](#4-optional--rebuild-the-hif8-npu-quant-kernel-only-on-import-error)). |
| RL reward init crashes on startup                   | `multi_score` in `ospnext/rewards/rewards.py` currently calls `videoalign_score(device)` without a checkpoint path — wire the actual path through `load_from_pretrained=` and re-run.       |

### 📚 Environment variables worth knowing

All shipped launch scripts (`scripts/{infer,train}/{gpu,npu}/*.sh`) read these
variables via `${VAR:-default}`, so you can override them inline on the command
line without touching the scripts.

| Variable                      | Used by              | Default                       | Purpose                                                            |
|-------------------------------|----------------------|-------------------------------|--------------------------------------------------------------------|
| `MASTER_ADDR`                 | infer + train        | `127.0.0.1`                   | torchrun rendezvous host                                            |
| `MASTER_PORT`                 | infer + train        | `29505` (infer) / `29501` (train) | torchrun rendezvous port                                        |
| `NPRC_PER_NODE` ⚠️             | infer + train        | `8`                           | Processes per node (see typo note below)                            |
| `NNODES`                      | **infer only**       | `1`                           | Total nodes — used by the inference scripts                         |
| `PET_NNODES`                  | **train only**       | `1`                           | Total nodes — used by the training scripts (legacy `pet`-style name) |
| `RANK`                        | **train only**       | `0`                           | This node's rank (`0` for single-node, `0..N-1` for multi-node)     |
| `WANDB_MODE`                  | train                | `offline` (preset in scripts) | Set to `online` to enable WandB upload, or keep `offline`           |
| `PYTORCH_NPU_ALLOC_CONF`      | NPU train            | `expandable_segments:True`    | NPU memory allocator (preset in NPU scripts)                        |

> ⚠️  **`NPRC_PER_NODE` is a typo**, but it's the variable the launch scripts
> actually look for. If you set `NPROC_PER_NODE` (the conventional spelling)
> it will be **silently ignored** and the script will fall back to `8`. We
> kept the typo to preserve backward-compat with existing run histories — a
> proper rename is tracked as a follow-up. **Always override with
> `NPRC_PER_NODE=...`** until that is fixed.

> 🔀 **Inference vs training use different multi-node vars.** Inference scripts
> read `NNODES`, training scripts read `PET_NNODES` + `RANK`. See the
> "single / multi-node" snippets in the
> [Inference Pipeline](#-inference-pipeline) and
> [Training Pipeline](#-training-pipeline) sections for copy-paste examples.

---

## 🙏 Acknowledgements

OSP-Next stands on the shoulders of giants. We gratefully build on:

- 🌊 [**Wan**](https://github.com/Wan-Video/Wan2.1) — the WAN-VAE and T5 backbone
  components that power our text-to-video stack.
- 🎬 [**Open-Sora-Plan**](https://github.com/PKU-YuanGroup/Open-Sora-Plan) — the open-source
  video diffusion ecosystem this project directly extends.
- 🏅 [**VideoAlign**](https://github.com/KwaiVGI/VideoAlign) — the multi-axis video-quality
  reward model used during RL post-training.
- 🎯 [**Mix-GRPO**](https://arxiv.org/abs/2507.21802) — the mixed ODE-SDE flow-matching RL
  algorithm at the heart of our sparse-model post-training pipeline.

We also welcome contributions of every size — bug reports, feature requests, and
PRs all go a long way! Please file an [issue](https://github.com/PKU-YuanGroup/OSP-Next/issues)
or open a pull request.

---

## 📝 Citation

If you find OSP-Next useful in your research, please consider citing:

```bibtex
@article{ge2026ospnext,
  title        = {OSP-Next: Efficient High-Quality Video Generation with Sparse Sequence Parallelism, HiF8 Quantization, and Reinforcement Learning},
  author       = {Ge, Yunyang and He, Xianyi and Zhang, Zezhong and Lin, Bin and Zhu, Bin and Cheng, Xinhua and Yuan, Li},
  journal      = {arXiv preprint arXiv:<ARXIV_ID>},
  year         = {2026},
  eprint       = {<ARXIV_ID>},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV},
  url          = {<ARXIV_URL>},
}
```

> 🔁 The two `<ARXIV_ID>` placeholders (in `journal` and `eprint`) need to be
> filled with the same arXiv number, e.g. `2606.12345`.

Related work this project builds on:

```bibtex
@article{wan2025wan,
  title={Wan: Open and advanced large-scale video generative models},
  author={Wan, Team and Wang, Ang and Ai, Baole and Wen, Bin and Mao, Chaojie and Xie, Chen-Wei and Chen, Di and Yu, Feiwu and Zhao, Haiming and Yang, Jianxiao and others},
  journal={arXiv preprint arXiv:2503.20314},
  year={2025}
}

@article{lin2024open,
  title={Open-sora plan: Open-source large video generation model},
  author={Lin, Bin and Ge, Yunyang and Cheng, Xinhua and Li, Zongjian and Zhu, Bin and Wang, Shaodong and He, Xianyi and Ye, Yang and Yuan, Shenghai and Chen, Liuhan and others},
  journal={arXiv preprint arXiv:2412.00131},
  year={2024}
}

@article{li2025mixgrpo,
  title={Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde},
  author={Li, Junzhe and Cui, Yutao and Huang, Tao and Ma, Yinping and Fan, Chun and Cheng, Yiming and Yang, Miles and Zhong, Zhao and Bo, Liefeng},
  journal={arXiv preprint arXiv:2507.21802},
  year={2025}
}
```

---

## 📄 License

See [`LICENSE.txt`](LICENSE.txt).

---

## ⭐ Star History

<div align="center">

<a href="https://star-history.com/#PKU-YuanGroup/OSP-Next&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)"  srcset="https://api.star-history.com/svg?repos=PKU-YuanGroup/OSP-Next&type=Date&theme=dark">
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=PKU-YuanGroup/OSP-Next&type=Date">
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=PKU-YuanGroup/OSP-Next&type=Date" width="720">
  </picture>
</a>

<br/>

<sub>If this project helped you, a ⭐ goes a long way 🙌</sub>

</div>

