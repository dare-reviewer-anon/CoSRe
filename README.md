Good catch — there are **three concrete Markdown bugs** that make this README fail to render:

### What’s broken (precisely)

1. **Duplicate section header**

   ```md
   ## 1. Repository Structure
   At the top level:
   ## 1. Repository Structure
   ```

   → breaks structure and confuses renderers.

2. **Unclosed code block**
   Your repository tree block:

   ````md
   ```text
   CoSRe/
   ...
   └── README.md
   ````

   ❌ **missing closing triple backticks** before `---`.

3. **Mismatched fence at the end**

   * You open with ```markdown
   * You close with ```
     → GitHub treats everything as code.

---

Below is a **fully corrected, clean, GitHub-renderable README**.
You can **copy–paste this verbatim**.

---

## ✅ **FIXED & IMPROVED README.md (RENDERABLE)**

````markdown
# CoSRe: Cosine-Compressed Shared–Residual Factorization for Efficient Multimodal Spatial Reasoning

This repository contains the implementation of **CoSRe** (Cosine-Compressed Shared–Residual Factorization), built on top of the **Anole** multimodal LLM and the **MVoT** interleaved visual–text reasoning framework.

CoSRe enables **efficient multi-hop multimodal reasoning** by:
- Compressing intermediate visual thoughts in the **cosine domain**
- Caching **shared semantics once**
- Preserving **lightweight modality-specific residual evidence**

CoSRe is primarily an **inference-time KV-cache management method**:
- It does **not** modify transformer attention kernels
- It can be applied **without retraining the backbone**

It includes:
- A baseline Anole training script (`traino.py`) with DeepSpeed ZeRO-3
- CoSRe modules (`model_utils/cosre/`) for cosine compression and shared–residual factorization
- A CoSRe driver (`train_CoSRe.py`) enabling CoSRe decoding during evaluation/prediction
- Utilities and configs for spatial reasoning benchmarks (Maze / MiniBehavior / FrozenLake)

---

## 1. Repository Structure

At the top level:

```text
CoSRe/
├── cfg/                       # YAML / config files (model, data, optimization, deepspeed, etc.)
├── model_utils/
│   ├── cosre/                 # Core CoSRe implementation
│   │   ├── cosine_codec.py    # Blockwise cosine transform + quantization + reconstruction
│   │   ├── shared_residual.py # Shared-slot extraction and modality residual formation
│   │   ├── controller.py      # CoSRe controller / decoding interfaces
│   │   ├── hooks.py           # Optional generation hooks
│   │   └── __init__.py
│   ├── logging.py             # Logging helpers
│   └── wrapped_visualizer.py  # Optional visualization / debugging utilities
├── prompt/                    # Prompt templates and instruction formats
├── utils/                     # Common utilities (data loading, evaluation)
├── traino.py                  # Baseline Anole training (no CoSRe)
├── train_CoSRe.py             # CoSRe driver (baseline + CoSRe decoding)
├── traino.sh                  # Baseline launcher
├── train_CoSRe.sh             # CoSRe launcher
├── requirements.txt           # Full dependency list
├── requirements_clean.txt     # Minimal dependency list
└── README.md                  # This file
````

---

## 2. Environment Setup

We recommend **conda** with **Python 3.10**.

```bash
conda create -n CoSRe python=3.10 -y
conda activate CoSRe

pip install torch==2.4.0
pip install -r requirements.txt --user
# or minimal:
# pip install -r requirements_clean.txt --user
```

**Hardware:** Experiments are typically run on multi-GPU systems (e.g., **4× A100 40GB**).
NCCL and DeepSpeed must be available for `torchrun`.

---

## 3. Data Preparation

After unpacking `data-samples.zip`:

```text
~/Datasets/data-samples/
├── frozenlake-datasets/
├── maze-datasets/
└── minibehavior-dataset/
```

Set:

```bash
DATA_ROOT=~/Data/datasets/data-samples
```

### Maze datasets

```text
${DATA_ROOT}/maze-datasets/
├── grid3/
├── grid3_paths.json
├── grid4/
├── grid4_paths.json
├── grid5/
├── grid5_paths.json
├── grid6/
└── grid6_paths.json
```

* `grid*/` contain rendered frames and metadata
* `*_paths.json` store ground-truth trajectories

Processed datasets are referenced via:

```bash
--data interleaved_maze
--data_dir ${DATA_ROOT}
```

Expected layout:

```text
${DATA_ROOT}/interleaved_maze/
├── train.jsonl
├── val.jsonl
├── images/
└── ...
```

---

## 4. Step 1 – Baseline Anole Training (No CoSRe)

```bash
torchrun --nproc_per_node=4 traino.py \
  --model anole \
  --data interleaved_maze \
  --data_dir ${DATA_ROOT} \
  --decoder_type anole \
  --do_train \
  --do_eval \
  --cfg_path cfg \
  --output outputs/anole7b_zero3_4gpusoutput \
  --report_to none \
  --train_bz 2 \
  --val_bz 2 \
  --grad_acc 32 \
  --image_seq_length 1024 \
  --note "anole7b_zero3_4gpus_"
```

Checkpoint produced at:

```text
outputs/anole7b_zero3_4gpusoutput/
```

---

## 5. Step 2 – CoSRe Evaluation / Prediction

Enable CoSRe during inference:

```bash
torchrun --nproc_per_node=4 train_CoSRe.py \
  --model anole \
  --data interleaved_maze \
  --data_dir ${DATA_ROOT} \
  --decoder_type anole \
  --image_seq_length 1024 \
  --input_format anole \
  --do_eval \
  --cfg_path cfg \
  --output outputs/cosre-anole7b-maze \
  --note "cosre-maze-" \
  --report_to none \
  --model_ckpt outputs/anole7b_zero3_4gpusoutput \
  --load_last_checkpoint \
  --enable_CoSRe \
  --cosre_block 8 \
  --cosre_keep_h 4 \
  --cosre_keep_w 4 \
  --cosre_slots 32 \
  --cosre_base_delta 0.5
```

### CoSRe-specific flags

* `--enable_CoSRe`
  Enable CoSRe decoding.

* `--cosre_block`
  Cosine block size `b`.

* `--cosre_keep_h`, `--cosre_keep_w`
  Low-frequency spatial retention per block.

* `--cosre_slots`
  Number of shared semantic slots `K`.

* `--cosre_base_delta`
  Base quantization step.

> Tip: start conservatively (`keep_h=6 keep_w=6 base_delta=0.25`) and tighten.

---

## 6. Implementation Notes

### Integration point

CoSRe operates at **generation time**:

* `CustomizeSeq2SeqTrainer` switches decoding when `enable_CoSRe=True`, or
* `VisualizationEvaluator` calls a CoSRe decoding wrapper

### KV-cache behavior

* **Training:** `use_cache=False` (memory efficient)
* **Eval / Predict:** `use_cache=True` (CoSRe compresses cache growth)

---

## 7. License & Acknowledgements

Built on:

* **Anole** multimodal LLM
* **MVoT** interleaved visual–text reasoning framework

<!-- Please cite the corresponding works when using this repository.

````

---

### ✅ Summary of fixes
- ✔ Removed duplicate headers  
- ✔ Closed **all** code blocks  
- ✔ Removed outer ```markdown fence  
- ✔ GitHub-safe, Markdown-compliant  

If you want, next I can:
- Align this README **exactly** to your real `model_utils/` files
- Generate a **diff vs DARE README** (for NeurIPS appendix)
- Tighten language for **camera-ready submission**
```` -->
