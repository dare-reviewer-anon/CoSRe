```markdown
# CoSRe: Cosine-Compressed Shared–Residual Factorization for Efficient Multimodal Spatial Reasoning

This repository contains the implementation of **CoSRe** (Cosine-Compressed Shared–Residual Factorization), built on top of the **Anole** multimodal LLM and the **MVoT** interleaved visual–text reasoning framework.  
CoSRe is designed to make **multi-hop multimodal reasoning** efficient by **compressing intermediate visual thoughts in the cosine domain** and **caching shared semantics once** while preserving **lightweight modality-specific residual evidence**.

CoSRe is primarily an **inference-time KV-cache management method**:
- It does **not** require modifying transformer attention kernels.
- It can be applied **without retraining the backbone** (optionally you may fine-tune lightweight components if your implementation supports it).

It includes:
- A baseline Anole training script (`traino.py`) with DeepSpeed ZeRO-3.
- CoSRe modules (`model_utils/cosre/`) for blockwise cosine compression and shared–residual factorization.
- A CoSRe launcher (`train_CoSRe.py`) that trains the baseline normally and enables **CoSRe decoding** during evaluation/prediction.
- Configuration files and utilities to reproduce experiments on dynamic spatial reasoning tasks (Maze / MiniBehavior / FrozenLake-style).

---

## 1. Repository Structure

At the top level:

```

CoSRe/
├── cfg/                       # YAML / config files (model, data, optimization, deepspeed, etc.)
├── model_utils/
│   ├── cosre/                 # Core CoSRe implementation
│   │   ├── cosine_codec.py    # Blockwise cosine transform + quantization + reconstruction
│   │   ├── shared_residual.py # Shared-slot extraction and modality residual formation
│   │   ├── controller.py      # CoSRe controller / interfaces used by decoding
│   │   ├── hooks.py           # Optional hooks to integrate CoSRe into generation
│   │   └── **init**.py
│   ├── logging.py             # Logging helpers
│   └── wrapped_visualizer.py  # Optional visualization / debugging utilities
├── prompt/                    # Prompt templates and instruction formats
├── utils/                     # Common utilities (data loading, evaluation, helpers)
├── traino.py                  # Baseline Anole training (no CoSRe), with DeepSpeed ZeRO-3
├── train_CoSRe.py             # CoSRe driver: baseline training + CoSRe decoding in eval/predict
├── traino.sh                  # Example launcher for baseline training
├── train_CoSRe.sh             # Example launcher for CoSRe runs
├── requirements.txt           # Full dependency list
├── requirements_clean.txt     # Minimal / cleaned dependency list
└── README.md                  # This file

````

---

## 2. Environment Setup

We recommend using **conda** and **Python 3.10**.

```bash
# 1) Create & activate environment
conda create -n CoSRe python=3.10 -y
conda activate CoSRe

# 2) Install PyTorch (adjust CUDA tags for your cluster)
pip install torch==2.4.0

# 3) Install remaining dependencies
pip install -r requirements.txt --user
# or minimal:
# pip install -r requirements_clean.txt --user
````

**Hardware:** Experiments are typically run on multi-GPU machines (e.g., **4× A100 40GB**). You need NCCL working for `torchrun` and DeepSpeed (installed via `requirements.txt`).

---

## 3. Data Preparation

We provide a small portion of each dataset used in experiments as a single archive, e.g. `data-samples.zip`.
After unpacking, the layout looks like:

```
~/Datasets/data-samples/
├── frozenlake-datasets/
├── maze-datasets/
└── minibehavior-dataset/
```

We refer to this folder as:

```bash
DATA_ROOT=~/Data/datasets/data-samples
```

### Maze datasets

The raw Maze data is stored under:

```
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

* `grid3/`, `grid4/`, `grid5/`, `grid6/` contain per-instance assets (frames, metadata).
* `*_paths.json` contain the ground-truth shortest paths and trajectory-level annotations.

These raw datasets are used to build **interleaved multimodal sequences** (image + text thoughts) consumed by **Anole + MVoT**.
In this codebase, the conversion from `maze-datasets/` into training-ready JSONL / HF format is handled by preprocessing scripts under `utils/` and dataset-specific scripts.

The final processed dataset is referenced via:

```bash
--data interleaved_maze
--data_dir ${DATA_ROOT}
```

i.e., the training scripts will look for:

```
${DATA_ROOT}/interleaved_maze/
    train.jsonl
    val.jsonl
    images/
    ...
```

---

## 4. Step 1 – Baseline Anole Training (No CoSRe)

The first stage is to fine-tune Anole on the provided datasets **without CoSRe**.
This uses `traino.py` and DeepSpeed ZeRO-3.

Example (4 GPUs):

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

Key arguments:

* `--model anole` uses the Anole backbone (see configs in `cfg/`).
* `--data interleaved_maze` selects the interleaved Maze dataset.
* `--data_dir ${DATA_ROOT}` points to where you unpacked `data-samples.zip`.
* `--image_seq_length 1024` sets the max number of visual tokens per example.
* `--output ...` where checkpoints and logs are written.

After this step finishes, your baseline checkpoint will be in:

```
outputs/anole7b_zero3_4gpusoutput/
```

---

## 5. Step 2 – CoSRe Evaluation / Prediction (CoSRe Decoding Enabled)

CoSRe is mainly an **inference-time** method. Conceptually, enabling CoSRe decoding:

1. **Compresses intermediate visual token grids** via **blockwise cosine compression** (transform → quantize → retain low-frequency coefficients → reconstruct a smaller grid).
2. Builds a compact memory using **shared–residual factorization**:

   * Cache **shared semantics** once (K latent slots).
   * Cache only **modality-specific residual evidence** for text and vision.
3. Updates the decoding context / KV-cache with the compact memory instead of full dense thoughts.

### Run CoSRe (4 GPUs)

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
  Enables CoSRe decoding / KV-cache compression during eval/predict.

* `--cosre_block`
  Block size `b` for blockwise cosine compression (e.g., `8`).

* `--cosre_keep_h`, `--cosre_keep_w`
  Amount of low-frequency content retained per block (larger values preserve quality but use more tokens).

* `--cosre_slots`
  Number of shared semantic slots `K`.

* `--cosre_base_delta`
  Base quantization step for frequency-dependent quantization.

> Tip: start conservative (less aggressive compression): `keep_h=6 keep_w=6 base_delta=0.25`, then tighten.

---

## 6. Optional: Training + CoSRe Eval in One Script

If you want to train baseline and then evaluate with CoSRe in the same run:

```bash
torchrun --nproc_per_node=4 train_CoSRe.py \
  --model anole \
  --data interleaved_maze \
  --data_dir ${DATA_ROOT} \
  --decoder_type anole \
  --input_format anole \
  --do_train \
  --do_eval \
  --cfg_path cfg \
  --output outputs/cosre-anole7b-maze \
  --note "cosre-maze-" \
  --report_to none \
  --train_bz 2 \
  --val_bz 2 \
  --grad_acc 32 \
  --image_seq_length 1024 \
  --enable_CoSRe \
  --cosre_block 8 \
  --cosre_keep_h 4 \
  --cosre_keep_w 4 \
  --cosre_slots 32 \
  --cosre_base_delta 0.5
```

---

## 7. Implementation Notes

### Where CoSRe integrates

CoSRe typically plugs into **generation** (evaluation/prediction), not supervised training:

* `CustomizeSeq2SeqTrainer` switches decoding when `args.enable_CoSRe=True`, or
* `VisualizationEvaluator` calls a CoSRe decoding wrapper.

### KV-cache behavior

* During **training**, KV cache is often disabled (`model.config.use_cache=False`) to save memory.
* During **CoSRe eval/predict**, KV cache should be enabled (`use_cache=True`) because CoSRe’s benefit is reducing cached token growth.

---

## 8. Reproducing Results

The default workflow:

1. **Train baseline** with `traino.py`.
2. **Evaluate with CoSRe decoding** using `train_CoSRe.py --enable_CoSRe ...`.

You can repeat this for other datasets by replacing `--data interleaved_maze` with your dataset name (e.g., `interleaved_minibehavior`, `interleaved_frozenlake`) provided your preprocessing produces the expected `train.jsonl/val.jsonl/test.jsonl` structure.

---

## 9. License and Acknowledgements

This code is built on top of:

* **Anole** multimodal LLM
* **MVoT** interleaved visual–text reasoning pipeline

Please cite the corresponding works when using this repository.

```

If you want, I can also generate a **repo-accurate file tree** (with your exact filenames) once you paste `ls -R` for `model_utils/` and `utils/`, and I’ll align the README to what actually exists (so no placeholder modules like `cosine_codec.py` unless you want them).
```
