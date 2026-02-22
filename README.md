# BTBR: Bayesian-Theory-Based Bias Removal — Reproduction Guide

> **Abstract.** > Large language models (LLMs) often encapsulate knowledge they do not explicitly recognize, including subtle manifestations of biases internalized from heterogeneous training corpora. This phenomenon—what LLMs "do not know they know"—raises fundamental challenges for fairness and interpretability. We define the implicit bias problem within a fuzzy-systems framework and propose **Bayesian-Theory-based Bias Removal (BTBR)**. BTBR applies likelihood ratio screening to identify entries in biased datasets, automatically constructs semantically relevant knowledge triples, and removes bias via targeted model editing (e.g., MEMIT, EMMET). Extensive experiments demonstrate that BTBR substantially reduces bias effects while maintaining general capability.

---

## Table of Contents

* [Overview](#overview)
* [Hardware & Model Requirements](#hardware--model-requirements)
* [Setup & Installation](#setup--installation)
* [Credential Configuration](#credential-configuration)
* [Step 1 — Fine-Tune the Biased Model (BAdam)](#step-1--fine-tune-the-biased-model-badam)
* [Step 2 — Configure Model Editor](#step-2--configure-model-editor)
* [Step 3 — Run the BTBR Pipeline](#step-3--run-the-btbr-pipeline)
* [Outputs & File Structure](#outputs--file-structure)
* [Tips & Troubleshooting](#tips--troubleshooting)
* [Acknowledgments](#acknowledgments)

---

## Overview

To reproduce the experiments:

1. **Fine-tune a "Biased" proxy model** on your bias dataset using the memory-efficient BAdam optimizer.
2. **Compute DB(x) Likelihood Scores** for each data point by comparing the base model and the biased model.
3. **Fuzzy Alpha-Cut Selection** to dynamically select the most representative biased samples ($K_{edit}$).
4. **Extract Knowledge Triples** (Subject-Relation-Object) using an LLM API.
5. **Apply Model Editing** (EMMET / MEMIT) to explicitly remove these traces from the model's weights.

> **Recommendation**
> Use **FP16** precision for both BAdam fine-tuning and model editing. BAdam allows 8B models to be fine-tuned on a single consumer GPU (e.g., RTX 3090/4090).

---

## Hardware & Model Requirements

* **GPU**: FP16 support required.
* **Typical single-GPU VRAM** (approx.):

| Phase | Model Size | VRAM Needed | Notes |
| ----- | ---------- | ----------- | ----- |
| BAdam Fine-Tuning | Llama-3 8B | ~24 GB | Mixed-precision, block coordinate descent. |
| DB(x) Scoring | Llama-3 8B | ~32 GB | Requires alternating loads if VRAM < 32GB. |
| Model Editing | Llama-3 8B | ~30-40 GB | Memory-intensive due to covariance matrices. |

---

## Setup & Installation

We highly recommend using `uv` for blazing-fast dependency resolution and installation.

```bash
# 1. Create a virtual environment using uv
uv venv btbr_env

# 2. Activate the environment
# On Linux/macOS:
source btbr_env/bin/activate
# On Windows:
# btbr_env\Scripts\activate

# 3. Install dependencies
uv pip install -r requirements.txt

```

---

## Credential Configuration

BTBR uses an LLM API (e.g., OpenAI GPT-4o) to accurately extract Subject-Relation-Object (SRO) triples from biased text.

You must expose your API key as an environment variable. **Do not hardcode it in the scripts.**

**Linux / macOS:**

```bash
export OPENAI_API_KEY="sk-YOUR_API_KEY_HERE"

```

**Windows (PowerShell):**

```powershell
$env:OPENAI_API_KEY="sk-YOUR_API_KEY_HERE"

```

---

## Step 1 — Fine-Tune the Biased Model (BAdam)

To calculate the Bayesian evidence score (), you first need to generate a biased version of your base model. We use BAdam to do this efficiently.

Ensure your `bias_dataset.json` is located in the `./data/` folder, then run:

```bash
python train_badam.py

```

This script will output the fine-tuned model to `./biased_llama3_model`, which will be used in the next step.

---

## Step 2 — Configure Model Editor

We utilize `EasyEdit` for modifying the LLM's weights. You must provide a strict YAML configuration file mapping out the model's architecture.

For **Llama-3-8B**, create the file at `./hparams/EMMET/Meta-Llama-3-8B-Instruct.yaml`.
*(Note: As established in our experiments, editing `layer 1` with a `batch_size: 1` yields the best stability).*

```yaml
# ./hparams/EMMET/Meta-Llama-3-8B-Instruct.yaml
model_name: "meta-llama/Meta-Llama-3-8B-Instruct"
model_class: AutoModelForCausalLM
tokenizer_class: AutoTokenizer
tokenizer_name: "meta-llama/Meta-Llama-3-8B-Instruct"
layers: [1]
v_num_grad_steps: 25
v_lr: 5e-1
v_loss_layer: 31
v_weight_decay: 1e-3
clamp_norm_factor: 4
kl_factor: 0.0625
mom2_adjustment: true
mom2_update_weight: 100000
rewrite_module_tmp: "model.layers.{}.mlp.down_proj"
layer_module_tmp: "model.layers.{}"
mlp_module_tmp: "model.layers.{}.mlp"
attn_module_tmp: "model.layers.{}.self_attn"
ln_f_module: "model.norm"
lm_head_module: "lm_head"
mom2_dataset: "wikipedia"
mom2_n_samples: 100000
mom2_dtype: "float32"
batch_size: 1
max_length: 512

```

---

## Step 3 — Run the BTBR Pipeline

With the biased model trained and the hyperparameters configured, you can execute the end-to-end pipeline:

```bash
python main.py

```

**What happens under the hood?**

1. Calculates  for both the base and biased models.
2. Computes the fuzzy  membership.
3. Applies a budgeted -cut to select the top  samples.
4. Extracts SRO triples via API.
5. Injects `none` placeholders into the target model via EMMET/MEMIT.
6. Saves the debiased model to `./btbr_edited_model`.

---

## Outputs & File Structure

```text
btbr_project/
├── data/
│   └── bias_dataset.json             # Input data
├── hparams/
│   ├── EMMET/
│   │   └── Meta-Llama-3-8B-Instruct.yaml # Editor configuration
├── train_badam.py                    # Step 1 script
├── config.py                         # Global parameters
├── data_loader.py                    
├── scorer.py                         # DB(x) Calculation
├── selector.py                       # Fuzzy Alpha-cut logic
├── extractor.py                      # LLM Triple Extraction
├── editor.py                         # EasyEdit wrapper
└── main.py                           # Orchestrator (Step 3)

```

---

## Tips & Troubleshooting

> **OOM / Out of Memory during Scoring**
> If your GPU has less than 32GB of VRAM, the `scorer.py` might crash when loading both the base and biased Llama-3 models. Modify `scorer.py` to instantiate `base_model`, calculate all its scores, `del base_model`, `torch.cuda.empty_cache()`, and *then* instantiate `biased_model`.

> **EasyEdit File Not Found**
> Ensure the name of your `.yaml` file exactly matches the base name of your `BASE_MODEL_NAME` in `config.py` (e.g., if the model is `meta-llama/Meta-Llama-3-8B-Instruct`, the file must be named `Meta-Llama-3-8B-Instruct.yaml`).

> **API Triple Extraction Failing**
> Check your console output. If the script prints *"Warning: No valid triples were extracted"*, your API key is either invalid, out of credits, or you are facing severe network/proxy blocking.

---

## Acknowledgments

Parts of our pipeline are heavily based on the excellent work from the open-source community. We sincerely thank the authors and maintainers of:

* **[BAdam](https://github.com/Ledzy/BAdam)**: For enabling memory-efficient full-parameter optimization.
* **[EasyEdit](https://github.com/zjunlp/EasyEdit)**: For providing a robust, unified framework for LLM knowledge editing.




