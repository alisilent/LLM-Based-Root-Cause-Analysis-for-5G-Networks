# LLM-Based Root Cause Analysis for 5G Networks

**Qwen2.5-7B-Instruct · QLoRA · Network Diagnostics · Unseen Root-Cause Generalization**

> **Research status: Initial version — ongoing.** This repository presents our first set of experiments on LLM-assisted root cause analysis (RCA) for 5G throughput degradation. We are extending this work to more realistic, diverse, and challenging datasets, with the aim of developing more reliable methods for identifying both known and previously unseen network faults.

## Overview

Diagnosing throughput degradation in 5G networks requires interpreting radio measurements, mobility indicators, interference conditions, and cell configuration parameters. This project investigates how an instruction-tuned large language model can use structured network features and textual root-cause definitions to identify the likely cause of a performance issue.

The repository contains **two complementary experiments**:

1. **Eight-class RCA:** Fine-tune Qwen2.5-7B-Instruct with 4-bit quantization and Low-Rank Adaptation (QLoRA) to classify incidents into eight predefined root causes.
2. **Unseen root-cause generalization:** Train on six classes and test whether supplying descriptions of two additional classes **at inference time** enables the model to identify those previously unseen classes. This experiment includes a Random Forest baseline, prompt ablations, and a supplementary few-shot Random Forest study.

The two experiments share the same data, fixed split assignments, and 18-feature input representation, but **train separate QLoRA adapters**. The second notebook is not a continuation from the first notebook's trained checkpoint.

## Experiments

### 1. Eight-class RCA with QLoRA

**Notebook:** [`01_qwen25_7b_lora_5g_rca.ipynb`](01_qwen25_7b_lora_5g_rca.ipynb)

- Uses **Qwen2.5-7B-Instruct** with structured prompts describing the network measurements and eight candidate fault definitions.
- Represents each incident using **18 selected diagnostic features** from a set of 69 engineered features.
- Applies **4-bit NF4 quantization** and **LoRA fine-tuning** on labeled incidents.
- Evaluates both the base model and the fine-tuned model, and reports accuracy, macro-F1, balanced accuracy, and per-class metrics.

**Recorded results:**

| Model | Split | Incidents | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: | ---: |
| Base Qwen2.5-7B-Instruct | Validation | 339 | 11.21% | 0.0252 |
| Qwen2.5-7B + QLoRA | Validation | 339 | **98.82%** | **0.9880** |
| Qwen2.5-7B + QLoRA | Test | 334 | **98.80%** | **0.9877** |

These results measure performance on the **eight known classes in this dataset**. They do not establish generalization to unknown fault categories or to other network environments.

### 2. Definition-guided generalization to unseen root causes

**Notebook:** [`02_qwen25_7b_unseen_root_cause_generalization.ipynb`](02_qwen25_7b_unseen_root_cause_generalization.ipynb)

The model is fine-tuned using only six root causes: **C1, C2, C4, C5, C6, and C8**. The two withheld causes, **C3 and C7**, are excluded from the LLM's supervised fine-tuning. At inference time, the candidate set expands to all eight classes, with the new classes introduced through natural-language definitions.

This notebook includes:

- A **six-class Random Forest** baseline using the same 18 features.
- A **base Qwen2.5-7B** reference evaluated on withheld incidents before task-specific fine-tuning.
- A separately trained **six-class QLoRA** model, evaluated on both seen and unseen classes.
- An ablation comparing inference prompts **with root-cause descriptions** against **labels only**.
- A supplementary Random Forest adaptation experiment with **1, 5, 10, 20, or 50 labeled examples per new class** from the original training partition.

**Recorded QLoRA results:**

| Evaluation | Incidents | Accuracy |
| --- | ---: | ---: |
| Seen classes — validation | 243 | **98.35%** |
| Unseen classes (C3/C7) — validation, with definitions | 96 | **0.00%** |
| Unseen classes (C3/C7) — validation, labels only | 96 | **0.00%** |
| Seen classes — test | 241 | **100.00%** |
| Unseen classes (C3/C7) — test | 93 | **0.00%** |

On the unseen-class validation subset, **base Qwen achieved 19.79% accuracy before fine-tuning**. The six-class Random Forest obtained 0% unseen-class accuracy because it cannot predict a label absent from its training set. With additional labeled examples, a retrained Random Forest achieved **68.75% accuracy with one example per new class** and **96.88% with 50 per class** on that validation subset; these are *few-shot* results, not zero-shot results.

**Key finding:** In this initial experiment, strong performance on familiar root causes did **not** transfer to the two new root causes. Providing their definitions at inference time was insufficient for successful recognition after six-class QLoRA training. The negative result motivates the next stage of the research; it should not be interpreted as evidence of successful unseen-class generalization.

## Dataset

All experiments operate on **2,400 labeled 5G incidents** spanning eight root-cause categories. The feature table contains **69 engineered diagnostic features** per incident; the notebooks use the same **18 preselected features** covering signal quality, antenna configuration, mobility, neighboring-cell relationships, throughput changes, and resource scheduling. The feature-selection procedure itself is not reproduced in these notebooks.

| File | Contents |
| --- | --- |
| [`all_2400_incidents_69_features.csv`](all_2400_incidents_69_features.csv) | Incident IDs and 69 engineered numerical features |
| [`split_assignments.csv`](split_assignments.csv) | Incident IDs, grouping information, and fixed train/validation/test assignments |
| [`train.csv`](train.csv) | Original incident questions and `answer` labels (C1–C8); the notebooks read only the `ID` and `answer` columns |

The label file is named `train.csv`, but it contains labels for **all 2,400 incidents**. The notebooks use `split_assignments.csv` to determine which incidents enter training versus validation and testing.

| Split | Incidents |
| --- | ---: |
| Training | 1,727 |
| Validation | 339 |
| Test | 334 |
| **Total** | **2,400** |

The eight labels correspond to issues involving downtilt, overshooting coverage, a better neighboring-cell connection, overlapping coverage, handovers, PCI-related interference, excessive vehicle speed, and insufficient scheduled resource blocks.

## Repository structure

```text
llm-5g-root-cause-analysis/
├── README.md
├── requirements.txt
├── 01_qwen25_7b_lora_5g_rca.ipynb
├── 02_qwen25_7b_unseen_root_cause_generalization.ipynb
├── all_2400_incidents_69_features.csv
├── split_assignments.csv
└── train.csv
```

The notebooks and datasets deliberately reside in the **same directory**, matching the notebooks' existing `DATA_DIR = Path(".")` configuration. Their saved execution outputs remain available for browsing without retraining.

## Setup and reproduction

### 1. Environment

The notebooks were originally run with **Python 3.11**, **CUDA**, and an **NVIDIA GeForce RTX 3080 Ti (12 GB class)** using 4-bit QLoRA. GPU memory and runtime requirements can vary with software versions and configuration. CUDA-enabled PyTorch and a supported NVIDIA GPU are required by the model-loading cells; CPU-only execution is not supported by the current notebooks.

Create a Python environment and install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

On Windows, use `.venv\Scripts\activate` instead of the Unix activation command. If needed, install a **CUDA-compatible PyTorch build** according to the [official PyTorch instructions](https://pytorch.org/get-started/locally/) before installing the other dependencies.

### 2. Obtain the base model

Download **[`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)** locally and update the `MODEL_NAME` variable near the start of **each** notebook to point to its local directory. For example:

```python
MODEL_NAME = "/absolute/path/to/Qwen2.5-7B-Instruct"
```

**Important:** The notebooks currently contain the original machine-specific model path (`/home/azaddel/models/Qwen2.5-7B-Instruct`) and call `from_pretrained(..., local_files_only=True)`. The model is **not included** in this repository. Leaving the original path unchanged will cause model loading to fail on other machines.

### 3. Run the notebooks

Start Jupyter from the repository's **root directory** so the current working directory contains all three CSVs. Open either notebook and run its cells in order:

1. [`01_qwen25_7b_lora_5g_rca.ipynb`](01_qwen25_7b_lora_5g_rca.ipynb) — eight-class classification.
2. [`02_qwen25_7b_unseen_root_cause_generalization.ipynb`](02_qwen25_7b_unseen_root_cause_generalization.ipynb) — six-class training and unseen-class evaluation.

Both notebooks are **independent**. The second does not require the first notebook's adapter. Training can take a substantial amount of GPU time; opening the notebooks on GitHub does **not** require installing dependencies or rerunning them.

After execution, the notebooks save model adapters/checkpoints and prediction or metrics CSV files in the working directory. These generated artifacts are **not included** here.

## Limitations and ongoing work

This repository represents a **basic research implementation**, not a finished or deployed 5G diagnostic system. The current experiments establish a useful starting point while revealing important limitations:

- **Dataset scope:** Performance has been measured on a single structured incident dataset; results may not transfer to different operators, environments, network configurations, or fault distributions.
- **Generalization:** The current six-class QLoRA setup failed to recognize the two withheld fault categories, despite receiving their natural-language descriptions at inference time.
- **Feature dependence:** Several labels may be closely associated with individual engineered diagnostic features. In particular, the notebooks identify potentially strong class-specific behavior in `mob__serving_pci_change_rate_per_second`. Further feature-ablation and leakage/robustness checks are needed before drawing broad conclusions from the high known-class accuracy.
- **Evaluation breadth:** Additional out-of-distribution tests, stronger classical and LLM baselines, uncertainty analysis, and broader unseen-class protocols remain future work.

**Ongoing research directions** include:

1. Evaluating the approach on **larger, more realistic, and more complex 5G/6G network datasets**, including harder failure cases and distribution shifts.
2. Improving **generalization to new root causes**, rather than relying only on fixed-label supervised fine-tuning.
3. Investigating better **feature representations, prompting strategies, and adaptation methods** for robust network diagnosis.
4. Expanding comparisons and ablations to clarify **when and why LLMs help** relative to conventional machine-learning methods.

Results and implementations in this repository document the **current stage**; the items above are planned research directions, not completed claims.

## Data and sharing note

The supplied `train.csv` includes incident-level drive-test and engineering-parameter text, including location and cell identifiers. Before publishing these files in a public repository, verify that redistribution is permitted and that any sensitive network/location information has been appropriately handled. Dataset provenance and licensing are not established by the notebooks alone.
