# Complete Pipeline Scripts

This folder contains end-to-end pipeline scripts for reproducing all four methods in the paper.

## Overview

| Script | Method | Preprocessing? | Training Time | GPU Required |
|--------|--------|---------------|---------------|--------------|
| `run_bart_full_pipeline.sh` | BART Baseline | No | ~4 hours | Yes (1x A100) |
| `run_bart_yake_full_pipeline.sh` | BART + YAKE | Minimal | ~4 hours | Yes (1x A100) |
| `run_yake_dbpedia_full_pipeline.sh` | OpenBioLaySumm | Yes (hours) | ~4 hours | Yes (1x A100) |
| `run_umls_full_pipeline.sh` | UMLS (Goldsack) | Yes (hours) | ~2 hours | Yes (1x A100) |

## Quick Start

### 1. BART Baseline (Simplest)
```bash
bash scripts/pipelines/run_bart_full_pipeline.sh
```

**No preprocessing needed!** Uses dataset splits directly.

---

### 2. BART + YAKE
```bash
bash scripts/pipelines/run_bart_yake_full_pipeline.sh
```

**Minimal preprocessing:** Keywords are prepended to abstracts.

---

### 3. OpenBioLaySumm (YAKE + DBpedia) - **Our Method**

**Prerequisites:**
- Ollama with DeepSeek-R1:7b: `ollama pull deepseek-r1:7b`
- Internet connection (DBpedia API)
```bash
bash scripts/pipelines/run_yake_dbpedia_full_pipeline.sh
```

**Full preprocessing included:** Keyword extraction, DBpedia querying, graph construction.

⚠️ **Note:** First run takes several hours for preprocessing. Script prompts whether to run preprocessing or use existing data.

---

### 4. UMLS (Goldsack Reproduction)

**Prerequisites:**
1. Obtain UMLS license: https://www.nlm.nih.gov/research/umls/
2. Download UMLS 2023AA
3. Place in project root as `2023AA/`
4. Install QuickUMLS: `pip install quickumls`
5. Index UMLS: `python -m quickumls.install /path/to/2023AA ./QuickUMLS_index`
```bash
bash scripts/pipelines/run_umls_full_pipeline.sh
```

**Full preprocessing included:** UMLS concept extraction, similarity computation, graph construction.

---

## Running on HPC (SLURM)

For Deucalion or other SLURM systems:
```bash
# Wrap in SLURM job
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=bart_pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=6:00:00
#SBATCH --mem=32G

bash scripts/pipelines/run_bart_full_pipeline.sh
EOF
```

---

## Skip Preprocessing (Training Only)

If you already have preprocessed data:

**YAKE+DBpedia:**
```bash
# Edit script, set SKIP_PREPROCESSING=true
# Or run training directly:
bash scripts/training/run_train_gpu.sh
```

**UMLS:**
```bash
# Edit script, respond 'n' to preprocessing prompt
# Or run training directly:
bash scripts/training/run_train_umls_gpu.sh
```

---

## Comparing All Methods

After running all pipelines:
```bash
# Compare results
python analysis/visualize_results.py \
    --results results/bart_baseline_results.json \
             results/bart_yake_results.json \
             results/yake_dbpedia_results.json \
             results/umls_results.json
```

---

## Troubleshooting

**"Ollama not found"**
- Install: https://ollama.ai/
- Pull model: `ollama pull deepseek-r1:7b`
- Start: `ollama serve`

**"UMLS data not found"**
- See main README for UMLS setup instructions

**"Out of memory"**
- Reduce batch_size in script
- Use smaller GPU or CPU (slower)

**"ModuleNotFoundError"**
- Install requirements: `pip install -r requirements.txt`