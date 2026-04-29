# OpenBioLaySumm: Biomedical Lay Summarization with YAKE and DBpedia

<!--[![Paper](https://img.shields.io/badge/Paper-CL4Health%202026-blue)](link-to-paper)-->
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

This repository contains the implementation of **OpenBioLaySumm**, a novel approach for biomedical lay summarization that combines YAKE keyword extraction with DBpedia knowledge graphs and Graph Attention Networks (GAT).

**Paper:** "OpenBioLaySumm: An Open-Resource Alternative to UMLS for Biomedical Lay Summarization using YAKE and DBpedia" (CL4Health 2026)

**Authors:** João Pedro Veloso and Evelin Amorim, INESC TEC & University of Porto

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Complete Pipelines](#complete-pipelines)
- [Datasets](#datasets)
- [Methods Comparison](#methods-comparison)
- [Manual Reproduction](#manual-reproduction)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

Biomedical lay summarization aims to make complex scientific articles accessible to general audiences. Our work compares four approaches:

1. **BART Baseline** - Standard sequence-to-sequence model
2. **BART + YAKE** - BART with keyword augmentation
3. **OpenBioLaySumm (YAKE + DBpedia)** - Our proposed method using open resources
4. **UMLS Reproduction** - Goldsack et al.'s approach using licensed UMLS data

### Key Innovation

OpenBioLaySumm achieves **97% of UMLS performance** while using only **open-access resources** (YAKE + DBpedia), eliminating the need for costly UMLS licenses.

---

## ✨ Key Features

- ✅ **Open-source alternative to UMLS** - No license required
- ✅ **Four complete implementations** - All baselines and our method
- ✅ **End-to-end pipelines** - One command to reproduce each approach
- ✅ **Graph-based enhancement** - GAT networks for concept relationships
- ✅ **LLM-based classification** - DeepSeek-R1 for biomedical concept filtering
- ✅ **Comprehensive evaluation** - ROUGE, BLEU, readability metrics
- ✅ **HPC-ready** - SLURM scripts for cluster deployment

---

## 🚀 Quick Start

### Prerequisites
```bash
# Clone repository
git clone https://github.com/your-org/OpenBioLaySumm.git
cd OpenBioLaySumm

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Pipeline (Choose One)
```bash
# 1. BART Baseline (simplest, ~4 hours)
bash scripts/pipelines/run_bart_full_pipeline.sh

# 2. BART + YAKE (~4 hours)
bash scripts/pipelines/run_bart_yake_full_pipeline.sh

# 3. OpenBioLaySumm - Our Method (~8-12 hours first run)
bash scripts/pipelines/run_yake_dbpedia_full_pipeline.sh

# 4. UMLS Reproduction (~8-12 hours, requires UMLS license)
bash scripts/pipelines/run_umls_full_pipeline.sh
```

See [Complete Pipelines](#complete-pipelines) for detailed instructions.

---

## 📦 Installation

### 1. Basic Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt
```

### 2. Additional Requirements for Each Method

#### BART & BART+YAKE (No additional setup)
```bash
pip install transformers torch yake
```

#### OpenBioLaySumm (YAKE + DBpedia)
```bash
# Install Ollama for LLM classification
# Visit: https://ollama.ai/

# Pull DeepSeek-R1 model
ollama pull deepseek-r1:7b

# Start Ollama server
ollama serve
```

#### UMLS Reproduction (Requires License)

**⚠️ UMLS License Required**

1. **Obtain UMLS License** (free but requires registration):
   - Visit: https://www.nlm.nih.gov/research/umls/
   - Sign up for UTS account
   - Accept license agreement

2. **Download UMLS 2023AA**:
   - Download full UMLS release
   - Extract to project root as `2023AA/`

3. **Install QuickUMLS**:
```bash
   pip install quickumls
   
   # Index UMLS data (takes ~30 minutes)
   python -m quickumls.install 2023AA/ ./QuickUMLS_index
```

4. **Update paths** in `src/data_preprocessing/extract_umls_concepts.py`:
```python
   mrconso_path = "2023AA/META/MRCONSO.RRF"
   quickumls_fp = "./QuickUMLS_index"
```

---

## 🔬 Complete Pipelines

We provide end-to-end scripts that handle preprocessing, training, and evaluation for each method.

### Pipeline Overview

| Method | Script | Preprocessing | Training Time | GPU |
|--------|--------|--------------|---------------|-----|
| BART Baseline | `run_bart_full_pipeline.sh` | ❌ None | ~4 hours | A100 |
| BART + YAKE | `run_bart_yake_full_pipeline.sh` | ✅ Minimal | ~4 hours | A100 |
| OpenBioLaySumm | `run_yake_dbpedia_full_pipeline.sh` | ✅ Full | ~8-12 hours | A100 |
| UMLS | `run_umls_full_pipeline.sh` | ✅ Full | ~8-12 hours | A100 |

### Running Pipelines

#### 1. BART Baseline
```bash
bash scripts/pipelines/run_bart_full_pipeline.sh
```

**What it does:**
- Trains BART on train_filtered.json
- Evaluates on test.json
- Saves results to `results/bart_baseline_results.json`

**No preprocessing needed** - uses dataset splits directly.

---

#### 2. BART + YAKE
```bash
bash scripts/pipelines/run_bart_yake_full_pipeline.sh
```

**What it does:**
- Prepends YAKE keywords to abstracts
- Trains BART on augmented data
- Evaluates on test.json
- Saves results to `results/bart_yake_results.json`

**Minimal preprocessing** - keyword prepending only.

---

#### 3. OpenBioLaySumm (YAKE + DBpedia) - **Our Method**
```bash
bash scripts/pipelines/run_yake_dbpedia_full_pipeline.sh
```

**What it does:**
1. **Keyword Extraction** - YAKE extracts biomedical concepts
2. **DBpedia Querying** - Fetches descriptions for each keyword
3. **LLM Classification** - DeepSeek-R1 filters biomedical concepts
4. **Similarity Computation** - SciBERT encodes concepts
5. **Graph Construction** - Builds concept graphs with GAT
6. **Training** - Trains graph-enhanced BART model
7. **Evaluation** - Tests on held-out set

**Prerequisites:**
- Ollama with DeepSeek-R1:7b
- Internet connection (DBpedia API)

**First run:** 8-12 hours (includes preprocessing)  
**Subsequent runs:** ~4 hours (uses cached preprocessing)

The script prompts whether to run preprocessing or use existing data.

---

#### 4. UMLS (Goldsack Reproduction)
```bash
bash scripts/pipelines/run_umls_full_pipeline.sh
```

**What it does:**
1. **UMLS Extraction** - QuickUMLS extracts medical concepts
2. **Similarity Computation** - SciBERT encodes concepts
3. **Graph Construction** - Builds UMLS concept graphs
4. **Training** - Trains UMLS-enhanced model
5. **Evaluation** - Tests on held-out set

**Prerequisites:**
- UMLS 2023AA license and data
- QuickUMLS installed and indexed

**First run:** 8-12 hours (includes preprocessing)  
**Subsequent runs:** ~2 hours (uses cached preprocessing)

---

### Running on HPC (SLURM)

For Deucalion or other SLURM clusters:
```bash
# Submit BART baseline
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=bart_pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/bart_%j.out

module load python/3.8
source venv/bin/activate

bash scripts/pipelines/run_bart_full_pipeline.sh
EOF
```
```bash
# Submit OpenBioLaySumm
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=yake_dbpedia
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/yake_dbpedia_%j.out

module load python/3.8
source venv/bin/activate

bash scripts/pipelines/run_yake_dbpedia_full_pipeline.sh
EOF
```


---

## 📊 Datasets

### eLife Lay Summarization Dataset

**Source:** Goldsack et al. (2022) "Making Science Simple"  
**Paper:** https://aclanthology.org/2022.emnlp-main.724/  
**Original Repository:** https://github.com/TGoldsack1/Corpora_for_Lay_Summarisation

### Dataset Splits

| Split | Articles | Description |
|-------|----------|-------------|
| **train_filtered.json** | 1,582 | Training set (filtered for graph availability) |
| **val.json** | 241 | Validation set |
| **test.json** | 241 | Test set |
| **train_yake_bart.json** | 1,582 | Training set with prepended keywords |

### Data Format

Each JSON file contains articles with:
```json
{
  "id": "article_id",
  "title": "Article Title",
  "abstract": ["Abstract paragraph 1", "Abstract paragraph 2"],
  "summary": ["Lay summary paragraph 1", "Lay summary paragraph 2"],
  "sections": [["Section 1 text"], ["Section 2 text"]]
}
```

### Downloading Original Dataset
```bash
python src/data_preprocessing/download_datasets.py
```

See `data/README.md` for detailed dataset documentation.

---

## 🔍 Methods Comparison

| Method | Resources | Graph-Based | Performance (ROUGE-1) | License Required |
|--------|-----------|-------------|----------------------|------------------|
| **BART Baseline** | None | ❌ No | Baseline | ❌ No |
| **BART + YAKE** | YAKE | ❌ No | +2.3% | ❌ No |
| **OpenBioLaySumm** | YAKE + DBpedia | ✅ GAT | +4.1% | ❌ No |
| **UMLS (Goldsack)** | UMLS 2023AA | ✅ GAT | +4.2% | ✅ **Yes** |

### Key Finding

**OpenBioLaySumm achieves 97% of UMLS performance using only open resources!**

---

## 🔧 Manual Reproduction

If you prefer step-by-step control instead of complete pipelines:

### OpenBioLaySumm (YAKE + DBpedia)
```bash
# 1. Extract keywords and query DBpedia
mkdir -p data/yakepreprocess/
python src/data_preprocessing/yake_preprocess.py 1 1582 --input-file train_filtered.json --output_folder yake_preprocess
python src/data_preprocessing/yake_preprocess.py --split val
python src/data_preprocessing/yake_preprocess.py --split test

# 2. Compute concept similarities (edge weights)
python src/data_preprocessing/compute_dbpedia_similarities.py --split train
python src/data_preprocessing/compute_dbpedia_similarities.py --split val
python src/data_preprocessing/compute_dbpedia_similarities.py --split test

# 3. Build graph PKLs
python src/data_preprocessing/create_pkls.py

# 4. Train model
python src/training/train.py \
    --config configs/train_config.json \
    --train data/train_filtered.json \
    --val data/val.json \
    --epochs 20

# 5. Evaluate
python src/evaluation/evaluate.py \
    --model_dir models/yake_dbpedia \
    --test data/test.json
```

### BART Baseline
```bash
# Train
python src/training/train_bart.py \
    --config configs/train_config_bart.json \
    --train data/train_filtered.json \
    --val data/val.json

# Evaluate
python src/evaluation/evaluate_bart.py \
    --model_dir models/bart_baseline \
    --test data/test.json
```

### BART + YAKE
```bash
# application of YAKE and searching in DBPedia API
mkdir -p data/yakepreprocess/
python src/data_preprocessing/yake_preprocess.py 1 1582 --input-file data/train_filtered.json --output-folder /data/yakepreprocess/

# Preprocess
python src/data_preprocessing/create_yake_bart_data.py

# Train
python src/training/train_yake_bart.py \
    --config configs/train_config_yake_bart.json \
    --train data/train_yake_bart.json \
    --val data/val.json

# Evaluate
python src/evaluation/evaluate_bart.py \
    --model_dir models/bart_yake \
    --test data/test.json
```

### UMLS Reproduction
```bash
# 1. Extract UMLS concepts (requires QuickUMLS)
python src/data_preprocessing/extract_umls_concepts.py --split train
python src/data_preprocessing/extract_umls_concepts.py --split val
python src/data_preprocessing/extract_umls_concepts.py --split test

# 2. Compute concept similarities
python src/data_preprocessing/compute_umls_similarities.py --split train
python src/data_preprocessing/compute_umls_similarities.py --split val
python src/data_preprocessing/compute_umls_similarities.py --split test

# 3. Build UMLS graph PKLs
python src/data_preprocessing/PKLS_from_UMLS.py

# 4. Train
python src/training/train_umls.py \
    --config configs/train_config_umls.json \
    --train data/train_filtered.json \
    --val data/val.json

# 5. Evaluate
python src/evaluation/evaluate_umls.py \
    --model_dir models/umls \
    --test data/test.json
```

---

## 📈 Results

### Main Results (Test Set)

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|--------|---------|---------|---------|------|
| BART Baseline | 45.2 | 18.3 | 28.7 | 12.4 |
| BART + YAKE | 47.5 | 19.8 | 30.1 | 13.9 |
| **OpenBioLaySumm** | **49.3** | **21.2** | **31.8** | **15.1** |
| UMLS (Goldsack) | 49.5 | 21.4 | 32.0 | 15.3 |

### Key Insights

1. **OpenBioLaySumm vs UMLS:** 97% performance without license requirements
2. **Graph Enhancement:** +4.1% ROUGE-1 over BART baseline
3. **Open Resources:** DBpedia proves competitive with UMLS
4. **Keyword Augmentation:** YAKE alone improves BART by +2.3%


---

## 📄 Citation

If you use this code or dataset, please cite:
```bibtex
@inproceedings{souza2026openbiolaysumm,
  title={OpenBioLaySumm: An Open-Resource Alternative to UMLS for Biomedical Lay Summarization using YAKE and DBpedia},
  author={Veloso, João Pedro, and Amorim, Evelin},
  booktitle={Proceedings of the Clinical NLP Workshop (CL4Health) at LREC 2026},
  year={2026},
    publisher={European Language Resources Association (ELRA)},
note={Co-located with LREC-COLING 2026}
}
```

**Original eLife Dataset:**
```bibtex
@inproceedings{goldsack2022making,
  title={Making Science Simple: Corpora for the Lay Summarisation of Scientific Literature},
  author={Goldsack, Tomas and Zhang, Zhihao and Lin, Chenghua and Scarton, Carolina},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  pages={10589--10604},
  year={2022}
}
```

---

## ⚠️ Data License Notice

### UMLS Data

This repository does **NOT** include UMLS (Unified Medical Language System) data due to license restrictions. UMLS is free to use but requires registration and cannot be redistributed.

**To reproduce the UMLS baseline:**
1. Obtain license: https://www.nlm.nih.gov/research/umls/
2. Download UMLS 2023AA
3. Follow setup instructions in [Installation](#installation)

### eLife Dataset

The eLife lay summarization dataset is openly available. We provide preprocessed splits in the `data/` folder.


---

## 🙏 Acknowledgments

- **eLife dataset:** Goldsack et al. (2022)
- **UMLS:** U.S. National Library of Medicine
- **DBpedia:** DBpedia Association
- **YAKE:** Campos et al. (2020)
- **Computing resources:** Deucalion supercomputer (MACC, University of Minho) through projects 2025.00013.AIvLAB.DEUCALION and 2025.00017.AIvLAB.DEUCALION, funded by EuroHPC Joint Undertaking and FCT Portugal
- **Funding:** COST Action CA23147 GOBLIN (Global Network on Large-Scale, Cross-domain and Multilingual Open Knowledge Graphs)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note:** While our code is MIT licensed, please respect:
- UMLS license terms if using UMLS data
- eLife dataset license
- Third-party library licenses


---

## 📚 Additional Resources

### Documentation
- [Pipeline Scripts Documentation](scripts/pipelines/README.md)
<!-- - [Dataset Documentation](data/README.md)-->
<!--- [Preprocessing Guide](src/data_preprocessing/README.md)-->

### Tutorials
- [Getting Started with OpenBioLaySumm](docs/getting_started.md) *(coming soon)*
- [Using DBpedia for Concept Graphs](docs/dbpedia_guide.md) *(coming soon)*
- [Adapting to New Datasets](docs/new_datasets.md) *(coming soon)*


<!--**Paper:** [Link to CL4Health 2026 proceedings]-->

---

<div align="center">
  <p>Made with ❤️ by the LIAAD team at INESC TEC</p>
  <p>
    <a href="https://www.inesctec.pt/">INESC TEC</a> •
    <a href="https://www.up.pt/">University of Porto</a> •
    <a href="https://liaad.github.io/">LIAAD</a>
  </p>
</div>
