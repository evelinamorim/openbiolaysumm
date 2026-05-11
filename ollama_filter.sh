#!/bin/bash
#SBATCH --job-name=ollama_filter
#SBATCH --account=f202500017aivlabdeucaliong
#SBATCH --exclusive
#SBATCH --gpus=1
#SBATCH --partition=normal-a100-40
#SBATCH --time=16:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/ollama_filter_%j.out
#SBATCH --error=logs/ollama_filter_%j.err

echo "=========================================="
echo "Ollama Filter - Biomedical Classification"
echo "Started: $(date)"
echo "=========================================="

module load Python/3.11.3-GCCcore-12.3.0

source /projects/F202600026AIVLABDEUCALION/evelinamorim/venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export HF_HOME=/projects/F202600026AIVLABDEUCALION/evelinamorim/hf_cache

python src/data_preprocessing/ollama_filter.py \
    --input-file data/yakepreprocess/processed_train_filtered_bigrams.json \
    --output-folder data/yakepreprocess/ \
    --batch-size 32
