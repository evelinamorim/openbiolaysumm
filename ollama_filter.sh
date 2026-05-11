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
module load ollama/0.20.3-GCCcore-14.2.0-CUDA-12.8.0

source /projects/F202600026AIVLABDEUCALION/evelinamorim/venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export HF_HOME="$(pwd)/hf_cache"
export OLLAMA_MODELS=/projects/F202600026AIVLABDEUCALION/evelinamorim/ollama_models/

# Start Ollama on the compute node (has GPU)
ollama serve &
OLLAMA_PID=$!

# Wait until ready
echo "Waiting for Ollama..."
for i in $(seq 1 30); do
    curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1 && echo "Ollama ready." && break
    sleep 2
done

ollama pull deepseek-r1:7b

python src/data_preprocessing/ollama_filter.py --input-file /projects/F202600026AIVLABDEUCALION/evelinamorim/openbiolaysumm/data/yakepreprocess/processed_train_filtered_bigrams.json --output-folder /projects/F202600026AIVLABDEUCALION/evelinamorim/openbiolaysumm/data/ollamafilter/

kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null

echo "Completed: $(date)"
