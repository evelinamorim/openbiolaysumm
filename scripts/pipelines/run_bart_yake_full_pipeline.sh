#!/bin/bash
#
# Complete Pipeline: BART + YAKE
#
# This script runs the full pipeline for BART with YAKE keyword augmentation.
# Preprocessing pipeline:
#   1. yake_preprocess.py     — YAKE keyword extraction + DBpedia enrichment
#                               (LOGIN NODE, requires internet)
#   2. ollama_filter.py       — Biomedical classification via DeepSeek-R1
#                               (SLURM NODE, requires GPU + Ollama)
#   3. create_yake_bart_data.py — Formats data for BART training
#                               (LOGIN NODE)
#
# Usage:
#   Login node  : bash scripts/pipelines/run_bart_yake_full_pipeline.sh --preprocess
#   SLURM node  : bash scripts/pipelines/run_bart_yake_full_pipeline.sh --filter
#   Full train  : sbatch scripts/pipelines/run_bart_yake_full_pipeline.sh
#
# Note: --preprocess and --filter must be completed before submitting the
#       SLURM training job.
#

set -e

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EPOCHS=20
BATCH_SIZE=4
LEARNING_RATE=2e-6

INPUT_FILE=../data/train_filtered.json
YAKE_OUTPUT=../data/yakepreprocess
BIOMEDICAL_FILE=${YAKE_OUTPUT}/processed_train_filtered_biomedical.json
BART_DATA=data/train_yake_bart.json

OLLAMA_BIN=PUT_YOUR_OLLAMA_BINARY_HERE
OLLAMA_MODELS=PUT_YOUR_OLLAMA_DIR_MODEL_HERE

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
PREPROCESS=false
FILTER=false

for arg in "$@"; do
    case $arg in
        --preprocess) PREPROCESS=true ;;
        --filter)     FILTER=true ;;
    esac
done

# ---------------------------------------------------------------------------
# Step 1: YAKE + DBpedia (login node)
# ---------------------------------------------------------------------------
if [ "$PREPROCESS" = true ]; then
    echo "================================================"
    echo "Step 1: YAKE + DBpedia Preprocessing"
    echo "Run this on the LOGIN NODE (requires internet)"
    echo "================================================"

    mkdir -p $YAKE_OUTPUT

    BIGRAMS_FILE=${YAKE_OUTPUT}/processed_train_filtered_bigrams.json
    if [ ! -f "$BIGRAMS_FILE" ]; then
        echo "Running YAKE + DBpedia extraction..."
        python src/data_preprocessing/yake_preprocess.py 1 1582 \
            --input-file $INPUT_FILE \
            --output-folder $YAKE_OUTPUT
    else
        echo "✓ DBpedia bigrams file already exists, skipping."
    fi

    echo ""
    echo "Step 1 complete. Next: run with --filter on a SLURM node."
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 2: Ollama biomedical filter (SLURM node with GPU)
# ---------------------------------------------------------------------------
if [ "$FILTER" = true ]; then
    echo "================================================"
    echo "Step 2: Ollama Biomedical Filter"
    echo "Run this on a SLURM NODE (requires GPU)"
    echo "================================================"

    BIGRAMS_FILE=${YAKE_OUTPUT}/processed_train_filtered_bigrams.json
    if [ ! -f "$BIGRAMS_FILE" ]; then
        echo "Error: bigrams file not found: $BIGRAMS_FILE"
        echo "Run --preprocess first on the login node."
        exit 1
    fi

    if [ ! -f "$BIOMEDICAL_FILE" ]; then
        echo "Starting Ollama..."
        export OLLAMA_MODELS=$OLLAMA_MODELS
        $OLLAMA_BIN serve &
        OLLAMA_PID=$!

        echo "Waiting for Ollama to be ready..."
        for i in $(seq 1 30); do
            curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1 && echo "Ollama ready." && break
            sleep 2
        done

        python src/data_preprocessing/ollama_filter.py \
            --input-file $BIGRAMS_FILE \
            --output-folder $YAKE_OUTPUT

        kill $OLLAMA_PID
        wait $OLLAMA_PID 2>/dev/null
    else
        echo "✓ Biomedical file already exists, skipping."
    fi

    echo ""
    echo "Step 2 complete. Next: run the full SLURM training job."
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 3 onwards: Training pipeline (SLURM job)
# ---------------------------------------------------------------------------
echo "================================================"
echo "BART + YAKE - FULL PIPELINE"
echo "================================================"

# --- Step 3: Create BART-ready data ---
echo ""
echo "Step 3: Creating YAKE+BART training data..."
if [ ! -f "$BART_DATA" ]; then
    if [ ! -f "$BIOMEDICAL_FILE" ]; then
        echo "Error: biomedical file not found: $BIOMEDICAL_FILE"
        echo "Run --preprocess and --filter steps first."
        exit 1
    fi
    python src/data_preprocessing/create_yake_bart_data.py \
        --input $INPUT_FILE \
        --yake-biomedical $BIOMEDICAL_FILE \
        --output $BART_DATA
else
    echo "  ✓ $BART_DATA already exists, skipping."
fi

# --- Step 4: Train ---
echo ""
echo "Step 4: Training BART+YAKE..."
echo "  - Epochs       : $EPOCHS"
echo "  - Batch size   : $BATCH_SIZE"
echo "  - Learning rate: $LEARNING_RATE"
echo ""

python src/training/train_yake_bart.py \
    --config configs/train_config_yake_bart.json \
    --train $BART_DATA \
    --val data/val.json \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --output_dir models/bart_yake

# --- Step 5: Evaluate ---
echo ""
echo "Step 5: Evaluating on Test Set..."
python src/evaluation/evaluate_bart.py \
    --model_dir models/bart_yake \
    --test data/test.json \
    --output_file results/bart_yake_results.json

echo ""
echo "================================================"
echo "✓ BART+YAKE Pipeline Complete!"
echo "================================================"
echo "Results saved to: results/bart_yake_results.json"