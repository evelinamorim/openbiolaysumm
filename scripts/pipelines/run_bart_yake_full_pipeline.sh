#!/bin/bash
#
# Complete Pipeline: BART + YAKE
#
# This script runs the full pipeline for BART with YAKE keyword augmentation.
# Preprocessing: Keywords are prepended to abstracts
#
# Usage: bash scripts/pipelines/run_bart_yake_full_pipeline.sh
#

set -e

echo "================================================"
echo "BART + YAKE - FULL PIPELINE"
echo "================================================"

# Configuration
EPOCHS=20
BATCH_SIZE=4
LEARNING_RATE=2e-6

echo ""
echo "Step 1/3: Preprocessing - Creating YAKE+BART Data..."
echo "  (Prepending keywords to abstracts)"
echo ""



# Preprocess (if not already done)
if [ ! -f "data/train_yake_bart.json" ]; then
    python src/data_preprocessing/yake_preprocess.py 1 1582\
     --input-file ../data/train_filtered.json \
     --output-folder ../data/yakepreprocess/
    python src/data_preprocessing/create_yake_bart_data.py \
        --input data/train_filtered.json \
        --output data/train_yake_bart.json
else
    echo "  ✓ data/train_yake_bart.json already exists, skipping preprocessing"
fi

echo ""
echo "Step 2/3: Training BART+YAKE..."
echo "  - Epochs: $EPOCHS"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Learning rate: $LEARNING_RATE"
echo ""

# Train
python src/training/train_yake_bart.py \
    --config configs/train_config_yake_bart.json \
    --train data/train_yake_bart.json \
    --val data/val.json \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --output_dir models/bart_yake

echo ""
echo "Step 3/3: Evaluating on Test Set..."
echo ""

# Evaluate
python src/evaluation/evaluate_bart.py \
    --model_dir models/bart_yake \
    --test data/test.json \
    --output_file results/bart_yake_results.json

echo ""
echo "================================================"
echo "✓ BART+YAKE Pipeline Complete!"
echo "================================================"
echo "Results saved to: results/bart_yake_results.json"
echo ""