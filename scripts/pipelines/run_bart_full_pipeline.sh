#!/bin/bash
#
# Complete Pipeline: BART Baseline
# 
# This script runs the full pipeline for the BART baseline approach.
# No preprocessing needed - uses data splits directly.
#
# Usage: bash scripts/pipelines/run_bart_full_pipeline.sh
#

set -e  # Exit on error

echo "================================================"
echo "BART BASELINE - FULL PIPELINE"
echo "================================================"

# Configuration
EPOCHS=20
BATCH_SIZE=4
LEARNING_RATE=2e-6

echo ""
echo "Step 1/2: Training BART Baseline..."
echo "  - Epochs: $EPOCHS"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Learning rate: $LEARNING_RATE"
echo ""

# Train
python src/training/train_bart.py \
    --config configs/train_config_bart.json \
    --train data/train_filtered.json \
    --val data/val.json \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --output_dir models/bart_baseline

echo ""
echo "Step 2/2: Evaluating on Test Set..."
echo ""

# Evaluate
python src/evaluation/evaluate_bart.py \
    --model_dir models/bart_baseline \
    --test data/test.json \
    --output_file results/bart_baseline_results.json

echo ""
echo "================================================"
echo "✓ BART Baseline Pipeline Complete!"
echo "================================================"
echo "Results saved to: results/bart_baseline_results.json"
echo ""