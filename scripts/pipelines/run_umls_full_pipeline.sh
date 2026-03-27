#!/bin/bash
#
# Complete Pipeline: UMLS-based (Goldsack Reproduction)
#
# This script runs the full pipeline for the UMLS-based approach.
# Includes: UMLS concept extraction, similarity computation, graph construction,
# and training.
#
# Prerequisites:
#   - UMLS 2023AA license and data (see README)
#   - QuickUMLS installed and indexed
#   - Update paths in extract_umls_concepts.py
#
# Usage: bash scripts/pipelines/run_umls_full_pipeline.sh
#

set -e

echo "================================================"
echo "UMLS (GOLDSACK REPRODUCTION) - FULL PIPELINE"
echo "================================================"

# Configuration
EPOCHS=5  # Note: UMLS approach is memory-intensive, uses fewer epochs
BATCH_SIZE=2
LEARNING_RATE=2e-6

# Check prerequisites
echo ""
echo "Checking prerequisites..."
if [ ! -d "2023AA" ]; then
    echo "❌ ERROR: UMLS data not found (2023AA/ folder missing)"
    echo ""
    echo "You need to:"
    echo "  1. Obtain UMLS license: https://www.nlm.nih.gov/research/umls/"
    echo "  2. Download UMLS 2023AA"
    echo "  3. Place in project root as 2023AA/"
    echo ""
    exit 1
fi

echo "✓ UMLS data found"

echo ""
echo "================================================"
echo "PREPROCESSING (Steps 1-3)"
echo "================================================"
echo ""
read -p "Run full preprocessing? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then

    echo ""
    echo "Step 1/6: UMLS Concept Extraction..."
    echo "  - Extracting UMLS concepts with QuickUMLS"
    echo "  - Processing abstracts and summaries"
    echo ""
    
    for split in train val test; do
        echo "  Processing $split split..."
        python src/data_preprocessing/extract_umls_concepts.py \
            --split $split \
            --output DSplit/
    done
    
    echo ""
    echo "Step 2/6: Computing UMLS Concept Similarities..."
    echo "  - Encoding concepts with SciBERT"
    echo "  - Computing pairwise similarities"
    echo ""
    
    for split in train val test; do
        echo "  Processing $split split..."
        python src/data_preprocessing/compute_umls_similarities.py \
            --split $split \
            --model allenai/scibert_scivocab_uncased
    done
    
    echo ""
    echo "Step 3/6: Building UMLS Graph PKLs..."
    echo ""
    
    python src/data_preprocessing/PKLS_from_UMLS.py \
        --splits train val test \
        --output_dir graphs/Created_PKLS/
    
    echo ""
    echo "✓ Preprocessing complete!"
    echo ""
else
    echo "Skipping preprocessing - using existing data"
fi

echo ""
echo "================================================"
echo "TRAINING (Step 4/6)"
echo "================================================"
echo ""
echo "Training UMLS Model..."
echo "  - Epochs: $EPOCHS (lower due to memory constraints)"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Learning rate: $LEARNING_RATE"
echo ""

# Train
python src/training/train_umls.py \
    --config configs/train_config_umls.json \
    --train data/train_filtered.json \
    --val data/val.json \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --output_dir models/umls

echo ""
echo "Step 5/6: Generating Predictions..."
echo ""

python src/evaluation/generate_predictions.py \
    --model_dir models/umls \
    --test data/test.json \
    --model_type umls \
    --output predictions/umls_preds.txt

echo ""
echo "Step 6/6: Evaluating on Test Set..."
echo ""

python src/evaluation/evaluate_umls.py \
    --model_dir models/umls \
    --test data/test.json \
    --output_file results/umls_results.json

echo ""
echo "================================================"
echo "✓ UMLS (Goldsack) Pipeline Complete!"
echo "================================================"
echo "Results saved to: results/umls_results.json"
echo ""