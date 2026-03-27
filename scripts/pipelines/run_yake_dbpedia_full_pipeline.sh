#!/bin/bash
#
# Complete Pipeline: OpenBioLaySumm (YAKE + DBpedia)
#
# This script runs the full pipeline for the YAKE+DBpedia approach.
# Includes: keyword extraction, DBpedia querying, similarity computation,
# graph construction, and training.
#
# Prerequisites:
#   - Ollama with DeepSeek-R1:7b running (for LLM classification)
#   - Internet connection (for DBpedia API)
#
# Usage: bash scripts/pipelines/run_yake_dbpedia_full_pipeline.sh
#
# Note: This is a LONG process. Consider running preprocessing steps separately
#       if you only want to retrain models.
#

set -e

echo "================================================"
echo "OPENBIOLAYSUMM (YAKE + DBpedia) - FULL PIPELINE"
echo "================================================"

# Configuration
EPOCHS=20
BATCH_SIZE=2
LEARNING_RATE=2e-6

# Check prerequisites
echo ""
echo "Checking prerequisites..."
if ! command -v ollama &> /dev/null; then
    echo "⚠️  WARNING: Ollama not found. You'll need it for keyword classification."
    echo "   Install: https://ollama.ai/"
fi

echo ""
echo "================================================"
echo "PREPROCESSING (Steps 1-4)"
echo "================================================"
echo ""
echo "⚠️  Note: Preprocessing can take several hours!"
echo "   If you already have preprocessed data, you can skip to training."
read -p "Run full preprocessing? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then

    echo ""
    echo "Step 1/7: YAKE Keyword Extraction + DBpedia Querying..."
    echo "  - Extracting keywords from abstracts"
    echo "  - Querying DBpedia for each keyword"
    echo "  - Classifying as biomedical/non-biomedical"
    echo ""
    
    # Run for each split
    for split in train val test; do
        echo "  Processing $split split..."
        python src/data_preprocessing/yake_preprocess.py \
            --split $split \
            --output YakePreProcess/Files_pre_processed/
    done
    
    echo ""
    echo "Step 2/7: Computing DBpedia Similarities (Edge Weights)..."
    echo "  - Encoding keywords and descriptions with SciBERT"
    echo "  - Computing cosine similarities"
    echo ""
    
    for split in train val test; do
        echo "  Processing $split split..."
        python src/data_preprocessing/compute_dbpedia_similarities.py \
            --split $split \
            --model allenai/scibert_scivocab_uncased \
            --threshold 0.7
    done
    
    echo ""
    echo "Step 3/7: Building Graph Structures..."
    echo "  - Creating nodes (concepts) and edges (similarities)"
    echo "  - Applying Graph Attention Network (GAT)"
    echo ""
    
    python src/data_preprocessing/create_pkls.py \
        --splits train val test \
        --output_dir graphs/
    
    echo ""
    echo "✓ Preprocessing complete!"
    echo ""
else
    echo "Skipping preprocessing - using existing data"
fi

echo ""
echo "================================================"
echo "TRAINING (Step 5/7)"
echo "================================================"
echo ""
echo "Training GNN Model (YAKE+DBpedia)..."
echo "  - Epochs: $EPOCHS"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Learning rate: $LEARNING_RATE"
echo ""

# Train
python src/training/train.py \
    --config configs/train_config.json \
    --train data/train_filtered.json \
    --val data/val.json \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --output_dir models/yake_dbpedia

echo ""
echo "Step 6/7: Generating Predictions..."
echo ""

python src/evaluation/generate_predictions.py \
    --model_dir models/yake_dbpedia \
    --test data/test.json \
    --output predictions/yake_dbpedia_preds.txt

echo ""
echo "Step 7/7: Evaluating on Test Set..."
echo ""

python src/evaluation/evaluate.py \
    --model_dir models/yake_dbpedia \
    --test data/test.json \
    --output_file results/yake_dbpedia_results.json

echo ""
echo "================================================"
echo "✓ OpenBioLaySumm (YAKE+DBpedia) Pipeline Complete!"
echo "================================================"
echo "Results saved to: results/yake_dbpedia_results.json"
echo ""