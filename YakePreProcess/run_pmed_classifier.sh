#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --partition=normal-a100-40
#SBATCH --mem=0
#SBATCH --account=f202500017aivlabdeucaliong
#SBATCH --output=results_pubmedberts/pmclass_%x_%j.out
#SBATCH --job-name=pm_classifier

# --- edit flags as needed ---
TRAIN_ONLY="YakePreProcess/filter_train/merged_train_dbpedia_all_keywords.json"
VAL_ONLY="YakePreProcess/filter_val/merged_val_dbpedia_all_keywords.json"
TEST_ONLY="YakePreProcess/filter_test/merged_test_dbpedia_all_keywords.json"
COMBINED="YakePreProcess/combined/combined_keywords_ordered.json"
MODEL="./models/pubmedbert_elifeabstracts"
OUT_JSON="YakePreProcess/combined/combined_with_preds.json"
OUT_CSV="YakePreProcess/combined/combined_preds_for_manual_check.csv"
NEW_OUT_CSV="YakePreProcess/newcombined/combined_preds_for_manual_check.csv"
NEW_OUT_JSON="YakePreProcess/newcombined/combined_with_preds.json"
TRAIN_ONLY_OUT="YakePreProcess/filter_train/train_classified.json"
TRAIN_ONLY_OUT_CSV="YakePreProcess/filter_train/train_classified.csv"
VAL_ONLY_OUT="YakePreProcess/filter_val/val_classified.json"
VAL_ONLY_OUT_CSV="YakePreProcess/filter_val/val_classified.csv"
TEST_ONLY_OUT="YakePreProcess/filter_test/test_classified.json"
TEST_ONLY_OUT_CSV="YakePreProcess/filter_test/test_classified.csv"
BATCH_SIZE=64
MAX_LENGTH=256
BIOMED_INDEX=1

# Toggle these: set to --only-keyword or empty string
ONLY_KEYWORD_FLAG=""        # use "--only-keyword" to ignore description/link/title
INCLUDE_TITLE_FLAG=""       # use "--include-title" to add title
INCLUDE_ID_FLAG=""          # use "--include-id" to add id
INCLUDE_LINK_FLAG="--include-link"   # set to "--include-link" to include URL/link
NO_DESCRIPTION_FLAG=""      # set to "--no-description" to skip description
# -----------------------

cd /projects/F202500017AIVLABDEUCALION/joaoveloso/Biomedical_Summary_Enhanced

# load modules / env
ml OpenMPI/5.0.3-GCC-13.3.0 CUDA/11.8.0
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/projects/F202500017AIVLABDEUCALION/joaoveloso/hf_cache
source /projects/F202500017AIVLABDEUCALION/joaoveloso/GoldSack_py311_legacy/bin/activate

echo "Host: $(hostname)"
echo "Python: $(which python3)"
echo "Model dir: ${MODEL}"
echo "biomed-index: ${BIOMED_INDEX}"
echo "only-keyword: ${ONLY_KEYWORD_FLAG}"
echo "include-title: ${INCLUDE_TITLE_FLAG}"
echo "include-id: ${INCLUDE_ID_FLAG}"
echo "include-link: ${INCLUDE_LINK_FLAG}"
echo "no-description: ${NO_DESCRIPTION_FLAG}"

[ ! -d "${MODEL}" ] && { echo "ERROR: model folder not found: ${MODEL}"; exit 3; }

# Process TRAIN split
echo ""
echo "=== Processing TRAIN split ==="
echo "Input: ${TRAIN_ONLY}"
echo "Output JSON: ${TRAIN_ONLY_OUT}"
echo "Output CSV: ${TRAIN_ONLY_OUT_CSV}"
[ ! -f "${TRAIN_ONLY}" ] && { echo "ERROR: train file not found: ${TRAIN_ONLY}"; exit 2; }

srun python3 YakePreProcess/PMedClassifier.py \
  --model-dir "${MODEL}" \
  --input "${TRAIN_ONLY}" \
  --output-json "${TRAIN_ONLY_OUT}" \
  --output-csv "${TRAIN_ONLY_OUT_CSV}" \
  --batch-size ${BATCH_SIZE} \
  --max-length ${MAX_LENGTH} \
  --biomed-index ${BIOMED_INDEX} \
  ${ONLY_KEYWORD_FLAG} ${INCLUDE_TITLE_FLAG} ${INCLUDE_ID_FLAG} ${INCLUDE_LINK_FLAG} ${NO_DESCRIPTION_FLAG} \
  --use-cuda

# Process VAL split
echo ""
echo "=== Processing VAL split ==="
echo "Input: ${VAL_ONLY}"
echo "Output JSON: ${VAL_ONLY_OUT}"
echo "Output CSV: ${VAL_ONLY_OUT_CSV}"
[ ! -f "${VAL_ONLY}" ] && { echo "ERROR: val file not found: ${VAL_ONLY}"; exit 2; }

srun python3 YakePreProcess/PMedClassifier.py \
  --model-dir "${MODEL}" \
  --input "${VAL_ONLY}" \
  --output-json "${VAL_ONLY_OUT}" \
  --output-csv "${VAL_ONLY_OUT_CSV}" \
  --batch-size ${BATCH_SIZE} \
  --max-length ${MAX_LENGTH} \
  --biomed-index ${BIOMED_INDEX} \
  ${ONLY_KEYWORD_FLAG} ${INCLUDE_TITLE_FLAG} ${INCLUDE_ID_FLAG} ${INCLUDE_LINK_FLAG} ${NO_DESCRIPTION_FLAG} \
  --use-cuda

# Process TEST split
echo ""
echo "=== Processing TEST split ==="
echo "Input: ${TEST_ONLY}"
echo "Output JSON: ${TEST_ONLY_OUT}"
echo "Output CSV: ${TEST_ONLY_OUT_CSV}"
[ ! -f "${TEST_ONLY}" ] && { echo "ERROR: test file not found: ${TEST_ONLY}"; exit 2; }

srun python3 YakePreProcess/PMedClassifier.py \
  --model-dir "${MODEL}" \
  --input "${TEST_ONLY}" \
  --output-json "${TEST_ONLY_OUT}" \
  --output-csv "${TEST_ONLY_OUT_CSV}" \
  --batch-size ${BATCH_SIZE} \
  --max-length ${MAX_LENGTH} \
  --biomed-index ${BIOMED_INDEX} \
  ${ONLY_KEYWORD_FLAG} ${INCLUDE_TITLE_FLAG} ${INCLUDE_ID_FLAG} ${INCLUDE_LINK_FLAG} ${NO_DESCRIPTION_FLAG} \
  --use-cuda

echo ""
echo "=== All splits processed successfully ==="

# end
