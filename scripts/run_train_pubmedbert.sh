#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --mem=0
#SBATCH --account=f202500017aivlabdeucaliong
#SBATCH --output=results_pubmedberts/%x_%j.out
#SBATCH --job-name=pubmedbert_finetune

# -----------------------
# Edit these variables as needed
TRAIN="scripts/Dset_downsample/train.csv"
TEST="scripts/Dset_downsample/test.csv"
TRAIN_WOUTSAMPLE="scripts/elife_raw/train.csv"
TEST_WOUTSAMPLE="scripts/elife_raw/test.csv"
TRAIN_WIKI="scripts/data/elife_plus_wiki_single/train.csv"
TEST_WIKI="scripts/data/elife_plus_wiki_single/test.csv"
TRAIN_ELIFEABS="scripts/elifeabstracts_combined/train.csv"
TEST_ELIFEABS="scripts/elifeabstracts_combined/test.csv"
OUTDIR="models/pubmedbert_downsample"
OUTDIR_WOUTSAMPLE="models/pubmedbert_woutsample"
OUTDIR_WIKI="models/pubmedbert_wiki_single"
OUTDIR_ELIFEABS="models/pubmedbert_elifeabstracts"
# point to the local model folder you downloaded on Deucalion:
MODEL="/projects/F202500017AIVLABDEUCALION/joaoveloso/Biomedical_Summary_Enhanced/pubmedbert_local"
MODEL_NAME="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
EPOCHS=3
BATCH_SIZE=8
MAX_LENGTH=256
LR=2e-5
FP16="--fp16"   # set empty string if you don't want fp16
# -----------------------

cd /projects/F202500017AIVLABDEUCALION/joaoveloso/Biomedical_Summary_Enhanced

# load modules (keep as in your environment)
ml OpenMPI/5.0.3-GCC-13.3.0 CUDA/11.8.0 NCCL/2.20.5-GCCcore-13.3.0-CUDA-12.4.0

export CUDA_VISIBLE_DEVICES=0

# HF cache locations (adjust if needed)
export HF_HOME=/projects/F202500017AIVLABDEUCALION/joaoveloso/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export HF_METRICS_CACHE=$HF_HOME

# activate python environment
source /projects/F202500017AIVLABDEUCALION/joaoveloso/GoldSack_py311_legacy/bin/activate

# print info
echo "Running on host $(hostname)"
echo "PYTHON: $(which python3)"
echo "Training file: ${TRAIN_ELIFEABS}"
echo "Test file: ${TEST_ELIFEABS}"
echo "Output dir: ${OUTDIR_ELIFEABS}"
echo "Model path: ${MODEL}"
echo "Absolute outdir: $(realpath ${OUTDIR_ELIFEABS} 2>/dev/null || printf \"%s/%s\" $(pwd) ${OUTDIR_ELIFEABS})"

# quick checks
if [ ! -d "${MODEL}" ]; then
  echo "ERROR: model folder not found: ${MODEL}"
  echo "Ensure you set MODEL to the local model folder (contains config.json, pytorch_model.bin, tokenizer files)."
  exit 2
fi

mkdir -p "${OUTDIR_ELIFEABS}"

# run training
srun python3 scripts/train_pubmedbert_finetune.py \
  --train "${TRAIN}" \
  --test "${TEST}" \
  --output-dir "${OUTDIR}" \
  --model-name "${MODEL}" \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --max-length ${MAX_LENGTH} \
  --lr ${LR} \
  ${FP16}

# end
