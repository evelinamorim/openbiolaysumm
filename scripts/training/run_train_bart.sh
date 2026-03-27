#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --mem=0
#SBATCH --account=f202500017aivlabdeucaliong
#SBATCH --output=results_bart/bart_%j.out

export HF_HOME=/projects/F202500017AIVLABDEUCALION/joaoveloso/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(cd ../.. && pwd)"
cd /projects/F202500017AIVLABDEUCALION/joaoveloso/Biomedical_Summary_Enhanced
source /projects/F202500017AIVLABDEUCALION/joaoveloso/GoldSack_py311_legacy/bin/activate
srun python3 train_bart.py train_config_bart.json
