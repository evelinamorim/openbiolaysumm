#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --mem=0
#SBATCH --account=f202500017aivlabdeucaliong
#SBATCH --output=results/%j.out
cd /projects/F202500017AIVLABDEUCALION/joaoveloso/Biomedical_Summary_Enhanced
ml OpenMPI/5.0.3-GCC-13.3.0 CUDA/11.8.0 NCCL/2.20.5-GCCcore-13.3.0-CUDA-12.4.0
export HF_HOME=/projects/F202500017AIVLABDEUCALION/joaoveloso/hf_cache
export TRANSFORMERS_CACHE=/projects/F202500017AIVLABDEUCALION/joaoveloso/hf_cache
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(cd ../.. && pwd)"
# Activate your Python environment if needed
source /projects/F202500017AIVLABDEUCALION/joaoveloso/GoldSack_py311_legacy/bin/activate
# Run your training script
srun python train.py train_config.json
