#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --mem=0
#SBATCH --account=f202500017aivlabdeucaliong
#SBATCH --output=results/test%j.out

cd /projects/F202500017AIVLABDEUCALION/joaoveloso/Biomedical_Summary_Enhanced
ml OpenMPI/5.0.3-GCC-13.3.0 CUDA/11.8.0 NCCL/2.20.5-GCCcore-13.3.0-CUDA-12.4.0
export CUDA_VISIBLE_DEVICES=0
source /projects/F202500017AIVLABDEUCALION/joaoveloso/GoldSack_py311/bin/activate

# point HF to shared cache and force offline
export HF_HOME=/projects/F202500017AIVLABDEUCALION/joaoveloso/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export TRANSFORMERS_OFFLINE=1

# call the eval script with config and model directory
srun python3 Evaluate_test_BART.py train_config_bart.json ./models/bart_baseline/Best_epoch
