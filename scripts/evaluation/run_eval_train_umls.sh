#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --mem=0
#SBATCH --account=f202500013aivlabdeucaliong
#SBATCH --output=results_test/%j.out
cd /projects/F202500013AIVLABDEUCALION/Biomedical_Summary_Enhanced
ml OpenMPI/5.0.3-GCC-13.3.0 CUDA/11.8.0 NCCL/2.20.5-GCCcore-13.3.0-CUDA-12.4.0
export CUDA_VISIBLE_DEVICES=0
source /projects/F202500013AIVLABDEUCALION/GoldSack_py311/bin/activate
srun python Evaluate_test_UMLS.py
