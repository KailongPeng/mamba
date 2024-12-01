#!/bin/bash
#SBATCH -p psych_gpu #psych_day  srun --pty -p psych_gpu --time=06:00:00 --gpus 1 --mem=50g bash
#SBATCH --job-name=eval
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --mem=50g
#SBATCH --gpus 1 

set -e
nvidia-smi

cd /gpfs/milgram/project/turk-browne/projects/SoftHebb/mamba/mamba 
module load miniconda 
conda activate mamba-env4

echo python -u evals/lm_harness_eval.py --model mamba --model_args pretrained=state-spaces/mamba-130m --tasks arc_easy --device cuda --batch_size 64
python -u evals/lm_harness_eval.py --model mamba --model_args pretrained=state-spaces/mamba-130m --tasks arc_easy --device cuda --batch_size 64

echo "done"


