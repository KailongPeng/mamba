#!/bin/bash
#SBATCH -p psych_gpu #psych_day  srun --pty -p psych_gpu --time=06:00:00 --gpus 1 --mem=50g bash
#SBATCH --job-name=generate
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

echo python benchmarks/benchmark_generation_mamba_simple.py --model-name "state-spaces/mamba-130m" --batch 128
python benchmarks/benchmark_generation_mamba_simple.py --model-name "state-spaces/mamba-130m" --batch 128

echo "done"

# """
# cd /gpfs/milgram/project/turk-browne/projects/SoftHebb/mamba/mamba 
# sbatch --array=1-1 kailong/run_generate.sh
# log 27896840_1 
# log 27896846_1 

