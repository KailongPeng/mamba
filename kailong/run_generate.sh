#!/bin/bash
#SBATCH -p psych_gpu
#SBATCH --job-name=generate
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --mem=50g
#SBATCH --gpus 1 

set -e
# nvidia-smi

cd /gpfs/milgram/project/turk-browne/projects/SoftHebb/mamba/mamba 
module load miniconda 
conda activate mamba-env4

echo python -u benchmarks/benchmark_generation_mamba_simple.py --model-name "state-spaces/mamba-130m" --batch 128
python -u benchmarks/benchmark_generation_mamba_simple.py --model-name "state-spaces/mamba-130m" --batch 128

echo "done"


# """
# # cd /gpfs/milgram/project/turk-browne/projects/SoftHebb/mamba/mamba 
# # sbatch --array=1-1 kailong/run_generate.sh
# # log 27896840_1 
# # log 27896846_1 
# # log 27896868_1

# Sun Dec  1 02:37:22 2024       
# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 555.42.06              Driver Version: 555.42.06      CUDA Version: 12.5     |
# |-----------------------------------------+------------------------+----------------------+
# | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
# |                                         |                        |               MIG M. |
# |=========================================+========================+======================|
# |   0  NVIDIA GeForce RTX 2080 Ti     On  |   00000000:3B:00.0 Off |                  N/A |
# | 22%   29C    P8             15W /  250W |       1MiB /  11264MiB |      0%      Default |
# |                                         |                        |                  N/A |
# +-----------------------------------------+------------------------+----------------------+
                                                                                         
# +-----------------------------------------------------------------------------------------+
# | Processes:                                                                              |
# |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
# |        ID   ID                                                               Usage      |
# |=========================================================================================|
# |  No running processes found                                                             |
# +-----------------------------------------------------------------------------------------+

# CondaError: Run 'conda init' before 'conda deactivate'

# python benchmarks/benchmark_generation_mamba_simple.py --model-name state-spaces/mamba-130m --batch 128
# Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
# Loading model state-spaces/mamba-130m
# Number of parameters: 129135360

# # 遍历模型的参数，打印每一层的名称、参数数量和维度信息
# # for name, param in model.named_parameters(): print(f'{name}: {param.numel()} elements, Shape: {param.size()}')

# backbone.embedding.weight: 38615040 elements, Shape: torch.Size([50280, 768])
# backbone.layers.0.mixer.A_log: 24576 elements, Shape: torch.Size([1536, 16])
# backbone.layers.0.mixer.D: 1536 elements, Shape: torch.Size([1536])
# backbone.layers.0.mixer.in_proj.weight: 2359296 elements, Shape: torch.Size([3072, 768])
# backbone.layers.0.mixer.conv1d.weight: 6144 elements, Shape: torch.Size([1536, 1, 4])
# backbone.layers.0.mixer.conv1d.bias: 1536 elements, Shape: torch.Size([1536])
# backbone.layers.0.mixer.x_proj.weight: 122880 elements, Shape: torch.Size([80, 1536])
# backbone.layers.0.mixer.dt_proj.weight: 73728 elements, Shape: torch.Size([1536, 48])
# backbone.layers.0.mixer.dt_proj.bias: 1536 elements, Shape: torch.Size([1536])
# backbone.layers.0.mixer.out_proj.weight: 1179648 elements, Shape: torch.Size([768, 1536])
# backbone.layers.0.norm.weight: 768 elements, Shape: torch.Size([768])

# ...


# backbone.layers.23.mixer.A_log: 24576 elements, Shape: torch.Size([1536, 16])
# backbone.layers.23.mixer.D: 1536 elements, Shape: torch.Size([1536])
# backbone.layers.23.mixer.in_proj.weight: 2359296 elements, Shape: torch.Size([3072, 768])
# backbone.layers.23.mixer.conv1d.weight: 6144 elements, Shape: torch.Size([1536, 1, 4])
# backbone.layers.23.mixer.conv1d.bias: 1536 elements, Shape: torch.Size([1536])
# backbone.layers.23.mixer.x_proj.weight: 122880 elements, Shape: torch.Size([80, 1536])
# backbone.layers.23.mixer.dt_proj.weight: 73728 elements, Shape: torch.Size([1536, 48])
# backbone.layers.23.mixer.dt_proj.bias: 1536 elements, Shape: torch.Size([1536])
# backbone.layers.23.mixer.out_proj.weight: 1179648 elements, Shape: torch.Size([768, 1536])
# backbone.layers.23.norm.weight: 768 elements, Shape: torch.Size([768])

# backbone.norm_f.weight: 768 elements, Shape: torch.Size([768])
# Prompt length: 100, generation length: 100
# state-spaces/mamba-130m prompt processing + decoding time: 540ms
# done
# """