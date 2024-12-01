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


# """
# cd /gpfs/milgram/project/turk-browne/projects/SoftHebb/mamba/mamba 
# sbatch --array=1-1 kailong/run_eval.sh


# log
# (mamba-env4)[kp578@login2.milgram mamba]$ vim logs/27896839_1.out
# Sun Dec  1 02:12:38 2024
# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 555.42.06              Driver Version: 555.42.06      CUDA Version: 12.5     |
# |-----------------------------------------+------------------------+----------------------+
# | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
# |                                         |                        |               MIG M. |
# |=========================================+========================+======================|
# |   0  NVIDIA GeForce RTX 2080 Ti     On  |   00000000:3B:00.0 Off |                  N/A |
# | 23%   31C    P8             14W /  250W |       1MiB /  11264MiB |      0%      Default |
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

# python -u evals/lm_harness_eval.py --model mamba --model_args pretrained=state-spaces/mamba-130m --tasks arc_easy --device cuda --batch_size 64
# 2024-12-01:02:12:42,507 INFO     [utils.py:148] Note: NumExpr detected 36 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
# 2024-12-01:02:12:42,507 INFO     [utils.py:160] NumExpr defaulting to 8 threads.
# 2024-12-01:02:12:42,781 INFO     [config.py:58] PyTorch version 1.13.0+cu116 available.
# 2024-12-01:02:12:44,110 INFO     [__main__.py:132] Verbosity set to INFO
# 2024-12-01:02:12:52,495 INFO     [__main__.py:205] Selected Tasks: ['arc_easy']
# 2024-12-01:02:12:52,499 WARNING  [evaluator.py:93] generation_kwargs specified through cli, these settings will be used over set parameters in yaml tasks.
# Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
# 2024-12-01:02:12:57,797 INFO     [task.py:355] Building contexts for task on rank 0...
# 2024-12-01:02:13:00,037 INFO     [evaluator.py:320] Running loglikelihood requests
# ^M  0%|          | 0/9501 [00:00<?, ?it/s]Processing batch 1 with size 64
# Batched input shape:batched_inps.shape= torch.Size([64, 158])
# ^M  0%|          | 1/9501 [00:03<9:37:57,  3.65s/it]Processing batch 2 with size 64
# Batched input shape:batched_inps.shape= torch.Size([64, 87])
# ^M  1%|          | 65/9501 [00:03<06:27, 24.35it/s] Processing batch 3 with size 64
# Batched input shape:batched_inps.shape= torch.Size([64, 78])
# ^M  1%|▏         | 138/9501 [00:03<02:35, 60.35it/s]Processing batch 4 with size 64
# Batched input shape:batched_inps.shape= torch.Size([64, 73])
# Processing batch 5 with size 64
# Batched input shape:batched_inps.shape= torch.Size([64, 70])
# ^M  3%|▎         | 257/9501 [00:04<01:11, 129.65it/s]Processing batch 6 with size 64
# Batched input shape:batched_inps.shape= torch.Size([64, 67])
# Processing batch 7 with size 64
# Batched input shape:batched_inps.shape= torch.Size([64, 64])
# ^M  4%|▍         | 385/9501 [00:04<00:42, 212.04it/s]Processing batch 8 with size 64
# "logs/27896839_1.out" 352L, 22766C                                                                                                          4,1           Top




# Batched input shape:batched_inps.shape= torch.Size([64, 13])
# ^M 98%|█████████▊| 9345/9501 [00:13<00:00, 1164.35it/s]Processing batch 148 with size 64
# Batched input shape:batched_inps.shape= torch.Size([64, 13])
# Processing batch 149 with size 29
# Batched input shape:batched_inps.shape= torch.Size([29, 11])
# Total batches: 149
# ^M100%|██████████| 9501/9501 [00:13<00:00, 709.98it/s]
# Total time to process 9501 requests: 15.376252174377441 seconds
# Throughput: 617.9008962816181 requests per second
# 处理 9501 个请求耗时 15.38 秒
# 吞吐量约为 617.90 请求/秒
# 输入大小为: torch.Size([64, 158])
# 总MACs: 708102586368.0
# 输入大小为: torch.Size([1, 158])
# dummy_input.size()=torch.Size([1, 158])时的总MACs: 11064102912.0
# 输入大小为: torch.Size([1, 79])
# dummy_input.size()=torch.Size([1, 79])时的总MACs: 5532272640.0
# 输入大小为: torch.Size([1, 1])
# dummy_input.size()=torch.Size([1, 1])时的总MACs: 70465536.0
# mamba (pretrained=state-spaces/mamba-130m), gen_kwargs: (), limit: None, num_fewshot: None, batch_size: 64
# | Tasks  |Version|Filter|n-shot| Metric |Value |   |Stderr|
# |--------|-------|------|-----:|--------|-----:|---|-----:|
# |arc_easy|Yaml   |none  |     0|acc     |0.4798|±  |0.0103|
# |        |       |none  |     0|acc_norm|0.4196|±  |0.0101|

# done

# """