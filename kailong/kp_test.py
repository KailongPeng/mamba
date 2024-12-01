import torch
from mamba_ssm import Mamba
"""
    d_model(768)
    d_inner(1536)
    dt_rank(48) 
    d_state(16)
    
    parameters from 
        python evals/lm_harness_eval.py --model hf --model_args pretrained=EleutherAI/pythia-160m --tasks arc_easy --device cuda --batch_size 64
    
    运行： 
    interactive_gpu
    cd_mamba
    python -u kp_test.py
    
"""
batch, length, dim = 64, 158, 768  # batch, length, dim # from arc_easy
x = torch.randn(batch, length, dim).to("cuda")  #(B, L, D)
model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")
y = model(x)
import pdb; pdb.set_trace()
assert y.shape == x.shape


"""
# milgram

module load miniconda
conda env list
# conda environments:
#
base                     /gpfs/milgram/apps/avx2/software/miniconda/24.7.1
GLMsingle                /gpfs/milgram/project/turk-browne/kp578/conda_envs/GLMsingle
analysis                 /gpfs/milgram/project/turk-browne/kp578/conda_envs/analysis
brainiak                 /gpfs/milgram/project/turk-browne/kp578/conda_envs/brainiak
brainiakTest             /gpfs/milgram/project/turk-browne/kp578/conda_envs/brainiakTest
brainpy-env              /gpfs/milgram/project/turk-browne/kp578/conda_envs/brainpy-env
env_fmri                 /gpfs/milgram/project/turk-browne/kp578/conda_envs/env_fmri
glmsingle                /gpfs/milgram/project/turk-browne/kp578/conda_envs/glmsingle
glmsingle_backup         /gpfs/milgram/project/turk-browne/kp578/conda_envs/glmsingle_backup
harmonic                 /gpfs/milgram/project/turk-browne/kp578/conda_envs/harmonic
mamba-env                /gpfs/milgram/project/turk-browne/kp578/conda_envs/mamba-env
mamba-env2               /gpfs/milgram/project/turk-browne/kp578/conda_envs/mamba-env2
mamba-env3               /gpfs/milgram/project/turk-browne/kp578/conda_envs/mamba-env3
mamba-env4               /gpfs/milgram/project/turk-browne/kp578/conda_envs/mamba-env4
mybrainiak               /gpfs/milgram/project/turk-browne/kp578/conda_envs/mybrainiak
myrtSynth_rt             /gpfs/milgram/project/turk-browne/kp578/conda_envs/myrtSynth_rt
nsdFS                    /gpfs/milgram/project/turk-browne/kp578/conda_envs/nsdFS
psychRNN                 /gpfs/milgram/project/turk-browne/kp578/conda_envs/psychRNN
py36                     /gpfs/milgram/project/turk-browne/kp578/conda_envs/py36
py36_jupyter             /gpfs/milgram/project/turk-browne/kp578/conda_envs/py36_jupyter
pydeface                 /gpfs/milgram/project/turk-browne/kp578/conda_envs/pydeface
rtAtten                  /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtAtten
rtSynth_rt               /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtSynth_rt
rtcloud                  /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtcloud
softhebb_test1           /gpfs/milgram/project/turk-browne/kp578/conda_envs/softhebb_test1
softhebb_test2           /gpfs/milgram/project/turk-browne/kp578/conda_envs/softhebb_test2
stylegan_env             /gpfs/milgram/project/turk-browne/kp578/conda_envs/stylegan_env
synthesis_env            /gpfs/milgram/project/turk-browne/kp578/conda_envs/synthesis_env
tensorflow112            /gpfs/milgram/project/turk-browne/kp578/conda_envs/tensorflow112
testenv                  /gpfs/milgram/project/turk-browne/kp578/conda_envs/testenv
tf1                      /gpfs/milgram/project/turk-browne/kp578/conda_envs/tf1
tf1_copy                 /gpfs/milgram/project/turk-browne/kp578/conda_envs/tf1_copy
tf2                      /gpfs/milgram/project/turk-browne/kp578/conda_envs/tf2
torch                    /gpfs/milgram/project/turk-browne/kp578/conda_envs/torch
venv                     /gpfs/milgram/project/turk-browne/kp578/conda_envs/venv
                         /gpfs/milgram/project/turk-browne/users/kp578/CONDA/rtcloud
                         /home/kp578/miniconda3

interactive_gpu
alias cd_mamba="cd /gpfs/milgram/project/turk-browne/projects/SoftHebb/mamba/mamba ; module load miniconda ; conda activate mamba-env4"
cd_mamba







"""