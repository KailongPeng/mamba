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
