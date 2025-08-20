# building on top of the following code:
# https://huggingface.co/blog/AviSoori1x/makemoe-from-scratch

import torch
import torch.nn as nn
from router import Router
from experts import Expert


num_experts = 8
topk = 2
emb_dim = 16
dropout=0.1

mh_output = torch.randn(2, 4, emb_dim) # batch_size, seq_len, emb_dim
print("mh_output: ", mh_output)


class TinyMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([Expert(in_dim=emb_dim, out_dim=emb_dim) for _ in range(num_experts)])
        self.router = Router(emb_dim, num_experts)

    def forward(self, x):
        gate_out, topk_idx = self.router(x, topk)
        moe_out = torch.zeros_like(x)
        xflat = x.view(-1, emb_dim)
        flatgate = gate_out.view(-1, num_experts) #flatten router output to 2d
        

        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i).any(dim=-1)

            if mask.any():

                # flatten so each value represents a single token
                flatmask = mask.view(-1)               

                # this removes the tokens that are at the false index of the flatmask
                expert_tokens = xflat[flatmask]
                expert_out = expert(expert_tokens)
                
                
                gater_score = flatgate[flatmask, i].unsqueeze(1) # ith column represents each expert
                weighted_expert_out = expert_out * gater_score
                # insert back into final moe output matrix
                moe_out[mask] += weighted_expert_out
                print("moe_out: ", moe_out)
        
        return moe_out
                
                
            
        
            

# call forward
tinymoe = TinyMoE()
tinymoe(mh_output)





