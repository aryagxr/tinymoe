# building on top of the following code:
# https://huggingface.co/blog/AviSoori1x/makemoe-from-scratch

import torch
import torch.nn as nn
from router import Router
from experts import Expert


num_experts = 3
emb_dim = 6
topk = 2

mh_output = torch.randn(2, 4, emb_dim)
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
        flatgate = gate_out.view(-1, num_experts)

        print("indices: ", topk_idx)

        for i, expert in enumerate(self.experts):
            # print("i: ", i, "expert: ", expert)
            mask = (topk_idx == i).any(dim=-1)
            print("mask: ", mask)

            # flatten so each value represents a single token
            flatmask = mask.view(-1)
            print("flatmask: ", flatmask)
            print("xflat: ", xflat)

            # this removes the tokens that are at the false index of the flatmask
            expert_tokens = xflat[flatmask]
            print("expert_tokens: ", expert_tokens)

            expert_out = expert(expert_tokens)
            print("expert_out: ", expert_out)
        
            






# call forward
tinymoe = TinyMoE()
tinymoe(mh_output)





