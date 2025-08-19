import torch
import torch.nn as nn
import torch.nn.functional as F


# router is a linear layer, no nonlinearity
# taken in mha output of shape (B, T, emb_dim)
# outputs tensor of shape (B, T, num_experts)
# each element is a score for the expert


# emb_dim = 6
# num_experts = 3
# topk = 2
# mh_output = torch.randn(2, 4, emb_dim)
# print(mh_output)

class Router(nn.Module):
    def __init__(self, emb_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(emb_dim, num_experts)

        # noise
        self.gaussian_noise = nn.Linear(emb_dim, num_experts)

    def forward(self, mha_out, topk):
        logits = self.gate(mha_out)

        # load balancing
        #https://arxiv.org/pdf/1701.06538
        noise_logits = self.gaussian_noise(mha_out)
        stdnorm = torch.randn_like(logits)
        softplus = F.softplus(noise_logits)
        noise = stdnorm * softplus
        noisy_logits = logits + noise


        topk_logits, topk_idx = torch.topk(noisy_logits, dim=-1, k=topk)
        print("noisy_logits: ", noisy_logits)
        print("topk_logits: ", topk_logits)
        print("topk_idx: ", topk_idx)

        mask = torch.full_like(noisy_logits, -float('inf'))
        masked = mask.scatter(dim=-1, index=topk_idx, src=topk_logits)
        print("masked: ", masked)
        out = torch.softmax(masked, dim=-1)
        print("out: ", out)

        return out, topk_idx
    







# router = Router(emb_dim, num_experts)
# print(router)

# router_out = router(mh_output)
# print(router_out)


