import torch

class EWC:
    def __init__(self, model, beta=0.001):
        self.beta = float(beta)
        self.ref = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = {n: torch.ones_like(p.detach()) for n, p in model.named_parameters() if p.requires_grad}

    def penalty(self, model):
        loss = torch.zeros((), device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.ref:
                loss = loss + (self.fisher[n] * (p - self.ref[n]).pow(2)).sum()
        return self.beta * loss

    def refresh_reference(self, model):
        self.ref = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
