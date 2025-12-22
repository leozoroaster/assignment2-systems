import torch
import torch.nn as nn
import torch.distributed as dist

class DDP(nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module=module
        self.handles=[]

        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self.prepare_grad)

    def prepare_grad(self,p):
        with torch.no_grad():
            p.grad/=dist.get_world_size()

        handle = dist.all_reduce(p.grad.data,op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append(handle)

    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

