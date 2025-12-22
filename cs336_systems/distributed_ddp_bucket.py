import torch
import torch.nn as nn
import torch.distributed as dist

class Bucket:
    def __init__(self, param_num: int):
        self.params=[]
        self.param_num=param_num

    def add_param(self,p):
        self.params.append(p)
        #when all params ready
        if len(self.params)==self.param_num:
            output=self.all_reduce_bucket()
            #clean up
            self.params = []
            return output
        return None

    def all_reduce_bucket(self):
        #return the handle, params and the flattened concat grad
        flattened_grad=[p.grad.data.contiguous().view(-1) for p in self.params]
        flattened_grad=torch.concat(flattened_grad)
        flattened_grad /= dist.get_world_size()
        handle = dist.all_reduce(flattened_grad, op=dist.ReduceOp.SUM, async_op=True)
        return handle, list(self.params), flattened_grad

class DDPBucket(nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb:float):
        super().__init__()
        self.module=module
        self.handles=[]
        self.bytes_bound=bucket_size_mb*1024*1024

        self.curr_bytes=0
        self.curr_bucket=[]
        self.p2b=dict()

        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
            if not p.requires_grad:
                continue
            self.curr_bytes+=p.numel() * p.element_size()
            self.curr_bucket.append(p)
            if self.curr_bytes>=self.bytes_bound:
                curr_bucket=Bucket(len(self.curr_bucket))
                for param in self.curr_bucket:
                    self.p2b[param]=curr_bucket
                for param in self.curr_bucket:
                    param.register_post_accumulate_grad_hook(self.prepare_grad)
                self.curr_bytes = 0
                self.curr_bucket = []

        if len(self.curr_bucket)>0:
            curr_bucket=Bucket(len(self.curr_bucket))
            for param in self.curr_bucket:
                self.p2b[param] = curr_bucket
            for param in self.curr_bucket:
                param.register_post_accumulate_grad_hook(self.prepare_grad)

    def prepare_grad(self,p):
        bucket = self.p2b[p]
        result = bucket.add_param(p)
        if result is not None:
            handle, params, flattened_grad=result
            self.handles.append((handle, params, flattened_grad))

    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle, params, flattened_grad in self.handles:
            handle.wait()
            #unflatten
            index=0
            for p in params:
                length = p.grad.numel()
                p.grad.copy_(flattened_grad[index:index + length].view_as(p.grad))
                index += length
        self.handles.clear()


