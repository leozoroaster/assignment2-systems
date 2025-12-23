import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

class OSS(optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):

        self.rank=dist.get_rank()
        self.world_size=dist.get_world_size()
        self.global_params=list(params)
        self.handles=[]

        #partition params
        self.local_params=[]
        index=0
        for p in self.global_params:
            if index==self.rank:
                self.local_params.append(p)
            index=(index+1)%self.world_size

        self.param_groups = []

        super().__init__(self.local_params, {})

        # initialize optimizer
        self.local_optimizer = optimizer_cls(self.param_groups, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

    def step(self,closure=None,**kwargs):
        self.local_optimizer.step(closure, **kwargs)

        #sync params, register handles
        for i, p in enumerate(self.global_params):
            rank= i % self.world_size
            handle=dist.broadcast(p.data,src=rank, async_op=True)
            self.handles.append(handle)

        #wait
        self.finish_gradient_synchronization()

    def add_param_group(self,param_group):
        self.param_groups.append(param_group)

