import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import math

def setup(rank, world_size, device_type):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if device_type=="nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend=device_type, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def model_fn():
    torch.manual_seed(0)
    return torch.nn.Linear(1,1)

def optimizer_fn(params):
    return torch.optim.SGD(params, lr=0.1)

def ddp_training(rank, x_dataset, y_dataset, loss, num_epoches, world_size, device_type):
    #device setup
    if device_type=="gloo":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{rank}")

    setup(rank, world_size, device_type)

    # initialize model and optimizer
    local_model = model_fn().to(device)
    local_optimizer = optimizer_fn(local_model.parameters())

    #partition dataset
    num_data=x_dataset.shape[0]
    local_num_data=math.ceil(num_data/world_size)
    x=x_dataset[local_num_data*rank: min(local_num_data*(rank+1),num_data)].to(device)
    y=y_dataset[local_num_data*rank: min(local_num_data*(rank+1),num_data)].to(device)

    #local training
    local_model.train()
    for epoch in range(num_epoches):
        #local forward/backward step
        local_optimizer.zero_grad()
        y_pred=local_model(x)
        local_loss=loss(y, y_pred)
        local_loss.backward()

        #all reduce gradient
        #with torch.no_grad():
            #for p in local_model.parameters():
                #if p.grad is None:
                    #continue
                #dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                #p.grad.div_(world_size)

        #all reduce flattened gradient
        with torch.no_grad():
            flattened_grad=[]
            for p in local_model.parameters():
                if p.grad is not None:
                    flattened_grad.append(p.grad.contiguous().view(-1))

            flattened_grad=torch.concat(flattened_grad)

            dist.all_reduce(flattened_grad, op=dist.ReduceOp.SUM)
            flattened_grad.div_(world_size)

            index=0
            for p in local_model.parameters():
                if p.grad is not None:
                    length=p.grad.numel()
                    p.grad.copy_(flattened_grad[index:index+length].view_as(p.grad))
                    index+=length

        #opt step
        local_optimizer.step()

    print("rank", rank , "preds", local_model(x_dataset))

    cleanup()

def ddp_training_wrapper(x_dataset, y_dataset, loss, num_epoches, world_size, device_type):
    mp.spawn(fn=ddp_training, args=(x_dataset, y_dataset, loss, num_epoches, world_size, device_type), nprocs=world_size, join=True)

if __name__ == "__main__":
    #configs
    x_dataset=torch.rand(10,1)
    y_dataset = torch.rand(10,1)
    loss=torch.nn.MSELoss()
    num_epoches=5
    world_size=2
    device_type="gloo"

    ddp_training_wrapper(x_dataset, y_dataset, loss, num_epoches, world_size, device_type)