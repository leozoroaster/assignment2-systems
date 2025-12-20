import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit

def setup(rank, world_size, device_type="gloo"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(device_type, rank=rank, world_size=world_size)

def distributed_demo(rank, world_size, data_size, device_type):
    global all_reduce_time
    timer = timeit.default_timer
    setup(rank, world_size, device_type)

    if device_type == "nccl":
        torch.cuda.set_device(rank)
        data = torch.rand((data_size,), device="cuda", dtype = torch.float32)
    else:
        data = torch.rand((data_size,), dtype = torch.float32)

    #warm-up
    for _ in range(5):
        dist.all_reduce(data, async_op=False)

    #time all_reduce
    t0=timer()
    for _ in range(20):
        dist.all_reduce(data, async_op=False)
    if device_type=="nccl":
        torch.cuda.synchronize()
    t1=timer()-t0

    if rank==0:
        avg=t1/20
        print(f"world={world_size}, size={data_size}, backend={device_type}, time={avg:.6f}s")

def bench_all_reduce(world_size, data_size, device_type):
    mp.spawn(fn=distributed_demo, args=(world_size,data_size, device_type), nprocs=world_size, join=True)

if __name__ == "__main__":
    world_sizes=[2,4,6]
    data_sizes = [256, 2560, 25600, 256000]
    #data_sizes=[256000, 2560000,25600000, 256000000]
    device_types=["gloo", "nccl"]
    for world_size in world_sizes:
        for data_size in data_sizes:
            for device_type in device_types:
                bench_all_reduce(world_size, data_size, device_type)