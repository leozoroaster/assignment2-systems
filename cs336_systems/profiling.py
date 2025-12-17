import cs336_basics.data
import cs336_basics.model
import cs336_basics.nn_utils
import cs336_basics.optimizer
import numpy as np
from collections.abc import Callable, Iterable
import torch
import timeit
import pickle

def initiate_model(vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float):
    lm_model=cs336_basics.model.BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    return lm_model

def initiate_optimizer(
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
    lm_optimizer=cs336_basics.optimizer.AdamW(params, lr, betas, eps, weight_decay)
    return lm_optimizer

def make_random_dataset(num_tokens: int, vocab_size: int) -> np.ndarray:
    random_array = np.random.randint(0, vocab_size-1, size=num_tokens)
    return random_array

def bench_model_time(model_size, context_length: int, num_tokens: int, vocab_size: int, w: int, n: int, mode="both", device=None, mixed=False, memory=None):
    d_model, d_ff, num_layers, num_heads=model_size
    new_dtype=torch.bfloat16

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device

    lm_model=initiate_model(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, 10000).to(device)
    lm_optimizer=initiate_optimizer(lm_model.parameters())
    dataset=make_random_dataset(num_tokens, vocab_size)

    lm_model.train()
    timer = timeit.default_timer

    forward_times=[]
    backward_times=[]

    print(device)

    for t in range(w):
        train_input, train_pred = cs336_basics.data.get_batch(dataset, 4, context_length, device)
        x = train_input.to(device)
        y = train_pred.to(device)
        lm_optimizer.zero_grad()
        logits = lm_model(x)
        loss_per_token = cs336_basics.nn_utils.cross_entropy(logits, y)
        loss = loss_per_token.mean()

        loss.backward()

        lm_optimizer.step()

    if memory is not None:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    for t in range(n):
        train_input, train_pred = cs336_basics.data.get_batch(dataset, 4, context_length, device)
        x = train_input.to(device)
        y = train_pred.to(device)
        lm_optimizer.zero_grad()

        if mode!="backward":
            t0 = timer()
            if mixed:
                with torch.autocast(device_type=device, dtype=new_dtype):
                    logits = lm_model(x)
                    loss_per_token = cs336_basics.nn_utils.cross_entropy(logits, y)
            else:
                logits = lm_model(x)
                loss_per_token = cs336_basics.nn_utils.cross_entropy(logits, y)
            torch.cuda.synchronize() if x.is_cuda else None
            t1 = timer() - t0
            forward_times.append(t1)
        else:
            logits = lm_model(x)
            loss_per_token = cs336_basics.nn_utils.cross_entropy(logits, y)

        if memory is not None and memory=="forward":
            torch.cuda.memory._dump_snapshot("memory_snapshot_forward.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)

        loss = loss_per_token.mean()

        if mode!="forward":
            t0 = timer()
            if mixed:
                with torch.autocast(device_type=device, dtype=new_dtype):
                    loss.backward()
            else:
                loss.backward()
            torch.cuda.synchronize() if x.is_cuda else None
            t1 = timer() - t0
            backward_times.append(t1)
        else:
            loss.backward()

        if memory is not None and memory=="both":
            torch.cuda.memory._dump_snapshot("memory_snapshot_both.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)

        lm_optimizer.step()

    print(model_size)
    if len(forward_times)>0:
        time_mean= float(np.mean(forward_times))
        time_std = float(np.std(forward_times))
        print(f"forward time mean {time_mean}")
        print(f"forward time std {time_std}")

    if len(backward_times) > 0:
        time_mean = float(np.mean(backward_times))
        time_std = float(np.std(backward_times))
        print(f"backward time mean {time_mean}")
        print(f"backward time std {time_std}")

    return forward_times, backward_times

if __name__ == "__main__":
    bench_model_time((16, 32, 2, 2), 16, 10000, 10000, 2, 4, mixed=True, memory="forward")
    #bench_model_time((768,3072,12,12),256,100000,10000, 5, 10, mixed=True)
    #bench_model_time((1024, 4096, 24, 16), 256, 100000, 10000, 5, 10, mixed=True)
    #bench_model_time((1280, 5120, 36, 20), 256, 100000, 10000, 5, 10, mixed=True)
    #bench_model_time((1600, 6400, 48, 25), 256, 100000, 10000, 5, 10, mixed=True)
    #bench_model_time((2560, 10240, 32, 32), 256, 100000, 10000, 5, 10, mixed=False, memory="forward")
    #bench_model_time((2560, 10240, 32, 32), 256, 100000, 10000, 5, 10, mixed=False, memory="both")

    with open("memory_snapshot_forward.pickle", "rb") as f:
        obj = pickle.load(f)

    print(obj)

    #with open("memory_snapshot_both.pickle", "rb") as f:
        #obj = pickle.load(f)

    #print(obj)
