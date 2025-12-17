import cs336_basics.data
import cs336_basics.model
import cs336_basics.nn_utils
import cs336_basics.optimizer
import numpy as np
from collections.abc import Callable, Iterable
import torch
import timeit
import pickle


def bench_attention(batch_size, device=None):
    timer = timeit.default_timer
    forward_times = dict()
    backward_times = dict()
    d_models=[16,32,64,128]
    seq_lengths=[256,1024, 4096,8192,16384]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device

    print(device)

    for d_model in d_models:
        for seq_length in seq_lengths:
            Q = torch.rand(batch_size, seq_length, d_model, device=device, requires_grad=True)
            K = torch.rand(batch_size, seq_length, d_model, device=device, requires_grad=True)
            V = torch.rand(batch_size, seq_length, d_model, device=device, requires_grad=True)

            for t in range(5):
                O =cs336_basics.model.scaled_dot_product_attention(Q,K,V)
                loss = torch.pow(O,2).sum()
                loss.backward()
                Q.grad = None; K.grad = None; V.grad = None

            torch.cuda.synchronize()

            t0 = timer()
            for t in range(100):
                O =cs336_basics.model.scaled_dot_product_attention(Q,K,V)
                torch.cuda.synchronize()
            t1 = timer() - t0
            print(f"forward time of d_model {d_model} seq_length {seq_length} is {t1}")
            forward_times[(d_model,seq_length)]=t1

            t0 = timer()
            for t in range(100):
                O = cs336_basics.model.scaled_dot_product_attention(Q, K, V)
                loss = torch.pow(O, 2).sum()
                loss.backward()
                torch.cuda.synchronize()
            t1 = timer() - t0
            print(f"backward time of d_model {d_model} seq_length {seq_length} is {t1}")
            backward_times[(d_model, seq_length)] = t1

    return forward_times, backward_times

if __name__ == "__main__":
    bench_attention(8)