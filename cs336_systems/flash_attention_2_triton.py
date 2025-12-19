import triton
import triton.language as tl
import torch
import math
import numpy as np

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(0, 1),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(0, 1),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(0, 1),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(0, 1),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    #initiate temporary buffers Oi, li, mi
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)

    #fetch Q
    Qi = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)

    scale=scale.to(tl.float32)

    #iterating over key tiles
    for j in range(tl.cdiv(N_KEYS,K_TILE_SIZE)):
        # fetch K, V tile Kj, Vj, with boundary check only on key dim
        Kj = tl.load(K_block_ptr,boundary_check=(0,1),padding_option="zero").to(tl.float32)
        Vj = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)

        # compute Sij, reusing S
        #S = tl.dot(Qi , tl.trans(Kj))
        S=tl.sum(Qi[:,:,None]*tl.trans(Kj)[None,:,:], axis=1)
        S = S *scale

        #check query key index
        starting_column = j * K_TILE_SIZE
        column_index = starting_column + tl.arange(0, K_TILE_SIZE)[None, :]
        starting_row = query_tile_index * Q_TILE_SIZE
        row_index = starting_row + tl.arange(0, Q_TILE_SIZE)[:, None]
        k_in = column_index < N_KEYS
        q_in = row_index < N_QUERIES
        final_mask=k_in & q_in

        #casual mask, the upper triangle (w.o. diag) of the global S should be set as -inf
        #that is, for row<column we set as -inf
        if is_causal:
            causal=column_index<=row_index
            final_mask=final_mask & causal

        S=tl.where(final_mask,S,-float("inf"))

        #compute mi
        S_row_max = tl.max(S, axis=1)
        new_mi = tl.maximum(mi, S_row_max)

        # compute Pi
        Pi = tl.exp(S - new_mi[:,None])

        # compute li
        expm=tl.exp(mi - new_mi)
        li = expm * li + tl.sum(Pi, axis=1)

        # compute Oi
        Pi = Pi.to(Vj.dtype)
        #Oi = Oi * expm[:,None] + tl.dot(Pi , Vj)
        Oi = Oi * expm[:, None] + tl.sum(Pi[:,:,None] *Vj[None,:,:], axis=1)

        # update mi
        mi = new_mi

        #update K,V ptr
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    #compute O_i, L_i
    Oi=Oi/li[:,None]
    li=mi+tl.log(li)

    # write O_i, L_i to O, L
    tl.store(O_block_ptr, Oi, boundary_check=(0,1))
    tl.store(L_block_ptr, li, boundary_check=(0,))

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        # dimensions
        d = Q.shape[-1]
        Nq = Q.shape[-2]
        Nk = K.shape[-2]
        batch_shape = Q.shape[:-2]
        batch_size=1
        for n in batch_shape:
            batch_size*=n

        #initialize scale, Bq, Bk
        Bq=16
        Bk=16
        scale=1/math.sqrt(d)

        #initialize O,L
        O = torch.empty(*batch_shape, Nq, d, device=Q.device, dtype=torch.float32)
        L = torch.empty(*batch_shape, Nq, device=Q.device,dtype=torch.float32)

        #handle ctx
        ctx.save_for_backward(L)
        ctx.Q_TILE_SIZE=Bq
        ctx.K_TILE_SIZE = Bk
        ctx.D = d
        ctx.is_causal = is_causal

        #flash attention
        flash_fwd_kernel[(math.ceil(Nq/ctx.Q_TILE_SIZE),batch_size)](
            Q, K, V,
            O, L,
            Q.stride(-3), Q.stride(-2), Q.stride(-1),
            K.stride(-3), K.stride(-2), K.stride(-1),
            V.stride(-3), V.stride(-2), V.stride(-1),
            O.stride(-3), O.stride(-2), O.stride(-1),
            L.stride(-2), L.stride(-1),
            Nq, Nk,
            scale,
            d,
            Bq,
            Bk,
            is_causal,
        )

        return O

    @staticmethod
    def backward(ctx, dO, dL):
        raise NotImplementedError
