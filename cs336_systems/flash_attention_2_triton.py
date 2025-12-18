import triton
import triton.language as tl

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
        offsets=(query_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(query_tile_index * K_TILE_SIZE, 0),
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
        offsets=(query_tile_index,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    #initiate temporary buffers Oi, li, mi
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE,),-float("inf"), dtype=tl.float32)

    #fetch Q
    Qi = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")

    #iterating over key tiles
    for j in range(tl.cdiv(N_KEYS,K_TILE_SIZE)):
        # fetch K, V tile Kj, Vj, with boundary check only on key dim
        Kj = tl.load(K_block_ptr,boundary_check=(0,),padding_option="zero")
        Vj = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

        # compute Sij, reusing S
        S = tl.dot(Qi , Kj)
        S = S *scale

        #compute mi
        S_row_max = tl.max(S, axis=1)
        new_mi = tl.maximum(mi, S_row_max)

        # compute Pi
        Pi = tl.exp(S - new_mi[:,None])

        # compute li
        expm=tl.exp(mi - new_mi)
        li = expm * li + tl.sum(Pi, axis=1)

        # compute Oi
        Oi = Oi * expm[:,None] + tl.dot(Pi , Vj)

        # update mi
        mi = new_mi

        #update K,V ptr
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # write O_i, L_i to O, L, TBD
    tl.store(O_block_ptr, Oi, boundary_check=(0,))
    tl.store(L_block_ptr, li, boundary_check=(0,))

    #update Q_ptr
    Q_block_ptr=Q_block_ptr.advance((Q_TILE_SIZE,0))

    #update O,L ptr, TBD