import torch
import math

class FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, Bq = 16, Bk = 16, is_causal=False):
        #initialize Bq, Bk
        if not isinstance(Bq,int) or Bq<=0:
            Bq=16
        if not isinstance(Bk,int) or Bk<=0:
            Bk=16

        #tile sizes
        ctx.tile_size_q = Bq
        ctx.tile_size_k = Bk

        #dimensions
        d = Q.shape[-1]
        Nq = Q.shape[-2]
        Nk = K.shape[-2]
        Tq = math.ceil(Nq / Bq)
        Tk = math.ceil(Nk / Bk)
        batch_shape = Q.shape[:-2]

        #initialize O,L
        O=torch.empty(*batch_shape, Nq,d)
        L=torch.empty(*batch_shape, Nq)

        #iterate over tiles of Q
        for i in range(Tq):
            # fetch Q tile Qi
            Qi = Q[...,Bq*i:min(Nq,Bq*(i+1)),:]

            #compute the row tile size
            Bq_real=min(Nq,Bq*(i+1))-Bq*i

            #initialize Oi, li, mi
            Oi=torch.zeros(*batch_shape,Bq_real, d)
            li=torch.zeros(*batch_shape,Bq_real)
            mi=torch.empty(*batch_shape,Bq_real)
            mi.fill_(float("-inf"))

            #iterate over tiles of K,V
            for j in range(Tk):
                # fetch K, V tile Kj, Vj
                Kj=K[...,Bk*j:min(Nk,Bk*(j+1)),:]
                Vj = V[...,Bk * j:min(Nk, Bk * (j + 1)), :]

                #compute Sij, reusing S
                S=Qi @ Kj.transpose(-2,-1)
                S=S/math.sqrt(d)

                #compute mi
                S_row_max= S.max(dim=-1).values
                new_mi=torch.maximum(mi,S_row_max)

                #compute Pi
                Pi=torch.exp(S-new_mi.unsqueeze(-1))

                #compute li
                li=torch.exp(mi-new_mi)*li+Pi.sum(dim=-1)

                #compute Oi
                scale=torch.exp(mi-new_mi)
                Oi= Oi * scale.unsqueeze(-1) + Pi @ Vj

                #update mi
                mi=new_mi

            #compute O_i, L_i
            O_i = Oi / li.unsqueeze(-1)
            L_i = mi + torch.log(li)

            #write O_i, L_i to O, L
            O[...,Bq*i:min(Nq,Bq*(i+1)),:] = O_i
            L[...,Bq*i:min(Nq,Bq*(i+1))] = L_i

        ctx.save_for_backward(Q,K,V,O,L)
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        #dims
        d=Q.shape[-1]

        #rowsum of O * dO
        D=torch.sum(O * dO, dim=-1, keepdim=True)

        #compute S
        S= Q @ K.transpose(-2,-1) / math.sqrt(d)

        #compute P
        P = torch.exp(S-L.unsqueeze(-1))

        #compute dV
        dV=P.transpose(-2,-1) @ dO

        #compute dP
        dP=dO @ V.transpose(-2,-1)

        #compute dS
        dS=P * (dP- D)

        #compute dQ
        dQ=dS @ K /math.sqrt(d)

        #compute dK
        dK=dS.transpose(-2,-1) @ Q/math.sqrt(d)

        return dQ, dK, dV, None

        raise NotImplementedError
