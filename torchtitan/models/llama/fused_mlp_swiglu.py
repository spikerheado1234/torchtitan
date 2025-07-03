## This is a single-GPU fused MLP. ##
import torch
from typing import Tuple
import triton
import triton.language as tl
from src.ops.tune import extract_fwd_params, extract_bwd_params_dx_dw2, \
        extract_bwd_params_dw1_dw3, extract_bwd_params_dx, extract_bwd_params_dw2, \
        extract_bwd_params_dw2_par_ffn, extract_bwd_params_dw1_dw3_par_ffn, \
        extract_bwd_params_dw1_dw2_dw3_par_ffn
import pdb
import time

## Tanh not supported for some reason. ##
# @triton.jit
# def _gelu_fwd(x):
#     # tanh approximation form of GELU is computed with:
#     # 0.5 * a * (1 + tanh(sqrt(2 / pi) * (a + 0.044715 * a^3)))
#     sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
#     x_cubed = x * x * x
#     tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
#     tanh_result = tl.math.tanh(tanh_arg)
#     gelu = 0.5 * x * (1 + tanh_result)
#     return gelu

def get_cuda_autotune_config_fwd():
    block_y_params = [2**i for i in range(5, 9)]
    block_x_params = [2**i for i in range(5, 9)]
    d_m_block_params = [32, 64, 128]
    num_stages = [i for i in range(2, 5)]
    warps = [2**i for i in range(2, 5)]

    cnfgs = []

    ## Seems like the best configuration so far is: BLOCK_X = 256, BLOCK_Y = 128, d_m_block = 128, num_warps = 16, num_ctas = 1, num_stages = 2. ##

    #for y in block_y_params:
    #    for x in block_x_params:
    #        for d_m in d_m_block_params:
    #            for w in warps:
    #                cnfgs.append(triton.Config(
    #                    kwargs={'BLOCK_Y': y, 'BLOCK_X': x, 'd_m_block': d_m},
    #                    num_stages=2,
    #                    num_warps=w,
    #                    num_ctas=1))

    print(f'testing: {len(cnfgs)} configurations...')
    #cnfgs.append(triton.Config(
    #    kwargs={'BLOCK_Y': 64, 'BLOCK_X': 256, 'd_m_block': 64},
    #    num_stages=2,
    #    num_warps=8,
    #    num_ctas=1))
    cnfgs.append(triton.Config(
        kwargs={'BLOCK_Y': 32, 'BLOCK_X': 32, 'd_m_block': 32},
        num_stages=2,
        num_warps=8,
        num_ctas=1))
    return cnfgs


def get_cuda_autotune_config_bwd():
    block_y_params = [2**i for i in range(5, 9)]
    block_x_params = [2**i for i in range(5, 9)]
    d_m_block_params = [32, 64, 128]
    num_stages = [i for i in range(2, 5)]
    warps = [2**i for i in range(2, 5)]

    cnfgs = []

    ## Dump best configuration here: 

    for y in block_y_params:
        for x in block_x_params:
            for d_m in d_m_block_params:
                for w in warps:
                    cnfgs.append(triton.Config(
                        kwargs={'BLOCK_Y': y, 'BLOCK_X': x, 'd_m_block': d_m},
                        num_stages=2,
                        num_warps=w,
                        num_ctas=1))

    print(f'testing: {len(cnfgs)} configurations...')

    return cnfgs

@triton.jit
def _gelu_grad(x):
    pass

@triton.jit
def _relu_fwd(x):
    return tl.where(x >= 0, x, 0)

@triton.jit
def _relu_grad(x):
    ## Is this correct, I am setting the gradient at 0 to 0. ##
    return tl.where(x >= 0, 1, 0).to(x.type.element_ty)

## Custom load add store, better than contention free atomic_add + no typecasting required. ##
## Invariant: tiling should ensure contention freeness (i.e. no data-races). ##
@triton.jit
def _load_add_store(values_smem, ptrs: tl.constexpr, mask: tl.constexpr):
    loaded_values = tl.load(ptrs, mask=mask, other=0.0)
    stored_vals = loaded_values + values_smem
    tl.store(ptrs, stored_vals.to(loaded_values.type.element_ty), mask=mask)

@triton.jit
def _load_add_atomic_write(values_smem, ptrs: tl.constexpr, mask: tl.constexpr):
    loaded_values = tl.load(ptrs, mask=mask, other=0.0)
    stored_vals = loaded_values + values_smem
    tl.atomic_add(ptrs, stored_vals.to(tl.float32), mask=mask, sem="relaxed")

#@triton.autotune(
#    configs=get_cuda_autotune_config_fwd(),
#    key=['N', 'd_ffn', 'd_m']
#)
@triton.jit
def _fwd_kernel(
    ## Input activations. ##
    inp_ptr,
    ## Weight matrices. ##
    w1_ptr, w2_ptr, w3_ptr,
    ## Output activation matrix. ##
    output_ptr,
    N: tl.constexpr, d_ffn: tl.constexpr, d_m: tl.constexpr,
    ## Strides. ##

    ## Strides of input activations. ##
    stride_inp_a: tl.constexpr, stride_inp_b: tl.constexpr, stride_inp_c: tl.constexpr,

    ## Strides of weight matrices. ##
    stride_w1_a: tl.constexpr, stride_w1_b: tl.constexpr,
    stride_w2_a: tl.constexpr, stride_w2_b: tl.constexpr,

    ## Strides of output activation matrix. ##
    stride_output_a: tl.constexpr, stride_output_b: tl.constexpr, stride_output_c: tl.constexpr,

    ## Lastly, the activation function. ##
    ACTIVATION: tl.constexpr, data_type: tl.constexpr, use_opt: tl.constexpr,
    ## Block Sizes. We default to parallelising over batch dimension. ##
    BLOCK_Y: tl.constexpr, BLOCK_X: tl.constexpr, d_m_block: tl.constexpr):
    """
    Input sizes:
    inp_ptr (original activations): [b, s, d_m] matrix.
    w1_ptr (first weight matrix): [d_m, d_ffn] matrix.
    w2_ptr (second weight matrix): [d_ffn, d_m] matrix.
    w3_ptr (third weight matrix): [d_m, d_ffn] matrxi (identical to w1).
    output_ptr (final output activations): [b, s, d_m] matrix.

    The kerenl performs the computation:
    (silu(inp @ w1) * inp @ w3) @w2
    """
    bid_x = tl.program_id(axis=0)
    batch = tl.program_id(axis=1)

    hid_tiles = tl.num_programs(axis=2)
    hid_num = tl.program_id(axis=2)
    for j in tl.range(hid_num, tl.cdiv(d_ffn, BLOCK_X), hid_tiles):
        output_ptrs_consec = tl.max_contiguous(tl.arange(0, d_m_block) * stride_output_c + \
                                               bid_x * BLOCK_Y * stride_output_b,
                                               d_m_block)
        output_ptrs_consec = output_ptrs_consec[None, :]
        output_ptrs = output_ptr + batch*stride_output_a \
                        + tl.arange(0, BLOCK_Y)[:, None]*stride_output_b \
                        + output_ptrs_consec
                        # + tl.arange(0, d_m_block)[None, :]*stride_output_c \
                        # + bid_x*BLOCK_Y*stride_output_b + step_size*i*stride_output_b
        ## We load the data from the input activations. ##
        activation_ptrs_consec = tl.max_contiguous(
            tl.arange(0, d_m_block) * stride_inp_c \
                        + bid_x*BLOCK_Y*stride_inp_b, d_m_block
        )
        activation_ptrs_consec = activation_ptrs_consec[None, :]
        activation_ptrs = inp_ptr + batch*stride_inp_a \
                            + tl.arange(0, BLOCK_Y)[:, None]*stride_inp_b \
                            + activation_ptrs_consec
                            # + tl.arange(0, d_m_block)[None, :]*stride_inp_c \
                            # + bid_x*BLOCK_Y*stride_inp_b + step_size*i*stride_inp_b
        ## We additionally tile across the hidden dimension to reduce shmem consumption. ##
        ## We pull out the base pointer's compute arithmetic. ##
        weight_one_ptrs_consec = tl.max_contiguous(
            tl.arange(0, BLOCK_X)*stride_w1_b, BLOCK_X
        )
        weight_one_ptrs_consec = weight_one_ptrs_consec[None, :]
        weight_one_ptrs = w1_ptr + j*BLOCK_X*stride_w1_b \
                            + tl.arange(0, d_m_block)[:, None]*stride_w1_a \
                            + weight_one_ptrs_consec

        weight_two_ptrs_consec = tl.max_contiguous(
            tl.arange(0, d_m_block) * stride_w2_b, d_m_block
        )
        weight_two_ptrs_consec = weight_two_ptrs_consec[None, :]
        weight_two_ptrs = w2_ptr + j * BLOCK_X * stride_w2_a \
                            + tl.arange(0, BLOCK_X)[:, None] * stride_w2_a \
                            + weight_two_ptrs_consec

        weight_three_ptrs_consec = tl.max_contiguous(
            tl.arange(0, BLOCK_X)*stride_w1_b, BLOCK_X
        )
        weight_three_ptrs_consec = weight_three_ptrs_consec[None, :]
        ## Weight 3's shape is identical to w1's.
        weight_three_ptrs = w3_ptr + j*BLOCK_X*stride_w1_b \
                            + tl.arange(0, d_m_block)[:, None]*stride_w1_a \
                            + weight_three_ptrs_consec

        gate_accum = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
        non_gate_accum = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(d_m, d_m_block)):
            ## First, we load the data from the input activations. ##
            activations = tl.load(activation_ptrs, mask=( \
                                    (tl.arange(0, BLOCK_Y)[:, None]+ bid_x*BLOCK_Y < N) \
                                    & (tl.arange(0, d_m_block)[None, :] + k*d_m_block < d_m)), other=0.0)
            ## Next, we load the first & third weight matrices. ##
            weight_one = tl.load(weight_one_ptrs, mask=( \
                                    (tl.arange(0, d_m_block)[:, None] + k*d_m_block < d_m) & \
                                    (tl.arange(0, BLOCK_X)[None, :] +j*BLOCK_X < d_ffn)), other=0.0)

            weight_three = tl.load(weight_three_ptrs, mask=( \
                                    (tl.arange(0, d_m_block)[:, None] + k*d_m_block < d_m) & \
                                    (tl.arange(0, BLOCK_X)[None, :] +j*BLOCK_X < d_ffn)), other=0.0)

            ## We accumulate the partial results. ##
            gate_accum += tl.dot(activations, weight_one)
            non_gate_accum += tl.dot(activations, weight_three)

            ## Increment pointers. ##
            activation_ptrs += d_m_block
            weight_one_ptrs += d_m_block*stride_w1_a
            weight_three_ptrs += d_m_block*stride_w1_a

        activ_accum = gate_accum * tl.sigmoid(gate_accum)
        activ_accum *= non_gate_accum

        ## Next, we have to downcast the intermediate activations to the pertinent data-type prior to second mat-mul. ##
        if data_type == "bfloat16":
            activ_accum = activ_accum.to(tl.bfloat16)

        ## Next, we load the data from the second weight matrix to do the matmul. We again tile this across the hidden dimension. ##
        for k in tl.range(0, tl.cdiv(d_m, d_m_block)):

            weight_two = tl.load(weight_two_ptrs, mask=( \
                                    (tl.arange(0, BLOCK_X)[:, None] +j*BLOCK_X < d_ffn) & \
                                    (tl.arange(0, d_m_block)[None, :] + k*d_m_block < d_m)), other=0.0)

            output_activations = tl.dot(activ_accum, weight_two)

            ## We have to store the final output to HBM incrementally. ##
            ## To do so we need atomic adds since triton has no other option. ##
            if use_opt == 'true':
                ## We have to commend out one or the other depending on num_par since python doesn't have macros. Predicating this forces the compiler to maket the conservative approach that atomic_add branch could be taken depsite par <=1.
                _load_add_store(output_activations, output_ptrs, mask=( \
                                    (tl.arange(0, BLOCK_Y)[:, None]+bid_x*BLOCK_Y < N) \
                                    & (tl.arange(0, d_m_block)[None, :] + k*d_m_block < d_m)))
                ## With a parallel scheme over the hid-dimension, we now have to atomically add to main memory. ##
                #tl.atomic_add(output_ptrs, output_activations.to(tl.float32),
                #            mask=(\
                #                (tl.arange(0, BLOCK_Y)[:, None]+bid_x*BLOCK_Y < N) \
                #                    & (tl.arange(0, d_m_block)[None, :] + k*d_m_block < d_m)), sem="relaxed")

            else:
                tl.atomic_add(output_ptrs, output_activations.to(tl.float32),
                            mask=(\
                                (tl.arange(0, BLOCK_Y)[:, None]+bid_x*BLOCK_Y < N) \
                                    & (tl.arange(0, d_m_block)[None, :] + k*d_m_block < d_m)), sem="relaxed")

            weight_two_ptrs += d_m_block*stride_w2_b
            output_ptrs += d_m_block*stride_output_c

## Deprecated, bad parallel scheme. ##
@triton.jit
def _bwd_kernel_dw1_dw2_dw3_par_ffn(
    ## Input activations. ##
    inp_ptr,
    ## Weight matrices. ##
    w1_ptr, w2_ptr, w3_ptr,
    ## Gradient matrices. ##
    incoming_grads_ptr, 
    dw1_ptr, dw2_ptr, dw3_ptr,
    ## Dimensions of the overall computation. ##
    N: tl.constexpr, d_ffn: tl.constexpr, d_m: tl.constexpr, batch_size: tl.constexpr,
    ## Strides. ##

    ## Strides of input activations. ##
    ## Inputs reshaped to 2-d tensor. ##
    stride_inp_a: tl.constexpr, stride_inp_b: tl.constexpr, 

    ## Strides of weight matrices. ##
    stride_w1_a: tl.constexpr, stride_w1_b: tl.constexpr,
    stride_w2_a: tl.constexpr, stride_w2_b: tl.constexpr,
    stride_w3_a: tl.constexpr, stride_w3_b: tl.constexpr,

    ## Strides of gradient matrices. ##
    stride_incoming_grads_a: tl.constexpr, stride_incoming_grads_b: tl.constexpr, 
    stride_dw1_a: tl.constexpr, stride_dw1_b: tl.constexpr,
    stride_dw2_a: tl.constexpr, stride_dw2_b: tl.constexpr,
    stride_dw3_a: tl.constexpr, stride_dw3_b: tl.constexpr,

    ## Lastly, the activation function. ##
    ACTIVATION: tl.constexpr,
    ## Block Sizes. We default to parallelising over batch dimension. ##
    BLOCK_Y: tl.constexpr, BLOCK_X: tl.constexpr, d_m_block: tl.constexpr):
    """
    Input sizes:
    inp_ptr (original activations): [b, s, d_m] matrix.
    w1_ptr (first weight matrix): [d_m, d_ffn] matrix.
    w2_ptr (second weight matrix): [d_ffn, d_m] matrix.
    w3_ptr (third weight matrix): [d_m, d_ffn] matrxi (identical to w1).
    output_ptr (final output activations): [b, s, d_m] matrix.

    This kernel computes the gradients of w1 and w3 in the following computation:
    (silu(inp @ w1) * inp @ w3) @ w2
    """
    bid_x = tl.program_id(axis=0)

    for j in tl.range(0, tl.cdiv(N * batch_size, BLOCK_Y)):
        ## This is necessary to compute dw2.
        ## We load the data from the input activations. ##
        activation_ptrs_consec = tl.max_contiguous(
            tl.arange(0, d_m_block) * stride_inp_b \
                        + j * BLOCK_Y * stride_inp_a, d_m_block
        )
        activation_ptrs_consec = activation_ptrs_consec[None, :]
        activation_ptrs = inp_ptr \
                            + tl.arange(0, BLOCK_Y)[:, None] * stride_inp_a \
                            + activation_ptrs_consec

        ## We additionally tile across the hidden dimension to reduce shmem consumption. ##
        ## We pull out the base pointer's compute arithmetic. ##
        weight_one_ptrs_consec = tl.max_contiguous(
            tl.arange(0, BLOCK_X) * stride_w1_b, BLOCK_X
        )
        weight_one_ptrs_consec = weight_one_ptrs_consec[None, :]
        weight_one_ptrs = w1_ptr + bid_x * BLOCK_X * stride_w1_b \
                            + tl.arange(0, d_m_block)[:, None] * stride_w1_a \
                            + weight_one_ptrs_consec

        weight_three_ptrs_consec = tl.max_contiguous(
            tl.arange(0, BLOCK_X) * stride_w3_b, BLOCK_X
        )
        weight_three_ptrs_consec = weight_three_ptrs_consec[None, :]
        weight_three_ptrs = w3_ptr + bid_x * BLOCK_X * stride_w3_b \
                            + tl.arange(0, d_m_block)[:, None] * stride_w3_a \
                            + weight_three_ptrs_consec

        do5_ptrs = incoming_grads_ptr + tl.arange(0, BLOCK_Y)[:, None] * stride_incoming_grads_a \
                    + tl.arange(0, d_m_block)[None, :] * stride_incoming_grads_b \
                    + j * BLOCK_Y * stride_incoming_grads_a \

        weight_two_ptrs = w2_ptr + tl.arange(0, BLOCK_X)[:, None] * stride_w2_a \
                            + tl.arange(0, d_m_block)[None, :] * stride_w2_b \
                            + bid_x * BLOCK_X * stride_w2_a

        o1 = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
        o2 = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
        do4 = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(d_m, d_m_block)):  
            ## First, we load the data from the input activations. ##
            activations = tl.load(activation_ptrs, mask=( \
                                    (tl.arange(0, BLOCK_Y)[:, None] + j * BLOCK_Y < N * batch_size) \
                                    & (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m)), other=0.0)
            ## Next, we load the first & third weight matrices. ##
            weight_one = tl.load(weight_one_ptrs, mask=( \
                                    (tl.arange(0, d_m_block)[:, None] + k * d_m_block < d_m) & \
                                    (tl.arange(0, BLOCK_X)[None, :] + bid_x * BLOCK_X < d_ffn)), other=0.0)

            weight_three = tl.load(weight_three_ptrs, mask=( \
                                    (tl.arange(0, d_m_block)[:, None] + k * d_m_block < d_m) & \
                                    (tl.arange(0, BLOCK_X)[None, :] + bid_x * BLOCK_X < d_ffn)), other=0.0)
            ## Finally we load do5 and w2 to compute do4. ##
            do5s = tl.load(do5_ptrs, mask=(\
                            (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m) &
                            (tl.arange(0, BLOCK_Y)[:, None] + j * BLOCK_Y < N * batch_size)
                ), other=0.0)

            weight_two = tl.load(weight_two_ptrs, mask=(\
                                    (tl.arange(0, BLOCK_X)[:, None] + bid_x * BLOCK_X < d_ffn) & \
                                    (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m)
                ))

            do4 += tl.dot(do5s, tl.trans(weight_two))

            ## We accumulate the partial results. ##
            o1 += tl.dot(activations, weight_one)
            o2 += tl.dot(activations, weight_three)

            ## Increment pointers. ##
            activation_ptrs += d_m_block
            weight_one_ptrs += d_m_block * stride_w1_a
            weight_three_ptrs += d_m_block * stride_w3_a
            do5_ptrs += d_m_block * stride_incoming_grads_b
            weight_two_ptrs += d_m_block * stride_w2_b

        ## All of these are of size: (BLOCK_Y, BLOCK_X).
        sig_o1 = tl.sigmoid(o1)
        o3 = o1 * sig_o1
        o4 = o3 * o2 
        do2 = do4 * o3
        do3 = do4 * o2
        do1 = do3 * (sig_o1 + o3 * (tl.full((BLOCK_Y, BLOCK_X), 1, tl.float32) - sig_o1))
        ## Next, we have to downcast the intermediate activations to the pertinent data-type prior to second mat-mul. ##
        if ACTIVATION == "bfloat16":
            o4 = o4.to(tl.bfloat16) ## Size: (BLOCK_Y, BLOCK_X)
            do2 = do2.to(tl.bfloat16) ## Size: (BLOCK_Y, BLOCK_X)
            do1 = do1.to(tl.bfloat16) ## Size: (BLOCK_Y, BLOCK_X)

        ## Load the input activations. ##
        activation_ptrs_consec = tl.max_contiguous(
            tl.arange(0, d_m_block) * stride_inp_b \
                        + j * BLOCK_Y * stride_inp_a, d_m_block
        )
        activation_ptrs_consec = activation_ptrs_consec[None, :]
        activation_ptrs = inp_ptr \
                            + tl.arange(0, BLOCK_Y)[:, None] * stride_inp_a \
                            + activation_ptrs_consec

        dw3_ptrs = dw3_ptr + tl.arange(0, d_m_block)[:, None] * stride_dw3_a \
                            + tl.arange(0, BLOCK_X)[None, :] * stride_dw3_b \
                            + bid_x * BLOCK_X * stride_dw3_b

        dw1_ptrs = dw1_ptr + tl.arange(0, d_m_block)[:, None] * stride_dw1_a \
                            + tl.arange(0, BLOCK_X)[None, :] * stride_dw1_b \
                            + bid_x * BLOCK_X * stride_dw1_b
        ## This is for computing dw2. ##
        do5_ptrs = incoming_grads_ptr + tl.arange(0, BLOCK_Y)[:, None] * stride_incoming_grads_a \
                    + tl.arange(0, d_m_block)[None, :] * stride_incoming_grads_b \
                    + j * BLOCK_Y * stride_incoming_grads_a \

        dw2_ptrs = dw2_ptr + tl.arange(0, d_m_block)[None, :] * stride_dw2_b \
                        + tl.arange(0, BLOCK_X)[:, None] * stride_dw2_a \
                        + bid_x * BLOCK_X * stride_dw2_a 

        ## Compute the answers. ##
        for k in tl.range(0, tl.cdiv(d_m, d_m_block)):

            activs = tl.load(activation_ptrs, mask=(\
                        (tl.arange(0, BLOCK_Y)[:, None] + j * BLOCK_Y < batch_size * N) & \
                        (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m) 
                ), other=0.0)
            do5s = tl.load(do5_ptrs, mask= \
                            (tl.arange(0, BLOCK_Y)[:, None] + j * BLOCK_Y < N * batch_size) & \
                            (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m), other=0.0)

            ## Mat-mul. ##
            dw2s = tl.dot(tl.trans(o4), do5s) # Output is size: (BLOCK_X, d_m_block)

            dw3s = tl.dot(tl.trans(activs), do2) ## Size: (d_m_block, BLOCK_X)
            dw1s = tl.dot(tl.trans(activs), do1) ## Size: (d_m_block, BLOCK_X)

            ## Write to dw3 and dw2 memory locations. ##
            _load_add_store(dw1s, dw1_ptrs, mask=(\
                            (tl.arange(0, BLOCK_X)[None, :] + bid_x * BLOCK_X < d_ffn) & \
                            (tl.arange(0, d_m_block)[:, None] + k * d_m_block < d_m) 
                ))

            _load_add_store(dw3s, dw3_ptrs, mask=(\
                            (tl.arange(0, BLOCK_X)[None, :] + bid_x * BLOCK_X < d_ffn) & \
                            (tl.arange(0, d_m_block)[:, None] + k * d_m_block < d_m) 
                ))
            _load_add_store(dw2s, dw2_ptrs, mask=(\
                    (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m) & \
                    (tl.arange(0, BLOCK_X)[:, None] + bid_x * BLOCK_X < d_ffn)
                ))

            activation_ptrs += d_m_block * stride_inp_b
            dw1_ptrs += d_m_block * stride_dw1_a
            dw3_ptrs += d_m_block * stride_dw3_a
            do5_ptrs += d_m_block * stride_incoming_grads_b
            dw2_ptrs += d_m_block * stride_dw2_b



@triton.jit
def _bwd_kernel_dw1_dw3_par_ffn(
    ## Input activations. ##
    inp_ptr,
    ## Weight matrices. ##
    w1_ptr, w2_ptr, w3_ptr,
    ## Gradient matrices. ##
    incoming_grads_ptr, 
    dw1_ptr, dw3_ptr,
    ## Dimensions of the overall computation. ##
    N: tl.constexpr, d_ffn: tl.constexpr, d_m: tl.constexpr, batch_size: tl.constexpr,
    ## Strides. ##

    ## Strides of input activations. ##
    ## Inputs reshaped to 2-d tensor. ##
    stride_inp_a: tl.constexpr, stride_inp_b: tl.constexpr, 

    ## Strides of weight matrices. ##
    stride_w1_a: tl.constexpr, stride_w1_b: tl.constexpr,
    stride_w2_a: tl.constexpr, stride_w2_b: tl.constexpr,
    stride_w3_a: tl.constexpr, stride_w3_b: tl.constexpr,

    ## Strides of gradient matrices. ##
    stride_incoming_grads_a: tl.constexpr, stride_incoming_grads_b: tl.constexpr, 
    stride_dw1_a: tl.constexpr, stride_dw1_b: tl.constexpr,
    stride_dw3_a: tl.constexpr, stride_dw3_b: tl.constexpr,

    ## Lastly, the activation function. ##
    ACTIVATION: tl.constexpr,
    ## Block Sizes. We default to parallelising over batch dimension. ##
    BLOCK_Y: tl.constexpr, BLOCK_X: tl.constexpr, d_m_block: tl.constexpr):
    """
    Input sizes:
    inp_ptr (original activations): [b, s, d_m] matrix.
    w1_ptr (first weight matrix): [d_m, d_ffn] matrix.
    w2_ptr (second weight matrix): [d_ffn, d_m] matrix.
    w3_ptr (third weight matrix): [d_m, d_ffn] matrxi (identical to w1).
    output_ptr (final output activations): [b, s, d_m] matrix.

    This kernel computes the gradients of w1 and w3 in the following computation:
    (silu(inp @ w1) * inp @ w3) @ w2
    """
    bid_x = tl.program_id(axis=0)

    for j in tl.range(0, tl.cdiv(N * batch_size, BLOCK_Y)):
        ## This is necessary to compute dw2.
        ## We load the data from the input activations. ##
        activation_ptrs_consec = tl.max_contiguous(
            tl.arange(0, d_m_block) * stride_inp_b \
                        + j * BLOCK_Y * stride_inp_a, d_m_block
        )
        activation_ptrs_consec = activation_ptrs_consec[None, :]
        activation_ptrs = inp_ptr \
                            + tl.arange(0, BLOCK_Y)[:, None] * stride_inp_a \
                            + activation_ptrs_consec

        ## We additionally tile across the hidden dimension to reduce shmem consumption. ##
        ## We pull out the base pointer's compute arithmetic. ##
        weight_one_ptrs_consec = tl.max_contiguous(
            tl.arange(0, BLOCK_X) * stride_w1_b, BLOCK_X
        )
        weight_one_ptrs_consec = weight_one_ptrs_consec[None, :]
        weight_one_ptrs = w1_ptr + bid_x * BLOCK_X * stride_w1_b \
                            + tl.arange(0, d_m_block)[:, None] * stride_w1_a \
                            + weight_one_ptrs_consec

        weight_three_ptrs_consec = tl.max_contiguous(
            tl.arange(0, BLOCK_X) * stride_w3_b, BLOCK_X
        )
        weight_three_ptrs_consec = weight_three_ptrs_consec[None, :]
        weight_three_ptrs = w3_ptr + bid_x * BLOCK_X * stride_w3_b \
                            + tl.arange(0, d_m_block)[:, None] * stride_w3_a \
                            + weight_three_ptrs_consec

        do5_ptrs = incoming_grads_ptr + tl.arange(0, BLOCK_Y)[:, None] * stride_incoming_grads_a \
                    + tl.arange(0, d_m_block)[None, :] * stride_incoming_grads_b \
                    + j * BLOCK_Y * stride_incoming_grads_a \

        weight_two_ptrs = w2_ptr + tl.arange(0, BLOCK_X)[:, None] * stride_w2_a \
                            + tl.arange(0, d_m_block)[None, :] * stride_w2_b \
                            + bid_x * BLOCK_X * stride_w2_a

        o1 = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
        o2 = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
        do4 = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(d_m, d_m_block)):  
            ## First, we load the data from the input activations. ##
            activations = tl.load(activation_ptrs, mask=( \
                                    (tl.arange(0, BLOCK_Y)[:, None] + j * BLOCK_Y < N * batch_size) \
                                    & (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m)), other=0.0)
            ## Next, we load the first & third weight matrices. ##
            weight_one = tl.load(weight_one_ptrs, mask=( \
                                    (tl.arange(0, d_m_block)[:, None] + k * d_m_block < d_m) & \
                                    (tl.arange(0, BLOCK_X)[None, :] + bid_x * BLOCK_X < d_ffn)), other=0.0)

            weight_three = tl.load(weight_three_ptrs, mask=( \
                                    (tl.arange(0, d_m_block)[:, None] + k * d_m_block < d_m) & \
                                    (tl.arange(0, BLOCK_X)[None, :] + bid_x * BLOCK_X < d_ffn)), other=0.0)
            ## Finally we load do5 and w2 to compute do4. ##
            do5s = tl.load(do5_ptrs, mask=(\
                            (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m) &
                            (tl.arange(0, BLOCK_Y)[:, None] + j * BLOCK_Y < N * batch_size)
                ), other=0.0)

            weight_two = tl.load(weight_two_ptrs, mask=(\
                                    (tl.arange(0, BLOCK_X)[:, None] + bid_x * BLOCK_X < d_ffn) & \
                                    (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m)
                ))

            do4 += tl.dot(do5s, tl.trans(weight_two))

            ## We accumulate the partial results. ##
            o1 += tl.dot(activations, weight_one)
            o2 += tl.dot(activations, weight_three)

            ## Increment pointers. ##
            activation_ptrs += d_m_block
            weight_one_ptrs += d_m_block * stride_w1_a
            weight_three_ptrs += d_m_block * stride_w3_a
            do5_ptrs += d_m_block * stride_incoming_grads_b
            weight_two_ptrs += d_m_block * stride_w2_b

        ## All of these are of size: (BLOCK_Y, BLOCK_X).
        sig_o1 = tl.sigmoid(o1)
        o3 = o1 * sig_o1
        o4 = o3 * o2 
        do2 = do4 * o3
        do3 = do4 * o2
        do1 = do3 * (sig_o1 + o3 * (tl.full((BLOCK_Y, BLOCK_X), 1, tl.float32) - sig_o1))
        ## Next, we have to downcast the intermediate activations to the pertinent data-type prior to second mat-mul. ##
        if ACTIVATION == "bfloat16":
            o4 = o4.to(tl.bfloat16) ## Size: (BLOCK_Y, BLOCK_X)
            do2 = do2.to(tl.bfloat16) ## Size: (BLOCK_Y, BLOCK_X)
            do1 = do1.to(tl.bfloat16) ## Size: (BLOCK_Y, BLOCK_X)

        ## Load the input activations. ##
        activation_ptrs_consec = tl.max_contiguous(
            tl.arange(0, d_m_block) * stride_inp_b \
                        + j * BLOCK_Y * stride_inp_a, d_m_block
        )
        activation_ptrs_consec = activation_ptrs_consec[None, :]
        activation_ptrs = inp_ptr \
                            + tl.arange(0, BLOCK_Y)[:, None] * stride_inp_a \
                            + activation_ptrs_consec

        dw3_ptrs = dw3_ptr + tl.arange(0, d_m_block)[:, None] * stride_dw3_a \
                            + tl.arange(0, BLOCK_X)[None, :] * stride_dw3_b \
                            + bid_x * BLOCK_X * stride_dw3_b

        dw1_ptrs = dw1_ptr + tl.arange(0, d_m_block)[:, None] * stride_dw1_a \
                            + tl.arange(0, BLOCK_X)[None, :] * stride_dw1_b \
                            + bid_x * BLOCK_X * stride_dw1_b

        ## Compute the answers. ##
        for k in tl.range(0, tl.cdiv(d_m, d_m_block)):

            activs = tl.load(activation_ptrs, mask=(\
                        (tl.arange(0, BLOCK_Y)[:, None] + j * BLOCK_Y < batch_size * N) & \
                        (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m) 
                ), other=0.0)

            dw3s = tl.dot(tl.trans(activs), do2) ## Size: (d_m_block, BLOCK_X)
            dw1s = tl.dot(tl.trans(activs), do1) ## Size: (d_m_block, BLOCK_X)

            ## Write to dw3 and dw2 memory locations. ##
            _load_add_store(dw1s, dw1_ptrs, mask=(\
                            (tl.arange(0, BLOCK_X)[None, :] + bid_x * BLOCK_X < d_ffn) & \
                            (tl.arange(0, d_m_block)[:, None] + k * d_m_block < d_m) 
                ))

            _load_add_store(dw3s, dw3_ptrs, mask=(\
                            (tl.arange(0, BLOCK_X)[None, :] + bid_x * BLOCK_X < d_ffn) & \
                            (tl.arange(0, d_m_block)[:, None] + k * d_m_block < d_m) 
                ))

            activation_ptrs += d_m_block * stride_inp_b
            dw1_ptrs += d_m_block * stride_dw1_a
            dw3_ptrs += d_m_block * stride_dw3_a

## Depecrated, bad parallel scheme. ##
#@triton.autotune(
#    configs=get_cuda_autotune_config_bwd(),
#    key=['N', 'd_ffn', 'd_m']
#)
@triton.jit
def _bwd_kernel_dw1_dw3(
    ## Input activations. ##
    inp_ptr,
    ## Weight matrices. ##
    w1_ptr, w2_ptr, w3_ptr,
    ## Gradient matrices. ##
    incoming_grads_ptr, 
    dw1_ptr, dw3_ptr,
    ## Dimensions of the overall computation. ##
    N: tl.constexpr, d_ffn: tl.constexpr, d_m: tl.constexpr, batch_size: tl.constexpr,
    ## Strides. ##

    ## Strides of input activations. ##
    ## Inputs reshaped to 2-d tensor. ##
    stride_inp_a: tl.constexpr, stride_inp_b: tl.constexpr, 

    ## Strides of weight matrices. ##
    stride_w1_a: tl.constexpr, stride_w1_b: tl.constexpr,
    stride_w2_a: tl.constexpr, stride_w2_b: tl.constexpr,
    stride_w3_a: tl.constexpr, stride_w3_b: tl.constexpr,

    ## Strides of gradient matrices. ##
    stride_incoming_grads_a: tl.constexpr, stride_incoming_grads_b: tl.constexpr, 
    stride_dw1_a: tl.constexpr, stride_dw1_b: tl.constexpr,
    stride_dw3_a: tl.constexpr, stride_dw3_b: tl.constexpr,

    ## Lastly, the activation function. ##
    ACTIVATION: tl.constexpr,
    ## Block Sizes. We default to parallelising over batch dimension. ##
    BLOCK_Y: tl.constexpr, BLOCK_X: tl.constexpr, d_m_block: tl.constexpr):
    """
    Input sizes:
    inp_ptr (original activations): [b, s, d_m] matrix.
    w1_ptr (first weight matrix): [d_m, d_ffn] matrix.
    w2_ptr (second weight matrix): [d_ffn, d_m] matrix.
    w3_ptr (third weight matrix): [d_m, d_ffn] matrxi (identical to w1).
    output_ptr (final output activations): [b, s, d_m] matrix.

    This kernel computes the gradients of w1 and w3 in the following computation:
    (silu(inp @ w1) * inp @ w3) @ w2
    """
    bid_x = tl.program_id(axis=0)
    batch = tl.program_id(axis=1)

    for j in tl.range(0, tl.cdiv(d_ffn, BLOCK_X)):
        ## This is necessary to compute dw2.
        ## We load the data from the input activations. ##
        activation_ptrs_consec = tl.max_contiguous(
            tl.arange(0, d_m_block) * stride_inp_b \
                        + bid_x * BLOCK_Y * stride_inp_a, d_m_block
        )
        activation_ptrs_consec = activation_ptrs_consec[None, :]
        activation_ptrs = inp_ptr + batch * N * d_m \
                            + tl.arange(0, BLOCK_Y)[:, None]*stride_inp_a \
                            + activation_ptrs_consec

        ## We additionally tile across the hidden dimension to reduce shmem consumption. ##
        ## We pull out the base pointer's compute arithmetic. ##
        weight_one_ptrs_consec = tl.max_contiguous(
            tl.arange(0, BLOCK_X) * stride_w1_b, BLOCK_X
        )
        weight_one_ptrs_consec = weight_one_ptrs_consec[None, :]
        weight_one_ptrs = w1_ptr + j * BLOCK_X * stride_w1_b \
                            + tl.arange(0, d_m_block)[:, None] * stride_w1_a \
                            + weight_one_ptrs_consec

        weight_three_ptrs_consec = tl.max_contiguous(
            tl.arange(0, BLOCK_X) * stride_w3_b, BLOCK_X
        )
        weight_three_ptrs_consec = weight_three_ptrs_consec[None, :]
        weight_three_ptrs = w3_ptr + j * BLOCK_X * stride_w3_b \
                            + tl.arange(0, d_m_block)[:, None] * stride_w3_a \
                            + weight_three_ptrs_consec

        ## This is necessary to compute dX.
        do5_ptrs = incoming_grads_ptr + tl.arange(0, BLOCK_Y)[:, None] * stride_incoming_grads_a \
                    + tl.arange(0, d_m_block)[None, :] * stride_incoming_grads_b \
                    + bid_x * BLOCK_Y * stride_incoming_grads_a \
                    + batch * N * d_m

        weight_two_ptrs = w2_ptr + tl.arange(0, BLOCK_X)[:, None] * stride_w2_a \
                            + tl.arange(0, d_m_block)[None, :] * stride_w2_b \
                            + j * BLOCK_X * stride_w2_a

        o1 = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
        o2 = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
        do4 = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(d_m, d_m_block)):  
            ## First, we load the data from the input activations. ##
            activations = tl.load(activation_ptrs, mask=( \
                                    (tl.arange(0, BLOCK_Y)[:, None] + bid_x * BLOCK_Y + batch * N < N * batch_size) \
                                    & (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m)), other=0.0)
            ## Next, we load the first & third weight matrices. ##
            weight_one = tl.load(weight_one_ptrs, mask=( \
                                    (tl.arange(0, d_m_block)[:, None] + k * d_m_block < d_m) & \
                                    (tl.arange(0, BLOCK_X)[None, :] + j * BLOCK_X < d_ffn)), other=0.0)

            weight_three = tl.load(weight_three_ptrs, mask=( \
                                    (tl.arange(0, d_m_block)[:, None] + k * d_m_block < d_m) & \
                                    (tl.arange(0, BLOCK_X)[None, :] + j * BLOCK_X < d_ffn)), other=0.0)
            ## Finally we load do5 and w2 to compute do4. ##
            do5s = tl.load(do5_ptrs, mask=(\
                            (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m) &
                            (tl.arange(0, BLOCK_Y)[:, None] + bid_x * BLOCK_Y + batch * N < N * batch_size)
                ), other=0.0)

            weight_two = tl.load(weight_two_ptrs, mask=(\
                                    (tl.arange(0, BLOCK_X)[:, None] + j * BLOCK_X < d_ffn) & \
                                    (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m)
                ))

            do4 += tl.dot(do5s, tl.trans(weight_two))

            ## We accumulate the partial results. ##
            o1 += tl.dot(activations, weight_one)
            o2 += tl.dot(activations, weight_three)

            ## Increment pointers. ##
            activation_ptrs += d_m_block
            weight_one_ptrs += d_m_block * stride_w1_a
            weight_three_ptrs += d_m_block * stride_w3_a
            do5_ptrs += d_m_block * stride_incoming_grads_b
            weight_two_ptrs += d_m_block * stride_w2_b

        ## All of these are of size: (BLOCK_Y, BLOCK_X).
        sig_o1 = tl.sigmoid(o1)
        o3 = o1 * sig_o1
        o4 = o3 * o2 
        do2 = do4 * o3
        do3 = do4 * o2
        do1 = do3 * (sig_o1 + o3 * (tl.full((BLOCK_Y, BLOCK_X), 1, tl.float32) - sig_o1))
        ## Next, we have to downcast the intermediate activations to the pertinent data-type prior to second mat-mul. ##
        if ACTIVATION == "bfloat16":
            o4 = o4.to(tl.bfloat16) ## Size: (BLOCK_Y, BLOCK_X)
            do2 = do2.to(tl.bfloat16) ## Size: (BLOCK_Y, BLOCK_X)
            do1 = do1.to(tl.bfloat16) ## Size: (BLOCK_Y, BLOCK_X)

        ## Load the input activations. ##
        activation_ptrs_consec = tl.max_contiguous(
            tl.arange(0, d_m_block) * stride_inp_b \
                        + bid_x * BLOCK_Y * stride_inp_a, d_m_block
        )
        activation_ptrs_consec = activation_ptrs_consec[None, :]
        activation_ptrs = inp_ptr + batch * N * d_m \
                            + tl.arange(0, BLOCK_Y)[:, None] * stride_inp_a \
                            + activation_ptrs_consec

        dw3_ptrs = dw3_ptr + tl.arange(0, d_m_block)[:, None] * stride_dw3_a \
                            + tl.arange(0, BLOCK_X)[None, :] * stride_dw3_b \
                            + j * BLOCK_X * stride_dw3_b

        dw1_ptrs = dw1_ptr + tl.arange(0, d_m_block)[:, None] * stride_dw1_a \
                            + tl.arange(0, BLOCK_X)[None, :] * stride_dw1_b \
                            + j * BLOCK_X * stride_dw1_b

        ## Compute the answers. ##
        for k in tl.range(0, tl.cdiv(d_m, d_m_block)):

            activs = tl.load(activation_ptrs, mask=(\
                        (tl.arange(0, BLOCK_Y)[:, None] + bid_x * BLOCK_Y + batch * N < batch_size * N) & \
                        (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m) 
                ), other=0.0)
            dw3s = tl.dot(tl.trans(activs), do2) ## Size: (d_m_block, BLOCK_X)
            dw1s = tl.dot(tl.trans(activs), do1) ## Size: (d_m_block, BLOCK_X)

            ## Write to dw3 and dw2 memory locations. ##
            tl.atomic_add(dw1_ptrs, dw1s.to(tl.float32), mask=(\
                            (tl.arange(0, BLOCK_X)[None, :] + j * BLOCK_X < d_ffn) & \
                            (tl.arange(0, d_m_block)[:, None] + k * d_m_block < d_m) 
                ))

            tl.atomic_add(dw3_ptrs, dw3s.to(tl.float32), mask=(\
                            (tl.arange(0, BLOCK_X)[None, :] + j * BLOCK_X < d_ffn) & \
                            (tl.arange(0, d_m_block)[:, None] + k * d_m_block < d_m) 
                ))

            activation_ptrs += d_m_block * stride_inp_b
            dw1_ptrs += d_m_block * stride_dw1_a
            dw3_ptrs += d_m_block * stride_dw3_a


@triton.jit
def _bwd_kernel_dx(
    ## Input activations. ##
    inp_ptr,
    ## Weight matrices. ##
    w1_ptr, w2_ptr, w3_ptr,
    ## Gradient matrices. ##
    incoming_grads_ptr, 
    dx_ptr, 
    ## Dimensions of the overall computation. ##
    N: tl.constexpr, d_ffn: tl.constexpr, d_m: tl.constexpr, batch_size: tl.constexpr,
    ## Strides. ##

    ## Strides of input activations. ##
    ## Inputs reshaped to 2-d tensor. ##
    stride_inp_a: tl.constexpr, stride_inp_b: tl.constexpr, 

    ## Strides of weight matrices. ##
    stride_w1_a: tl.constexpr, stride_w1_b: tl.constexpr,
    stride_w2_a: tl.constexpr, stride_w2_b: tl.constexpr,
    stride_w3_a: tl.constexpr, stride_w3_b: tl.constexpr,

    ## Strides of gradient matrices. ##
    stride_incoming_grads_a: tl.constexpr, stride_incoming_grads_b: tl.constexpr, 
    stride_dx_a: tl.constexpr, stride_dx_b: tl.constexpr,

    ## Lastly, the activation function. ##
    ACTIVATION: tl.constexpr,
    ## Block Sizes. We default to parallelising over batch dimension. ##
    BLOCK_Y: tl.constexpr, BLOCK_X: tl.constexpr, d_m_block: tl.constexpr):
    """
    Input sizes:
    inp_ptr (original activations): [b, s, d_m] matrix.
    w1_ptr (first weight matrix): [d_m, d_ffn] matrix.
    w2_ptr (second weight matrix): [d_ffn, d_m] matrix.
    w3_ptr (third weight matrix): [d_m, d_ffn] matrxi (identical to w1).
    output_ptr (final output activations): [b, s, d_m] matrix.

    This kernel computes the gradients of inp and w2 in the following computation:
    (silu(inp @ w1) * inp @ w3) @ w2
    """
    bid_x = tl.program_id(axis=0)
    batch = tl.program_id(axis=1)

    for j in tl.range(0, tl.cdiv(d_ffn, BLOCK_X)):
        ## This is necessary to compute dw2.
        ## We load the data from the input activations. ##
        activation_ptrs_consec = tl.max_contiguous(
            tl.arange(0, d_m_block) * stride_inp_b \
                        + bid_x * BLOCK_Y * stride_inp_a, d_m_block
        )
        activation_ptrs_consec = activation_ptrs_consec[None, :]
        activation_ptrs = inp_ptr + batch * N * d_m \
                            + tl.arange(0, BLOCK_Y)[:, None]*stride_inp_a \
                            + activation_ptrs_consec

        ## We additionally tile across the hidden dimension to reduce shmem consumption. ##
        ## We pull out the base pointer's compute arithmetic. ##
        weight_one_ptrs_consec = tl.max_contiguous(
            tl.arange(0, BLOCK_X)*stride_w1_b, BLOCK_X
        )
        weight_one_ptrs_consec = weight_one_ptrs_consec[None, :]
        weight_one_ptrs = w1_ptr + j * BLOCK_X * stride_w1_b \
                            + tl.arange(0, d_m_block)[:, None]*stride_w1_a \
                            + weight_one_ptrs_consec

        weight_three_ptrs_consec = tl.max_contiguous(
            tl.arange(0, BLOCK_X) * stride_w3_b, BLOCK_X
        )
        weight_three_ptrs_consec = weight_three_ptrs_consec[None, :]
        weight_three_ptrs = w3_ptr + j * BLOCK_X * stride_w3_b \
                            + tl.arange(0, d_m_block)[:, None] * stride_w3_a \
                            + weight_three_ptrs_consec

        ## This is necessary to compute dX.
        do5_ptrs = incoming_grads_ptr + tl.arange(0, BLOCK_Y)[:, None] * stride_incoming_grads_a \
                    + tl.arange(0, d_m_block)[None, :] * stride_incoming_grads_b \
                    + bid_x * BLOCK_Y * stride_incoming_grads_a \
                    + batch * N * d_m

        weight_two_ptrs = w2_ptr + tl.arange(0, BLOCK_X)[:, None] * stride_w2_a \
                            + tl.arange(0, d_m_block)[None, :] * stride_w2_b \
                            + j * BLOCK_X * stride_w2_a

        o1 = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
        o2 = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
        do4 = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(d_m, d_m_block)):  
            ## First, we load the data from the input activations. ##
            activations = tl.load(activation_ptrs, mask=( \
                                    (tl.arange(0, BLOCK_Y)[:, None] + bid_x * BLOCK_Y + batch * N < N * batch_size) \
                                    & (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m)), other=0.0)
            ## Next, we load the first & third weight matrices. ##
            weight_one = tl.load(weight_one_ptrs, mask=( \
                                    (tl.arange(0, d_m_block)[:, None] + k * d_m_block < d_m) & \
                                    (tl.arange(0, BLOCK_X)[None, :] + j * BLOCK_X < d_ffn)), other=0.0)

            weight_three = tl.load(weight_three_ptrs, mask=( \
                                    (tl.arange(0, d_m_block)[:, None] + k * d_m_block < d_m) & \
                                    (tl.arange(0, BLOCK_X)[None, :] + j * BLOCK_X < d_ffn)), other=0.0)
            ## Finally we load do5 and w2 to compute do4. ##
            do5s = tl.load(do5_ptrs, mask=(\
                            (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m) &
                            (tl.arange(0, BLOCK_Y)[:, None] + bid_x * BLOCK_Y + batch * N < N * batch_size)
                ), other=0.0)

            weight_two = tl.load(weight_two_ptrs, mask=(\
                                    (tl.arange(0, BLOCK_X)[:, None] + j * BLOCK_X < d_ffn) & \
                                    (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m)
                ))

            do4 += tl.dot(do5s, tl.trans(weight_two))

            ## We accumulate the partial results. ##
            o1 += tl.dot(activations, weight_one)
            o2 += tl.dot(activations, weight_three)

            ## Increment pointers. ##
            activation_ptrs += d_m_block
            weight_one_ptrs += d_m_block * stride_w1_a
            weight_three_ptrs += d_m_block * stride_w3_a
            do5_ptrs += d_m_block * stride_incoming_grads_b
            weight_two_ptrs += d_m_block * stride_w2_b

        ## All of these are of size: (BLOCK_Y, BLOCK_X).
        sig_o1 = tl.sigmoid(o1)
        o3 = o1 * sig_o1
        do2 = do4 * o3
        do3 = do4 * o2
        do1 = do3 * (sig_o1 + o3 * (tl.full((BLOCK_Y, BLOCK_X), 1, tl.float32) - sig_o1))
        ## Next, we have to downcast the intermediate activations to the pertinent data-type prior to second mat-mul. ##
        if ACTIVATION == "bfloat16":
            do2 = do2.to(tl.bfloat16)
            do1 = do1.to(tl.bfloat16)

        ## This is for computing dX. ##
        weight_three_ptrs = w3_ptr + tl.arange(0, d_m_block)[:, None] * stride_w3_a \
                                   + tl.arange(0, BLOCK_X)[None, :] * stride_w3_b \
                                   + j * BLOCK_X

        weight_one_ptrs = w1_ptr + tl.arange(0, d_m_block)[:, None] * stride_w1_a \
                                 + tl.arange(0, BLOCK_X)[None, :] * stride_w1_b \
                                 + j * BLOCK_X
        
        dx_ptrs = dx_ptr + tl.arange(0, BLOCK_Y)[:, None] * stride_dx_a \
                         + tl.arange(0, d_m_block)[None, :] * stride_dx_b \
                         + bid_x * BLOCK_Y * stride_dx_a \
                         + batch * N * d_m 

        ## Compute the answers. ##
        for k in tl.range(0, tl.cdiv(d_m, d_m_block)):
            w3s = tl.load(weight_three_ptrs, mask=(\
                            (tl.arange(0, d_m_block)[:, None] + k * d_m_block < d_m) & \
                            (tl.arange(0, BLOCK_X)[None, :] + j * BLOCK_X < d_ffn)
                ))

            w1s = tl.load(weight_one_ptrs, mask=(\
                            (tl.arange(0, d_m_block)[:, None] + k * d_m_block < d_m) & \
                            (tl.arange(0, BLOCK_X)[None, :] + j * BLOCK_X < d_ffn) 
                ))

            ## Mat-mul. ##
            dx = tl.dot(do2, tl.trans(w3s)) + tl.dot(do1, tl.trans(w1s)) # Output is size: (BLOCK_Y, d_m_block)

            ## Load add store to dx. ##
            _load_add_store(dx, dx_ptrs, mask=(\
                    (tl.arange(0, BLOCK_Y)[:, None] + bid_x * BLOCK_Y + batch * N < N * batch_size) & \
                    (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m) 
                ))

            weight_three_ptrs += d_m_block * stride_w3_a
            weight_one_ptrs += d_m_block * stride_w1_a
            dx_ptrs += d_m_block * stride_dx_b

## This dw2 parallelises over ffn and sequentially tiles over the N x batch dimension.
## Doing so enables us to not use atomic_adds within hot inner loops.
@triton.jit
def _bwd_kernel_dw2_par_ffn(
    ## Input activations. ##
    inp_ptr,
    ## Weight matrices. ##
    w1_ptr, w3_ptr,
    ## Gradient matrices. ##
    incoming_grads_ptr, 
    dw2_ptr,
    ## Dimensions of the overall computation. ##
    N: tl.constexpr, d_ffn: tl.constexpr, d_m: tl.constexpr, batch_size: tl.constexpr,
    ## Strides. ##

    ## Strides of input activations. ##
    ## Inputs reshaped to 2-d tensor. ##
    stride_inp_a: tl.constexpr, stride_inp_b: tl.constexpr, 

    ## Strides of weight matrices. ##
    stride_w1_a: tl.constexpr, stride_w1_b: tl.constexpr,
    stride_w3_a: tl.constexpr, stride_w3_b: tl.constexpr,

    ## Strides of gradient matrices. ##
    stride_incoming_grads_a: tl.constexpr, stride_incoming_grads_b: tl.constexpr, 
    stride_dw2_a: tl.constexpr, stride_dw2_b: tl.constexpr,

    ## Lastly, the activation function. ##
    ACTIVATION: tl.constexpr,
    ## Block Sizes. We default to parallelising over batch dimension. ##
    BLOCK_Y: tl.constexpr, BLOCK_X: tl.constexpr, d_m_block: tl.constexpr):
    """
    Input sizes:
    inp_ptr (original activations): [b, s, d_m] matrix.
    w1_ptr (first weight matrix): [d_m, d_ffn] matrix.
    w2_ptr (second weight matrix): [d_ffn, d_m] matrix.
    w3_ptr (third weight matrix): [d_m, d_ffn] matrxi (identical to w1).
    output_ptr (final output activations): [b, s, d_m] matrix.

    This kernel computes the gradients of w2 in the following computation:
    (silu(inp @ w1) * inp @ w3) @ w2
    """
    bid_x = tl.program_id(axis=0)

    for j in tl.range(0, tl.cdiv(N * batch_size, BLOCK_Y)):
        ## This is necessary to compute dw2.
        ## We load the data from the input activations. ##
        activation_ptrs_consec = tl.max_contiguous(
            tl.arange(0, d_m_block) * stride_inp_b \
                        + j * BLOCK_Y * stride_inp_a, d_m_block
        )
        activation_ptrs_consec = activation_ptrs_consec[None, :]
        activation_ptrs = inp_ptr \
                            + tl.arange(0, BLOCK_Y)[:, None] * stride_inp_a \
                            + activation_ptrs_consec

        ## We additionally tile across the hidden dimension to reduce shmem consumption. ##
        ## We pull out the base pointer's compute arithmetic. ##
        weight_one_ptrs_consec = tl.max_contiguous(
            tl.arange(0, BLOCK_X) * stride_w1_b, BLOCK_X
        )
        weight_one_ptrs_consec = weight_one_ptrs_consec[None, :]
        weight_one_ptrs = w1_ptr + bid_x * BLOCK_X * stride_w1_b \
                            + tl.arange(0, d_m_block)[:, None] * stride_w1_a \
                            + weight_one_ptrs_consec

        weight_three_ptrs_consec = tl.max_contiguous(
            tl.arange(0, BLOCK_X) * stride_w3_b, BLOCK_X
        )
        weight_three_ptrs_consec = weight_three_ptrs_consec[None, :]
        weight_three_ptrs = w3_ptr + bid_x * BLOCK_X * stride_w3_b \
                            + tl.arange(0, d_m_block)[:, None] * stride_w3_a \
                            + weight_three_ptrs_consec

        o1 = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
        o2 = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(d_m, d_m_block)):  
            ## First, we load the data from the input activations. ##
            activations = tl.load(activation_ptrs, mask=( \
                                    (tl.arange(0, BLOCK_Y)[:, None] + j * BLOCK_Y < N * batch_size) \
                                    & (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m)), other=0.0)
            ## Next, we load the first & third weight matrices. ##
            weight_one = tl.load(weight_one_ptrs, mask=( \
                                    (tl.arange(0, d_m_block)[:, None] + k * d_m_block < d_m) & \
                                    (tl.arange(0, BLOCK_X)[None, :] + bid_x * BLOCK_X < d_ffn)), other=0.0)

            weight_three = tl.load(weight_three_ptrs, mask=( \
                                    (tl.arange(0, d_m_block)[:, None] + k * d_m_block < d_m) & \
                                    (tl.arange(0, BLOCK_X)[None, :] + bid_x * BLOCK_X < d_ffn)), other=0.0)

            ## We accumulate the partial results. ##
            o1 += tl.dot(activations, weight_one)
            o2 += tl.dot(activations, weight_three)

            ## Increment pointers. ##
            activation_ptrs += d_m_block * stride_inp_b
            weight_one_ptrs += d_m_block * stride_w1_a
            weight_three_ptrs += d_m_block * stride_w3_a

        ## All of these are of size: (BLOCK_Y, BLOCK_X).
        sig_o1 = tl.sigmoid(o1)
        o3 = o1 * sig_o1
        o4 = o3 * o2 
        ## Next, we have to downcast the intermediate activations to the pertinent data-type prior to second mat-mul. ##
        if ACTIVATION == "bfloat16":
            o4 = o4.to(tl.bfloat16) ## Size: (BLOCK_Y, BLOCK_X)

        ## This is for computing dw2. ##
        do5_ptrs = incoming_grads_ptr + tl.arange(0, BLOCK_Y)[:, None] * stride_incoming_grads_a \
                    + tl.arange(0, d_m_block)[None, :] * stride_incoming_grads_b \
                    + j * BLOCK_Y * stride_incoming_grads_a \

        dw2_ptrs = dw2_ptr + tl.arange(0, d_m_block)[None, :] * stride_dw2_b \
                        + tl.arange(0, BLOCK_X)[:, None] * stride_dw2_a \
                        + bid_x * BLOCK_X * stride_dw2_a 

        ## Compute the answers. ##
        for k in tl.range(0, tl.cdiv(d_m, d_m_block)):
            ## Load do5. ##
            do5s = tl.load(do5_ptrs, mask= \
                            (tl.arange(0, BLOCK_Y)[:, None] + j * BLOCK_Y < N * batch_size) & \
                            (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m), other=0.0)

            ## Mat-mul. ##
            dw2s = tl.dot(tl.trans(o4), do5s) # Output is size: (BLOCK_X, d_m_block)
            ## add atomic_write to main memory. ##
            _load_add_store(dw2s, dw2_ptrs, mask=(\
                    (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m) & \
                    (tl.arange(0, BLOCK_X)[:, None] + bid_x * BLOCK_X < d_ffn)
                ))

            do5_ptrs += d_m_block * stride_incoming_grads_b
            dw2_ptrs += d_m_block * stride_dw2_b

## Deprecate this, the parallel scheme is not good. ##
@triton.jit
def _bwd_kernel_dw2(
    ## Input activations. ##
    inp_ptr,
    ## Weight matrices. ##
    w1_ptr, w3_ptr,
    ## Gradient matrices. ##
    incoming_grads_ptr, 
    dw2_ptr,
    ## Dimensions of the overall computation. ##
    N: tl.constexpr, d_ffn: tl.constexpr, d_m: tl.constexpr, batch_size: tl.constexpr,
    ## Strides. ##

    ## Strides of input activations. ##
    ## Inputs reshaped to 2-d tensor. ##
    stride_inp_a: tl.constexpr, stride_inp_b: tl.constexpr, 

    ## Strides of weight matrices. ##
    stride_w1_a: tl.constexpr, stride_w1_b: tl.constexpr,
    stride_w3_a: tl.constexpr, stride_w3_b: tl.constexpr,

    ## Strides of gradient matrices. ##
    stride_incoming_grads_a: tl.constexpr, stride_incoming_grads_b: tl.constexpr, 
    stride_dw2_a: tl.constexpr, stride_dw2_b: tl.constexpr,

    ## Lastly, the activation function. ##
    ACTIVATION: tl.constexpr,
    ## Block Sizes. We default to parallelising over batch dimension. ##
    BLOCK_Y: tl.constexpr, BLOCK_X: tl.constexpr, d_m_block: tl.constexpr):
    """
    Input sizes:
    inp_ptr (original activations): [b, s, d_m] matrix.
    w1_ptr (first weight matrix): [d_m, d_ffn] matrix.
    w2_ptr (second weight matrix): [d_ffn, d_m] matrix.
    w3_ptr (third weight matrix): [d_m, d_ffn] matrxi (identical to w1).
    output_ptr (final output activations): [b, s, d_m] matrix.

    This kernel computes the gradients of inp and w2 in the following computation:
    (silu(inp @ w1) * inp @ w3) @ w2
    """
    bid_x = tl.program_id(axis=0)
    batch = tl.program_id(axis=1)

    for j in tl.range(0, tl.cdiv(d_ffn, BLOCK_X)):
        ## This is necessary to compute dw2.
        ## We load the data from the input activations. ##
        activation_ptrs_consec = tl.max_contiguous(
            tl.arange(0, d_m_block) * stride_inp_b \
                        + bid_x * BLOCK_Y * stride_inp_a, d_m_block
        )
        activation_ptrs_consec = activation_ptrs_consec[None, :]
        activation_ptrs = inp_ptr + batch * N * d_m \
                            + tl.arange(0, BLOCK_Y)[:, None]*stride_inp_a \
                            + activation_ptrs_consec

        ## We additionally tile across the hidden dimension to reduce shmem consumption. ##
        ## We pull out the base pointer's compute arithmetic. ##
        weight_one_ptrs_consec = tl.max_contiguous(
            tl.arange(0, BLOCK_X)*stride_w1_b, BLOCK_X
        )
        weight_one_ptrs_consec = weight_one_ptrs_consec[None, :]
        weight_one_ptrs = w1_ptr + j * BLOCK_X * stride_w1_b \
                            + tl.arange(0, d_m_block)[:, None]*stride_w1_a \
                            + weight_one_ptrs_consec

        weight_three_ptrs_consec = tl.max_contiguous(
            tl.arange(0, BLOCK_X) * stride_w3_b, BLOCK_X
        )
        weight_three_ptrs_consec = weight_three_ptrs_consec[None, :]
        weight_three_ptrs = w3_ptr + j * BLOCK_X * stride_w3_b \
                            + tl.arange(0, d_m_block)[:, None] * stride_w3_a \
                            + weight_three_ptrs_consec

        o1 = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
        o2 = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(d_m, d_m_block)):  
            ## First, we load the data from the input activations. ##
            activations = tl.load(activation_ptrs, mask=( \
                                    (tl.arange(0, BLOCK_Y)[:, None] + bid_x * BLOCK_Y + batch * N < N * batch_size) \
                                    & (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m)), other=0.0)
            ## Next, we load the first & third weight matrices. ##
            weight_one = tl.load(weight_one_ptrs, mask=( \
                                    (tl.arange(0, d_m_block)[:, None] + k * d_m_block < d_m) & \
                                    (tl.arange(0, BLOCK_X)[None, :] + j * BLOCK_X < d_ffn)), other=0.0)

            weight_three = tl.load(weight_three_ptrs, mask=( \
                                    (tl.arange(0, d_m_block)[:, None] + k * d_m_block < d_m) & \
                                    (tl.arange(0, BLOCK_X)[None, :] + j * BLOCK_X < d_ffn)), other=0.0)

            ## We accumulate the partial results. ##
            o1 += tl.dot(activations, weight_one)
            o2 += tl.dot(activations, weight_three)

            ## Increment pointers. ##
            activation_ptrs += d_m_block
            weight_one_ptrs += d_m_block * stride_w1_a
            weight_three_ptrs += d_m_block * stride_w3_a

        ## All of these are of size: (BLOCK_Y, BLOCK_X).
        sig_o1 = tl.sigmoid(o1)
        o3 = o1 * sig_o1
        o4 = o3 * o2 
        ## Next, we have to downcast the intermediate activations to the pertinent data-type prior to second mat-mul. ##
        if ACTIVATION == "bfloat16":
            o4 = o4.to(tl.bfloat16) ## Size: (BLOCK_Y, BLOCK_X)

        ## This is for computing dw2. ##
        do5_ptrs = incoming_grads_ptr + tl.arange(0, BLOCK_Y)[:, None] * stride_incoming_grads_a \
                    + tl.arange(0, d_m_block)[None, :] * stride_incoming_grads_b \
                    + bid_x * BLOCK_Y * stride_incoming_grads_a \
                    + batch * N * d_m

        dw2_ptrs = dw2_ptr + tl.arange(0, d_m_block)[None, :] * stride_dw2_b \
                        + tl.arange(0, BLOCK_X)[:, None] * stride_dw2_a \
                        + j * BLOCK_X * stride_dw2_a 

        ## Compute the answers. ##
        for k in tl.range(0, tl.cdiv(d_m, d_m_block)):
            ## Load do5. ##
            do5s = tl.load(do5_ptrs, mask= \
                            (tl.arange(0, BLOCK_Y)[:, None] + bid_x * BLOCK_Y + batch * N < N * batch_size) & \
                            (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m), other=0.0)

            ## Mat-mul. ##
            dw2s = tl.dot(tl.trans(o4), do5s) # Output is size: (BLOCK_X, d_m_block)
            ## add atomic_write to main memory. ##
            tl.atomic_add(dw2_ptrs, dw2s, mask=(\
                    (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m) & \
                    (tl.arange(0, BLOCK_X)[:, None] + j * BLOCK_X < d_ffn)
                ))

            do5_ptrs += d_m_block * stride_incoming_grads_b
            dw2_ptrs += d_m_block * stride_dw2_b

## Deprecate this, the parallel scheme is not good. ##
#@triton.autotune(
#    configs=get_cuda_autotune_config_bwd(),
#    key=['N', 'd_ffn', 'd_m']
#)
@triton.jit
def _bwd_kernel_dx_dw2(
    ## Input activations. ##
    inp_ptr,
    ## Weight matrices. ##
    w1_ptr, w2_ptr, w3_ptr,
    ## Gradient matrices. ##
    incoming_grads_ptr, 
    dx_ptr, dw2_ptr,
    ## Dimensions of the overall computation. ##
    N: tl.constexpr, d_ffn: tl.constexpr, d_m: tl.constexpr, batch_size: tl.constexpr,
    ## Strides. ##

    ## Strides of input activations. ##
    ## Inputs reshaped to 2-d tensor. ##
    stride_inp_a: tl.constexpr, stride_inp_b: tl.constexpr, 

    ## Strides of weight matrices. ##
    stride_w1_a: tl.constexpr, stride_w1_b: tl.constexpr,
    stride_w2_a: tl.constexpr, stride_w2_b: tl.constexpr,
    stride_w3_a: tl.constexpr, stride_w3_b: tl.constexpr,

    ## Strides of gradient matrices. ##
    stride_incoming_grads_a: tl.constexpr, stride_incoming_grads_b: tl.constexpr, 
    stride_dx_a: tl.constexpr, stride_dx_b: tl.constexpr,
    stride_dw2_a: tl.constexpr, stride_dw2_b: tl.constexpr,

    ## Lastly, the activation function. ##
    ACTIVATION: tl.constexpr,
    ## Block Sizes. We default to parallelising over batch dimension. ##
    BLOCK_Y: tl.constexpr, BLOCK_X: tl.constexpr, d_m_block: tl.constexpr):
    """
    Input sizes:
    inp_ptr (original activations): [b, s, d_m] matrix.
    w1_ptr (first weight matrix): [d_m, d_ffn] matrix.
    w2_ptr (second weight matrix): [d_ffn, d_m] matrix.
    w3_ptr (third weight matrix): [d_m, d_ffn] matrxi (identical to w1).
    output_ptr (final output activations): [b, s, d_m] matrix.

    This kernel computes the gradients of inp and w2 in the following computation:
    (silu(inp @ w1) * inp @ w3) @ w2
    """
    bid_x = tl.program_id(axis=0)
    batch = tl.program_id(axis=1)

    for j in tl.range(0, tl.cdiv(d_ffn, BLOCK_X)):
        ## This is necessary to compute dw2.
        ## We load the data from the input activations. ##
        activation_ptrs_consec = tl.max_contiguous(
            tl.arange(0, d_m_block) * stride_inp_b \
                        + bid_x * BLOCK_Y * stride_inp_a, d_m_block
        )
        activation_ptrs_consec = activation_ptrs_consec[None, :]
        activation_ptrs = inp_ptr + batch * N * d_m \
                            + tl.arange(0, BLOCK_Y)[:, None]*stride_inp_a \
                            + activation_ptrs_consec

        ## We additionally tile across the hidden dimension to reduce shmem consumption. ##
        ## We pull out the base pointer's compute arithmetic. ##
        weight_one_ptrs_consec = tl.max_contiguous(
            tl.arange(0, BLOCK_X)*stride_w1_b, BLOCK_X
        )
        weight_one_ptrs_consec = weight_one_ptrs_consec[None, :]
        weight_one_ptrs = w1_ptr + j * BLOCK_X * stride_w1_b \
                            + tl.arange(0, d_m_block)[:, None]*stride_w1_a \
                            + weight_one_ptrs_consec

        weight_three_ptrs_consec = tl.max_contiguous(
            tl.arange(0, BLOCK_X) * stride_w3_b, BLOCK_X
        )
        weight_three_ptrs_consec = weight_three_ptrs_consec[None, :]
        weight_three_ptrs = w3_ptr + j * BLOCK_X * stride_w3_b \
                            + tl.arange(0, d_m_block)[:, None] * stride_w3_a \
                            + weight_three_ptrs_consec

        ## This is necessary to compute dX.
        do5_ptrs = incoming_grads_ptr + tl.arange(0, BLOCK_Y)[:, None] * stride_incoming_grads_a \
                    + tl.arange(0, d_m_block)[None, :] * stride_incoming_grads_b \
                    + bid_x * BLOCK_Y * stride_incoming_grads_a \
                    + batch * N * d_m

        weight_two_ptrs = w2_ptr + tl.arange(0, BLOCK_X)[:, None] * stride_w2_a \
                            + tl.arange(0, d_m_block)[None, :] * stride_w2_b \
                            + j * BLOCK_X * stride_w2_a

        o1 = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
        o2 = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
        do4 = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(d_m, d_m_block)):  
            ## First, we load the data from the input activations. ##
            activations = tl.load(activation_ptrs, mask=( \
                                    (tl.arange(0, BLOCK_Y)[:, None] + bid_x * BLOCK_Y + batch * N < N * batch_size) \
                                    & (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m)), other=0.0)
            ## Next, we load the first & third weight matrices. ##
            weight_one = tl.load(weight_one_ptrs, mask=( \
                                    (tl.arange(0, d_m_block)[:, None] + k * d_m_block < d_m) & \
                                    (tl.arange(0, BLOCK_X)[None, :] + j * BLOCK_X < d_ffn)), other=0.0)

            weight_three = tl.load(weight_three_ptrs, mask=( \
                                    (tl.arange(0, d_m_block)[:, None] + k * d_m_block < d_m) & \
                                    (tl.arange(0, BLOCK_X)[None, :] + j * BLOCK_X < d_ffn)), other=0.0)
            ## Finally we load do5 and w2 to compute do4. ##
            do5s = tl.load(do5_ptrs, mask=(\
                            (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m) &
                            (tl.arange(0, BLOCK_Y)[:, None] + bid_x * BLOCK_Y + batch * N < N * batch_size)
                ), other=0.0)

            weight_two = tl.load(weight_two_ptrs, mask=(\
                                    (tl.arange(0, BLOCK_X)[:, None] + j * BLOCK_X < d_ffn) & \
                                    (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m)
                ))

            do4 += tl.dot(do5s, tl.trans(weight_two))

            ## We accumulate the partial results. ##
            o1 += tl.dot(activations, weight_one)
            o2 += tl.dot(activations, weight_three)

            ## Increment pointers. ##
            activation_ptrs += d_m_block
            weight_one_ptrs += d_m_block * stride_w1_a
            weight_three_ptrs += d_m_block * stride_w3_a
            do5_ptrs += d_m_block * stride_incoming_grads_b
            weight_two_ptrs += d_m_block * stride_w2_b

        ## All of these are of size: (BLOCK_Y, BLOCK_X).
        sig_o1 = tl.sigmoid(o1)
        o3 = o1 * sig_o1
        o4 = o3 * o2 
        do2 = do4 * o3
        do3 = do4 * o2
        do1 = do3 * (sig_o1 + o3 * (tl.full((BLOCK_Y, BLOCK_X), 1, tl.float32) - sig_o1))
        ## Next, we have to downcast the intermediate activations to the pertinent data-type prior to second mat-mul. ##
        if ACTIVATION == "bfloat16":
            o4 = o4.to(tl.bfloat16) ## Size: (BLOCK_Y, BLOCK_X)
            do2 = do2.to(tl.bfloat16)
            do1 = do1.to(tl.bfloat16)

        ## This is for computing dw2. ##
        do5_ptrs = incoming_grads_ptr + tl.arange(0, BLOCK_Y)[:, None] * stride_incoming_grads_a \
                    + tl.arange(0, d_m_block)[None, :] * stride_incoming_grads_b \
                    + bid_x * BLOCK_Y * stride_incoming_grads_a \
                    + batch * N * d_m

        dw2_ptrs = dw2_ptr + tl.arange(0, d_m_block)[None, :] * stride_dw2_b \
                        + tl.arange(0, BLOCK_X)[:, None] * stride_dw2_a \
                        + j * BLOCK_X * stride_dw2_a 

        ## This is for computing dX. ##
        weight_three_ptrs = w3_ptr + tl.arange(0, d_m_block)[:, None] * stride_w3_a \
                                   + tl.arange(0, BLOCK_X)[None, :] * stride_w3_b \
                                   + j * BLOCK_X

        weight_one_ptrs = w1_ptr + tl.arange(0, d_m_block)[:, None] * stride_w1_a \
                                 + tl.arange(0, BLOCK_X)[None, :] * stride_w1_b \
                                 + j * BLOCK_X
        
        dx_ptrs = dx_ptr + tl.arange(0, BLOCK_Y)[:, None] * stride_dx_a \
                         + tl.arange(0, d_m_block)[None, :] * stride_dx_b \
                         + bid_x * BLOCK_Y * stride_dx_a \
                         + batch * N * d_m 

        ## Compute the answers. ##
        for k in tl.range(0, tl.cdiv(d_m, d_m_block)):
            ## Load do5. ##
            do5s = tl.load(do5_ptrs, mask= \
                            (tl.arange(0, BLOCK_Y)[:, None] + bid_x * BLOCK_Y + batch * N < N * batch_size) & \
                            (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m), other=0.0)

            w3s = tl.load(weight_three_ptrs, mask=(\
                            (tl.arange(0, d_m_block)[:, None] + k * d_m_block < d_m) & \
                            (tl.arange(0, BLOCK_X)[None, :] + j * BLOCK_X < d_ffn)
                ))

            w1s = tl.load(weight_one_ptrs, mask=(\
                            (tl.arange(0, d_m_block)[:, None] + k * d_m_block < d_m) & \
                            (tl.arange(0, BLOCK_X)[None, :] + j * BLOCK_X < d_ffn) 
                ))

            ## Mat-mul. ##
            dw2s = tl.dot(tl.trans(o4), do5s) # Output is size: (BLOCK_X, d_m_block)
            dx = tl.dot(do2, tl.trans(w3s)) + tl.dot(do1, tl.trans(w1s)) # Output is size: (BLOCK_Y, d_m_block)
            ## add atomic_write to main memory. ##
            tl.atomic_add(dw2_ptrs, dw2s, mask=(\
                    (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m) & \
                    (tl.arange(0, BLOCK_X)[:, None] + j * BLOCK_X < d_ffn)
                ))
            ## Load add store to dx. ##
            _load_add_store(dx, dx_ptrs, mask=(\
                    (tl.arange(0, BLOCK_Y)[:, None] + bid_x * BLOCK_Y + batch * N < N * batch_size) & \
                    (tl.arange(0, d_m_block)[None, :] + k * d_m_block < d_m) 
                ))

            do5_ptrs += d_m_block * stride_incoming_grads_b
            dw2_ptrs += d_m_block * stride_dw2_b
            weight_three_ptrs += d_m_block * stride_w3_a
            weight_one_ptrs += d_m_block * stride_w1_a
            dx_ptrs += d_m_block * stride_dx_b

def _fwd(inp, w1, w2, w3, use_opt=True):
    assert inp.dtype == w1.dtype and w1.dtype == w2.dtype, 'Incorrect dtypes passed in.'
    assert inp.dtype == torch.float32 or inp.dtype == torch.bfloat16, 'Incorrect dtypes passed in, must be float32 or bfloat16'
    assert w1.shape == w3.shape, 'Incorrect weight matrices passed in.'
    ## We need to extract a good triton configuration here. ##
    num_warps, num_blocks, BLOCK_Y, BLOCK_X, d_m_block_size, num_stages = extract_fwd_params(inp.shape[1], w1.shape[1], w1.shape[0])
    num_hid_par = 1  ## Unfortunately, this optimization doesn't work. 
    grid = (
        triton.cdiv(inp.shape[1], BLOCK_Y),
        inp.shape[0], num_hid_par
        )

    output = torch.zeros_like(inp, dtype=inp.dtype if num_hid_par <= 1 else torch.float32)

    _fwd_kernel[grid](
        inp, w1, w2, w3, output, inp.shape[1], w1.shape[1], w1.shape[0],
        inp.stride(0), inp.stride(1), inp.stride(2),
        w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        "relu", "float32" if inp.dtype == torch.float32 else "bfloat16",
        use_opt='true' if use_opt else 'false',
        BLOCK_Y=BLOCK_Y,BLOCK_X=BLOCK_X,d_m_block=d_m_block_size
        )

    if num_hid_par <= 1:
        output = output.to(inp.dtype)

    return output

def _bwd(
        incoming_gradients, input_activations,
        w1, w2, w3, use_opt=True 
        ):
    assert incoming_gradients.dtype == input_activations.dtype, 'Incorrect dtypes passed in.'
    assert incoming_gradients.dtype == torch.float32 or incoming_gradients.dtype == torch.bfloat16, 'Incorrect dtypes passed in, must be float32 or bfloat16'

    ## We gotta reshape the incoming_gradients. ##
    batch_size, N, d_m = incoming_gradients.shape
    d_ffn = w1.shape[-1]
    incoming_gradients = incoming_gradients.view(-1, incoming_gradients.shape[-1])
    input_activations = input_activations.view(-1, input_activations.shape[-1])

    outgoing_gradients = torch.zeros_like(incoming_gradients, dtype=incoming_gradients.dtype)
    w1_gradients = torch.zeros_like(w1, dtype=torch.float32)
    w2_gradients = torch.zeros_like(w2, dtype=torch.float32)
    w3_gradients = torch.zeros_like(w3, dtype=torch.float32)


    if use_opt:
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        num_warps, num_blocks, BLOCK_Y, BLOCK_X, d_m_block_size, num_stages = extract_bwd_params_dx(
            incoming_gradients.shape[1],
            w1.shape[1], w1.shape[0]
            )
        
        grid_dx = (
            triton.cdiv(N, BLOCK_Y), batch_size, 1
        )

        with torch.cuda.stream(s1):
            _bwd_kernel_dx[grid_dx](
                input_activations,
                w1, w2, w3,
                incoming_gradients, outgoing_gradients, 
                N, d_ffn, d_m, batch_size,
                input_activations.stride(0), input_activations.stride(1),
                w1.stride(0), w1.stride(1),
                w2.stride(0), w2.stride(1),
                w3.stride(0), w3.stride(1),
                incoming_gradients.stride(0), incoming_gradients.stride(1),
                outgoing_gradients.stride(0), outgoing_gradients.stride(1),
                ACTIVATION="bfloat16" if input_activations.dtype == torch.bfloat16 else "float32", 
                BLOCK_Y=BLOCK_Y, BLOCK_X=BLOCK_X, d_m_block=d_m_block_size
                )

        num_warps, num_blocks, BLOCK_Y, BLOCK_X, d_m_block_size, num_stages = extract_bwd_params_dw2_par_ffn(
            incoming_gradients.shape[1],
            w1.shape[1], w1.shape[0]
            )

        grid_dw2_par_ffn = (
            triton.cdiv(d_ffn, BLOCK_X), 1, 1
        )

        with torch.cuda.stream(s1):
            _bwd_kernel_dw2_par_ffn[grid_dw2_par_ffn](
                input_activations,
                w1, w3,
                incoming_gradients, w2_gradients,
                N, d_ffn, d_m, batch_size,
                input_activations.stride(0), input_activations.stride(1),
                w1.stride(0), w1.stride(1),
                w3.stride(0), w3.stride(1),
                incoming_gradients.stride(0), incoming_gradients.stride(1),
                w2_gradients.stride(0), w2_gradients.stride(1),
                ACTIVATION="bfloat16" if input_activations.dtype == torch.bfloat16 else "float32", 
                BLOCK_Y=BLOCK_Y, BLOCK_X=BLOCK_X, d_m_block=d_m_block_size
                )

        num_warps, num_blocks, BLOCK_Y, BLOCK_X, d_m_block_size, num_stages = extract_bwd_params_dw1_dw3_par_ffn(
            incoming_gradients.shape[1],
            w1.shape[1], w1.shape[0]
            )

        grid_dw1_dw3 = (
            triton.cdiv(d_ffn, BLOCK_Y), 1, 1
        )

        with torch.cuda.stream(s2):
            _bwd_kernel_dw1_dw3_par_ffn[grid_dw1_dw3](
                input_activations,
                w1, w2, w3,
                incoming_gradients, w1_gradients, w3_gradients,
                N, d_ffn, d_m, batch_size,
                input_activations.stride(0), input_activations.stride(1),
                w1.stride(0), w1.stride(1),
                w2.stride(0), w2.stride(1),
                w3.stride(0), w3.stride(1),
                incoming_gradients.stride(0), incoming_gradients.stride(1),
                w1_gradients.stride(0), w1_gradients.stride(1),
                w3_gradients.stride(0), w3_gradients.stride(1),
                ACTIVATION="bfloat16" if input_activations.dtype == torch.bfloat16 else "float32", 
                BLOCK_Y=BLOCK_Y, BLOCK_X=BLOCK_X, d_m_block=d_m_block_size
                )
    else:
        num_warps, num_blocks, BLOCK_Y, BLOCK_X, d_m_block_size, num_stages = extract_bwd_params_dx_dw2(
            incoming_gradients.shape[1],
            w1.shape[1], w1.shape[0]
            )

        grid_dx_dw2 = (
            triton.cdiv(N, BLOCK_Y), batch_size, 1
        )

        _bwd_kernel_dx_dw2[grid_dx_dw2](
            input_activations,
            w1, w2, w3,
            incoming_gradients, outgoing_gradients, w2_gradients,
            N, d_ffn, d_m, batch_size,
            input_activations.stride(0), input_activations.stride(1),
            w1.stride(0), w1.stride(1),
            w2.stride(0), w2.stride(1),
            w3.stride(0), w3.stride(1),
            incoming_gradients.stride(0), incoming_gradients.stride(1),
            outgoing_gradients.stride(0), outgoing_gradients.stride(1),
            w2_gradients.stride(0), w2_gradients.stride(1),
            ACTIVATION="bfloat16" if input_activations.dtype == torch.bfloat16 else "float32", 
            BLOCK_Y=BLOCK_Y, BLOCK_X=BLOCK_X, d_m_block=d_m_block_size
            )

        num_warps, num_blocks, BLOCK_Y, BLOCK_X, d_m_block_size, num_stages = extract_bwd_params_dw1_dw3(
            incoming_gradients.shape[1],
            w1.shape[1], w1.shape[0]
            )

        grid_dw1_dw3 = (
            triton.cdiv(N, BLOCK_Y), batch_size, 1
        )

        _bwd_kernel_dw1_dw3[grid_dw1_dw3](
            input_activations,
            w1, w2, w3,
            incoming_gradients, w1_gradients, w3_gradients,
            N, d_ffn, d_m, batch_size,
            input_activations.stride(0), input_activations.stride(1),
            w1.stride(0), w1.stride(1),
            w2.stride(0), w2.stride(1),
            w3.stride(0), w3.stride(1),
            incoming_gradients.stride(0), incoming_gradients.stride(1),
            w1_gradients.stride(0), w1_gradients.stride(1),
            w3_gradients.stride(0), w3_gradients.stride(1),
            ACTIVATION="bfloat16" if input_activations.dtype == torch.bfloat16 else "float32", 
            BLOCK_Y=BLOCK_Y, BLOCK_X=BLOCK_X, d_m_block=d_m_block_size
            )

    ## All the necessary typecasting and reshaping. ##
    w1_gradients = w1_gradients.to(w1.dtype)
    w2_gradients = w2_gradients.to(w2.dtype)
    w3_gradients = w3_gradients.to(w3.dtype)
    outgoing_gradients = outgoing_gradients.view(batch_size, N, d_m)
    input_activations = input_activations.view(batch_size, N, d_m)

    return outgoing_gradients, w1_gradients, w2_gradients, w3_gradients

class FusedMLP(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx,
            input : torch.Tensor,
            w1 : torch.Tensor,
            w2 : torch.Tensor,
            w3 : torch.Tensor,
            ) -> torch.Tensor:
        ctx.input_activations = input
        ctx.w1 = w1
        ctx.w2 = w2
        ctx.w3 = w3
        return _fwd(input, w1, w2, w3, use_opt=True)

    @staticmethod
    def backward(ctx, gradients : torch.Tensor) -> Tuple[torch.Tensor]:

        outgoing_grads, w1_grads, w2_grads, w3_grads = _bwd(gradients, ctx.input_activations, ctx.w1, ctx.w2, ctx.w3)

        return outgoing_grads, w1_grads, w2_grads, w3_grads

FusedMLP = FusedMLP.apply


class FusedSwigluLayer(torch.nn.Module):
    """
    Fuses the fwd and bwd pass for the compute of:
    (silu(inp @ w1) + (inp @ w3)) @ w2 
    """
    def __init__(
            self, d_m: int, d_ffn: int, 
            w1: torch.nn.Parameter=None, w2: torch.nn.Parameter=None, 
            w3: torch.nn.Parameter=None):
        super().__init__()
        ## We initialize the w1, w2 and w3 weight matrices. ##

        ## First, if w1, w2 and w3 are not none. ##
        if w1 is None and w2 is None and w3 is None:
            ## Otherwise, we manually initialize. ##
            self.w1 = torch.nn.Parameter(data=torch.nn.init.trunc_normal_(
                torch.zeros((d_m, d_ffn), device="cuda" if torch.cuda.is_available() else "cpu"), 
                mean=0.0, std=0.02))
            self.w2 = torch.nn.Parameter(data=torch.nn.init.trunc_normal_(
                torch.zeros((d_ffn, d_m), device="cuda" if torch.cuda.is_available() else "cpu"), 
                mean=0.0, std=0.02))
            self.w3 = torch.nn.Parameter(data=torch.nn.init.trunc_normal_(
                torch.zeros((d_m, d_ffn), device="cuda" if torch.cuda.is_available() else "cpu"), 
                mean=0.0, std=0.02))
        elif w1 is not None and w2 is not None and w3 is not None: 
            self.w1 = w1
            self.w2 = w2
            self.w3 = w3
        else:
            raise Exception("Issue with init statements.")

    def forward(self, X):
        return FusedMLP(X, self.w1, self.w2, self.w3)
