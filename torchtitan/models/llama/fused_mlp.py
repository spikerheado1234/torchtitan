## This is a single-GPU fused MLP. ##
import torch
from typing import Tuple
import triton
import triton.language as tl
from src.ops.tune import extract_fwd_params, extract_bwd_params
import torch.distributed as dist
import pdb

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

@triton.jit
def _fwd_kernel(
    ## Input activations. ##
    inp_ptr,
    ## Weight matrices. ##
    w1_ptr, w2_ptr,
    ## Output activation matrix. ##
    output_ptr,
    N: tl.constexpr, d_ffn: tl.constexpr, d_m: tl.constexpr, d_m_block: tl.constexpr,
    ## Strides. ##

    ## Strides of input activations. ##
    stride_inp_a: tl.constexpr, stride_inp_b: tl.constexpr, stride_inp_c: tl.constexpr,

    ## Strides of weight matrices. ##
    stride_w1_a: tl.constexpr, stride_w1_b: tl.constexpr,
    stride_w2_a: tl.constexpr, stride_w2_b: tl.constexpr,

    ## Strides of output activation matrix. ##
    stride_output_a: tl.constexpr, stride_output_b: tl.constexpr, stride_output_c: tl.constexpr,

    ## Block Sizes. We default to parallelising over batch dimension. ##
    BLOCK_Y: tl.constexpr, BLOCK_X: tl.constexpr,
    ## Lastly, the activation function. ##
    ACTIVATION: tl.constexpr, data_type: tl.constexpr):
    """
    Input sizes:
    inp_ptr (original activations): [b, s, d_m] matrix.
    w1_ptr (first weight matrix): [d_m, d_ffn] matrix.
    w2_ptr (second weight matrix): [d_ffn, d_m] matrix.
    output_ptr (final output activations): [b, s, d_m] matrix.
    """
    bid_x = tl.program_id(axis=0)
    batch = tl.program_id(axis=1)
    num_prog = tl.num_programs(axis=0)
    ## Uncover how many iterations we traverse over the rows of O_2. ##
    step_size = num_prog*BLOCK_Y
    for i in tl.range(0, tl.cdiv(N, step_size), num_stages=1):
        for j in tl.range(0, tl.cdiv(d_ffn, BLOCK_X), num_stages=1):
            output_ptrs = output_ptr + batch*stride_output_a \
                            + tl.arange(0, BLOCK_Y)[:, None]*stride_output_b \
                            + tl.arange(0, d_m_block)[None, :]*stride_output_c \
                            + bid_x*BLOCK_Y*stride_output_b + step_size*i*stride_output_b
            ## We load the data from the input activations. ##
            activation_ptrs = inp_ptr + batch*stride_inp_a \
                                + tl.arange(0, BLOCK_Y)[:, None]*stride_inp_b \
                                + tl.arange(0, d_m_block)[None, :]*stride_inp_c \
                                + bid_x*BLOCK_Y*stride_inp_b + step_size*i*stride_inp_b
            ## We additionally tile across the hidden dimension to reduce shmem consumption. ##
            ## We pull out the base pointer's compute arithmetic. ##
            weight_one_ptrs = w1_ptr + j*BLOCK_X*stride_w1_b \
                                + tl.arange(0, d_m_block)[:, None]*stride_w1_a \
                                + tl.arange(0, BLOCK_X)[None, :]*stride_w1_b

            weight_two_ptrs = w2_ptr + j*BLOCK_X*stride_w2_a \
                                + tl.arange(0, BLOCK_X)[:, None]*stride_w2_a \
                                + tl.arange(0, d_m_block)[None, :]*stride_w2_b

            activ_accum = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
            for k in tl.range(0, tl.cdiv(d_m, d_m_block), num_stages=1):
                ## First, we load the data from the input activations. ##
                activations = tl.load(activation_ptrs, mask=( \
                                        (tl.arange(0, BLOCK_Y)[:, None]+ bid_x*BLOCK_Y + step_size*i < N) \
                                        & (tl.arange(0, d_m_block)[None, :] + k*d_m_block < d_m)), other=0.0)
                ## Next, we load the first weight matrix. ##
                weight_one = tl.load(weight_one_ptrs, mask=( \
                                        (tl.arange(0, d_m_block)[:, None] + k*d_m_block < d_m) & \
                                        (tl.arange(0, BLOCK_X)[None, :] +j*BLOCK_X < d_ffn)), other=0.0)

                ## We accumulate the partial results. ##
                activ_accum += tl.dot(activations, weight_one)

                ## Increment pointers. ##
                activation_ptrs += d_m_block
                weight_one_ptrs += d_m_block*stride_w1_a

            ## Fuse with activation. ##
            # if ACTIVATION == "gelu":
            #     activ_accum = _gelu_fwd(activ_accum)
            if ACTIVATION == "relu":
                activ_accum = _relu_fwd(activ_accum)

            ## Next, we have to downcast the intermediate activations to the pertinent data-type prior to second mat-mul. ##
            if data_type == "bfloat16":
                activ_accum = activ_accum.to(tl.bfloat16)

            ## Next, we load the data from the second weight matrix to do the matmul. We again tile this across the hidden dimension. ##
            for k in tl.range(0, tl.cdiv(d_m, d_m_block), num_stages=1):

                weight_two = tl.load(weight_two_ptrs, mask=( \
                                        (tl.arange(0, BLOCK_X)[:, None] +j*BLOCK_X < d_ffn) & \
                                        (tl.arange(0, d_m_block)[None, :] + k*d_m_block < d_m)), other=0.0)

                output_activations = tl.dot(activ_accum, weight_two)

                ## We have to store the final output to HBM incrementally. ##
                ## To do so we need atomic adds since triton has no other option. ##
                tl.atomic_add(output_ptrs, output_activations.to(tl.float32),
                              mask=(\
                                  (tl.arange(0, BLOCK_Y)[:, None]+bid_x*BLOCK_Y + step_size*i < N) \
                                    & (tl.arange(0, d_m_block)[None, :] + k*d_m_block < d_m)))

                weight_two_ptrs += d_m_block*stride_w2_b
                output_ptrs += d_m_block*stride_output_c

def _fwd(inp, w1, w2):
    output = torch.zeros_like(inp, dtype=torch.float32)
    assert inp.dtype == w1.dtype and w1.dtype == w2.dtype, 'Incorrect dtypes passed in.'
    assert inp.dtype == torch.float32 or inp.dtype == torch.bfloat16, 'Incorrect dtypes passed in, must be float32 or bfloat16'
    ## We need to extract a good triton configuration here. ##
    num_warps, num_blocks, BLOCK_Y, BLOCK_X, d_m_block_size, num_stages = extract_fwd_params(inp.shape[1], w1.shape[1], w1.shape[0])

    _fwd_kernel[(num_blocks,inp.shape[0],1)](
        inp, w1, w2, output, inp.shape[1], w1.shape[1], w1.shape[0], d_m_block_size,
        inp.stride(0), inp.stride(1), inp.stride(2),
        w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_Y, BLOCK_X, "relu", "float32" if inp.dtype == torch.float32 else "bfloat16", num_warps=num_warps, num_stages=num_stages
        )

    output = output.to(inp.dtype)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return output

## This is currently a naive kernel. TODO(ahangupta): performance optimise and neaten up the code after correctness is assured. ##
@triton.jit
def _bwd_kernel(
    ## Input activations & gradient. ##
    inp_ptr, inp_grad_ptr,
    ## Weight matrices. ##
    w1_ptr, w2_ptr,
    ## Output, w1 and w2 grad matrix. ##
    output_grad_ptr, w1_grad_ptr, w2_grad_ptr,
    N: tl.constexpr, d_ffn: tl.constexpr, d_m: tl.constexpr, d_m_block: tl.constexpr,
    ## Strides. ##

    ## Strides of input activations & gradients (same size). ##
    stride_inp_grad_a: tl.constexpr, stride_inp_grad_b: tl.constexpr, stride_inp_grad_c: tl.constexpr,

    ## Strides of weight matrices. ##
    stride_w1_a: tl.constexpr, stride_w1_b: tl.constexpr,
    stride_w2_a: tl.constexpr, stride_w2_b: tl.constexpr,

    ## Strides of output grad/activation matrix. ##
    stride_output_grad_a: tl.constexpr, stride_output_grad_b: tl.constexpr,
    stride_output_grad_c: tl.constexpr,

    ## Block Sizes. We default to parallelising over batch dimension. ##
    BLOCK_Y: tl.constexpr, BLOCK_X: tl.constexpr,
    ## Lastly, the activation function. ##
    ACTIVATION: tl.constexpr, data_type: tl.constexpr):
    """
    Input sizes:
    inp_grad_ptr/inp_ptr (incoming gradients from bwd pass & activations from fwd pass): [b, s, d_m] matrix.
    w1_grad_ptr/w2_ptr (first weight matrix and its gradients): [d_m, d_ffn] matrix.
    w2_grad_ptr/w2_ptr (second weight matrix and its gradients): [d_ffn, d_m] matrix.
    output_grad_ptr (final output gradients): [b, s, d_m] matrix.
    """
    bid_x = tl.program_id(axis=0)
    batch = tl.program_id(axis=1)
    num_prog = tl.num_programs(axis=0)
    step_size = num_prog*BLOCK_Y
    for i in tl.range(0, tl.cdiv(N, step_size), num_stages=1):
        for j in tl.range(0, tl.cdiv(d_ffn, BLOCK_X), num_stages=1):
            ## Initialize the accumulators. ##
            m1 = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
            m2 = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
            inp_grad_ptrs = inp_grad_ptr + batch*stride_inp_grad_a \
                                + i*step_size*stride_inp_grad_b + bid_x*BLOCK_Y*stride_inp_grad_b \
                                + tl.arange(0, BLOCK_Y)[:, None]*stride_inp_grad_b \
                                + tl.arange(0, d_m_block)[None, :]*stride_inp_grad_c
            w1_ptrs = w1_ptr + j*BLOCK_X*stride_w1_b \
                        + tl.arange(0, d_m_block)[:, None]*stride_w1_a \
                        + tl.arange(0, BLOCK_X)[None, :]*stride_w1_b
            w2_ptrs = w2_ptr + j*BLOCK_X*stride_w2_a \
                        + tl.arange(0, BLOCK_X)[:, None]*stride_w2_a \
                        + tl.arange(0, d_m_block)[None, :]*stride_w2_b
            activ_ptrs = inp_ptr + batch*stride_inp_grad_a \
                            + i*step_size*stride_inp_grad_b + bid_x*BLOCK_Y*stride_inp_grad_b \
                            + tl.arange(0, BLOCK_Y)[:, None]*stride_inp_grad_b \
                            + tl.arange(0, d_m_block)[None, :]*stride_inp_grad_c
            for k in tl.range(0, tl.cdiv(d_m, d_m_block), num_stages=1):
                ## Next, we load the weights. ##
                inp_grads = tl.load(inp_grad_ptrs, mask=(\
                                        (tl.arange(0, BLOCK_Y)[:, None] + i*step_size + bid_x*BLOCK_Y < N) & \
                                        (tl.arange(0, d_m_block)[None, :] + k*d_m_block < d_m)
                ), other=0.0)
                activs = tl.load(activ_ptrs, mask=(\
                                    (tl.arange(0, BLOCK_Y)[:, None] + i*step_size + bid_x*BLOCK_Y < N) & \
                                    (tl.arange(0, d_m_block)[None, :] + k*d_m_block < d_m)
                ), other=0.0)
                w1s = tl.load(w1_ptrs, mask=( \
                            (tl.arange(0, d_m_block)[:, None] +  k*d_m_block < d_m) & \
                            (tl.arange(0, BLOCK_X)[None, :] + j*BLOCK_X < d_ffn)), other=0.0)
                w2s = tl.load(w2_ptrs, mask=( \
                            (tl.arange(0, BLOCK_X)[:, None] + j*BLOCK_X < d_ffn) & \
                            (tl.arange(0, d_m_block)[None, :] + k*d_m_block < d_m)), other=0.0)

                ## Here, we have to do a bunch of different mat-muls. ##
                m1 += tl.dot(inp_grads, tl.trans(w2s))
                m2 += tl.dot(activs, w1s)

                w1_ptrs += d_m_block*stride_w1_a
                w2_ptrs += d_m_block*stride_w2_b
                activ_ptrs += d_m_block*stride_inp_grad_c
                inp_grad_ptrs += d_m_block*stride_inp_grad_c
            ## We fuse with the activation and relu' (relu derivative). ##
            if ACTIVATION == "relu":
                inter_activs = _relu_fwd(m2)
                m3 = m1 * _relu_grad(m2)

            if data_type == "bfloat16":
                m1 = m1.to(tl.bfloat16)
                m2 = m2.to(tl.bfloat16)
                m3 = m3.to(tl.bfloat16)
                inter_activs = inter_activs.to(tl.bfloat16)

            activ_ptrs = inp_ptr + batch*stride_inp_grad_a \
                            + i*step_size*stride_inp_grad_b + bid_x*BLOCK_Y*stride_inp_grad_b \
                            + tl.arange(0, BLOCK_Y)[:, None]*stride_inp_grad_b \
                            + tl.arange(0, d_m_block)[None, :]*stride_inp_grad_c
            inp_grad_ptrs = inp_grad_ptr + batch*stride_inp_grad_a \
                                + i*step_size*stride_inp_grad_b + bid_x*BLOCK_Y*stride_inp_grad_b \
                                + tl.arange(0, BLOCK_Y)[:, None]*stride_inp_grad_b \
                                + tl.arange(0, d_m_block)[None, :]*stride_inp_grad_c
            w1s_grad = w1_grad_ptr + j*BLOCK_X*stride_w1_b \
                        + tl.arange(0, d_m_block)[:, None]*stride_w1_a \
                        + tl.arange(0, BLOCK_X)[None, :]*stride_w1_b
            w2s_grad = w2_grad_ptr + j*BLOCK_X*stride_w2_a \
                        + tl.arange(0, BLOCK_X)[:, None]*stride_w2_a \
                        + tl.arange(0, d_m_block)[None, :]*stride_w2_b
            w1_ptrs = w1_ptr + j*BLOCK_X*stride_w1_b \
                        + tl.arange(0, d_m_block)[:, None]*stride_w1_a \
                        + tl.arange(0, BLOCK_X)[None, :]*stride_w1_b
            output_grad_ptrs = output_grad_ptr + batch*stride_output_grad_a \
                                + i*step_size*stride_output_grad_b + bid_x*BLOCK_Y*stride_output_grad_b \
                                + tl.arange(0, BLOCK_Y)[:, None]*stride_output_grad_b \
                                + tl.arange(0, d_m_block)[None, :]*stride_output_grad_c
            for k in tl.range(0, tl.cdiv(d_m, d_m_block), num_stages=1):
                activs = tl.load(activ_ptrs, mask=( \
                                    (tl.arange(0, BLOCK_Y)[:, None] + i*step_size + bid_x*BLOCK_Y < N) & \
                                    (tl.arange(0, d_m_block)[None, :] + k*d_m_block < d_m)
                ), other=0.0)
                inp_grads = tl.load(inp_grad_ptrs, mask=( \
                                        (tl.arange(0, BLOCK_Y)[:, None] + i*step_size + bid_x*BLOCK_Y < N) & \
                                        (tl.arange(0, d_m_block)[None, :] + k*d_m_block < d_m)
                ), other=0.0)
                w1s = tl.load(w1_ptrs, mask=( \
                            (tl.arange(0, d_m_block)[:, None] + k*d_m_block < d_m) & \
                            (tl.arange(0, BLOCK_X)[None, :] + j*BLOCK_X < d_ffn)), other=0.0)
                ## Next, we accumulate the partial results. ##
                grad_w_1 = tl.dot(tl.trans(activs), m3)
                grad_w_2 = tl.dot(tl.trans(inter_activs), inp_grads)
                grad_output = tl.dot(m3, tl.trans(w1s))
                tl.atomic_add(w1s_grad, grad_w_1.to(tl.float32), mask=( \
                                (tl.arange(0, d_m_block)[:, None] + k*d_m_block < d_m) & \
                                (tl.arange(0, BLOCK_X)[None, :] +j*BLOCK_X < d_ffn)))
                tl.atomic_add(w2s_grad, grad_w_2.to(tl.float32), mask=( \
                                (tl.arange(0, BLOCK_X)[:, None] +j*BLOCK_X < d_ffn) & \
                                (tl.arange(0, d_m_block)[None, :] + k*d_m_block < d_m)))
                tl.atomic_add(output_grad_ptrs, grad_output.to(tl.float32), mask=( \
                                    (tl.arange(0, BLOCK_Y)[:, None] + i*step_size + bid_x*BLOCK_Y < N) & \
                                    (tl.arange(0, d_m_block)[None, :] + k*d_m_block < d_m)))

                w1s_grad += d_m_block*stride_w1_a
                w2s_grad += d_m_block*stride_w2_b
                output_grad_ptrs += d_m_block*stride_output_grad_c
                inp_grad_ptrs += d_m_block*stride_inp_grad_c
                activ_ptrs += d_m_block*stride_inp_grad_c
                w1_ptrs += d_m_block*stride_w1_a

def _bwd(
        incoming_gradients, input_activations,
        w1, w2
        ):
    assert incoming_gradients.dtype == input_activations.dtype, 'Incorrect dtypes passed in.'
    assert incoming_gradients.dtype == torch.float32 or incoming_gradients.dtype == torch.bfloat16, 'Incorrect dtypes passed in, must be float32 or bfloat16'

    outgoing_gradients = torch.zeros_like(incoming_gradients, dtype=torch.float32)
    w1_gradients = torch.zeros_like(w1, dtype=torch.float32)
    w2_gradients = torch.zeros_like(w2, dtype=torch.float32)

    num_warps, num_blocks, BLOCK_Y, BLOCK_X, d_m_block_size, num_stages = extract_bwd_params(
        incoming_gradients.shape[1],
        w1.shape[1], w1.shape[0]
        )

    ## Call to bwd kernel here. ##
    _bwd_kernel[(num_blocks, incoming_gradients.shape[0], 1)](
        input_activations, incoming_gradients, w1, w2,
        outgoing_gradients, w1_gradients, w2_gradients,
        incoming_gradients.shape[1], w1.shape[1], w1.shape[0],
        d_m_block_size, incoming_gradients.stride(0),
        incoming_gradients.stride(1), incoming_gradients.stride(2),
        w1.stride(0), w1.stride(1), w2.stride(0), w2.stride(1),
        outgoing_gradients.stride(0), outgoing_gradients.stride(1),
        outgoing_gradients.stride(2), BLOCK_Y, BLOCK_X,
        "relu", "bfloat16" if incoming_gradients.dtype == torch.bfloat16 else "float32", num_warps=num_warps, num_stages=num_stages
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    w1_gradients = w1_gradients.to(incoming_gradients.dtype)
    w2_gradients = w2_gradients.to(incoming_gradients.dtype)

    return outgoing_gradients, w1_gradients, w2_gradients

class FusedMLP(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx,
            input : torch.Tensor,
            w1 : torch.Tensor,
            w2 : torch.Tensor
            ) -> torch.Tensor:
        ctx.input_activations = input
        ctx.w1 = w1
        ctx.w2 = w2
        return _fwd(input, w1, w2)

    @staticmethod
    def backward(ctx, gradients : torch.Tensor) -> Tuple[torch.Tensor]:

        outgoing_grads, w1_grads, w2_grads = _bwd(gradients, ctx.input_activations, ctx.w1, ctx.w2)

        ## TODO(ahangupta): verify correctness. ##
        if dist.is_initialized():
            dist.all_reduce(w1_grads, op=dist.ReduceOp.SUM)
            dist.all_reduce(w2_grads, op=dist.ReduceOp.SUM)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return outgoing_grads, w1_grads, w2_grads

FusedMLP = FusedMLP.apply
