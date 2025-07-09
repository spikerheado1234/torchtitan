## This explores the interplay of fusion + AC. ##
import torch
from typing import Tuple
import triton
import triton.language as tl

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]

## We define a new set of kernels here that are simpler. ##
#@triton.autotune(
#    configs=get_cuda_autotune_config(),
#    key=['M', 'N', 'K'],
#)
@triton.jit
def matmul_dw1_dw3(
        # Pointers to matrices
        activation_ptr, o1_ptr, o2_ptr,
        do4_ptr, dw1_ptr, dw3_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  # activation_ptr stride.
        stride_bk, stride_bn,  # do4_ptr, o1_ptr, & o2_ptr strides.
        stride_cm, stride_cn,  # dw1_ptr, dw3_ptr strides.
        precision : tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = activation_ptr + (offs_am[None, :] * stride_ak + offs_k[:, None] * stride_am)

    b_ptrs = o1_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    c_ptrs = o2_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    d_ptrs = do4_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    dw1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    dw3 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        activations = tl.load(a_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        o1 = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        o2 = tl.load(c_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        do4 = tl.load(d_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        ## Here are some compute operators. ##
        o3 = o1 * tl.sigmoid(o1.to(tl.float32)) 
        do2 = do4 * o3
        do3 = do4 * o2
        do1 = do3 * (tl.sigmoid(o1.to(tl.float32)) + o3 * (tl.full((BLOCK_SIZE_K, BLOCK_SIZE_N), 1, dtype=tl.float32) - tl.sigmoid(o1.to(tl.float32))))
        if precision == "bfloat16":
            do2 = do2.to(tl.bfloat16)
            do1 = do1.to(tl.bfloat16)

        # We accumulate along the K dimension.
        dw1 = tl.dot(tl.trans(activations), do1, dw1)
        dw3 = tl.dot(tl.trans(activations), do2, dw3)
        # Advance the ptrs to the next K block.

        a_ptrs += BLOCK_SIZE_K * stride_am
        b_ptrs += BLOCK_SIZE_K * stride_bk
        c_ptrs += BLOCK_SIZE_K * stride_bk
        d_ptrs += BLOCK_SIZE_K * stride_bk

    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if precision == "bfloat16":
        dw1 = dw1.to(tl.bfloat16)
        dw3 = dw3.to(tl.bfloat16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw1_ptrs = dw1_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    dw3_ptrs = dw3_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    store_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(dw1_ptrs, dw1, mask=store_mask)
    tl.store(dw3_ptrs, dw3, mask=store_mask)

#@triton.autotune(
#    configs=get_cuda_autotune_config(),
#    key=['M', 'N', 'K'],
#)
@triton.jit
def matmul_dw2(
        # Pointers to matrices
        o1_ptr, o2_ptr, inc_gradient_ptr, 
        dw2_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  # o1 & o2 stride.
        stride_bk, stride_bn,  # inc_gradient stride.
        stride_cm, stride_cn,  # dw2_ptr stride.
        precision : tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = o1_ptr + (offs_am[None, :] * stride_ak + offs_k[:, None] * stride_am)
    b_ptrs = o2_ptr + (offs_am[None, :] * stride_ak + offs_k[:, None] * stride_am)

    c_ptrs = inc_gradient_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    dw2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        o1 = tl.load(a_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        o2 = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        do5 = tl.load(c_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        ## Here are some compute operators. ##
        o3 = o1 * tl.sigmoid(o1.to(tl.float32)) 
        o4 = o2 * o3

        if precision == "bfloat16":
            o4 = o4.to(tl.bfloat16)

        # We accumulate along the K dimension.
        dw2 = tl.dot(tl.trans(o4), do5, dw2)
        # Advance the ptrs to the next K block.

        a_ptrs += BLOCK_SIZE_K * stride_am
        b_ptrs += BLOCK_SIZE_K * stride_am
        c_ptrs += BLOCK_SIZE_K * stride_bk

    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if precision == "bfloat16":
        dw2 = dw2.to(tl.bfloat16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw2_ptrs = dw2_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    store_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(dw2_ptrs, dw2, mask=store_mask)


#@triton.autotune(
#    configs=get_cuda_autotune_config(),
#    key=['M', 'N', 'K'],
#)
@triton.jit
def matmul_dx(
        # Pointers to matrices
        o1_ptr, o2_ptr, w1_ptr, w3_ptr,
        do4_ptr, dx_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  # o1_ptr, o2_ptr stride.
        stride_bk, stride_bn,  # w1_ptr & w3_ptr stride.
        stride_cm, stride_cn,  # dx_ptr stride.
        precision : tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = o1_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = o2_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    c_ptrs = do4_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    d_ptrs = w1_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    e_ptrs = w3_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    dx = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.

        o1 = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        o2 = tl.load(b_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        do4 = tl.load(c_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)

        w1 = tl.load(d_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        w3 = tl.load(e_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        ## Here are some compute operators. ##
        o3 = o1 * tl.sigmoid(o1.to(tl.float32)) 
        do2 = do4 * o3
        do3 = do4 * o2
        do1 = do3 * (tl.sigmoid(o1.to(tl.float32)) + o3 * (tl.full((BLOCK_SIZE_M, BLOCK_SIZE_K), 1, dtype=tl.float32) - tl.sigmoid(o1.to(tl.float32))))
        if precision == "bfloat16":
            do2 = do2.to(tl.bfloat16)
            do1 = do1.to(tl.bfloat16)

        # We accumulate along the K dimension.
        dx += tl.dot(do2, w3) + tl.dot(do1, w1)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_ak
        c_ptrs += BLOCK_SIZE_K * stride_ak
        d_ptrs += BLOCK_SIZE_K * stride_bk
        e_ptrs += BLOCK_SIZE_K * stride_bk

    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if precision == "bfloat16":
        dx = dx.to(tl.bfloat16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dx_ptrs = dx_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    store_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(dx_ptrs, dx, mask=store_mask)


def _bwd(
        incoming_gradients, input_activations,
        w1, w2, w3, use_opt=True 
        ):
    assert incoming_gradients.dtype == input_activations.dtype, 'Incorrect dtypes passed in.'
    assert incoming_gradients.dtype == torch.float32 or incoming_gradients.dtype == torch.bfloat16, 'Incorrect dtypes passed in, must be float32 or bfloat16'

    ## We gotta reshape the incoming_gradients. ##
    batch_size, N, d_m = incoming_gradients.shape
    incoming_gradients = incoming_gradients.view(-1, incoming_gradients.shape[-1])
    input_activations = input_activations.view(-1, input_activations.shape[-1])

    ## We reconstruct certain tensors first. ##
    o1 = torch.einsum('sh,hf -> sf', input_activations, w1)
    o2 = torch.einsum('sh,hf -> sf', input_activations, w3)
    do4 = torch.einsum('sh, fh -> sf', incoming_gradients, w2)

    ## Then we directly compute everything else in a fused fashion. 
    ## Hopefully, this speeds up the kernels. 
    outgoing_gradients = torch.zeros_like(incoming_gradients, dtype=incoming_gradients.dtype)
    w1_gradients = torch.zeros_like(w1, dtype=torch.float32)
    w2_gradients = torch.zeros_like(w2, dtype=torch.float32)
    w3_gradients = torch.zeros_like(w3, dtype=torch.float32)

    ## Some transpositions. 
    #input_activations = torch.transpose(input_activations, 0, 1)
    w1 = torch.transpose(w1, 0, 1)
    w3 = torch.transpose(w3, 0, 1)

    dx_grid = (
        triton.cdiv(o1.shape[0], 64) * triton.cdiv(w1.shape[-1], 64), 
        1, 1
    )

    #dx_grid = lambda META: (
    #    triton.cdiv(o1.shape[0], META["BLOCK_SIZE_M"]) * triton.cdiv(w1.shape[-1], META["BLOCK_SIZE_N"]), 
    #    1, 1
    #)

    dw1_dw3_grid = (
        triton.cdiv(input_activations.shape[-1], 256) * triton.cdiv(o1.shape[-1], 64),
        1, 1
    )

    #dw1_dw3_grid = lambda META: (
    #    triton.cdiv(input_activations.shape[-1], META["BLOCK_SIZE_M"]) * triton.cdiv(o1.shape[-1], META["BLOCK_SIZE_N"]),
    #    1, 1
    #)

    ## First we launch and dx & dw1_dw3. ##
    matmul_dx[dx_grid](
        o1, o2, w1, w3,
        do4, outgoing_gradients,
        o1.shape[0], w1.shape[-1], o1.shape[-1],
        o1.stride(0), o1.stride(1),
        w1.stride(0), w1.stride(1),
        outgoing_gradients.stride(0), outgoing_gradients.stride(1),
        precision="bfloat16" if input_activations.dtype == torch.bfloat16 else "float32",
        ## Block stuff goes here. ##
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8,
        num_warps=4, num_stages=4
    )

    matmul_dw1_dw3[dw1_dw3_grid](
        input_activations, o1, o2,
        do4, w1_gradients, w3_gradients,
        input_activations.shape[-1], o1.shape[-1], input_activations.shape[0],
        input_activations.stride(0), input_activations.stride(1),
        o1.stride(0), o2.stride(1),
        w1_gradients.stride(0), w1_gradients.stride(1),
        precision="bfloat16" if input_activations.dtype == torch.bfloat16 else "float32",
        BLOCK_SIZE_M=256, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8,
        num_warps=8, num_stages=4
    )

    ## Next, we launch dw2, which requires transposing o1 & o2.
    dw2_grid = (
        triton.cdiv(o1.shape[-1], 64) * triton.cdiv(incoming_gradients.shape[-1], 256),
        1, 1
    )

    #dw2_grid = lambda META: (
    #    triton.cdiv(o1.shape[-1], META["BLOCK_SIZE_M"]) * triton.cdiv(incoming_gradients.shape[-1], META["BLOCK_SIZE_N"]),
    #    1, 1
    #)

    matmul_dw2[dw2_grid](
        o1, o2, incoming_gradients,
        w2_gradients,
        o1.shape[-1], incoming_gradients.shape[-1], o1.shape[0],
        o1.stride(0), o1.stride(1),
        incoming_gradients.stride(0), incoming_gradients.stride(1),
        w2_gradients.stride(0), w2_gradients.stride(1),
        precision="bfloat16" if input_activations.dtype == torch.bfloat16 else "float32",
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=256, BLOCK_SIZE_K=32, GROUP_SIZE_M=8,
        num_warps=4, num_stages=4
    )

    ## Undo views and transpositions. ##
    outgoing_gradients = outgoing_gradients.view(batch_size, N, d_m)
    input_activations = input_activations.view(batch_size, N, d_m)
    w1 = torch.transpose(w1, 0, 1)
    w3 = torch.transpose(w3, 0, 1)

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
        ## Forward pass is more simple. ##
        a = torch.einsum('bsh,hf -> bsf', input, w1)
        b = torch.einsum('bsh,hf -> bsf', input, w3)
        c = torch.nn.functional.silu(a)*b
        return torch.einsum('bsf,fh->bsh', c, w2)

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
