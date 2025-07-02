import triton

## This requires more tuning. ##
def extract_fwd_params(N, d_ffn, d_m):
    """ Returns tuple of (num warps, num blocks, BLOCK Y, BLOCK X, d_m block size, num stages)
    """

    ## Seems like the best configuration so far is:
    ## BLOCK_X = 256, BLOCK_Y = 128, d_m_block = 128,
    ##      num_warps = 16, num_ctas = 1, num_stages = 2. ##
    ## However, it doesn't allow us to use it for SWIGLU due to shmem constraints. ##

    BLOCK_Y = 64
    BLOCK_X = 128
    d_m_block = 64

    #d_m_block = min(triton.next_power_of_2(d_m), 16)
    num_stages = 2

    ## Now, we compute the number of thread-blocks. ##
    tbs = int(triton.cdiv(N, BLOCK_Y))

    ## Finally, we compute the number of warps. ##
    num_warps = 8
    num_stages = 2

    return (num_warps, tbs, BLOCK_Y, BLOCK_X, d_m_block, num_stages)

def extract_bwd_params(N, d_ffn, d_m):
    """ Returns tuple of (num warps, num blocks, BLOCK Y, BLOCK X, d_m block size, num stages)
    """
    ## This is a testing configuration only. ##
    BLOCK_Y = 16
    BLOCK_X = 16
    tbs = int(triton.cdiv(N,BLOCK_Y))
    d_m_block = 16
    num_stages = 1
    num_warps = 8
    return (num_warps, tbs, BLOCK_Y, BLOCK_X, d_m_block, num_stages)

def extract_bwd_params_dx(N, d_ffn, d_m):
    """ Returns tuple of (num warps, num blocks, BLOCK Y, BLOCK X, d_m block size, num stages)
    """
    ## This is a testing configuration only. ##
    BLOCK_Y = 256
    BLOCK_X = 32
    tbs = int(triton.cdiv(N,BLOCK_Y))
    d_m_block = 64

    num_stages = 1
    num_warps = 8
    return (num_warps, tbs, BLOCK_Y, BLOCK_X, d_m_block, num_stages)

def extract_bwd_params_dw2(N, d_ffn, d_m):
    """ Returns tuple of (num warps, num blocks, BLOCK Y, BLOCK X, d_m block size, num stages)
    """
    ## This is a testing configuration only. ##
    BLOCK_Y = 256
    BLOCK_X = 64
    tbs = int(triton.cdiv(N,BLOCK_Y))
    d_m_block = 64

    num_stages = 1
    num_warps = 8
    return (num_warps, tbs, BLOCK_Y, BLOCK_X, d_m_block, num_stages)

def extract_bwd_params_dw2_par_ffn(N, d_ffn, d_m):
    """ Returns tuple of (num warps, num blocks, BLOCK Y, BLOCK X, d_m block size, num stages)
    """
    ## This is a testing configuration only. ##
    BLOCK_Y = 64
    BLOCK_X = 128
    tbs = int(triton.cdiv(d_ffn,BLOCK_X))
    d_m_block = 64

    num_stages = 1
    num_warps = 8
    return (num_warps, tbs, BLOCK_Y, BLOCK_X, d_m_block, num_stages)

def extract_bwd_params_dx_dw2(N, d_ffn, d_m):
    """ Returns tuple of (num warps, num blocks, BLOCK Y, BLOCK X, d_m block size, num stages)
    """
    ## This is a testing configuration only. ##
    BLOCK_Y = 256
    BLOCK_X = 32
    tbs = int(triton.cdiv(N,BLOCK_Y))
    d_m_block = 64
    num_stages = 1
    num_warps = 8
    return (num_warps, tbs, BLOCK_Y, BLOCK_X, d_m_block, num_stages)

def extract_bwd_params_dw1_dw3(N, d_ffn, d_m):
    """ Returns tuple of (num warps, num blocks, BLOCK Y, BLOCK X, d_m block size, num stages)
    """
    ## This is a testing configuration only. ##
    BLOCK_Y = 256
    BLOCK_X = 64
    tbs = int(triton.cdiv(N,BLOCK_Y))
    d_m_block = 32
    num_stages = 1
    num_warps = 8
    return (num_warps, tbs, BLOCK_Y, BLOCK_X, d_m_block, num_stages)

def extract_bwd_params_dw1_dw3_par_ffn(N, d_ffn, d_m):
    """ Returns tuple of (num warps, num blocks, BLOCK Y, BLOCK X, d_m block size, num stages)
    """
    ## This is a testing configuration only. ##
    BLOCK_Y = 64
    BLOCK_X = 128
    tbs = int(triton.cdiv(d_ffn, BLOCK_X))
    d_m_block = 32
    num_stages = 1
    num_warps = 8
    return (num_warps, tbs, BLOCK_Y, BLOCK_X, d_m_block, num_stages)

## Here, the tiling for the two kernels is different. ##
def extract_bwd_params_opt(N, d_ffn, d_m):
    """ Returns tuple of (num warps, num blocks, BLOCK Y, BLOCK X, d_m block size, num stages)
    """
    BLOCK_Y = 32
    BLOCK_X = 256
    if N < 1024:
        tbs = int(triton.cdiv(N,BLOCK_Y))
    else:
        tbs = int(triton.cdiv(N, BLOCK_Y) // 2)
    d_m_block = 64
    num_stages = 2
    num_warps = 8
    return (num_warps, tbs, BLOCK_Y, BLOCK_X, d_m_block, num_stages)
