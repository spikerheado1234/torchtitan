import triton

## This requires more tuning. ##
def extract_fwd_params(N, d_ffn, d_m):
    """ Returns tuple of (num warps, num blocks, BLOCK Y, BLOCK X, d_m block size, num stages)
    """
    ## For now, something naive. ##
    BLOCK_Y = 32
    BLOCK_X = 256
    d_m_block = 64
    #d_m_block = min(triton.next_power_of_2(d_m), 16)
    num_stages = 2

    ## Now, we compute the number of thread-blocks. ##
    if N < 1024:
        tbs = int(triton.cdiv(N,BLOCK_Y))
    else:
        tbs = int(triton.cdiv(N, BLOCK_Y) // 2)

    ## Finally, we compute the number of warps. ##
    num_warps = 8
    num_stages = 2

    return (num_warps, tbs, BLOCK_Y, BLOCK_X, d_m_block, num_stages)

def extract_bwd_params(N, d_ffn, d_m):
    """ Returns tuple of (num warps, num blocks, BLOCK Y, BLOCK X, d_m block size, num stages)
    """
    BLOCK_Y = 32
    BLOCK_X = 128
    if N < 1024:
        tbs = int(triton.cdiv(N,BLOCK_Y))
    else:
        tbs = int(triton.cdiv(N, BLOCK_Y) // 2)
    d_m_block = 64
    num_stages = 2
    num_warps = 8
    return (num_warps, tbs, BLOCK_Y, BLOCK_X, d_m_block, num_stages)
