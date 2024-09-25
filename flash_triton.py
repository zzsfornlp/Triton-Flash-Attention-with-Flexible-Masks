#

# --
# An implementation of flash attention with triton, most based on:
# https://github.com/FlagOpen/FlagAttention/blob/main/src/flag_attn/flash.py
# https://github.com/triton-lang/kernels/blob/main/kernels/flash_attention.py
# --

import math
import torch
import triton
import triton.language as tl

# --
# kernels

# --
# note: scoring function mode
# mode0 [nope]
SCORE_FUNC_MODE0_NOPE = 0
# mode1 [individual-doc]: individually encoding if SS<0, otherwise attend to all
SCORE_FUNC_MODE1_DOC = 1
# --

@triton.jit
def _fwd_kernel(
    Q, K, V, softmax_scale,  # main inputs: [z, h, m_or_n, k], (m=qlen, n=kvlen)
    L, Out,  # main outputs: log_sum_exp=[z, h, m], output=[z, h, m, k]
    Eq, Ek,  # extra inputs for score modification: [z, h, m]
    stride_qz, stride_qh, stride_qm, stride_qk,  # strides for each one
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_eqz, stride_eqh, stride_eqm,  # strides for each E
    stride_ekz, stride_ekh, stride_ekn,
    Z, H, M, N, MN_GAP, num_groups,  # main dimension sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr, DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
    SCORE_FUNC: tl.constexpr,  # special scoring function
):
    input_dtype = Q.dtype.element_ty
    # --
    # this grid manages [z, h, BLOCK_M, N]
    start_m = tl.program_id(0)  # qlen // BLOCK_M
    # off_zh = tl.program_id(1)  # z*h
    # off_z, off_h = off_zh // H, off_zh % H  # [z, h]
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    # E^(s*scale) = 2^(s*scale*log2e)
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = softmax_scale * log2e
    # prepare base pointers [z, h]
    off_hk = off_h // num_groups  # group attention
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    Out += off_z * stride_oz + off_h * stride_oh
    L += (off_z * H + off_h) * M  # L is a continuous chunk [z, h, m]
    # prepare offsets (offsets are dimension-specific indexes)
    _idx_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    _idx_n = tl.arange(0, BLOCK_N)  # [BLOCK_N], simply take [0,N) and iter later
    _idx_k = tl.arange(0, BLOCK_DMODEL)  # [K]
    # init starting pointers (pointers actually point to data)
    q_ptrs = Q + (_idx_m[:, None] * stride_qm + _idx_k[None, :] * stride_qk)  # [BLOCK_M, K]
    o_ptrs = Out + (_idx_m[:, None] * stride_om + _idx_k[None, :] * stride_ok)  # [BLOCK_M, K]
    l_ptrs = L + _idx_m  # [BLOCK_M]
    # initialize pointer to m and l, fp32 for accumulators
    # m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)  # [BLOCK_M], maximum
    m_i = tl.full([BLOCK_M], value=-10000., dtype=tl.float32)  # [BLOCK_M], maximum (note: avoiding -inf)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # [BLOCK_M], log-sum-exp
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)  # [BLOCK_M, BLOCK_DMODEL], accumulated Vs
    # =====
    # note: special scoring
    if SCORE_FUNC != 0:
        Eq += off_z * stride_eqz + off_h * stride_eqh
        Ek += off_z * stride_ekz + off_h * stride_ekh
        eq_ptrs = Eq + _idx_m * stride_eqm  # [BLOCK_M]
        ek_ptrs = Ek + _idx_n * stride_ekn  # [BLOCK_N]
        t_extra_q = tl.load(eq_ptrs)  # [BLOCK_M]
    # =====
    # load query
    # if DIVISIBLE_M:
    #     q = tl.load(q_ptrs, cache_modifier=".cg")
    # else:
    #     mask_m = (offs_m < M)
    #     q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")
    query = tl.load(q_ptrs)  # [BLOCK_M, K], note: maybe OOB ones do not matter since this is Q
    query = (query * qk_scale).to(input_dtype)  # [BLOCK_M, K]
    # --
    # looping
    hi = tl.minimum(N, MN_GAP + (start_m + 1) * BLOCK_M) if IS_CAUSAL else N  # MN_GAP = max(0, N-M)
    k_ptrs = K + (_idx_k[:, None] * stride_vk + _idx_n[None, :] * stride_vn)  # [BLOCK_DMODEL, BLOCK_N]
    v_ptrs = V + (_idx_n[:, None] * stride_kn + _idx_k[None, :] * stride_kk)  # [BLOCK_N, BLOCK_DMODEL]
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)  # mark it
        # load kv
        key = tl.load(k_ptrs)  # [BLOCK_N, BLOCK_DMODEL]
        value = tl.load(v_ptrs)  # [BLOCK_N, BLOCK_DMODEL]
        # get score
        score = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        score += tl.dot(query, key)  # [BLOCK_M, BLOCK_N]
        _new_idx_n = start_n + _idx_n  # [BLOCK_N]
        if not DIVISIBLE_N:  # exclude invalid kv
            invalid_mask = (_new_idx_n < N)
            score = tl.where(invalid_mask, score, float("-inf"))
        if IS_CAUSAL:  # exclude causal
            causal_mask = (MN_GAP + _idx_m[:, None]) >= _new_idx_n[None, :]
            score = tl.where(causal_mask, score, float("-inf"))
        # =====
        # note: special scoring
        if SCORE_FUNC != 0:
            t_extra_k = tl.load(ek_ptrs)  # [BLOCK_N]
            if SCORE_FUNC == 1:  # mode1 [individual-doc]
                doc_mask = (t_extra_q[:, None] >= 0) | (t_extra_q[:, None] == t_extra_k[None, :])  # [BLOCK_M, BLOCK_N]
                score = tl.where(doc_mask, score, float("-inf"))
            ek_ptrs += BLOCK_N * stride_ekn  # next
        # =====
        # get max and rescale
        mi_new = tl.maximum(m_i, tl.max(score, 1))
        alpha = tl.math.exp2(m_i - mi_new)  # alpha=exp2(mi_old-mi_new)
        prob = tl.math.exp2(score - mi_new[:, None])  # [BLOCK_M, BLOCK_N], exp2(s'-mi_new) = prob * exp2(mi_new)
        # updates
        l_i = l_i * alpha + tl.sum(prob, 1)  # [BLOCK_M], \sum_exp2(s'-mi_new)
        m_i = mi_new  # [BLOCK_M]
        acc *= alpha[:, None]  # [BLOCK_M, BLOCK_DMODEL]
        acc += tl.dot(prob.to(input_dtype), value)  # [BLOCK_M, BLOCK_DMODEL], \sum_exp2(s'-mi_new)*v
        # advance kv pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
    # --
    # final write back
    acc = acc / l_i[:, None]  # out = \sum_exp2(s'-mi_new)*v / \sum_exp2(s'-mi_new)
    logsumexp = m_i + tl.math.log2(l_i)  # [BLOCK_M], \log2 \sum exp2(s'), note: exp2(s')==exp(s)
    if DIVISIBLE_M:
        tl.store(l_ptrs, logsumexp)
        tl.store(o_ptrs, acc.to(input_dtype))
    else:
        mask_m = (_idx_m < M)
        tl.store(l_ptrs, logsumexp, mask=mask_m)
        tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None])
    # --

@triton.jit
def _bwd_preprocess(
    Out, DO, Delta,  # [z, h, m, k], [z, h, k]
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dz, stride_dh, stride_dm,
    M,  # main dimension sizes
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr, DIVISIBLE_M: tl.constexpr
):
    # advance z & h first
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    Out += off_z * stride_oz + off_h * stride_oh
    DO += off_z * stride_doz + off_h * stride_doh
    Delta += off_z * stride_dz + off_h * stride_dh
    # load
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_k = tl.arange(0, D_HEAD)
    o = tl.load(Out + off_m[:, None] * stride_om + off_k[None, :] * stride_ok).to(tl.float32)  # [z, h, m, k]
    do = tl.load(DO + off_m[:, None] * stride_dom + off_k[None, :] * stride_dok).to(tl.float32)  # [z, h, m, k]
    # compute
    # note: scale can get cancelled, Of=Or*scale, Dr=Df*scale, Of*Df=Or*scale*Dr/scale=Or*Dr
    delta = tl.sum(o * do, axis=1)  # [z, h, m]
    # write-back
    d_ptrs = Delta + off_m * stride_dm
    if DIVISIBLE_M:
        tl.store(d_ptrs, delta)
    else:
        mask_m = (off_m < M)
        tl.store(d_ptrs, delta, mask=mask_m)  # [z, h, m]
    # --

@triton.jit
def _bwd_kv_kernel(
    Q, K, V, softmax_scale,  # main inputs: [z, h, m_or_n, k], (m=qlen, n=kvlen)
    DO, DK, DV,  # grads: [z, h, m_or_n, k]
    L, D,  # [z, h, m]
    Eq, Ek,  # extra inputs for score modification: [z, h, m]
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    stride_eqz, stride_eqh, stride_eqm,  # strides for each E
    stride_ekz, stride_ekh, stride_ekn,
    Z, H, M, N, MN_GAP, num_groups,  # main dimension sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr, DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
    SCORE_FUNC: tl.constexpr,  # special scoring function
):
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    start_n = tl.program_id(0)  # klen // BLOCK_N
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = softmax_scale * log2e
    # offset pointers for (batch, head)
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    DO += off_z * stride_doz + off_h * stride_doh
    # offset pointers for batch/head
    DK += off_z * stride_dkz + off_h * stride_dkh
    DV += off_z * stride_dvz + off_h * stride_dvh
    # offset pointers for batch/head
    D += (off_z * H + off_h) * M  # L/D is a continuous chunk [z, h, m]
    L += (off_z * H + off_h) * M
    # loop over a col
    if IS_CAUSAL:
        lo = tl.maximum(start_n * BLOCK_N - MN_GAP, 0)
        lo = (lo // BLOCK_M) * BLOCK_M
    else:
        lo = 0
    offs_m_init = lo + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_m_base = tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_k = tl.arange(0, BLOCK_DMODEL)  # [BLOCK_DMODEL]
    # initialize pointers to value-like data
    q_ptrs = Q + (offs_m_init[:, None] * stride_qm + offs_k[None, :] * stride_qk)  # (BLOCK_M, BLOCK_DMODEL)
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)  # (BLOCK_N, BLOCK_DMODEL)
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)  # (BLOCK_N, BLOCK_DMODEL)
    do_ptrs = DO + (offs_m_init[:, None] * stride_dom + offs_k[None, :] * stride_dok)  # (BLOCK_M, BLOCK_DMODEL)
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk)  # (BLOCK_N, BLOCK_DMODEL)
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_k[None, :] * stride_dvk)  # (BLOCK_N, BLOCK_DMODEL)
    # k and v stay in SRAM throughout
    if DIVISIBLE_N:
        v = tl.load(v_ptrs)
        k = tl.load(k_ptrs)
    else:
        mask_n = offs_n < N
        v = tl.load(v_ptrs, mask=mask_n[:, None])
        k = tl.load(k_ptrs, mask=mask_n[:, None])
    # initialize dk and dv
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    # =====
    # note: special scoring
    if SCORE_FUNC != 0:
        Eq += off_z * stride_eqz + off_h * stride_eqh
        Ek += off_z * stride_ekz + off_h * stride_ekh
        eq_ptrs = Eq + offs_m_init * stride_eqm  # [BLOCK_M]
        ek_ptrs = Ek + offs_n * stride_ekn  # [BLOCK_N]
        t_extra_k = tl.load(ek_ptrs)  # [BLOCK_N]
    # =====
    for start_m in range(lo, M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)  # mark it
        offs_m = start_m + offs_m_base
        valid_mask = (offs_m < M)[:, None]  # (BLOCK_M, 1)
        causal_mask = (MN_GAP + offs_m[:, None]) >= (offs_n[None, :])  # (BLOCK_M, BLOCK_N)
        # load q1, k1, q2, k2, v, do on-chip
        # if DIVISIBLE_M:
        #     q = tl.load(q_ptrs)
        # else:
        #     mask_m = offs_m < M
        #     valid_mask = mask_m[:, None] # & mask_n
        #     q = tl.load(q_ptrs, mask=mask_m[:, None])
        query = tl.load(q_ptrs)  # [BLOCK_M, BLOCK_DMODEL]
        # recompute p = softmax(qk * sm_scale, dim=-1)
        score = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        score += tl.dot(query, tl.trans(k))  # [BLOCK_M, BLOCK_N]
        logsumexp = tl.load(L + offs_m)  # [BLOCK_M]
        # --
        # p = exp(s)/sum_exp(s) = 2**(s*log2e - log2(sum_exp(s))) = 2**(s*log2e - logsumexp)
        prob = tl.math.exp2(score * qk_scale - logsumexp[:, None])  # [BLOCK_M, BLOCK_N]
        if not DIVISIBLE_M:
            prob = tl.where(valid_mask, prob, 0.0)
        if IS_CAUSAL:
            prob = tl.where(causal_mask, prob, 0.0)
        # =====
        # note: special scoring
        if SCORE_FUNC != 0:
            t_extra_q = tl.load(eq_ptrs)  # [BLOCK_M]
            if SCORE_FUNC == 1:  # mode1 [individual-doc]
                doc_mask = (t_extra_q[:, None] >= 0) | (t_extra_q[:, None] == t_extra_k[None, :])
                prob = tl.where(doc_mask, prob, 0.0)
            eq_ptrs += BLOCK_M * stride_eqm  # next
        # =====
        # --
        # compute dv = dot(p^T, do)
        # if DIVISIBLE_M:
        #     do = tl.load(do_ptrs)
        # else:
        #     mask_m = (offs_m < M)
        #     do = tl.load(do_ptrs, mask=mask_m[:, None])  # [BLOCK_M, BLOCK_DMODEL]
        do = tl.load(do_ptrs)  # [BLOCK_M, BLOCK_DMODEL]
        dv += tl.dot(tl.trans(prob).to(input_dtype), do) # [BLOCK_N, BLOCK_DMODEL]
        # compute dp = dot(v, do)
        # if DIVISIBLE_M:
        #     delta = tl.load(D + offs_m)
        # else:
        #     delta = tl.load(D + offs_m, mask=mask_m)
        delta = tl.load(D + offs_m)  # [BLOCK_M]
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)  # [BLOCK_M, BLOCK_N]
        dp += tl.dot(do, tl.trans(v))
        # compute ds = p * (dp - delta[:, None])
        ds = prob * (dp - delta[:, None])  # [BLOCK_M, BLOCK_N]
        # if not DIVISIBLE_M:  # no need here, already handled with prob
        #     ds = tl.where(valid_mask, ds, 0.0)
        # if IS_CAUSAL:
        #     ds = tl.where(causal_mask, ds, 0.0)
        ds = ds.to(input_dtype)
        # compute dk = dot(ds.T, q) masking
        dk += tl.dot(tl.trans(ds), query)  # [BLOCK_N, BLOCK_DMODEL]
        # increment pointers
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
    # --
    dk *= softmax_scale
    if DIVISIBLE_N:
        tl.store(dk_ptrs, dk.to(input_dtype))  # (BLOCK_N, BLOCK_DMODEL)
        tl.store(dv_ptrs, dv.to(input_dtype))  # (BLOCK_N, BLOCK_DMODEL)
    else:
        mask_n = (offs_n < N)
        tl.store(dk_ptrs, dk.to(input_dtype), mask=mask_n[:, None])  # (BLOCK_N, BLOCK_DMODEL)
        tl.store(dv_ptrs, dv.to(input_dtype), mask=mask_n[:, None])  # (BLOCK_N, BLOCK_DMODEL)
    # --

@triton.jit
def _bwd_q_kernel(
    Q, K, V, softmax_scale,  # main inputs: [z, h, m_or_n, k], (m=qlen, n=kvlen)
    DO, DQ,  # grads: [z, h, m, k]
    L, D,  # [z, h, m]
    Eq, Ek,  # extra inputs for score modification: [z, h, m, SS]
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_eqz, stride_eqh, stride_eqm,  # strides for each E
    stride_ekz, stride_ekh, stride_ekn,
    Z, H, M, N, MN_GAP, num_groups,  # main dimension sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr, DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
    SCORE_FUNC: tl.constexpr,  # special scoring function
):  # note: this one is very similar to forward (since both looping over kv)
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    start_m = tl.program_id(0)  # qlen // BLOCK_M
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = softmax_scale * log2e
    # offset pointers for (batch, head)
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    DO += off_z * stride_doz + off_h * stride_doh
    DQ += off_z * stride_dqz + off_h * stride_dqh
    D += (off_z * H + off_h) * M  # L/D is a continuous chunk [z, h, m]
    L += (off_z * H + off_h) * M
    # prepare offsets
    _idx_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    _idx_n = tl.arange(0, BLOCK_N)  # [BLOCK_N], simply take [0,N) and iter later
    _idx_k = tl.arange(0, BLOCK_DMODEL)  # [K]
    # init starting pointers (pointers actually point to data)
    q_ptrs = Q + (_idx_m[:, None] * stride_qm + _idx_k[None, :] * stride_qk)  # [BLOCK_M, K]
    do_ptrs = DO + (_idx_m[:, None] * stride_dom + _idx_k[None, :] * stride_dok)  # [BLOCK_M, K]
    dq_ptrs = DQ + (_idx_m[:, None] * stride_dqm + _idx_k[None, :] * stride_dqk)  # [BLOCK_M, K]
    # load query
    query = tl.load(q_ptrs)  # [BLOCK_M, K], note: maybe OOB ones do not matter since this is Q
    query = (query * qk_scale).to(input_dtype)  # [BLOCK_M, K]
    do = tl.load(do_ptrs)  # [BLOCK_M, BLOCK_DMODEL]
    logsumexp = tl.load(L + _idx_m)  # [BLOCK_M]
    delta = tl.load(D + _idx_m)  # [BLOCK_M]
    # --
    # loop over a row
    lo = 0
    hi = tl.minimum(N, MN_GAP + (start_m + 1) * BLOCK_M) if IS_CAUSAL else N  # MN_GAP = max(0, N-M)
    k_ptrs = K + (_idx_k[:, None] * stride_vk + _idx_n[None, :] * stride_vn)  # [BLOCK_DMODEL, BLOCK_N]
    v_ptrs = V + (_idx_n[:, None] * stride_kn + _idx_k[None, :] * stride_kk)  # [BLOCK_N, BLOCK_DMODEL]
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)  # (BLOCK_M, BLOCK_DMODEL)
    # =====
    # note: special scoring
    if SCORE_FUNC != 0:
        Eq += off_z * stride_eqz + off_h * stride_eqh
        Ek += off_z * stride_ekz + off_h * stride_ekh
        eq_ptrs = Eq + _idx_m * stride_eqm  # [BLOCK_M]
        ek_ptrs = Ek + _idx_n * stride_ekn  # [BLOCK_N]
        t_extra_q = tl.load(eq_ptrs)  # [BLOCK_M]
    # =====
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)  # mark it
        # load kv
        key = tl.load(k_ptrs)  # [BLOCK_DMODEL, BLOCK_N]
        value = tl.load(v_ptrs)  # [BLOCK_N, BLOCK_DMODEL]
        # recompute p = softmax(qk * sm_scale, dim=-1)
        score = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        score += tl.dot(query, key)  # [BLOCK_M, BLOCK_N]
        # p = ... = 2**(s*log2e - logsumexp)
        prob = tl.math.exp2(score - logsumexp[:, None])  # [BLOCK_M, BLOCK_N]
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)  # [BLOCK_M, BLOCK_N]
        dp += tl.dot(do, tl.trans(value))  # [BLOCK_M, BLOCK_N]
        # compute ds = p * (dp - delta[:, None])
        ds = prob * (dp - delta[:, None])  # [BLOCK_M, BLOCK_N]
        _new_idx_n = start_n + _idx_n  # [BLOCK_N]
        if not DIVISIBLE_N:
            valid_mask = (_new_idx_n < N)
            ds = tl.where(valid_mask, ds, 0.0)
        if IS_CAUSAL:
            causal_mask = (MN_GAP + _idx_m[:, None]) >= (_new_idx_n[None, :])  # (BLOCK_M, BLOCK_N)
            ds = tl.where(causal_mask, ds, 0.0)
        # =====
        # note: special scoring
        if SCORE_FUNC != 0:
            t_extra_k = tl.load(ek_ptrs)  # [BLOCK_N]
            if SCORE_FUNC == 1:  # mode1 [individual-doc]
                doc_mask = (t_extra_q[:, None] >= 0) | (t_extra_q[:, None] == t_extra_k[None, :])
                ds = tl.where(doc_mask, ds, 0.0)
            ek_ptrs += BLOCK_N * stride_ekn  # next
        # =====
        ds = ds.to(input_dtype)  # [BLOCK_M, BLOCK_N]
        # compute dq = dot(ds, k)
        # tl.static_print(111, ds, tl.trans(key))
        dq += tl.dot(ds, tl.trans(key))  # [BLOCK_M, BLOCK_DMODEL]
        # advance kv pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
    # --
    dq *= softmax_scale
    if DIVISIBLE_M:
        tl.store(dq_ptrs, dq.to(input_dtype))  # (BLOCK_M, BLOCK_DMODEL)
    else:
        mask_m = (_idx_m < M)
        tl.store(dq_ptrs, dq.to(input_dtype), mask=mask_m[:, None])  # (BLOCK_M, BLOCK_DMODEL)
    # --

# --
# helpers
def maybe_contiguous(x):
    # only when the innermost dimension is contiguous can LDGSTS instruction be used so inner-dimension contiguity is enforced.
    return x.contiguous() if x.stride(-1) != 1 else x

# --
# configs
# (too lazy to tune more, the FlagAttention ones already seem good with A100)

def get_fwd_config(B, H, M, N, D, causal):
    if torch.cuda.get_device_capability() == (8, 0):
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
            else:
                if M <= 1024:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 3, 4
                else:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 3, 8
        else:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 4, 4
            else:
                if M <= 1024:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
                else:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 3, 8
    elif torch.cuda.get_device_capability() == (8, 6):
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
        else: # causal
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 1, 4
    return (BLOCK_M, BLOCK_N, num_stages, num_warps)

def get_bwd_config(B, H, M, N, D, causal):
    if torch.cuda.get_device_capability() == (8, 0):
        if not causal:
            BLOCK_M = 128 if D <= 64 else 64
            BLOCK_N = 64
            num_stages = 2
            num_warps = 4
        else:
            BLOCK_M = 64
            BLOCK_N = 64
            num_stages = 3 if D <= 64 else 2
            num_warps = 4
    elif torch.cuda.get_device_capability() == (8, 6): # tune for RTX-3090, device_capability(8, 6)
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 8
        else:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 2, 4
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 1, 4
    return (BLOCK_M, BLOCK_N, num_stages, num_warps)
# --

# --
# Function
class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, eq, ek, score_func_mode):
        # check model dim
        Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Dq == Dk and Dk == Dv, "feature size of q, k, v should be equal"
        assert Dk in {16, 32, 64, 128}
        # check other dims
        B, H, M, D = q.shape
        N = k.shape[2]
        Hk, Hv = k.shape[1], v.shape[1]
        assert Hk == Hv, "num of heads in k and v should be equal"
        assert H % Hk == 0, "number of heads in q must be a multiple of that in k & v"
        num_groups = H // Hk
        assert (N >= M) or (not causal), "KV length should be larger than M length in causal mode!"
        MN_GAP = max(0, N-M)
        # scale
        if sm_scale is None:
            sm_scale = 1. / math.sqrt(D)
        q, k, v = maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v)
        # score mode
        assert (eq is None) == (ek is None)
        if eq is None:
            assert score_func_mode == SCORE_FUNC_MODE0_NOPE
            eq, ek = q, k  # simply pass something in!
        # get around device mismatch problem
        device = torch.cuda.device_of(q)
        with torch.cuda.device(device):
            config = get_fwd_config(B, H, M, N, D, causal)  # get heuristics
            BLOCK_M, BLOCK_N, num_stages, num_warps = config
            divisible_m = (M % BLOCK_M == 0)
            divisible_n = (N % BLOCK_N == 0)
            grid = (triton.cdiv(M, BLOCK_M), H, B)  # [qlen//BLOCK_M, z, h]
            Out = torch.empty_like(q)
            L = torch.empty((B, H, M), device=q.device, dtype=torch.float32)
            _fwd_kernel[grid](
                q, k, v, sm_scale,
                L, Out,
                eq, ek,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
                eq.stride(0), eq.stride(1), eq.stride(2),
                ek.stride(0), ek.stride(1), ek.stride(2),
                B, H, M, N, MN_GAP, num_groups,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
                IS_CAUSAL=causal, DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
                SCORE_FUNC=score_func_mode,
                num_warps=num_warps, num_stages=num_stages,
            )
        # autograd context maintenance
        ctx.save_for_backward(q, k, v, Out, L, eq, ek)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.shapes = (B, H, M, D, N, Hk, num_groups, MN_GAP)  # ...
        ctx.score_func_info = (score_func_mode, )
        # return
        return Out

    @staticmethod
    def backward(ctx, do):
        # get saved items
        q, k, v, o, L, eq, ek = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        B, H, M, D, N, Hk, num_groups, MN_GAP = ctx.shapes
        score_func_mode, = ctx.score_func_info
        # --
        # get around device mismatch problem
        device = torch.cuda.device_of(q)
        with torch.cuda.device(device):
            config = get_bwd_config(B, H, M, N, D, causal)  # get heuristics
            BLOCK_M, BLOCK_N, num_stages, num_warps = config
            divisible_m = (M % BLOCK_M == 0)
            divisible_n = (N % BLOCK_N == 0)
            # --
            # calculating d*do
            delta = torch.empty_like(L)  # [B, H, M]
            _bwd_preprocess[(triton.cdiv(M, BLOCK_M), H, B)](
                o, do, delta,
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                delta.stride(0), delta.stride(1), delta.stride(2),
                M,
                BLOCK_M=BLOCK_M, D_HEAD=D, DIVISIBLE_M=divisible_m
            )
            # calculating dkdv
            dk = torch.empty((B, H, N, D), dtype=k.dtype, device=q.device)
            dv = torch.empty((B, H, N, D), dtype=v.dtype, device=q.device)
            _bwd_kv_kernel[(triton.cdiv(N, BLOCK_N), H, B)](
                q, k, v, sm_scale,
                do, dk, dv,
                L, delta,
                eq, ek,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                eq.stride(0), eq.stride(1), eq.stride(2),
                ek.stride(0), ek.stride(1), ek.stride(2),
                B, H, M, N, MN_GAP, num_groups,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
                IS_CAUSAL=causal, DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
                SCORE_FUNC=score_func_mode,
                num_stages=num_stages, num_warps=num_warps,
            )
            # calculating dq
            dq = torch.zeros_like(q)
            _bwd_q_kernel[(triton.cdiv(M, BLOCK_M), H, B)](
                q, k, v, sm_scale,
                do, dq,
                L, delta,
                eq, ek,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                eq.stride(0), eq.stride(1), eq.stride(2),
                ek.stride(0), ek.stride(1), ek.stride(2),
                B, H, M, N, MN_GAP, num_groups,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
                IS_CAUSAL=causal, DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
                SCORE_FUNC=score_func_mode,
                num_stages=num_stages, num_warps = num_warps,
            )
            if num_groups > 1:
                dk = dk.reshape((B, Hk, num_groups, N, D)).sum(2)
                dv = dv.reshape((B, Hk, num_groups, N, D)).sum(2)
        return dq, dk, dv, None, None, None, None, None

def attention(q, k, v, causal=False, sm_scale=None, eq=None, ek=None, score_func_mode=SCORE_FUNC_MODE0_NOPE):
    return FlashAttention.apply(q, k, v, causal, sm_scale, eq, ek, score_func_mode)
