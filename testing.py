#

# test different versions of ATT

import math
import torch
import numpy as np
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
from triton.testing import do_bench

try:
    from mspx.utils import ZDEBUG, stat_analyze
except:
    import os
    # --
    ZDEBUG = lambda: os.environ.get("ZDEBUG", False)
    stat_analyze = None
    # --

# --
_device_capability = torch.cuda.get_device_capability("cuda")
is_sm75 = _device_capability == (7, 5)
is_sm8x = _device_capability[0] == 8
is_sm80 = _device_capability == (8, 0)
is_sm90 = _device_capability == (9, 0)
print(f"DEVICE-CAPA: {_device_capability}")
# --

# different versions of ATT

def _attn_torch(q, k, v, is_causal: bool, attn_mask=None, **kwargs):
    Lq, Lk = q.size(-2), k.size(-2)
    scale_factor = 1 / math.sqrt(q.size(-1))
    attn_weight = q @ k.transpose(-2, -1) * scale_factor  # [*, H, Lq, Lk]
    if is_causal:
        assert Lk >= Lq
        _posi_q = torch.arange(Lq).to(q)  # [Lq]
        _posi_kv = torch.arange(Lk).to(q) - (Lk - Lq)  # [Lk]
        attn_bias = torch.where((_posi_q.unsqueeze(-1) >= _posi_kv.unsqueeze(-2)), 0., float("-inf"))  # [Lq, Lk]
        attn_weight += attn_bias  # [*, H, Lq, Lk]
    if attn_mask is not None:
        attn_weight = torch.where((attn_mask > 0), attn_weight, float("-inf"))  # [Lq, Lk]
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32).to(v.dtype)  # [*, H, Lq, Lk]
    # attn_weight = torch.dropout(attn_weight, dropout_p, train=True)  # note: no dropout for now
    # breakpoint()
    return attn_weight @ v  # [*, Lq, D]

def _attn_sdpa(q, k, v, is_causal: bool, backend, attn_mask=None, **kwargs):
    with sdpa_kernel(backend):
        ret = scaled_dot_product_attention(q, k, v, is_causal=is_causal, attn_mask=attn_mask, **kwargs)
    return ret

_attn_flash=(lambda *args, **kwargs: _attn_sdpa(*args, **kwargs, backend=SDPBackend.FLASH_ATTENTION))
_attn_math=(lambda *args, **kwargs: _attn_sdpa(*args, **kwargs, backend=SDPBackend.MATH))
_attn_mem=(lambda *args, **kwargs: _attn_sdpa(*args, **kwargs, backend=SDPBackend.EFFICIENT_ATTENTION))
_attn_cudnn=(lambda *args, **kwargs: _attn_sdpa(*args, **kwargs, backend=SDPBackend.CUDNN_ATTENTION))

def _attn_flashattn(q, k, v, is_causal: bool, do_flashattn_transpose=True):
    from flash_attn import flash_attn_func
    if do_flashattn_transpose:
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ret = flash_attn_func(q, k, v, causal=is_causal)
    if do_flashattn_transpose:
        ret = ret.transpose(1, 2)
    return ret

def _attn_triton(q, k, v, is_causal: bool, version=None, **kwargs):
    if version == "og":
        from tests.flash.flash_og import attention
    elif version == "ops":
        from tests.flash.flash_ops import attention
    elif version == "06":
        from tests.flash.flash_06 import attention
    elif version == "flag":
        from tests.flash.flash_flag import attention
    elif version == "triton":
        try:
            from mspx.core.ops.flash_triton import attention
        except:
            from flash_triton import attention
    else:
        attention = None
    ret = attention(q, k, v, is_causal, q.shape[-1]**(-0.5), **kwargs)
    return ret

_attn_triton_og=(lambda *args, **kwargs: _attn_triton(*args, **kwargs, version="og"))
_attn_triton_ops=(lambda *args, **kwargs: _attn_triton(*args, **kwargs, version="ops"))
_attn_triton_06=(lambda *args, **kwargs: _attn_triton(*args, **kwargs, version="06"))
_attn_triton_flag=(lambda *args, **kwargs: _attn_triton(*args, **kwargs, version="flag"))
_attn_triton_triton=(lambda *args, **kwargs: _attn_triton(*args, **kwargs, version="triton"))

# --
def _calculate_flops(shape, is_causal, mode, one_name):
    flops_per_matmul = 2.0 * shape[0] * shape[1] * shape[2] * shape[2] * shape[3]  # 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5
    # flops
    if mode == "fwd":
        one_flops = total_flops
    elif mode in ["bwd", "fb"]:
        one_flops = total_flops * (2 + 0.5 * float(one_name not in ["math", "torch"]))  # 2.0(bwd) + 0.5(recompute)
        if mode == "fb":
            one_flops += total_flops
    else:
        raise RuntimeError()
    return one_flops

def _sample_emask_doc(full_shape, device, is_causal):
    bs, _, qlen, _ = full_shape
    eq = (torch.rand([bs, qlen], device=device) <= 0.05).cumsum(-1)  # [bs, L]
    eq = eq - eq[:, -1:] // 3  # [bs, L]
    emask = ((eq.unsqueeze(-1) >= 0) | (eq.unsqueeze(-1) == eq.unsqueeze(-2))).unsqueeze(-3)  # [bs, 1, Lq, Lk]
    if is_causal:
        _arange = torch.arange(qlen, device=device)
        emask = emask & (_arange[:, None] >= _arange[None, :])  # [bs, 1, Lq, Lk]
    eq = eq[:, None, :].expand(full_shape[:-1])  # [bs, H, L]
    # breakpoint()
    return eq, eq, emask

def _check_attn(one_attn, q, k, v, attn_kwargs, mode, do, ref_ts):
    one_output = one_attn(q, k, v, **attn_kwargs)  # [bs, H, L, D]
    one_ts = [one_output.detach().cpu()]
    if mode == "bwd":
        one_output.backward(do)
        one_dq, one_dk, one_dv = q.grad.detach().cpu(), k.grad.detach().cpu(), v.grad.detach().cpu()
        q.grad, k.grad, v.grad = None, None, None
        one_ts.extend([one_dq, one_dk, one_dv])
    if ref_ts is None:  # put it here for reference!
        ref_ts = one_ts
    # check results
    _diffs = []
    for x, y in zip(ref_ts, one_ts):
        _abs = (x - y).abs()
        # _diffs.append((_abs.max().item(), _abs.mean().item()))
        _diffs.append(_abs.max().item())
        # if _abs.max().item() > 0.1:
        #     breakpoint()
    return ref_ts, _diffs

def _bench_attn(one_attn, q, k, v, mode, do, attn_kwargs, one_flops, warmup_times, bench_times):
    if mode == "fwd":
        fn = lambda: one_attn(q, k, v, **attn_kwargs)
    elif mode == "bwd":
        o = one_attn(q, k, v, **attn_kwargs)
        fn = lambda: o.backward(do, retain_graph=True)
    elif mode == "fb":
        fn = lambda: one_attn(q, k, v, **attn_kwargs).backward(do)
    else:
        raise RuntimeError()
    ms, min_ms, max_ms = do_bench(fn, grad_to_none=[q,k,v], warmup=warmup_times, rep=bench_times, quantiles=[0.5, 0.2, 0.8])
    tflops_per_sec = [f"{one_flops / z * 1e-9:.2f}" for z in [ms, min_ms, max_ms]]
    return tflops_per_sec

def run_attn(func_names, shape, is_causal: bool, dtype, mask_doc: bool, check_times=3, bench_times=50, warmup_times=10):
    attns = [globals()[f"_attn_{z}"] for z in func_names]
    _dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    _all_check_modes, _all_bench_modes = ["bwd"], ["fwd", "bwd", "fb"]
    # --
    data_check, data_bench = [], []  # List[(K,V)]
    # check correctness
    if check_times > 0:
        print(f"# <<<<<\n# ===== Check correctness [{shape},{is_causal},{dtype}]")
        for rid in range(check_times):
            for mode in _all_check_modes:
                kwargs = {"dtype": _dtype, "device": "cuda"}
                if mode == "bwd":
                    kwargs["requires_grad"] = True
                q, k, v = torch.randn(shape, **kwargs), torch.randn(shape, **kwargs), torch.randn(shape, **kwargs)
                if mask_doc:
                    eq, ek, emask = _sample_emask_doc(shape, device="cuda", is_causal=is_causal)
                kwargs["requires_grad"] = False
                do = torch.randn(shape, **kwargs)
                ref_ts = None  # reference tensor
                for one_attn, one_name in zip(attns, func_names):
                    _run_kwargs = {"is_causal": is_causal}
                    if mask_doc:
                        if one_name == "triton_triton":
                            _run_kwargs.update({"eq": eq, "ek": ek, "score_func_mode": 1})
                        else:
                            _run_kwargs.update({"attn_mask": emask, "is_causal": False})  # already combined with mask
                            if one_name == "torch" and is_causal:
                                _run_kwargs.update({"is_causal": True})
                    if ZDEBUG():
                        ref_ts, _diffs = _check_attn(one_attn, q, k, v, _run_kwargs, mode, do, ref_ts)
                        print(f"Compare[{shape},{is_causal},{dtype},{rid},{mode},{one_name}]: diff={_diffs}")
                    else:
                        try:
                            ref_ts, _diffs = _check_attn(one_attn, q, k, v, _run_kwargs, mode, do, ref_ts)
                            print(f"Compare[{shape},{is_causal},{dtype},{rid},{mode},{one_name}]: diff={_diffs}")
                        except Exception as e:
                            _diffs = [-1.] * (4 if mode == "bwd" else 1)
                            print(f"Compare[{shape},{is_causal},{dtype},{rid},{mode},{one_name}]: ERR={e}")
                    data_check.append(({"shape": shape, "is_causal": is_causal, "dtype": str(dtype), "rid": rid, "mode": mode, "func_name": one_name}, _diffs))  # add results
    # --
    # benchmark speed
    if bench_times:
        print(f"# <<<<<\n# ===== Benchmark [{shape},{is_causal},{dtype}]")
        for mode in _all_bench_modes:
            kwargs = {"dtype": _dtype, "device": "cuda"}
            if mode in ["bwd", "fb"]:
                kwargs["requires_grad"] = True
            kwargs2 = kwargs.copy()
            kwargs2["requires_grad"] = False
            for one_attn, one_name in zip(attns, func_names):
                one_flops = _calculate_flops(shape, is_causal, mode, one_name)
                # --
                # prepare tensors
                _cur_shape = shape
                _run_kwargs = {"is_causal": is_causal}
                if mask_doc:
                    eq, ek, emask = _sample_emask_doc(shape, device="cuda", is_causal=is_causal)
                    if one_name == "triton_triton":
                        _run_kwargs.update({"eq": eq, "ek": ek, "score_func_mode": 1})
                    else:
                        _run_kwargs.update({"attn_mask": emask, "is_causal": False})  # already combined with mask
                        if one_name == "torch" and is_causal:
                            _run_kwargs.update({"is_causal": True})
                _tstr = ""  # print for checking
                if one_name in ["flashattn"]:
                    _cur_shape = (shape[:1] + [shape[2], shape[1]] + shape[3:])
                    _run_kwargs["do_flashattn_transpose"] = False
                    _tstr = "; T=False"
                q, k, v = torch.randn(_cur_shape, **kwargs), torch.randn(_cur_shape, **kwargs), torch.randn(_cur_shape, **kwargs)
                do = torch.randn(_cur_shape, **kwargs2)
                # --
                # run
                if ZDEBUG():
                    tpflops_per_sec = _bench_attn(one_attn, q, k, v, mode, do, _run_kwargs, one_flops, warmup_times, bench_times)
                    print(f"Throughput(tflops/s)[{shape},{is_causal},{dtype},{mode},{one_name}]: mean/max/min={tpflops_per_sec}{_tstr}")
                else:
                    try:
                        tpflops_per_sec = _bench_attn(one_attn, q, k, v, mode, do, _run_kwargs, one_flops, warmup_times, bench_times)
                        print(f"Throughput(tflops/s)[{shape},{is_causal},{dtype},{mode},{one_name}]: mean/max/min={tpflops_per_sec}{_tstr}")
                    except Exception as e:
                        tpflops_per_sec = (-1, -1, -1)
                        print(f"Throughput(tflops/s)[{shape},{is_causal},{dtype},{mode},{one_name}]: ERR={e}{_tstr}")
                data_bench.append(({"shape": shape, "is_causal": is_causal, "dtype": str(dtype), "mode": mode, "func_name": one_name}, tpflops_per_sec))  # add results
    # --
    # finished
    print(f"# >>>>> Finished")
    return data_check, data_bench

# --

# helpers
def _func_label(k):
    return f"Mode={k['mode']},C={k['is_causal']},F={k['func_name']}", f"L={k['shape'][2]:02d},D={k['dtype']}"

def _func_entry_check(d):
    ret = f"({len(d)})"
    _mean_results = np.asarray([z[1] for z in d]).mean(0).flatten().tolist()
    _mean_res = [f"{z:.4f}" for z in _mean_results]
    ret += "/".join(_mean_res)
    return ret

def _func_entry_bench(d):
    assert len(d) == 1
    _res = float(d[0][1][0])  # take the first entry
    ret = f"{_res:.2f}"
    return ret

def main():
    # func_names = ["flashattn", "math", "torch", "mem", "flash", "triton"]
    # func_names = ["mem", "flash", "triton_og", "triton_06", "triton_ops", "triton_flag"]
    # func_names = ["mem", "math", "torch", "flash", "triton_flag", "triton_triton"]
    func_names = ["mem", "math", "torch", "triton_triton"]
    # func_names = ["mem", "flash", "triton_triton"]
    shapes = [  # make total 16K
        # llama-7B: 32*128=4K hidden-size
        # [16//L, 32, 1024*L, 64] for L in [1, 2, 4, 8, 16]
        [16//L, 32, 1024*L, 128] for L in [1, 2, 4, 8, 16]
    ]
    # dtypes = ["fp32", "fp16", "bf16"]
    # dtypes = ["fp16", "bf16"]
    dtypes = ["bf16"]
    causals = [False, True]
    data_check, data_bench = [], []
    mask_doc = True
    for shape in shapes:
        for is_causal in causals:
            for dtype in dtypes:
                a, b = run_attn(func_names, shape, is_causal, dtype, mask_doc=mask_doc, check_times=1, bench_times=100)
                data_check.extend(a)
                data_bench.extend(b)
    # --
    if stat_analyze:
        df_check = stat_analyze(data_check, _func_label, _func_entry_check)
        df_bench = stat_analyze(data_bench, _func_label, _func_entry_bench)
        print(df_check.to_string())
        print(df_bench.to_string())
    breakpoint()

# TRITON_PTXAS_PATH=/root/.conda/envs/myenv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin/ptxas CUDA_VISIBLE_DEVICES=0 python testing.py
if __name__ == '__main__':
    main()
