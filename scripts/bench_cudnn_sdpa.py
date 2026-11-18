#!/usr/bin/env python3
"""SDPA forward time on torch's cuDNN attention backend at the per-rank self-attention shapes of the video cells, one GPU.

The 2026-11-16 round's vLLM-Omni got 2.8-3.3% faster per denoise step on every CUDNN_ATTN cell after torch 2.13.0 -> 2.14.0
(cuDNN 9.20.0.48 -> 9.24.0.43) while the TRTLLM_ATTN and Triton-VSA cells did not move. This isolates the attention call:
run it in a venv per torch/cuDNN pair, and swap the cuDNN wheel inside one venv (pip install --no-deps nvidia-cudnn-cu13==<v>)
to separate the cuDNN version from the torch version. EFFICIENT_ATTENTION is the control: it is not cuDNN, so it should not move.

    CUDA_VISIBLE_DEVICES=0 <venv>/bin/python scripts/bench_cudnn_sdpa.py [--iters 30]
"""
import argparse
import json
import statistics

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

# (batch, heads on this rank, sequence, head_dim): MiniMax-H3 T2VA 768x1344x124f has 37742 rows and 56 heads, so 28 per rank
# under TP2 or Ulysses 2; Wan2.2 A14B 720p 81f has 75600 rows and 40 heads, so 20 per rank under USP 2
SHAPES = {"h3_rank_28h_37742": (1, 28, 37742, 128), "wan_rank_20h_75600": (1, 20, 75600, 128)}
BACKENDS = {"cudnn": SDPBackend.CUDNN_ATTENTION, "efficient_control": SDPBackend.EFFICIENT_ATTENTION}


def bench(shape, backend, iters, warmup):
    torch.manual_seed(0)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    with sdpa_kernel(backend):
        for _ in range(warmup):
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    return {"median_ms": round(statistics.median(times), 3), "min_ms": round(min(times), 3), "max_ms": round(max(times), 3)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    out = {
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(0),
        "results": {},
    }
    for shape_name, shape in SHAPES.items():
        for backend_name, backend in BACKENDS.items():
            out["results"][f"{shape_name}/{backend_name}"] = bench(shape, backend, args.iters, args.warmup)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
