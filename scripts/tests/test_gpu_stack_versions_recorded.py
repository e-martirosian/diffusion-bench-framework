#!/usr/bin/env python3
"""A result names the GPU library wheels sglang ran on.

cuDNN 9.20 -> 9.24 alone moved every CUDNN_ATTN cell by ~3% between two rounds and neither
result carried a cuDNN version, so the attribution took a second box and a wheel swap.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from diffusion_bench import run_comparison as rc  # noqa: E402

installed = {"sglang": "0.0.0.dev1", "torch": "2.14.1", "nvidia-cudnn-cu13": "9.24.0.43", "nvidia-nccl-cu13": "2.30.7", "triton": "3.8.0"}
rc._installed_version = installed.get
meta = rc._collect_sglang_runtime_metadata()
fail = 0


def check(label, ok, detail=""):
    global fail
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f" -- {detail}" if not ok else ""))
    fail |= not ok


check("the sglang runtime block lists torch and the GPU wheels that are installed",
      meta.get("packages") == {k: v for k, v in installed.items() if k != "sglang"}, str(meta.get("packages")))
check("a wheel that is not installed is left out, not recorded as None", "nvidia-cudnn-cu12" not in meta.get("packages", {}))
check("the package version is still recorded", meta.get("package_version") == "0.0.0.dev1")
check("competitor venvs are asked for the same wheels", {"nvidia-cudnn-cu13", "nvidia-nccl-cu13", "triton"} <= set(rc.GPU_STACK))
sys.exit(fail)
