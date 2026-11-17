#!/usr/bin/env python3
"""A run that measured the environment instead of the frameworks is not published.

2026-11-01's H200 round lost every sglang cell to a rebuilt HF cache
(`Diffusers pipeline index not found`), had zero comparable rows, and still
replaced the last valid round as the page's headline: the zero count only
printed a warning.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("publish_bench_run", ROOT / "scripts" / "publish_bench_run.py")
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fail = 1


ok = lambda **m: {"status": "ok", **m}  # noqa: E731
bad = lambda why: {"status": "failed", "reason": why}  # noqa: E731


def section(*rows):
    return {"id": "probe", "rows": [{"case_id": f"c{i}", "cells": cells} for i, cells in enumerate(rows)]}


infra = "FileNotFoundError: Diffusers pipeline index not found"

check(
    "every cell failed (the 2026-11-01 shape)",
    pub._refusal(section(*[{"sglang": bad(infra), "vllm-omni": bad(infra)}] * 3)) is not None,
)
check(
    "a competitor is measured but sglang is not: still refused (nothing to compare)",
    pub._refusal(section({"sglang": bad(infra), "vllm-omni": ok(latency_s=1.0)})) is not None,
)
check(
    "no row has two measured frameworks",
    "two measured frameworks" in (pub._refusal(section({"sglang": ok(latency_s=1.0), "vllm-omni": bad("oom")})) or ""),
)
check(
    "sglang failing on most rows is refused even when a few rows compare",
    pub._refusal(
        section(
            {"sglang": ok(latency_s=1.0), "vllm-omni": ok(latency_s=1.1)},
            {"sglang": bad(infra), "vllm-omni": ok(latency_s=1.0)},
            {"sglang": bad(infra), "vllm-omni": ok(latency_s=1.0)},
        )
    )
    is not None,
)
check(
    "a healthy section passes",
    pub._refusal(
        section(
            {"sglang": ok(latency_s=1.0), "vllm-omni": ok(latency_s=1.1)},
            {"sglang": ok(latency_s=2.0), "vllm-omni": bad("unsupported")},
        )
    )
    is None,
)
check(
    "half the rows failing is tolerated (the cutoff is 'most')",
    pub._refusal(
        section(
            {"sglang": ok(latency_s=1.0), "vllm-omni": ok(latency_s=1.1)},
            {"sglang": bad("oom"), "vllm-omni": ok(latency_s=1.0)},
        )
    )
    is None,
)
check(
    "a throughput section compares on qps",
    pub._refusal(section({"sglang": ok(qps=0.5), "vllm-omni": ok(qps=0.4)})) is None,
)
check("a section without rows is not this gate's business", pub._refusal({"id": "probe", "rows": []}) is None)

sys.exit(fail)
