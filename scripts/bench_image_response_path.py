#!/usr/bin/env python3
"""Host cost of returning one image from sglang's images API: base64 plus the response serialization, old path against new.

Runs both routes in-process through FastAPI's TestClient with the real ImageResponse model and a stored PNG from the real
encoder, so the libraries (fastapi, pydantic, orjson, pybase64) are the ones the server image ships.

    PYTHONPATH=<patched sglang tree>/python taskset -c <one idle core> python3 scripts/bench_image_response_path.py [--reps 200]
"""
import argparse
import base64
import json
import statistics
import time

import numpy as np
import pybase64
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageResponse,
    ImageResponseData,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import _encode_png_stored
from sglang.srt.utils.json_response import orjson_response


def med(fn, reps):
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - start) * 1000)
    return round(statistics.median(times), 3), round(min(times), 3), out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=200)
    args = parser.parse_args()
    results = {}
    for width, height in ((1024, 1024), (1280, 720), (2048, 2048)):
        frame = np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)
        png = _encode_png_stored(frame)
        old_b64 = base64.b64encode(png).decode("utf-8")
        new_b64 = pybase64.b64encode(png).decode("utf-8")
        assert old_b64 == new_b64

        app = FastAPI()

        @app.post("/old", response_model=ImageResponse)
        async def old():
            return ImageResponse(id="x", data=[ImageResponseData(b64_json=old_b64)])

        @app.post("/new", response_model=ImageResponse)
        async def new():
            return orjson_response(ImageResponse(id="x", data=[ImageResponseData(b64_json=new_b64)]).model_dump(mode="json"))

        with TestClient(app) as client:
            for _ in range(10):
                client.post("/old"), client.post("/new")
            m_old, n_old, r_old = med(lambda: client.post("/old"), args.reps)
            m_new, n_new, r_new = med(lambda: client.post("/new"), args.reps)
            same_fields = json.loads(r_old.content) == {**json.loads(r_new.content), "created": json.loads(r_old.content)["created"]}
        b_old, bn_old, _ = med(lambda: base64.b64encode(png).decode("utf-8"), args.reps)
        b_new, bn_new, _ = med(lambda: pybase64.b64encode(png).decode("utf-8"), args.reps)
        results[f"{width}x{height}"] = {
            "png_mb": round(len(png) / 1e6, 2),
            "route_old_ms": {"median": m_old, "min": n_old},
            "route_new_ms": {"median": m_new, "min": n_new},
            "base64_stdlib_ms": {"median": b_old, "min": bn_old},
            "base64_pybase64_ms": {"median": b_new, "min": bn_new},
            "same_response_fields": same_fields,
            "body_bytes": [len(r_old.content), len(r_new.content)],
        }
    print(json.dumps(results))


if __name__ == "__main__":
    main()
