#!/usr/bin/env python3
"""Does the installed LightX2V accept what the harness sends it?

LightX2V builds its request model per deployed model (`lightx2v.server.schema.build_task_request`,
since the 2026-11-16 refactor #1673), and the model forbids unknown fields, so a renamed or
removed field fails the whole request with a bare 422 -- after a model load. This checks every
LightX2V cell the matrix would run for a hardware class in seconds: the harness's own startup
config and request payload against the request model that startup config deploys.

Two steps, because the harness and LightX2V live in different environments:

  # 1. harness environment (needs requests; downloads nothing)
  PYTHONPATH=src python3 scripts/validate_lightx2v_payloads.py --dump /tmp/lx2v_cells.json --hardware b300
  # 2. the LightX2V venv (needs lightx2v)
  <venv>/bin/python scripts/validate_lightx2v_payloads.py --check /tmp/lx2v_cells.json

Step 2 exits 1 if any cell is rejected. It validates the request, not the weights: whether the
startup loader accepts the config against a real checkpoint is what the first launch tells you.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def dump(args) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from diffusion_bench import run_comparison as rc
    from diffusion_bench.config_guard import apply_hardware_statuses

    config = json.loads(Path(args.config).read_text())
    hardware_metadata = {"hardware_profile_override": args.hardware}
    config, _ = apply_hardware_statuses(config, rc._hardware_profile_candidates(hardware_metadata))
    # The image bytes never reach a schema check; reading them would only need the asset.
    rc._get_ref_image_b64 = lambda _config, _case: "AAAA"
    cells = []
    for case in config["cases"]:
        if args.cases and case["id"] not in args.cases:
            continue
        fw_cfg = (case.get("frameworks") or {}).get("lightx2v")
        if not fw_cfg:
            continue
        resolved = rc._resolve_framework_config("lightx2v", fw_cfg, None, hardware_metadata)
        # A model path that is not a directory keeps the checkpoint resolver from downloading.
        startup = json.loads(
            Path(
                rc._write_lightx2v_config(case, resolved, "/nonexistent", rc._server_model_path(case, resolved))
            ).read_text()
        )
        serve = (resolved.get("serve_args") or "").split()
        is_image = case["task"] in rc.LIGHTX2V_IMAGE_TASKS
        payload = rc._lightx2v_payload(case, config, is_image)
        if not is_image:
            payload["save_result_path"] = "/tmp/lightx2v_probe.mp4"  # send_request_lightx2v adds it
        cells.append(
            {
                "case": case["id"],
                "profile": resolved["_benchmark_metadata"]["profile"],
                "model_cls": resolved["model_cls"],
                "task": resolved.get("lightx2v_task"),
                "model_variant": serve[serve.index("--model-variant") + 1] if "--model-variant" in serve else None,
                "endpoint": "/v1/tasks/image/sync" if is_image else "/v1/tasks/video/",
                "startup": startup,
                "payload": payload,
            }
        )
    Path(args.dump).write_text(json.dumps(cells, indent=1))
    print(f"wrote {len(cells)} LightX2V cell(s) for {args.hardware} to {args.dump}")
    return 0


def check(args) -> int:
    from pydantic import TypeAdapter, ValidationError

    from lightx2v.server.schema import build_task_request

    failed = 0
    for cell in json.loads(Path(args.check).read_text()):
        # What build_startup_config assembles, minus the checkpoint-dependent loader.
        config = dict(cell["startup"], model_cls=cell["model_cls"], task=cell["task"])
        if cell["model_variant"]:
            config["model_variant"] = cell["model_variant"]
        label = f"{cell['case']} [{cell['profile']}] {cell['model_cls']} task={cell['task']} {cell['endpoint']}"
        try:
            TypeAdapter(build_task_request(config)).validate_python(cell["payload"])
        except ValidationError as exc:
            failed += 1
            print(f"  FAIL {label}")
            for err in exc.errors()[:6]:
                print(f"       {'.'.join(str(p) for p in err['loc'])}: {err['msg']}")
        except Exception as exc:  # noqa: BLE001 - the point is to name what broke
            failed += 1
            print(f"  FAIL {label}\n       {type(exc).__name__}: {str(exc)[:300]}")
        else:
            print(f"  ok   {label}  ({len(cell['payload'])} field(s))")
    print(f"{failed} cell(s) rejected")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", help="step 1: write the cells to this JSON file")
    ap.add_argument("--check", help="step 2: validate the cells in this JSON file")
    ap.add_argument("--hardware", default="b300", help="hardware class to resolve profiles for (dump)")
    ap.add_argument("--config", default=str(ROOT / "configs" / "comparison_configs.json"))
    ap.add_argument("--cases", nargs="*", help="only these case ids (dump)")
    args = ap.parse_args()
    if bool(args.dump) == bool(args.check):
        ap.error("pass exactly one of --dump / --check")
    return dump(args) if args.dump else check(args)


if __name__ == "__main__":
    sys.exit(main())
