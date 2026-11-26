#!/usr/bin/env python3
"""Diffusion Cross-Platform Performance Dashboard.

Our dashboard view as a standalone artifact, leaving generate_dashboard.py's
own dashboard untouched. Differences from that dashboard:

  - one row per case, with consecutive rows sharing a model merged under one
    Model cell (markdown has no rowspan, so repeats leave the cell empty);
  - Task column with short labels (T2I / T2V / I2V / IE / ...);
  - per-case Supported column: did sglang produce a measurement in this run;
  - a coverage section listing the catalog's sglang-supported models this run
    never attempted at all ("supported for other backends -- not checked on
    NPU" when the artifact is an NPU run);
  - no throughput section (the single-request comparison is the point here).

Reads the merged.json that build_report_artifacts already writes plus the
benchmark catalog for the coverage section:

  python scripts/diffusion_cross_platform_performance_dashboard.py \
      --merged tmp/report/merged.json \
      --config configs/comparison_configs.json \
      --output-md tmp/report/diffusion_cross_platform_performance_dashboard.md
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusion_bench.generate_dashboard import (  # noqa: E402
    _assess_risk,
    _dispersed_cases,
    _extract_case_results,
    _fmt_latency,
    _fmt_speedup,
)

TASK_ABBREV = {
    "text-to-image": "T2I",
    "image-edit": "IE",
    "image-to-image": "I2I",
    "text-to-video": "T2V",
    "image-to-video": "I2V",
    "text-image-to-video": "TI2V",
    "video-super-resolution": "VSR",
    "text-to-3d": "T2-3D",
}


def _task_abbrev(task) -> str:
    """Short task label (T2I / T2V / I2V / ...); unknown tasks fall through."""
    if not task:
        return "-"
    return TASK_ABBREV.get(task, task)


def _md_cell(value) -> str:
    if value is None:
        return "-"
    text = str(value).replace("\n", " ").replace("|", "\\|")
    return text if text else "-"


def _case_catalog(config: dict) -> list[dict]:
    """Every catalog case's model with its sglang classification.

    The built config publishes command bodies only for supported frameworks,
    so an unsupported status lives in report_framework_statuses with its reason.
    """
    catalog = []
    for case in config.get("cases", []):
        catalog.append(
            {
                "id": case.get("id"),
                "model": case.get("model"),
                "task": case.get("task"),
                "sglang_status": (
                    "supported"
                    if "sglang" in (case.get("frameworks") or {})
                    else str(
                        (case.get("report_framework_statuses") or {}).get("sglang")
                        or "not_run"
                    )
                ),
                "sglang_note": (case.get("report_framework_reasons") or {}).get("sglang"),
            }
        )
    return catalog


def _is_npu(merged: dict) -> bool:
    devices = set(merged.get("devices") or [])
    devices.add(str((merged.get("hardware") or {}).get("device") or ""))
    return "npu" in devices


def _hw_label(merged: dict) -> str:
    """Platform + hardware for the sglang column header, e.g. "Ascend A3"."""
    hardware = merged.get("hardware") or {}
    device = str(hardware.get("device") or "gpu")
    profile = str(hardware.get("hardware_profile_override") or "").strip()
    platform = "Ascend" if device == "npu" else "GPU"
    return f"{platform} {profile.upper()}" if profile else platform


def build_dashboard(merged: dict, catalog: list[dict], history: list[dict]) -> str:
    lines = ["# Diffusion Cross-Platform Performance Dashboard", ""]
    ts = str(merged.get("timestamp", ""))[:10]
    sha = merged.get("commit_sha", "unknown")
    sha_s = sha[:7] if sha and sha != "unknown" else "?"
    lines.append(f"*Generated: {ts} | Commit: `{sha_s}`*\n")

    current_cases = _extract_case_results(merged)
    dispersed = _dispersed_cases(merged)
    dispersed_notes: list[str] = []

    # Frameworks present in this run, sglang first.
    all_frameworks: list[str] = []
    seen_fw: set[str] = set()
    for r in merged.get("results", []) + merged.get("throughput_results", []):
        fw = r.get("framework")
        if fw and fw not in seen_fw:
            all_frameworks.append(fw)
            seen_fw.add(fw)
    if "sglang" in all_frameworks:
        all_frameworks.remove("sglang")
        all_frameworks.insert(0, "sglang")
    other_frameworks = [fw for fw in all_frameworks if fw != "sglang"]

    # ---- Cross-platform comparison ----
    lines.append("## Cross-Platform Performance Comparison\n")

    risk_map: dict[str, tuple[str, str]] = {}
    for r in merged.get("results", []):
        cid = r["case_id"]
        if cid not in risk_map:
            risk_map[cid] = _assess_risk(cid, current_cases, history, other_frameworks)

    hw_label = _hw_label(merged)
    header = "| Model | Task |"
    sep = "|-------|------|"
    for fw in all_frameworks:
        if fw == "sglang":
            header += f" sglang {hw_label} (s) |"
        else:
            header += f" {fw} (s) |"
        sep += "---------|"
    for ofw in other_frameworks:
        header += f" vs {ofw} |"
        sep += "---------|"
    header += " Supported |"
    sep += "-----------|"
    lines.append(header)
    lines.append(sep)

    # One row per case (deduplicated by case_id). Consecutive rows sharing a
    # model show the name once: markdown has no rowspan, so the repeats leave
    # the Model cell empty and read as one merged block.
    seen_cases: set[str] = set()
    prev_model = None
    for r in merged.get("results", []):
        cid = r["case_id"]
        if cid in seen_cases:
            continue
        seen_cases.add(cid)

        case_fws = current_cases.get(cid, {})
        sg_lat = case_fws.get("sglang")

        model_name = r["model"].split("/")[-1]
        model_cell = "" if model_name == prev_model else model_name
        prev_model = model_name
        supported = "✅" if sg_lat is not None else "❌"
        row = f"| {model_cell} | {_task_abbrev(r.get('task'))} |"
        lats = {fw: case_fws.get(fw) for fw in all_frameworks}
        valid_lats = [v for v in lats.values() if v is not None]
        min_lat = min(valid_lats) if valid_lats else None
        for fw in all_frameworks:
            lat = lats[fw]
            mark = " ~" if (cid, fw) in dispersed else ""
            if lat is not None and min_lat is not None and lat == min_lat:
                row += f" **{_fmt_latency(lat)}**{mark} |"
            else:
                row += f" {_fmt_latency(lat)}{mark} |"
        for ofw in other_frameworks:
            row += f" {_fmt_speedup(sg_lat, case_fws.get(ofw))} |"
        row += f" {supported} |"
        lines.append(row)
        for fw in all_frameworks:
            samples = dispersed.get((cid, fw))
            if samples:
                dispersed_notes.append(
                    f"- `{cid}` / **{fw}**: repeats spread "
                    f"{(max(samples) - min(samples)) / min(samples) * 100:.0f}% "
                    "(" + ", ".join(f"{s:g}s" for s in samples) + ") — "
                    "median published; the distribution is wider than it"
                )

    if dispersed_notes:
        lines.append("\n`~` = the repeats behind this median spread 25% or more:\n")
        lines.extend(dispersed_notes)
        lines.append("")

    # ---- SGLang-supported models this run never tested ----
    # Every model the benchmark catalog classifies for sglang, minus the models
    # this artifact actually attempted (an errored attempt counts as tested --
    # it shows as ❌ above). These never ran at all: pay attention.
    untested_notes: list[str] = []
    if catalog:
        tested_models = {
            r.get("model") for r in merged.get("results", []) if r.get("model")
        }
        untested: list[dict] = []
        seen_untested: set[str] = set()
        for c in catalog:
            model = c.get("model")
            if not model or model in tested_models or model in seen_untested:
                continue
            seen_untested.add(model)
            untested.append(c)
        if untested:
            npu = _is_npu(merged)
            lines.append("\n## SGLang-Supported Models Not Tested In This Run\n")
            lines.append(
                "> [!IMPORTANT]\n> These sglang-supported models have no case in "
                "this run's results at all — not tested, not errored.\n"
            )
            lines.append("")
            lines.append("| Model | Task | sglang |")
            lines.append("|-------|------|--------|")
            for c in untested:
                status = str(c.get("sglang_status") or "not_run")
                note = c.get("sglang_note")
                if status == "supported":
                    # "Supported" is the catalog's sglang classification, which
                    # reflects other backends; say so where this run is NPU.
                    cell = (
                        "supported for other backends — not checked on NPU"
                        if npu
                        else "supported — not run"
                    )
                else:
                    cell = _md_cell(f"{status}: {note}" if note else status)
                lines.append(
                    f"| {str(c.get('model')).split('/')[-1]} "
                    f"| {_task_abbrev(c.get('task'))} | {cell} |"
                )
                untested_notes.append(
                    f"{c.get('model')}: sglang {status}, not tested in this run"
                )
            lines.append("")

    # ---- Risk Notification ----
    alert_cases = [
        (cid, emoji, reason)
        for cid, (emoji, reason) in risk_map.items()
        if emoji in ("⚠️", "🔴", "❌")
    ]
    if alert_cases:
        lines.append("\n> [!CAUTION]")
        lines.append("> **Action Required — Performance Alert**")
        lines.append(">")
        lines.append("> The following cases need attention:")
        for _cid, _emoji, reason in alert_cases:
            lines.append(f"> - {reason}")
        lines.append("")

    lines.append("\n---")
    lines.append(
        "*Generated by `scripts/diffusion_cross_platform_performance_dashboard.py`.*"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diffusion cross-platform performance dashboard: merged "
        "same-model rows, abbreviated tasks, the sglang column labeled with "
        "the run's platform/hardware, per-case Supported, and the "
        "sglang-supported models this run never tested."
    )
    parser.add_argument("--merged", required=True, type=Path, help="merged.json from build_report_artifacts")
    parser.add_argument("--config", default=Path("configs/comparison_configs.json"), type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    merged = json.loads(args.merged.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    text = build_dashboard(merged, _case_catalog(config), history=[])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(text, encoding="utf-8")
    print(args.output_md)


if __name__ == "__main__":
    main()
