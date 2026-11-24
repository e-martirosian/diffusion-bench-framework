#!/usr/bin/env python3
"""Group consecutive benchmark cases that share one model download.

Reads case ids in run order from argv, prints one group per line
(space-separated case ids that share a model). run_ascend.sh feeds each line
to ONE harness process (DBF_ONE_PROCESS=1) so a shared model downloads once
and is removed after the group's last case.

File-based on purpose: `python3 -` with a heredoc proved fragile on some
boxes (exit 1, no stderr), while `python3 script.py` is what the harness
itself already uses.

Non-adjacent repeats of a model do NOT share: the group is the unit of
sharing, and one harness process per group keeps the download lifecycle
in-memory only.
"""
import json
import sys
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "comparison_configs.json"


def main() -> None:
    case_ids = sys.argv[1:]
    if not case_ids:
        raise SystemExit("usage: group_cases.py CASE_ID...")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    model_of = {c["id"]: c.get("model") for c in cfg["cases"]}
    groups: list[list[str]] = []
    current: list[str] = []
    current_model: object = object()
    for cid in case_ids:
        if cid not in model_of:
            raise SystemExit(f"'{cid}' is not a case id in {CONFIG.name}")
        model = model_of[cid]
        if current and model != current_model:
            groups.append(current)
            current = []
        current.append(cid)
        current_model = model
    if current:
        groups.append(current)
    print("\n".join(" ".join(g) for g in groups))


if __name__ == "__main__":
    main()
