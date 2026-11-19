"""Per-step GPU kernel time by kernel name from torch-profiler chrome traces (.json.gz, one file per rank).

    python3 scripts/trace_kernel_summary.py <steps> <trace.json.gz> [...]

Works on sglang's traces (request field `profile: true` with `profile_all_stages: true`; SGLANG_DIFFUSION_TORCH_PROFILER_DIR) and on
vLLM-Omni's (`--profiler-config '{"profiler":"torch","torch_profiler_dir":...}'` then POST /start_profile, one request, POST
/stop_profile), so the two frameworks' kernel tables can be laid side by side. Compare shares and call counts, not sums: a profiler
that makes the host the bottleneck lowers GPU load, and kernels run faster at low load.
"""
import collections
import gzip
import json
import sys

steps = int(sys.argv[1])
for path in sys.argv[2:]:
    with gzip.open(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    kernels = collections.defaultdict(lambda: [0.0, 0])
    memops = collections.defaultdict(lambda: [0.0, 0])
    cpu_ops = collections.defaultdict(lambda: [0.0, 0])
    span = [None, None]
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat, dur = ev.get("cat"), ev.get("dur", 0)
        if cat == "kernel":
            kernels[ev["name"]][0] += dur
            kernels[ev["name"]][1] += 1
            start = ev["ts"]
            span[0] = start if span[0] is None else min(span[0], start)
            span[1] = start + dur if span[1] is None else max(span[1], start + dur)
        elif cat in ("gpu_memcpy", "gpu_memset"):
            memops[ev["name"]][0] += dur
            memops[ev["name"]][1] += 1
        elif cat == "cpu_op":
            cpu_ops[ev["name"]][0] += dur
            cpu_ops[ev["name"]][1] += 1
    total = sum(v[0] for v in kernels.values())
    count = sum(v[1] for v in kernels.values())
    print(f"== {path.rsplit('/', 1)[-1]}")
    if span[0] is None:
        print("   no kernel events")
        continue
    wall = span[1] - span[0]
    print(f"   kernel time {total / 1e3 / steps:.2f} ms/step in {count / steps:.0f} launches/step; first-to-last kernel wall {wall / 1e3 / steps:.2f} ms/step; "
          f"busy {100 * total / wall:.1f}%")
    for name, (dur, n) in sorted(kernels.items(), key=lambda kv: -kv[1][0])[:34]:
        print(f"   {dur / 1e3 / steps:8.3f} ms/step {n / steps:7.1f}x  {100 * dur / total:5.1f}%  {name[:110]}")
    for name, (dur, n) in sorted(memops.items(), key=lambda kv: -kv[1][0])[:4]:
        print(f"   memop {dur / 1e3 / steps:8.3f} ms/step {n / steps:7.1f}x  {name[:80]}")
    print("   -- top CPU ops by self-inclusive duration")
    for name, (dur, n) in sorted(cpu_ops.items(), key=lambda kv: -kv[1][0])[:14]:
        print(f"   {dur / 1e3 / steps:8.3f} ms/step {n / steps:7.1f}x  {name[:100]}")
