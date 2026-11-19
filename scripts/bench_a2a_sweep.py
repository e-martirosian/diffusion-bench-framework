"""Plain Ulysses all-to-all by payload size: NCCL against the N-rank copy-engine transport, same process group.
Run under torch.distributed.run with --nproc-per-node=WORLD. Each rank sends MB megabytes in total, split evenly over the ranks.
Rank 0 prints its own mean time per exchange (ms) after a barrier, device-synchronized, 3 repeats, the minimum."""
import os
import sys
import time

import torch
import torch.distributed as dist

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed.parallel_state import maybe_init_distributed_environment_and_model_parallel

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=world, ulysses_degree=world)

from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a_multi import IPC_A2A_MULTI
from sglang.multimodal_gen.runtime.layers.usp import _usp_all_to_all_single

sizes = [float(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "0.25,1,4,8,16,32,64,128,256,512,1024").split(",")]


def timed(x, calls):
    best = float("inf")
    for _ in range(3):
        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(calls):
            y = _usp_all_to_all_single(x, role="usp_input")
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - start) / calls * 1e3)
    return best, y


for mb in sizes:
    numel = int(mb * 2**20 / 2) // world * world
    torch.manual_seed(rank)
    x = torch.randn(numel, dtype=torch.bfloat16, device="cuda")
    calls = 200 if mb <= 16 else (60 if mb <= 128 else 20)
    row = {}
    outs = {}
    for name, flag in (("nccl", False), ("ce", True), ("nccl2", False), ("ce2", True)):
        envs.SGLANG_DIFFUSION_IPC_A2A_MULTI = flag
        before = IPC_A2A_MULTI.calls
        for _ in range(5):
            _usp_all_to_all_single(x, role="usp_input")
        row[name], out = timed(x, calls)
        outs[name] = out.clone()
        if flag and IPC_A2A_MULTI.calls == before:
            row[name] = float("nan")
    same = torch.equal(outs["nccl"], outs["ce"])
    if rank == 0:
        nccl, ce = min(row["nccl"], row["nccl2"]), min(row["ce"], row["ce2"])
        print(f"A2A world={world} {mb:8.2f} MB/rank: nccl {nccl:8.3f} ms  ce {ce:8.3f} ms  ce/nccl {ce / nccl:5.2f}  same={same}", flush=True)
    del x, outs
envs.SGLANG_DIFFUSION_IPC_A2A_MULTI = False
dist.barrier()
dist.destroy_process_group()
