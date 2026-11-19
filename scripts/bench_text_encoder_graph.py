"""A diffusers-layout text encoder's forward on one GPU: eager against a CUDA graph replay at a fixed token count.
argv: repo, tokens[, subfolder]. Mean ms per forward (30 calls after 6 warm calls) and whether the replay's states match."""
import sys
import time

import torch
from transformers import AutoModel

repo, tokens = sys.argv[1], int(sys.argv[2])
subfolder = sys.argv[3] if len(sys.argv) > 3 else "text_encoder"
model = AutoModel.from_pretrained(repo, subfolder=subfolder, torch_dtype=torch.bfloat16).cuda().eval()
layers = sum(1 for name, _ in model.named_modules() if name.endswith(".mlp") or name.endswith(".ffn"))
torch.manual_seed(0)
vocab = model.get_input_embeddings().num_embeddings
ids = torch.randint(0, min(vocab, 30000), (1, tokens), device="cuda")
mask = torch.ones(1, tokens, dtype=torch.long, device="cuda")
mask[:, tokens // 4:] = 0  # a short prompt padded to the fixed length


def forward(i, m):
    return model(input_ids=i, attention_mask=m, output_hidden_states=True).hidden_states[-2]


def timed(fn, calls=30, warm=6):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(calls):
        out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / calls * 1e3, out


with torch.no_grad():
    eager_ms, eager = timed(lambda: forward(ids, mask))
    static_ids, static_mask = ids.clone(), mask.clone()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            static_out = forward(static_ids, static_mask)
    except RuntimeError as error:
        print(f"TEXT {repo} {tokens} tokens: eager {eager_ms:.2f} ms; capture failed: {str(error).splitlines()[0][:120]}")
        raise SystemExit(0)

    def replay():
        static_ids.copy_(ids)
        static_mask.copy_(mask)
        graph.replay()
        return static_out.clone()

    torch.cuda.synchronize()
    held = (torch.cuda.memory_allocated() - before[0]) / 2**20, (torch.cuda.memory_reserved() - before[1]) / 2**20

    def forward_all(i, m):
        return model(input_ids=i, attention_mask=m, output_hidden_states=True).hidden_states

    graph_ms, replayed = timed(replay)
    ids2 = torch.randint(0, min(vocab, 30000), (1, tokens), device="cuda")
    static_ids.copy_(ids2)
    graph.replay()
    same_new = torch.equal(static_out, forward(ids2, mask))
print(f"TEXT {repo} {tokens} tokens ({layers} blocks): eager {eager_ms:.2f} ms, graph replay {graph_ms:.2f} ms, "
      f"same states {torch.equal(eager, replayed)}, same for new ids {same_new}, graph holds {held[0]:.0f} MiB allocated / {held[1]:.0f} MiB reserved", flush=True)
