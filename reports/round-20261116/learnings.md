# Round 2026-11-16: what the other frameworks' changes teach, and where sglang stood

Follow-up to the published round `b300x4-20261116`. Everything here was measured on one RunPod 4xB300 box on 2026-11-18/19
(arms through `scripts/run_plan_ab.sh`, a fresh server per arm, mirrored rounds) unless a line says otherwise. Numbers are
single-request end-to-end medians in seconds. Pins: sglang `25c1c43942` (round pin) and main `9fbab1f6ed`; vLLM-Omni
`24e5472849` on vllm 0.33.0; TRT-LLM 1.4.0rc3; LightX2V `395e2ac49f`. Raw results: `results/r1118-b300/` (untracked).

Status: the FLUX.3 work and the sglang checks are complete. Section 5's first rounds are in; the confirmation rounds
(`plan_b300_lxconfirm_20261119.txt`, `plan_b300_a2a_confirm_20261119.txt`) are still running where a row says "pending".

## 1. One cell where sglang was behind: FLUX.3 Klein Base (2 GPUs)

vLLM-Omni has served FLUX.3 since vllm-omni #9523 (2026-11-10), inside the round's pin. The case still classified it
`unsupported`, so the published round has no vLLM-Omni FLUX.3 row. Measured now (1024x1024, 50 steps, CFG 3.5, CFG parallel 2):

| framework, command | s |
|---|---:|
| sglang, round pin / main | 2.050 / 2.044-2.053 |
| vLLM-Omni `default` (CFG parallel 2, cuDNN attention) | 1.909 |
| vLLM-Omni + `--no-diffusion-compile-dynamic` | 1.901 / 1.903 / 1.905 |
| vLLM-Omni + FlashInfer attention (now its selected B300 profile) | 1.892 / 1.885 / 1.885 |
| vLLM-Omni TP2 / Ulysses 2 (batched CFG) | 2.254 / 4.851 |
| sglang main + the five PRs below | 1.897 / 1.894 (four of them, without the text cache: 1.902-1.906) |

sglang main with synchronized stage timers: text 0.044, denoise 1.916 (38.3 ms/step), decode 0.066, outside the stages 0.030.

What it was not: host launch cost (the breakable CUDA graph cut the host loop to 1.617 s and the request stayed at 2.051 s),
per-step host syncs (none in the trace), the output path (PNG 3.5 ms + file 0.8 ms on the worker), the attention backend
(`fa` 2.010 / 2.012 against 2.009 / 2.004), `--encoder-parallel dp` (2.046), `--async-output-save` (2.049), a channels_last
decoder (1.946 / 1.949 against 1.902 / 1.906: slower).

What it was, and the fixes (each measured alone against main, mirrored):

| fix | sglang PR | alone | image against main |
|---|---|---|---|
| Under CFG parallel every step recomputed the static text stream: the per-step slice dropped `StaticStreams` and the gate excluded CFG parallel | sgl-project/sglang#51380 | 2.046 / 2.048 -> 2.006 / 2.001 (-2.1%) | identical bytes |
| The native SwiGLU called `sgl_kernel.silu_and_mul` (101 us per call at these widths); the shared `SiluAndMul` layer's in-tree kernel takes 38 us | sgl-project/sglang#51387 | 2.050 / 2.049 -> 1.975 / 1.973 (-3.7%) | PSNR 39.4-39.9 dB (differs only for gates below -15) |
| The image VAE sharded its decode across the CFG ranks although `vae_sp` is false; the sharded decode is slower than the serial one from 512x512 to 2560x2560 (1024x1024: 0.069 against 0.050 s) | sgl-project/sglang#51406 | 2.044 / 2.047 -> 2.027 / 2.029 (-0.9%) | identical bytes |
| Image responses through pybase64 and orjson | sgl-project/sglang#51318 | 5.3 ms per 1024x1024 response (host component) | identical bytes |
| The text stage encoded the (constant, empty) negative prompt on every request; it now goes through the conditioning cache like the other text stages | sgl-project/sglang#51411 | 2.041 / 2.048 -> 2.037 / 2.039 (-0.3%); text stage 0.043 -> 0.036 s | identical bytes |
| all five | | 2.047 -> 1.894-1.897 (-7.4%); synchronized: denoise 1.792, decode 0.052, text 0.036, outside 0.028 | |

The SwiGLU fix came from laying the two frameworks' torch-profiler tables side by side: GEMMs and cuDNN SDPA were the same
kernels with the same call counts; vLLM-Omni's compiled SwiGLU ran 33 times for 1.75 ms per step, sglang's 38 times for 3.74 ms.

Where it stands: 0.4% ahead of vLLM-Omni's cuDNN command (1.903 / 1.905) and 0.1-0.6% behind its FlashInfer command
(1.885 / 1.885 / 1.892). Denoise is ahead (1.792 s against about 1.82 s from vLLM-Omni's progress bar). vLLM-Omni runs the
FLUX.3 VAE in bf16 (its log: fp32 weights loaded into bf16 parameters; no switch); the checkpoint and sglang are fp32, so the two
rows are not at the same VAE precision. sglang's `--component-precisions.vae` does not reach the FLUX.3 VAE (its dtype comes from
the package config), so a same-precision sglang arm is not measured. The cell should not be published before the five PRs land and
it is measured again.

## 2. Did sglang regress between rounds? No

Old `ca819c11cd` (2026-11-02), round pin, main, same box, two mirrored rounds, medians:

| cell | old | pin | main |
|---|---:|---:|---:|
| Ideogram 4 (2 GPUs) | 1.918 | 1.910 | 1.908 |
| LTX-2.5 two-stage | 5.158 | 5.172 | 5.178 |
| Cosmos3 T2V 189f | 22.726 | 22.750 | 22.724 |
| FastH3 V2 | 5.081 | 5.045 | 5.040 |
| FLUX.3 Klein Base | 2.044 | 2.051 | 2.052 |
| Z-Image Turbo | 0.347 | 0.318 | 0.319 |
| Qwen-Image-2.1 Edit | 2.569 | 2.553-2.559 | 2.554-2.590 |
| Qwen-Image-2.1 T2I | 2.127 | 2.126 | 2.126 |
| Qwen-Image-2.1 T2I 2 GPUs | 1.357 | 1.311 | 1.311 |
| Cosmos3 T2I | 0.509 / 0.510 | 0.516 / 0.548 | 0.512 / 0.522 |

The "Cosmos3 T2I got 5% slower" reading of the published tables was cross-box noise: five trees on one box agree within 1% by
the fastest sample, and sglang #49162 (the suspect) measures 0.512 before and 0.511 after. The long video cells are within
1% across the two rounds, and the MiniMax-H3 dp2 throughput p50 is 118.290 s (2026-11-02) and 118.278 s (this round): the
narrower throughput lead is all vLLM-Omni's own 2%.

## 3. Why the others got faster

| change | effect | does sglang need it |
|---|---|---|
| vLLM-Omni's stack moved to torch 2.14 / cuDNN 9.24 | cuDNN attention -3% per step in every `CUDNN_ATTN` video cell (cuDNN 9.20 -> 9.24, independent of the torch version) | has had it since 2026-10-04 |
| vllm-omni #4989 (FlashInfer attention as one custom op under `torch.compile`) | Z-Image -81 ms | compile-only |
| vllm-omni #7597 (one-launch QK-norm + RoPE) | Z-Image +18 ms under compile on B300 | sglang upper bound measured with an in-place fused kernel: -0.7%; not pursued |
| vllm-omni #10120 (stored PNG, pybase64, orjson) | outside-engine time 59 -> 16 ms | sgl-project/sglang#51318 (pybase64 + orjson): 5.3 ms per 1024x1024 response on the B300 host, not visible end to end |
| vllm-omni #9523 (FLUX.3) | a new cell | section 1 |
| vllm-omni `4b63581cf..24e547284` (2026-11-11 to 11-16; #9487 bf16 tensors over shared memory as raw bits, #10120, #10171 shared-memory wakeup, #9553 Wan-family VAE fast path) | same box, old and new stacks bucketed: FastH3 8.19-8.25 -> 7.70-7.71 (-6.2%), Qwen-Image-2.1 2 GPUs 1.498 -> 1.425-1.427 (-4.8%), Cosmos3 T2I 0.635 -> 0.586-0.603 (-6.5%); the stack move alone: Qwen -2.6%, Cosmos3 T2I -1.9%, FastH3 none | its own worker-to-API and response overhead; sglang's time outside the stages is 9-30 ms on the image cells and 4-45 ms on H3 (section 4) |
| TRT-LLM rc1 -> rc3 | per-request fixed cost about -1.1 s on H3; per-step unchanged | sglang's non-denoise time is already lower (section 4) |
| LightX2V `19c19614..395e2ac4` | no cell moved | one idea, below |

Checked and closed without a change: the Qwen-Image-2.1 VAE padding fold that B300 rejects costs at most 5 ms (forced on:
2.126 / 2.130 against 2.135 / 2.132); LightX2V's fused anti-aliased SnakeBeta for the H3 vocoder (#1721) targets an audio
decode that takes 0.080-0.085 s of a request in sglang (1.6% of FastH3, 0.25% of H3 on four GPUs); the `SiluAndMul` layer
Z-Image and Cosmos3 use is already on the fast kernel.

## 4. sglang's real stage split (synchronized timers)

| cell | denoise | decode | text / other | outside stages |
|---|---:|---:|---:|---:|
| Wan2.2 T2V (4) | 87.727 | 0.761 | 0.100 | 0.143 |
| MiniMax-H3 T2VA (2) | 60.211 | 1.169 | 0.080 | 0.015 |
| MiniMax-H3 Ref2VA (2) | 92.047 | 1.166 | 0.609 text + 0.482 visual | 0.073 |
| MiniMax-H3 T2VA (4) | 30.707 | 0.696 | 0.084 | 0.004 |
| FastH3 V2 (4) | 4.222 | 0.685 | 0.084 | 0.045 |
| Cosmos3 T2V (4) | 21.444 | 1.272 | 0.002 | 0.017 |
| LTX-2.5 two-stage (2) | 3.310 | 1.766 | 0.107 | 0 |
| Z-Image (2) | 0.253 | 0.022 | 0.032 | 0.012 |
| Cosmos3 T2I (2) | | | | 0.023 |
| Qwen-Image-2.1 T2I (2) | | | | 0.012 |

The default stage timers are asynchronous: a stage's GPU work is charged to whichever later stage first waits on it, so
"decode" and the client-minus-stages tail absorb the denoise tail (Wan's decode read 2.95 s that way; it is 0.761 s). In every
video cell sglang's time outside denoising is below vLLM-Omni's (derived from its logs) and TRT-LLM's (two-point fit).
sglang has no first-step penalty (H3 on four GPUs: 30.707 s against 49 x 0.625 s).

## 5. Commands we were not giving the competitors

| framework, cell | option | result |
|---|---|---|
| TRT-LLM, MiniMax-H3 T2VA 2 / 4 GPUs | `parallel_config.async_ulysses` | -1.72% / -1.65% and -1.59% / -1.50%: promoted (`b300-omp8-async`, `b300-fa4-omp8-async`); the published TRT-LLM H3 rows predate it |
| vLLM-Omni, FLUX.3 | the whole cell | section 1; B300 now selects `b300-cfg2-static-flashinfer` (1.885-1.892) |
| vLLM-Omni | `OMP_NUM_THREADS=8`, FA4, FastH3 static compile | not gaps (2026-11-18 first box) |
| vLLM-Omni | `--ulysses-a2a-permute` (copy-engine Ulysses exchange, #9922) | round 1: Qwen-Image-2.1 2 GPUs 1.435 -> 1.385 (-3.5%), FastH3 7.785 -> 7.338 (-5.7%), MiniMax-H3 4 GPUs 36.613 -> 34.779 (-5.0%), Wan2.2 97.710 -> 94.642 (-3.1%); sglang stays ahead on all four (1.311 / 5.040 / 31.530 / 88.967); second round and the other Ulysses cells pending |
| vLLM-Omni | denoise-step CUDA graph | no gain (Z-Image 0.449 -> 0.447, Wan2.2 97.710 -> 97.783) |
| vLLM-Omni | text-encoder TP (Qwen 2 GPUs), Wan `preencode_mp4`, Wan Ulysses 4 + batched CFG | 1.424 -> 1.434 (none); 97.517 -> 97.141 (-0.39%, one round); 98.464 (slower) |
| TRT-LLM, MiniMax-H3 Ref2VA | `async_ulysses` | -0.44% on the first box, +0.36% on the second (104.074 against 103.704): not promoted |
| TRT-LLM | CUDA graphs | Wan2.2 93.673 -> 93.655 (none); on Cosmos3 the server exits before the health check |
| TRT-LLM | token-sharded TP (Wan2.2), TP2 x Ulysses 2 (MiniMax-H3 4 GPUs), text-encoder TP (MiniMax-H3) | 99.685 against 93.673 (slower); 38.330 against 35.749 (slower); server exits |
| LightX2V | `PROFILING_DEBUG_LEVEL=0` on the non-H3 cells (it defaults to 2: a device synchronize and a log line around every step; only our H3 entries set 0) | Qwen-Image-2.1 2.483 / 2.480 -> 2.449 / 2.446 (-1.4%): promoted (`b300-sdpa-noprof`); Z-Image, Cosmos3 T2V, Wan2.2: no difference |
| LightX2V, MiniMax-H3 | VAE compile, packed Ulysses exchange (`seq_p_tensor_fusion`, Triton pre/post) | T2VA 2 GPUs, two rounds: 71.286 / 71.338 -> VAE compile 70.281 / 70.211, packed exchange 70.788 / 70.627, both 69.644 / 69.632 (-2.4%); 4 GPUs, one round: 36.799 -> 36.262 (-1.5%); Ref2VA, one round: 112.33 -> 110.13 (-2.0%): promoted (`b300-sp2-sdpa-fusion-vaec`, `b300-sp4-sdpa-fusion-vaec`); sglang on the same box 61.487 / 31.530 / 94.778 |
| LightX2V, MiniMax-H3 T2VA | head-parallel Ulysses, fused QKV, traceable RMSNorm | 84.761, 73.879, 72.244: slower |
| LightX2V, Wan2.2 | VAE compile (#1646) | two rounds: 99.904 / 99.274 -> 99.529 / 99.004 (-0.38% / -0.27%): promoted (`b300-sdpa-vae-compile`); sglang 88.967 |

Fairness note for the FLUX.3 cell: vLLM-Omni decodes with a bf16 VAE where the checkpoint is fp32.

## 6. Method notes worth keeping

- Short cells (0.3-0.5 s) differ by 2-5% between boxes and between arms on a shared host; judge a regression only by old and
  new trees interleaved on one box, by the fastest sample.
- A torch profiler that makes the host the bottleneck drops GPU load, and B300 kernels run faster at low load: the same GEMM
  read 0.167 ms in sglang's trace (70% busy) and 0.192 ms in vLLM-Omni's (96% busy). Compare shares and call counts.
- "They compile, we do not" was not the cause of the FLUX.3 gap; one slow kernel call was. Lay the kernel tables side by side
  before concluding.
