## Diffusion Benchmark Data - 2026-11-17T17:56:06.766093+00:00

| item | value |
| --- | --- |
| run_id | b300x4-20261116 |
| data | single_e2e, throughput |
| bench_commit | 083322e273273764174a4409039df59a1ca24a08 |
| gpu | 4 x NVIDIA B300 SXM6 AC, 275040 MiB, 595.91.07 |
| reproduce | scripts/generate_h200_report_artifacts.sh; inputs: run_b300img1116_cosmos3_nano_t2i_720p_cosmos3_nano_t2i_720p.json, run_b300img1116_flux3_klein_base_t2i_1024_flux3_klein_base_t2i_1024.json, run_b300img1116_ideogram4_t2i_1024_2gpu_tp_ideogram4_t2i_1024_2gpu_tp.json, run_b300img1116_ltx2.5_twostage_t2v_2gpus_ltx2.5_twostage_t2v_2gpus.json, run_b300img1116_qwen_image_21_edit_1024_qwen_image_21_edit_1024.json, run_b300img1116_qwen_image_21_t2i_1024_2gpu_qwen_image_21_t2i_1024_2gpu.json, run_b300img1116_qwen_image_21_t2i_1024_qwen_image_21_t2i_1024.json, run_b300img1116_zimage_turbo_t2i_1024_zimage_turbo_t2i_1024.json, run_b300vid1116_cosmos3_nano_i2v_720p_189f_cosmos3_nano_i2v_720p_189f.json, run_b300vid1116_cosmos3_nano_t2v_720p_189f_cosmos3_nano_t2v_720p_189f.json, run_b300vid1116_minimax_h3_fasth3_v2_t2va_5s_minimax_h3_fasth3_v2_t2va_5s.json, run_b300vid1116_minimax_h3_ref2va_5s_minimax_h3_ref2va_5s.json, run_b300vid1116_minimax_h3_t2va_5s_4gpu_minimax_h3_t2va_5s_4gpu.json, run_b300vid1116_minimax_h3_t2va_5s_minimax_h3_t2va_5s.json, run_b300vid1116_wan22_t2v_a14b_720p_wan22_t2v_a14b_720p.json, run_h3t1116_h-d-r1_minimax_h3_t2va_5s.json, run_h3t1116_h-v-r3_minimax_h3_t2va_5s.json, run_b300vid1116_wan22_t2v_a14b_720p_wan22_t2v_a14b_720p.json |

| framework | version/ref |
| --- | --- |
| SGLang-Diffusion | 0.0.0.dev1+g0b098e127 (25c1c43) |
| vLLM-Omni | vllm-omni 0.33.0rc2.dev319+g24e547284; vllm 0.33.0 |
| LightX2V | 0.5.0 (395e2ac) |
| trtllm-visual | 1.4.0rc3 |
| ComfyUI | master @ a308f3f |
| FastVideo | main @ c938a8a |

Ratio columns are framework value divided by SGLang-Diffusion value for the same case.
Statuses: `not_run` means configured but absent from this artifact; `unsupported` means unsupported by the tracked framework/version; `no_profile` means no validated aligned serving profile is tracked.

### Thinnest lead - single_e2e

| case | hardware | sglang_gpus | sglang_single_e2e_s | fastest_competitor | competitor_gpus | competitor_single_e2e_s | competitor/SGLang-Diffusion | diagnostic_client_s | stage_sum_s | unattributed_tail_s |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wan22_t2v_a14b_720p | 4 x NVIDIA B300 SXM6 AC, 275040 MiB, 595.91.07 | 4 | 89.552 | trtllm-visual | 4 | 94.086 | 1.051x | 89.608 | 89.584 | 0.024 |
| minimax_h3_ref2va_5s | 4 x NVIDIA B300 SXM6 AC, 275040 MiB, 595.91.07 | 2 | 95.844 | vLLM-Omni | 2 | 103.240 | 1.077x | 95.875 | 94.032 | 1.842 |
| minimax_h3_t2va_5s | 4 x NVIDIA B300 SXM6 AC, 275040 MiB, 595.91.07 | 2 | 62.342 | vLLM-Omni | 2 | 67.404 | 1.081x | 62.372 | 61.281 | 1.091 |

### Thinnest lead - throughput

| case | hardware | sglang_gpus | sglang_qps | fastest_competitor | competitor_gpus | competitor_qps | competitor/SGLang-Diffusion |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| minimax_h3_t2va_5s | 4 x NVIDIA B300 SXM6 AC, 275040 MiB, 595.91.07 | 2 | 0.0168 | vLLM-Omni | 2 | 0.0164 | 0.976x |
| qwen_image_21_t2i_1024 | 4 x NVIDIA B300 SXM6 AC, 275040 MiB, 595.91.07 | 1 | 0.4746 | vLLM-Omni | 1 | 0.4001 | 0.843x |

### flux3_klein_base_t2i_1024

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| black-forest-labs/flux-3-image-klein-base | text-to-image | 1024x1024 | 50 | gs=3.5 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 2 | 2.086 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | No FLUX.3 pipeline on vLLM-Omni main (2026-10-16). It is in the open, approved PR vllm-project/vllm-omni#9007; reclassify once it merges. Measured pre-merge on 2xH200 2026-10-16 at head 389a02a149ce with vllm==0.31.0 (the head's docker/Dockerfile.ci): CFG parallel 2 4.534 s, one GPU 9.101 s (regional compile, FLASH_ATTN). Its FLUX.3 loader reads only a local package directory (model_path = the snapshot dir), and requests need --served-model-name set to the case model. |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | LightX2V main has no FLUX.3 serving path (2026-10-15): no flux3 / flux-3-image code or PR in ModelTC/LightX2V; its FLUX coverage ends at FLUX.2. |
| trtllm-visual | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | No FLUX.3 pipeline on TensorRT-LLM main (2026-10-16); NVIDIA/TensorRT-LLM#20430 adds it and is still a draft. |
| ComfyUI | default | 2 | 3.255 | 1.560x | ok | - | - | - | - | - | - | - | - | not_run | - |
| FastVideo | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | FastVideo main has no FLUX.3 pipeline (2026-10-15): no flux3 code in hao-ai-lab/FastVideo. |

### qwen_image_21_t2i_1024

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Qwen/Qwen-Image-2.1 | text-to-image | 1024x1024 | 40 | gs=1.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | b300-1gpu-resident-fa-async-save | 1 | 2.130 | 1.000x | ok | 4/4 | 2 | 4.189 | 1.000x | 4.230 | 4.235 | 0.4746 | 1.000x | ok | - |
| vLLM-Omni | blackwell-cudnn-static | 1 | 2.440 | 1.146x | ok | 4/4 | 2 | 4.991 | 1.191x | 5.148 | 5.149 | 0.4001 | 0.843x | ok | - |
| LightX2V | blackwell-sdpa | 1 | 2.478 | 1.163x | ok | 4/4 | 2 | 5.013 | 1.197x | 5.017 | 5.017 | 0.3988 | 0.840x | ok | - |
| trtllm-visual | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | TensorRT-LLM VisualGen carries a Qwen-Image pipeline (tensorrt_llm/_torch/visual_gen/models/qwen_image/) but nothing for the 2.1 architecture: a code search for QwenImage21 / Qwen-Image-2.1 over NVIDIA/TensorRT-LLM returns no hits, and the model is absent from pipeline_registry.py. Qwen-Image-2.1 is a distinct architecture (QwenImage21Pipeline / QwenImage21Transformer2DModel / AutoencoderKLQwenImage21 / Qwen3-VL encoder), so the existing Qwen-Image pipeline does not cover it. |
| ComfyUI | blackwell-gpuonly-compile | 1 | 3.045 | 1.430x | ok | 4/4 | 2 | 6.106 | 1.458x | 6.396 | 6.435 | 0.3190 | 0.672x | ok | - |
| FastVideo | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | No Qwen-Image pipeline in FastVideo main (2b164405c3d1, 2026-10-07). |

### qwen_image_21_t2i_1024_2gpu

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Qwen/Qwen-Image-2.1 | text-to-image | 1024x1024 | 40 | gs=1.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | gb300-2gpu-tp2 | 2 | 1.319 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-cudnn-static | 2 | 1.453 | 1.102x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | No validated two-GPU LightX2V command for Qwen-Image-2.1 yet (its upstream configs for this model are single-GPU) (2026-10-10). |
| trtllm-visual | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | TensorRT-LLM VisualGen carries a Qwen-Image pipeline (tensorrt_llm/_torch/visual_gen/models/qwen_image/) but nothing for the 2.1 architecture: a code search for QwenImage21 / Qwen-Image-2.1 over NVIDIA/TensorRT-LLM returns no hits, and the model is absent from pipeline_registry.py. Qwen-Image-2.1 is a distinct architecture (QwenImage21Pipeline / QwenImage21Transformer2DModel / AutoencoderKLQwenImage21 / Qwen3-VL encoder), so the existing Qwen-Image pipeline does not cover it. |
| ComfyUI | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | ComfyUI's only multi-GPU path splits CFG branches, and this case runs without CFG, so a two-GPU cell would be its one-GPU cell. |
| FastVideo | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | No Qwen-Image pipeline in FastVideo main (2b164405c3d1, 2026-10-07). |

### qwen_image_21_edit_1024

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Qwen/Qwen-Image-2.1 | image-edit | 1024x1024 | 40 | gs=1.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | b200-1gpu-resident-fa | 1 | 2.562 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-cudnn-static | 1 | 2.951 | 1.152x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | LightX2V's Qwen-Image-2.1 integration is validated in this harness for t2i only (lightx2v_task t2i); its image-conditioned path has no validated profile yet (2026-10-10). |
| trtllm-visual | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | TensorRT-LLM VisualGen carries a Qwen-Image pipeline (tensorrt_llm/_torch/visual_gen/models/qwen_image/) but nothing for the 2.1 architecture: a code search for QwenImage21 / Qwen-Image-2.1 over NVIDIA/TensorRT-LLM returns no hits, and the model is absent from pipeline_registry.py. Qwen-Image-2.1 is a distinct architecture (QwenImage21Pipeline / QwenImage21Transformer2DModel / AutoencoderKLQwenImage21 / Qwen3-VL encoder), so the existing Qwen-Image pipeline does not cover it. |
| ComfyUI | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | The harness's Qwen-Image-2.1 graph is the official t2i template; an edit graph wiring the reference into its images.image_1 input has not been built and validated yet (2026-10-10). |
| FastVideo | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | No Qwen-Image pipeline in FastVideo main (2b164405c3d1, 2026-10-07). |

### zimage_turbo_t2i_1024

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Tongyi-MAI/Z-Image-Turbo | text-to-image | 1024x1024 | 9 | gs=0.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | b300-speed | 2 | 0.317 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-tp2-static | 2 | 0.457 | 1.442x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | blackwell-fa2-fusion | 2 | 0.852 | 2.688x | ok | - | - | - | - | - | - | - | - | not_run | - |
| trtllm-visual | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | Not yet classified. Confirm framework support and add a validated command profile, or mark unsupported. |
| ComfyUI | blackwell-gpuonly | 1 | 0.840 | 2.650x | ok | - | - | - | - | - | - | - | - | not_run | - |
| FastVideo | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | FastVideo main (2b164405c3d1, 2026-10-07) registers Z-Image-Turbo, but this harness integrates FastVideo for MiniMax-H3 only (2026-10-08); no command validated. |

### ideogram4_t2i_1024_2gpu_tp

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| ideogram-ai/ideogram-4-fp8 | text-to-image | 1024x1024 | 20 | - |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | b300-speed | 2 | 1.939 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | Latest vLLM-Omni has no Ideogram-4 serving path (image-gen coverage is FLUX/Qwen-Image/Wan). |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | Latest LightX2V has no Ideogram-4 serving path. |
| trtllm-visual | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | Latest TensorRT-LLM VisualGen has no Ideogram-4 serving path (coverage is FLUX/Wan/LTX-2/Qwen-Image/Cosmos3). |
| ComfyUI | blackwell-gpuonly-compile | 1 | 3.685 | 1.900x | ok | - | - | - | - | - | - | - | - | not_run | - |
| FastVideo | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | No Ideogram pipeline in FastVideo main (2b164405c3d1, 2026-10-07). |

### wan22_t2v_a14b_720p

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Wan-AI/Wan2.2-T2V-A14B-Diffusers | text-to-video | 1280x720x81 | 40 | gs=4.0,gs2=3.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | b300-noexp | 4 | 89.552 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-4gpu-cfg2-usp2-vaepp-devpost | 4 | 97.824 | 1.092x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | blackwell-sdpa | 4 | 100.441 | 1.122x | ok | - | - | - | - | - | - | - | - | not_run | - |
| trtllm-visual | blackwell-cudnn | 4 | 94.086 | 1.051x | ok | - | - | - | - | - | - | - | - | not_run | - |
| ComfyUI | blackwell-gpuonly | 2 | 521.849 | 5.827x | ok | - | - | - | - | - | - | - | - | not_run | - |
| FastVideo | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | FastVideo main (2b164405c3d1, 2026-10-07) registers Wan2.2-T2V-A14B (examples/serving/openai_wan22_t2v_a14b.yaml), but this harness integrates FastVideo for MiniMax-H3 only (2026-10-08); no command validated. |

### ltx2.5_twostage_t2v_2gpus

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| Lightricks/LTX-2.5-Diffusers | text-to-video | 1920x1088x121 | 8 | gs=1.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | b300-auto-bcg | 2 | 5.221 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-2gpu-distilled-usp2 | 2 | 5.861 | 1.123x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | LightX2V's LTX-2.5 runner (ltx2_5, main 19c19614 = HEAD on 2026-10-24) decodes only with the diffusion VAE: LTX25VideoVAE raises 'LTX25VideoVAE requires a CausalDiffusionVAE checkpoint' for anything else (lightx2v/models/video_encoders/hf/ltx2/model.py), and the runner class fixes video_vae_class, so this case's convolutional decode is not reachable from its config. Its distilled two-stage itself would match (one distilled DiT for both stages, the same 8 + 3 sigmas, CFG 1); a diffusion-VAE row would be different, heavier work. Checked 2026-10-24. |
| trtllm-visual | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | TensorRT-LLM 1.4.0rc1 (the newest release candidate, installed) has no LTX-2.5: its VisualGen ltx2 package serves Lightricks/LTX-2 (19B) only, and nothing under tensorrt_llm/_torch/visual_gen loads Gemma-4, LTX-2.5's text encoder. Checked 2026-10-24. |
| ComfyUI | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | ComfyUI master (1a82e6d0) ships an official LTX-2.5 T2V template (video_ltx2_5_t2v): the same distilled two-stage (8 sigmas, x2 latent upsampler, a 3-sigma tail, CFG 1), but on the int8-convrot DiT and text encoder, with the diffusion VAE (ltx-2.5-video-vae-bf16), a stage-2 schedule starting at 0.85, and a prompt-enhancer node. A bf16 graph with the conv VAE (vae/ltx-2.5-video-vae-conv-bf16) and the official stage-2 sigmas is not built or validated yet. 2026-10-24. |
| FastVideo | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | FastVideo main (fa2d1324) registers LTX-2.5 distilled (FastVideo/LTX-2.5-Distilled-Diffusers, preset ltx2_5_distilled_two_stage), but this harness integrates FastVideo for MiniMax-H3 only (2026-10-08); no command validated. |

### cosmos3_nano_t2i_720p

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| nvidia/Cosmos3-Nano | text-to-image | 1280x720 | 35 | gs=6.0 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | blackwell-2gpu-cfg-parallel | 2 | 0.528 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-cudnn-static | 2 | 0.592 | 1.121x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | blackwell-cfg2-sdpa | 2 | - | - | failed | - | - | - | - | - | - | - | - | failed | 500 Server Error: Internal Server Error for url: http://127.0.0.1:61016/v1/tasks/image/sync?poll_interval_seconds=0.02 -- {"detail":"Image inference did not return an encodable image."}; LightX2V 395e2ac4's cosmos3 runner answers /v1/tasks/image/sync with HTTP 500 'Image inference did not return an encodable image' (4xB300, 2026-11-17, re-measured after the refactor #1673): the T2I task returns a (1, 3, 1, 720, 1280) video-shaped tensor and the route's PNG encoder (pipeline_image_encode._tensor_to_png_bytes) rejects it ('Unsupported tensor shape'). |
| trtllm-visual | default | 2 | 0.599 | 1.134x | ok | - | - | - | - | - | - | - | - | not_run | - |
| ComfyUI | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | ComfyUI core has no Cosmos3: comfy/ldm/cosmos covers Cosmos Predict1 and Predict2 only (CosmosT2V/I2V, CosmosT2IPredict2/I2VPredict2 in comfy/supported_models.py) at master 88ab4a06, 2026-09-25. |
| FastVideo | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | FastVideo main (2b164405c3d1, 2026-10-07) has Cosmos Predict2/2.5 and GEN3C pipelines but no Cosmos3. |

### cosmos3_nano_t2v_720p_189f

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| nvidia/Cosmos3-Nano | text-to-video | 1280x720x189 | 35 | gs=6.0,neg=1 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | b300-noexp | 4 | 22.807 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-cudnn-vaepp | 4 | 26.290 | 1.153x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | blackwell-cfg2-sp2-sdpa | 4 | 44.559 | 1.954x | ok | - | - | - | - | - | - | - | - | not_run | - |
| trtllm-visual | default | 4 | 25.861 | 1.134x | ok | - | - | - | - | - | - | - | - | not_run | - |
| ComfyUI | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | ComfyUI core has no Cosmos3: comfy/ldm/cosmos covers Cosmos Predict1 and Predict2 only (CosmosT2V/I2V, CosmosT2IPredict2/I2VPredict2 in comfy/supported_models.py) at master 88ab4a06, 2026-09-25. |
| FastVideo | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | FastVideo main (2b164405c3d1, 2026-10-07) has Cosmos Predict2/2.5 and GEN3C pipelines but no Cosmos3. |

### cosmos3_nano_i2v_720p_189f

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| nvidia/Cosmos3-Nano | image-to-video | 1280x720x189 | 35 | gs=6.0,neg=1 |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | b300-noexp | 4 | 22.807 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-cudnn-vaepp | 4 | 25.804 | 1.131x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | blackwell-cfg2-sp2-sdpa | 4 | 47.052 | 2.063x | ok | - | - | - | - | - | - | - | - | not_run | - |
| trtllm-visual | default | 4 | 27.876 | 1.222x | ok | - | - | - | - | - | - | - | - | not_run | - |
| ComfyUI | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | ComfyUI core has no Cosmos3: comfy/ldm/cosmos covers Cosmos Predict1 and Predict2 only (CosmosT2V/I2V, CosmosT2IPredict2/I2VPredict2 in comfy/supported_models.py) at master 88ab4a06, 2026-09-25. |
| FastVideo | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | FastVideo main (2b164405c3d1, 2026-10-07) has Cosmos Predict2/2.5 and GEN3C pipelines but no Cosmos3. |

### minimax_h3_t2va_5s

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| MiniMaxAI/MiniMax-H3 | text-to-video | 1344x768x124 | 50 | - |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | single:b300-noexp; throughput:dp2-throughput | 2 | 62.342 | 1.000x | ok | 4/4 | 2 | 118.278 | 1.000x | 119.286 | 119.288 | 0.0168 | 1.000x | ok | - |
| vLLM-Omni | single:blackwell-2gpu-static; throughput:replica2-throughput | 2 | 67.404 | 1.081x | ok | 4/4 | 2 | 121.759 | 1.029x | 122.261 | 122.262 | 0.0164 | 0.976x | ok | - |
| LightX2V | blackwell-sp2-sdpa | 2 | 71.758 | 1.151x | ok | 4/4 | 2 | 143.398 | 1.212x | 144.290 | 144.413 | 0.0139 | 0.827x | ok | - |
| trtllm-visual | blackwell-omp8 | 2 | 68.334 | 1.096x | ok | 4/4 | 2 | 133.589 | 1.129x | 134.888 | 135.071 | 0.0149 | 0.887x | ok | - |
| ComfyUI | blackwell-gpuonly | 1 | 307.046 | 4.925x | ok | 4/4 | 2 | 610.177 | 5.159x | 610.357 | 610.370 | 0.0033 | 0.196x | ok | - |
| FastVideo | blackwell-2gpu | 2 | 75.105 | 1.205x | ok | 4/4 | 2 | 149.929 | 1.268x | 150.038 | 150.053 | 0.0133 | 0.792x | ok | - |

### minimax_h3_t2va_5s_4gpu

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| MiniMaxAI/MiniMax-H3 | text-to-video | 1344x768x124 | 50 | - |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | b300-noexp | 4 | 31.638 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-4gpu-static | 4 | 36.751 | 1.162x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | blackwell-sp4-sdpa | 4 | 36.973 | 1.169x | ok | - | - | - | - | - | - | - | - | not_run | - |
| trtllm-visual | blackwell-fa4-omp8 | 4 | 36.505 | 1.154x | ok | - | - | - | - | - | - | - | - | not_run | - |
| ComfyUI | - | - | - | - | not_run | - | - | - | - | - | - | - | - | not_run | ComfyUI has no sequence or tensor parallelism (its one multi-GPU path splits CFG branches, and H3 runs without CFG), so a four-GPU cell would be its one-GPU number; that number is in minimax_h3_t2va_5s. |
| FastVideo | blackwell-4gpu | 4 | 48.851 | 1.544x | ok | - | - | - | - | - | - | - | - | not_run | - |

### minimax_h3_ref2va_5s

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| MiniMaxAI/MiniMax-H3 | text-to-video | 1344x768x124 | 50 | - |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | b300-noexp | 2 | 95.844 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | blackwell-2gpu-static | 2 | 103.240 | 1.077x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | blackwell-sp2-sdpa | 2 | 113.840 | 1.188x | ok | - | - | - | - | - | - | - | - | not_run | - |
| trtllm-visual | blackwell-fa4-omp8 | 2 | 106.329 | 1.109x | ok | - | - | - | - | - | - | - | - | not_run | - |
| ComfyUI | blackwell-gpuonly | 1 | 518.199 | 5.407x | ok | - | - | - | - | - | - | - | - | not_run | - |
| FastVideo | blackwell-2gpu | 2 | 110.498 | 1.153x | ok | - | - | - | - | - | - | - | - | not_run | - |

### minimax_h3_fasth3_v2_t2va_5s

| model | task | dims | steps | cfg |
| --- | --- | --- | ---: | --- |
| FastVideo/FastVideo-FastH3-8-Step-V2 | text-to-video | 1344x768x124 | 8 | - |

| framework | profile | gpus | single_e2e_s | single/SGLang-Diffusion | single_status | done/reqs | concurrency | p50_s | p50/SGLang-Diffusion | p95_s | p99_s | qps | qps/SGLang-Diffusion | throughput_status | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SGLang-Diffusion | default | 4 | 5.070 | 1.000x | ok | - | - | - | - | - | - | - | - | not_run | - |
| vLLM-Omni | default | 4 | 7.738 | 1.526x | ok | - | - | - | - | - | - | - | - | not_run | - |
| LightX2V | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | Checked 2026-10-08 against LightX2V main 0c2edc12 (2026-10-05): LightX2V cannot serve this checkpoint under its trained contract (fastvideo_inference.json). V2 is trained for VIDEO_SPARSE_ATTN_H3 at 0.8 sparsity on 64-token tiles, and LightX2V has no VSA backend (its sparse attentions -- Sparge, SVG, radial, NBHD, Sol-Attn, sparse Sage/FA4 -- are other methods; dense attention would run a different model). Its H3 scheduler builds only a uniform shifted grid from infer_steps, so the explicit rungs [999, 874, ..., 125] cannot be given (an 8-step grid starts at 1000). LightX2V's own few-step H3 paths (configs/minimax_h3/dmd/: 4- and 8-step LoRAs on the base checkpoint, mostly FP8/INT8) are a different distillation, not this checkpoint, and are not substituted. |
| trtllm-visual | - | - | - | - | unsupported | - | - | - | - | - | - | - | - | unsupported | trtllm-visual's tracked scope in this harness is image cases; H3 is a video+audio model. |
| ComfyUI | - | - | - | - | no_profile | - | - | - | - | - | - | - | - | no_profile | Comfy-Org/workflow_templates templates/video_fastvideo_fasth3_t2v.json exists (since 2026-09-22), but it runs int8 dense attention and VSA keeping 10% of tiles from step 0.2 -- a different numeric class and a different sparsity from the case's BF16 VSA at the trained 80% (64-token tiles). Re-checked 2026-10-22. |
| FastVideo | blackwell-4gpu | 4 | 9.294 | 1.833x | ok | - | - | - | - | - | - | - | - | not_run | - |
