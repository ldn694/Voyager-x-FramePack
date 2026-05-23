# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo overview

This is a fork of Tencent's **HunyuanWorld-Voyager** (RGB-D video diffusion for explorable 3D scenes from a single image + camera path), extended with custom training infrastructure for fine-tuning on **RealEstate10K**. The original DiT backbone, VAE, and text encoders are preserved; on top of them this fork adds LoRA, custom patch-embed (`patch-adapter`), multi-kernel patchify, and a double-branch transformer — all switchable per-run via flags.

## Common commands

All commands assume the `voyager` conda env (`python==3.11.9`, CUDA 12.4, PyTorch 2.4.0, flash-attn, `xfuser==0.4.2`). Weights live under `ckpts/` (overridable via `MODEL_BASE` env var). Frame counts for the 3D VAE must be `4n+1` (typically 49).

### Inference

Single GPU:
```bash
python3 sample_image2video.py \
    --model HYVideo-T/2 \
    --input-path "examples/case1" \
    --prompt "..." \
    --i2v-stability --flow-reverse --flow-shift 7.0 \
    --infer-steps 50 --embedded-cfg-scale 6.0 --seed 0 \
    --use-cpu-offload --save-path ./results
```

Multi-GPU via xDiT (USP sequence parallel): set `ALLOW_RESIZE_FOR_SP=1`, prefix with `torchrun --nproc_per_node=N`, and pass `--ulysses-degree A --ring-degree B` with `A*B == N`.

Eval-on-dataset loop (auto-restart on OOM): `bash sample_realestate10K_addons_replay.sh` — wraps `sample_image2video_realestate10K.py` with LoRA + double-branch + multi-kernel flags. Edit the `--use-kernel-indices` lines (one per kernel set), `--ratio`, and checkpoint paths.

Gradio demo: `python3 app.py`.

### Training (DeepSpeed)

Primary entry point is `deepspeed_train_render.py` (CPU-bound point-cloud rendering moved into a DataLoader worker). Launch via `bash deepspeed_train.sh` — edit `--include localhost:<gpu_ids>` to pick GPUs. Key flags:

- `--global-batch-size 1` (effective batch after grad accumulation; per-GPU is derived)
- `--gradient-checkpoint` and `--zero-stage` for memory
- `--use-cache-text-encoder` reads from a precomputed text-embedding cache
- `--use-model-input-cache --model-input-cache-name <name>` reads cached latents/cond-latents so VAE+text-encoder are skipped at train time
- Adapter toggles: `--train-lora --lora-rank N`, `--patch_adapter_size pt ph pw`, `--train-multiple-kernels --kernel-sizes ... --kernel-indices ...`, `--use-double-branch --model-with-double-branch <cfg>`
- Resume *per adapter*, not from a single ckpt: `--resume-lora`, `--resume-patch-adapter`, `--resume-multi-kernel`, `--resume-double-branch`
- Output dirs auto-numbered as `<output-dir>/run_NNNNN/`; `*_last.pt` is the rolling checkpoint

Older single-GPU/non-render variants: `train.py`, `deepspeed_train.py` (kept; render variant is the active path).

### Cache builders (run before training)

- `python cache_clipl_text_embed.py --dataset-root ...` — precomputes text encoder outputs
- `python cache_model_input.py --dataset-root ... --model-input-cache-name <name>` — runs VAE + text encoder + renderer once, dumps `latents.pt / cond_latents.pt / partial_cond.pt / partial_mask.pt / sample_id_to_index.json` (see `voyager/cache/model_input_cache.py`)

### Dataset prep & eval

- `bash gather_realestate.sh` — `gather_realestate.py` writes a per-split folder with rendered partial conditions
- `python evaluation_summary.py`, `python gather_result_realestate10K.py` — post-processing for `sample_image2video_realestate10K.py` outputs

### Data engine (separate env)

`data_engine/` produces RGB-D + camera data for new videos using VGGT + MoGe + Metric3D. It needs its own `data_engine` conda env (Python 3.10) and external repos cloned in-place — see `data_engine/README.md`. The directory is gitignored.

## Architecture

### Entry points and the args pattern

Every script composes its argparse via the `add_*_args(parser)` builders in `voyager/config.py` (e.g. `add_network_args`, `add_i2v_args`, `add_lora_args`, `add_double_branch_args`). When adding a new training/inference flag, add it to the relevant builder so every script that calls that builder picks it up consistently. `voyager/config.py:parse_args()` is used by inference scripts; the training scripts call the builders directly and add their own `argparse.ArgumentParser`.

### DiT backbone and model variants (`voyager/modules/models.py`)

`HYVideoDiffusionTransformer` is the MMDiT-style backbone (double-stream MM blocks + single-stream blocks, flow-matching). Model presets live in the `HUNYUAN_VIDEO_CONFIG` dict at the bottom of the file. Notable variants used in this fork:

- `HYVideo-T/2` — the standard 3072-dim / 24-head / 20+40 block model
- `HYVideo-T/2-2branch-cross_attn` — adds a smaller (512-dim / 4-head) second branch alongside the main blocks, with a `scheduler` of `(query_block, kv_block)` tuples controlling where cross-attention happens between the two branches. Negative indices in the scheduler are interpreted by the double-branch code.
- `HYVideo-T/2-2branch-cross_attn-unidirectional-q_second` — same shape, but cross-attention only flows from the second branch into the first.

### Adapter modules (consistent apply/get/load pattern)

LoRA, patch-adapter, multi-kernel, and double-branch each live in their own file under `voyager/modules/` and expose the same four-function surface:

- `apply_<name>_to_hunyuan_video(model, ...)` — mutates the model in-place
- `get_<name>_parameters(model)` — returns the trainable subset for the optimizer
- `get_<name>_state_dict(model)` — pulls out only the adapter weights
- `load_<name>_state_dict(model, state_dict)` — restores them

This pattern is the answer for "where do I plug in another adapter?" — follow the same four-function surface and register it in `deepspeed_train_render.py` (apply, collect params, save in the checkpoint loop, load on resume).

`voyager/modules/multi_kernel.py` and `multi_kernel_transpose.py` implement patchify with multiple kernel sizes at chosen layer indices (`--kernel-sizes` and `--kernel-indices` are repeatable flags: each `--kernel-indices` group is paired with one `--kernel-sizes` group). `voyager/modules/custom_patch_embed.py` replaces the front-end `PatchEmbed` / `FinalLayer` with a different patch size (`--patch_adapter_size`).

### Pipeline composition

- `voyager/inference.py:HunyuanVideoSampler` and `load_models()` build the full inference stack (VAE → text encoders → DiT → flow scheduler → `HunyuanVideoPipeline`). `load_vae_only()` is used by the cache builders.
- `voyager/diffusion/` — `flow/transport.py` (training-time flow matching loss), `pipelines/pipeline_hunyuan_video.py` (sampling), `schedulers/scheduling_flow_match_discrete.py`. `voyager/diffusion/__init__.py:load_denoiser` builds the `Transport` for training.
- `voyager/vae/` — causal 3D VAE (`autoencoder_kl_causal_3d.py`). Tiling enabled by default.
- `voyager/text_encoder/__init__.py:TextEncoder` wraps LLaVA-style LLM + CLIP-L for the dual text-encoder setup.
- `voyager/cache/` — `TextEncoderCache` and `ModelInputCache` (load whole `.pt` tensors at construction; index by `sample_id`).
- `voyager/utils/geometry.py` — Plücker-coordinate camera embeddings consumed by the DiT.

### Dataset and rendering

- `dataset/RealEstate10K.py` — base loader (frames + depth + cameras + captions).
- `dataset/RealEstate10K_render.py` — variant that renders partial-view conditioning images inside the DataLoader so it overlaps with GPU compute (this is what the active training script uses; see recent commit history).
- `utils/render.py` — `Camera` and `Frame` classes do point-cloud reprojection from an RGB-D source frame into a target camera; `gather_realestate.norm_partial_render_output` normalizes the resulting partial images/depths/masks for the model.

### Conditioning shape

For a clip of `T` frames the model receives the latent target plus a concatenated partial-RGB + partial-depth condition and a binary mask. `sample_image2video.py` constructs `ref_images` (frame 0) and `partial_cond`/`partial_mask` lists (frames `000..T-1`) — these come from the input case folder's `rgb/`, `depth/`, and `mask/` subfolders. The `RealEstate10K_render` dataset produces the same tensors at training time by reprojecting GT frames through estimated cameras.

## Conventions and gotchas

- **`MODEL_BASE` env var** (default `ckpts`) controls where pretrained weights are loaded from. All `*_PATH` constants in `voyager/constants.py` interpolate it.
- **`--latent-channels` must match the VAE** (parsed from the VAE name, e.g. `884-16c-hy` → 16). `sanity_check_args` enforces this.
- **DeepSpeed runs** (see `deepspeed_train.sh`) typically set `DS_SKIP_CUDA_CHECK=2`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and redirect `TMPDIR/TMP/TEMP` to fast local storage. Keep these when adding new launch scripts.
- **`use_kernel_indices`** at inference is an *append* arg: each `--use-kernel-indices a b c` adds one group; pass multiple times to select different kernel sets for different layer groups, matching the `kernel_sizes` from the checkpoint's args.
- **Output naming**: training output dirs are auto-incremented `run_NNNNN`; checkpoint files are `<adapter>_last.pt` (rolling) plus periodic `<adapter>_step_<N>.pt`. Each adapter saves separately — there is no single combined checkpoint file.
- **`sample_image2video.py`** imports `spectralAnalyser` (spatial-frequency analysis) at the top level; if you trim that file, the import must come with you or be removed.
- The `data_engine/` directory and its sub-repos (Metric3D, MoGe, vggt) are gitignored — don't commit weights or third-party clones into them.
