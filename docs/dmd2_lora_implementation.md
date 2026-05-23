# DMD2 + LoRA distillation for HunyuanWorld-Voyager

Design document for adding **DMD2** (Distribution Matching Distillation v2, Yin et al. 2024) finetuning, packaged as a LoRA adapter on top of the frozen HunyuanWorld-Voyager DiT, to produce a few-step (default: 4-step) sampler.

The training entry point is the existing `deepspeed_train_render.py`. Passing `--dmd2-steps N` (with `N > 0`) switches the loop from standard flow-matching training to DMD2 distillation; `--train-lora` is required so the generator is a LoRA on the frozen base.

---

## 1. Background — DMD2 in one page

### 1.1 The problem

The base Voyager model is a flow-matching DiT that needs ~50 denoising steps to produce a sample. DMD2 distills that into a **few-step generator** `G_θ` (here: 4 steps) that approximately samples from the same distribution as the teacher, by directly minimizing a reverse KL between fake and real distributions at multiple noise levels.

### 1.2 The two scores

DMD2 defines:

- **Real score** `s_real(x_t, t)`: from the frozen pretrained model. For Voyager this is the base DiT with **LoRA disabled**, predicting flow-matching velocity `v_real`.
- **Fake score** `s_fake(x_t, t; φ)`: a separate trainable head that learns the score of the current generator's output distribution. In our design this is a **second LoRA adapter** on the same frozen base DiT.

The DMD gradient on the generator approximates ∇_θ KL(p_fake ‖ p_real):

```
∇_θ L_DMD  ∝  E_{z, ε, t} [ w(t) · stop_grad(v_fake − v_real) · ∂x̂₀/∂θ ]
```

where `x̂₀ = G_θ(z, c)` is the generator's predicted clean sample for some sampled noise `z`, conditioning `c`, and noise level `t`. We use the standard "x̂₀ surrogate loss" formulation:

```
L_DMD(θ) = mean( x̂₀ · stop_grad( weight · (v_fake − v_real) ) )
```

which has the same gradient as above and plays nicely with autograd. The weight `w(t)` is a per-sample normalization (see §2.4).

The fake score is trained with the **standard flow-matching loss on generator outputs**:

```
L_fake(φ) = MSE( v_fake_LoRA(x_t, t), v_target )
where x_t = α_t · stop_grad(x̂₀) + σ_t · ε
      v_target is the flow-matching velocity target for path(stop_grad(x̂₀), ε)
```

These two losses are alternated (DMD2's two-timescale rule: typically `K_fake = 5` fake-score updates per generator update; configurable).

### 1.3 What's different from DMD1

- **No regression-to-teacher-ODE loss.** Training is faster and avoids the offline trajectory-generation cost.
- **Two-timescale update.** Multiple fake-score updates per generator update stabilizes training.
- **Multi-step generator.** Generator denoises over `N` fixed timesteps (we use `N=4`) instead of a single shot. Samples for distillation are drawn at one randomly chosen step per minibatch.
- **(Optional) GAN head.** Deferred in v1; can be added behind a flag later.

### 1.4 Adaptation to flow matching

Voyager is trained with flow matching (`Transport` in `voyager/diffusion/flow/transport.py`, `--flow-reverse --flow-shift 7.0`), not DDPM. The mapping is:

| DDPM term | Flow-matching equivalent (this repo) |
|---|---|
| `ε`-prediction / `s = ∇ log p` | velocity `v` predicted by the DiT |
| `x_t = √ᾱ_t x₀ + √(1−ᾱ_t) ε` | `x_t = α_t x₁ + σ_t x₀` (with `x₁` = clean data, `x₀` = noise, linear path: `α_t = t`, `σ_t = 1−t` under `--flow-reverse`) |
| Score-matching loss | Flow-matching MSE on `v` |

For the DMD gradient, we use **velocity differences directly**: `v_fake − v_real` is proportional to the score difference up to a positive factor, so the formula above is correct for the rectified-flow / linear path used here. The proportionality constant is folded into `w(t)`.

### 1.5 Timestep grid (per user spec)

Generator timesteps are **uniformly spaced in [0, 1]**, *not* following the `--flow-shift` schedule. For `--dmd2-steps N`:

```
t_grid = [ (N - i) / N  for i in range(N) ]    # e.g. N=4 → [1.0, 0.75, 0.5, 0.25]
```

Convention (matches `--flow-reverse`): `t = 1.0` is pure noise, `t = 0.0` is clean data; the generator is invoked starting at each `t_i` and is responsible for denoising one full uniform step (size `1/N`). This is intentionally simple and decoupled from the teacher's shifted schedule — only the **teacher's velocity field** is consulted via `v_real`, not its sampler timesteps.

---

## 2. Algorithm in this repo

### 2.1 Single base, two LoRA adapters

We keep one frozen copy of the DiT in memory and attach **two named LoRA adapters**:

- `"gen"` — the **generator** LoRA. Trainable. This is the artifact we ship for 4-step inference.
- `"fake"` — the **fake-score** LoRA. Trainable, but discarded after distillation.

For each forward pass we **route** through exactly one adapter (or none, for the real-score pass) via a global `set_active_lora_adapter(model, name | None)` helper. The base weights are always frozen.

The real score is obtained by setting the active adapter to `None` (i.e. all `LoRALinear.lora_enabled = False`) and running the DiT.

### 2.2 Generator forward

For a sampled timestep `t_i` from `t_grid` (one per minibatch), the generator predicts the clean sample from a noisy input:

```python
# Pure-noise start when t_i == 1.0; otherwise α_t · x1_data + σ_t · ε
x_t = sample_x_t(noise, data_or_None, t_i)        # at t=1, x_t is pure noise
set_active_lora_adapter(model, "gen")
v_pred = model(x_t, t_i, conds)
x_hat_0 = x_t + (1.0 - t_i) * v_pred              # x̂₀ from velocity along linear path
```

For the linear path with `--flow-reverse` (`x_t = (1−s) · noise + s · data` where `s = 1 − t`), the closed-form clean estimate from velocity `v = data − noise` is `x̂₀ = x_t + (1 − t) · v_pred`. The helper `compute_x_hat_0_from_velocity(x_t, v_pred, t)` lives next to the loss.

### 2.3 DMD2 generator loss

```python
x_hat_0 = generator(z, c, t_i)                    # grad flows through θ

# Re-noise the prediction at a "perturbation" timestep t_p ∈ [0, 1] (uniform)
t_p = uniform(0, 1)
eps = randn_like(x_hat_0)
x_tp = alpha(t_p) * x_hat_0 + sigma(t_p) * eps

with torch.no_grad():
    set_active_lora_adapter(model, None)
    v_real = model(x_tp, t_p, c)
    set_active_lora_adapter(model, "fake")
    v_fake = model(x_tp, t_p, c)

# Surrogate loss — same gradient as ∇θ KL
grad_signal = (v_fake - v_real)                   # detached
weight      = dmd_weight(x_hat_0, t_p)            # see §2.4, detached
L_dmd       = (weight * grad_signal * x_hat_0).mean()
```

Note: `t_p` is sampled **independently** of `t_i`. The generator timestep `t_i` selects which "noise budget" the few-step generator is being trained at; `t_p` selects where in `[0, 1]` we compare scores.

### 2.4 Weight `w(t_p)`

Following DMD2, we use the "normalized magnitude" weight that decouples the loss scale from the noise level:

```python
# detached, per-sample (broadcast over spatial/temporal dims)
weight = 1.0 / (x_hat_0.detach().abs().mean(dim=spatial_dims, keepdim=True) + eps)
```

This is the default. A simpler `weight = 1.0` is available behind a flag for ablation.

### 2.5 Fake-score loss

```python
with torch.no_grad():
    x_hat_0 = generator(z, c, t_i)                # detached

t_p = uniform(0, 1)
eps = randn_like(x_hat_0)
x_tp, v_target = sample_path_and_target(x_hat_0, eps, t_p)   # from Transport

set_active_lora_adapter(model, "fake")
v_pred_fake = model(x_tp, t_p, c)

L_fake = mse(v_pred_fake, v_target)
```

This is the **existing flow-matching loss** computed against generator outputs instead of dataset ground truth. We reuse `Transport.training_losses` for this with a small wrapper.

### 2.6 Two-timescale loop

```
for step in range(total_steps):
    for _ in range(K_fake):                       # default K_fake = 5
        update_fake_score()                       # L_fake on detached x̂₀
    update_generator()                            # L_dmd
```

`K_fake` is the `--dmd2-fake-updates-per-gen` flag.

### 2.7 Initialization

Both LoRA adapters start from the standard LoRA init (`A` Kaiming, `B` zero). The **fake-score LoRA needs a short warmup** before the first generator update, otherwise `v_fake ≈ v_real` (both equal the base model) and the DMD gradient is ~0. We do `--dmd2-fake-warmup-steps` (default 200) of fake-only updates before the first generator step.

---

## 3. Surface area — files added & modified

### 3.1 New files

| File | Contents |
|---|---|
| `voyager/diffusion/dmd2/__init__.py` | Re-exports `DMD2Config`, `DMD2Trainer`, `compute_dmd2_generator_loss`, `compute_fake_score_loss`, `uniform_timestep_grid`. |
| `voyager/diffusion/dmd2/dmd2_config.py` | `@dataclass DMD2Config` holding `num_steps`, `fake_updates_per_gen`, `fake_warmup_steps`, `weight_mode`, `t_grid_kind` (default `"uniform"`), `cfg_scale_for_real`, `min_t_p`, `max_t_p`. Built from argparse in `voyager/config.py:add_dmd2_args`. |
| `voyager/diffusion/dmd2/timestep_grid.py` | `uniform_timestep_grid(N) → tensor([(N-i)/N for i in range(N)])`. Independent of `flow_shift` per the user's spec. Comment notes how to swap in a shifted variant if we ever want to A/B. |
| `voyager/diffusion/dmd2/dmd2_loss.py` | Pure functions: `compute_x_hat_0_from_velocity(x_t, v, t)`, `dmd_weight(x_hat_0, t_p, mode)`, `compute_dmd2_generator_loss(...)`, `compute_fake_score_loss(...)`. All take a callable `forward_fn(x_t, t, conds)` so they don't depend on the DiT class. |
| `voyager/diffusion/dmd2/dmd2_trainer.py` | `DMD2Trainer` thin orchestrator: holds the model engine, two parameter groups (gen + fake), the schedule of generator timesteps, and a `step()` method that runs `K_fake` fake updates + 1 generator update. Exposes `state_dict()` / `load_state_dict()` for resume. Does **not** own a DataLoader — the existing training script keeps that. |
| `voyager/diffusion/dmd2/sampler.py` | `DMD2Sampler.sample(z, conds, n_steps, t_grid)` — runs the `N`-step generator at inference. Used by `inference.py`. |
| `dmd2_train.sh` | Launch script (see §5). |
| `docs/dmd2_lora_implementation.md` | This document. |

### 3.2 Files modified

#### `voyager/modules/lora_layers.py` — multi-adapter support

Change `LoRALinear` from holding **one** `(lora_A, lora_B)` pair to a `nn.ModuleDict`-keyed set of `(A, B)` pairs, plus an `active_adapter: Optional[str]` field. Only the active adapter contributes to the forward pass; if `active_adapter is None`, the LoRA branch is skipped entirely (this is how we get `v_real`).

New / changed signatures:

- `LoRALinear.add_adapter(name: str, r: int, lora_alpha: float, lora_dropout: float)` — adds a named adapter in-place.
- `LoRALinear.set_active_adapter(name: Optional[str])` — sets which adapter is live.
- `set_active_lora_adapter(model, name: Optional[str])` — module-tree helper.
- `apply_lora_to_hunyuan_video(model, r, lora_alpha, lora_dropout, freeze_base=True, adapter_name="default")` — gains an `adapter_name` kwarg. Idempotent: if a `LoRALinear` already wraps the target layer, it just calls `add_adapter` instead of re-replacing.
- `get_lora_parameters(model, adapter_name=...)`, `get_lora_state_dict(model, adapter_name=...)`, `load_lora_state_dict(model, sd, adapter_name=...)` — all gain an `adapter_name` kwarg.
- `toggle_lora(model, enable, adapter_name=None)` — preserved for back-compat; when `adapter_name` is `None` it toggles whatever's currently active.

**Back-compat**: existing single-adapter LoRA training paths (`--train-lora` without `--dmd2-steps`) keep working by using `adapter_name="default"` everywhere implicitly.

#### `voyager/config.py` — `add_dmd2_args`

New parser group, plugged into `parse_args()` and into the bespoke parsers inside `deepspeed_train_render.py` / `deepspeed_train.py`:

```python
def add_dmd2_args(parser):
    g = parser.add_argument_group("DMD2 distillation")
    g.add_argument("--dmd2-steps", type=int, default=0,
                   help="Number of generator timesteps for DMD2 distillation. "
                        "0 disables DMD2 and runs standard flow-matching training.")
    g.add_argument("--dmd2-fake-updates-per-gen", type=int, default=5)
    g.add_argument("--dmd2-fake-warmup-steps", type=int, default=200)
    g.add_argument("--dmd2-fake-lora-rank", type=int, default=None,
                   help="LoRA rank for the fake-score adapter. Defaults to --lora-rank.")
    g.add_argument("--dmd2-fake-lora-alpha", type=float, default=None)
    g.add_argument("--dmd2-weight-mode", choices=["normalized", "uniform"], default="normalized")
    g.add_argument("--dmd2-min-tp", type=float, default=0.02)
    g.add_argument("--dmd2-max-tp", type=float, default=0.98,
                   help="Clamp range for the perturbation timestep t_p.")
    g.add_argument("--dmd2-cfg-scale-real", type=float, default=1.0,
                   help="Optional CFG scale on the real-score call. 1.0 disables.")
    g.add_argument("--dmd2-fake-lr-mult", type=float, default=1.0,
                   help="Multiplier on --lr for the fake-score optimizer.")
    g.add_argument("--resume-dmd2-fake-lora", type=str, default=None,
                   help="Resume path for the fake-score LoRA checkpoint.")
    return parser
```

`sanity_check_args` gains: if `--dmd2-steps > 0`, require `--train-lora` and `--flow-reverse`.

#### `deepspeed_train_render.py`

Branch at the top of the training loop:

```python
if args.dmd2_steps > 0:
    run_dmd2_training(model_engine, denoiser, dataloader, args, ...)
else:
    run_standard_training(...)   # existing path, unchanged
```

`run_dmd2_training(...)` lives in the same file (parallel to the existing inline loop) and:

1. After `apply_lora_to_hunyuan_video(model, ..., adapter_name="gen")`, also calls it again with `adapter_name="fake"` and `r = args.dmd2_fake_lora_rank or args.lora_rank`.
2. Builds **two DeepSpeed engines or two parameter groups** on the same model. We use **one DeepSpeed engine with two optimizers** (DeepSpeed supports multiple optimizer step calls per forward; we toggle `model_engine.optimizer` between gen / fake). Decision recorded in §6.
3. Pre-computes `t_grid = uniform_timestep_grid(args.dmd2_steps).to(device)` once.
4. Inside the per-batch loop:
   - Pull `x1` (clean latent), `cond_latents`, `partial_cond`, `partial_mask` exactly as today (the existing rendering DataLoader is unchanged).
   - For `_ in range(K_fake)`: call `compute_fake_score_loss(...)`, backprop, step fake optimizer.
   - After `--dmd2-fake-warmup-steps` initial fake-only steps, call `compute_dmd2_generator_loss(...)`, backprop, step generator optimizer.
5. Checkpoint writes both adapters: `lora_gen_last.pt`, `lora_fake_last.pt` (matches existing `<adapter>_last.pt` convention). Resume uses `--resume-lora` for the generator (back-compat) and `--resume-dmd2-fake-lora` for the fake-score.
6. Sample / preview path (`--sample-dir`): runs `DMD2Sampler` with `args.dmd2_steps`, prints quality so progress is visible.

#### `voyager/diffusion/__init__.py`

Add: `from .dmd2 import DMD2Config, DMD2Sampler`.

#### `voyager/inference.py`

`HunyuanVideoSampler.predict` gains an optional `dmd2_steps: int = 0` path. When non-zero, it instantiates `DMD2Sampler` and replaces the `HunyuanVideoPipeline` denoising loop. Activation: pass `--dmd2-steps 4` to `sample_image2video.py`; we also wire `--lora-path` (generator LoRA) and `--lora-adapter-name gen` so the same LoRA file produced by training is loaded.

`load_lora_for_pipeline` in `voyager/utils/lora_utils.py` is updated to accept `adapter_name`.

#### `voyager/diffusion/flow/transport.py`

Two small additions, no changes to existing behavior:

- `Transport.alpha_sigma(t)` — public helper returning `(α_t, σ_t)` for the current path. Used by `dmd2_loss.py` so it doesn't reimplement the path math.
- `Transport.compute_flow_target(x0, x1, t)` — public helper returning the velocity target the loss is computed against; used by `compute_fake_score_loss`.

---

## 4. Pseudocode — full DMD2 step

```python
# Once at startup
apply_lora_to_hunyuan_video(model, r=args.lora_rank, adapter_name="gen")
apply_lora_to_hunyuan_video(model, r=args.dmd2_fake_lora_rank or args.lora_rank,
                            adapter_name="fake")

gen_opt  = build_optimizer(get_lora_parameters(model, "gen"),  args)
fake_opt = build_optimizer(get_lora_parameters(model, "fake"),
                           args, lr_mult=args.dmd2_fake_lr_mult)

t_grid = uniform_timestep_grid(args.dmd2_steps).to(device)   # e.g. [1, .75, .5, .25]

for step, batch in enumerate(loader):
    x1, conds = prepare_model_inputs(batch)        # existing path

    # --- K_fake fake-score updates -----------------------------------
    for _ in range(args.dmd2_fake_updates_per_gen):
        t_i = t_grid[torch.randint(len(t_grid), (B,))]
        with torch.no_grad():
            set_active_lora_adapter(model, "gen")
            x_t  = make_xt_for_generator(noise, x1, t_i)
            v    = model(x_t, t_i, conds)
            x_hat_0 = compute_x_hat_0_from_velocity(x_t, v, t_i)

        L_fake = compute_fake_score_loss(
            model, transport, x_hat_0=x_hat_0, conds=conds,
            t_p_range=(args.dmd2_min_tp, args.dmd2_max_tp),
            adapter_name="fake",
        )
        fake_opt.zero_grad(); L_fake.backward(); fake_opt.step()

    if step < args.dmd2_fake_warmup_steps:
        continue  # skip generator updates until fake has warmed up

    # --- 1 generator update ------------------------------------------
    t_i = t_grid[torch.randint(len(t_grid), (B,))]
    set_active_lora_adapter(model, "gen")
    x_t  = make_xt_for_generator(noise, x1, t_i)
    v    = model(x_t, t_i, conds)
    x_hat_0 = compute_x_hat_0_from_velocity(x_t, v, t_i)

    L_dmd = compute_dmd2_generator_loss(
        model, transport, x_hat_0=x_hat_0, conds=conds,
        weight_mode=args.dmd2_weight_mode,
        t_p_range=(args.dmd2_min_tp, args.dmd2_max_tp),
        real_adapter=None, fake_adapter="fake",
        cfg_scale_real=args.dmd2_cfg_scale_real,
    )
    gen_opt.zero_grad(); L_dmd.backward(); gen_opt.step()

    if step % args.ckpt_every == 0:
        save_lora_checkpoint(model, "gen",  out_dir / "lora_gen_last.pt")
        save_lora_checkpoint(model, "fake", out_dir / "lora_fake_last.pt")
```

`make_xt_for_generator(noise, x1, t)`:

- At `t == 1.0`: returns pure noise (no data leakage — this is the standard "noise → 4-step" path the deployed model will see).
- At `t < 1.0`: returns `α_t · x1 + σ_t · noise` (so the generator also learns to denoise from partially-noised real data, which stabilizes training; this is consistent with DMD2's multi-step variant).

---

## 5. Training launch script

`dmd2_train.sh` (new, modeled on the existing `deepspeed_train.sh`):

```bash
#!/usr/bin/env bash
export TMPDIR=/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
export DS_SKIP_CUDA_CHECK=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

deepspeed --include localhost:3 --master_port=29506 deepspeed_train_render.py \
    --dataset-root /raid/hvtham/ldnhuan/data/RealEstate10K \
    --output-dir training_outputs \
    --sample-dir sample_outputs \
    --task-flag dmd2_4step_lora \
    --global-batch-size 1 \
    --width 768 --height 512 \
    --use-cpu-offload \
    --epochs 3000 --save-every 32 --backup-every 5000 \
    --flow-reverse --flow-shift 7.0 \
    --lr 1e-4 \
    --model HYVideo-T/2 --model-base ckpts \
    --gradient-checkpoint \
    --num-workers-render 8 --num-workers 4 \
    --use-cache-text-encoder \
    --use-model-input-cache \
    --model-input-cache-name model_input_refined_train_10pct_768x512 \
    --train-lora \
    --lora-rank 64 \
    --dmd2-steps 4 \
    --dmd2-fake-updates-per-gen 5 \
    --dmd2-fake-warmup-steps 200 \
    --dmd2-fake-lora-rank 64 \
    --dmd2-weight-mode normalized \
    --dmd2-min-tp 0.02 --dmd2-max-tp 0.98 \
    # --resume-lora training_outputs/run_XXXXX/lora_gen_last.pt \
    # --resume-dmd2-fake-lora training_outputs/run_XXXXX/lora_fake_last.pt
```

Inference with the resulting 4-step LoRA (existing `sample_image2video.py`, with two new flags wired through):

```bash
python3 sample_image2video.py \
    --model HYVideo-T/2 \
    --input-path "examples/case1" \
    --prompt "..." \
    --i2v-stability \
    --flow-reverse --flow-shift 7.0 \
    --seed 0 --embedded-cfg-scale 6.0 \
    --use-cpu-offload --save-path ./results \
    --model-base ckpts \
    --use-lora --lora-path training_outputs/run_XXXXX/lora_gen_last.pt \
    --dmd2-steps 4
```

(`--infer-steps` is ignored when `--dmd2-steps > 0`.)

---

## 6. Open decisions recorded

| Decision | Choice | Rationale |
|---|---|---|
| Single base, two LoRAs vs. two full DiT copies | **Two LoRAs on one frozen base.** | User-confirmed. Cuts VRAM roughly in half; fits the rigs targeted by `deepspeed_train.sh`. |
| GAN head | **Skipped in v1.** | User-confirmed. Add behind `--dmd2-use-gan` later if needed. |
| Timestep grid | **Uniform, no shift.** `t_i = (N − i) / N`. | User-confirmed. Decoupled from teacher's `--flow-shift`. The teacher only contributes via `v_real`. |
| Two optimizers on one DeepSpeed engine | Use one engine; swap `model_engine.optimizer` between gen / fake steps and call `.zero_grad / .step` accordingly. | Avoids running two full DeepSpeed engines on the same parameters (which DeepSpeed doesn't support cleanly). The base weights are frozen, so there's no contention. |
| Fake-score warmup | 200 fake-only steps before first gen update. | Without warmup `v_fake ≈ v_real` and the DMD gradient is ~0. 200 is a common heuristic; exposed as a flag. |
| Perturbation timestep `t_p` | Sampled uniformly in `[min_tp, max_tp]`, default `[0.02, 0.98]`. | Avoid degenerate endpoints where `α_t` or `σ_t` → 0. Independent of `t_i`. |

## 7. Validation plan (before declaring the implementation done)

1. **LoRA multi-adapter unit test**: same `LoRALinear` with two adapters returns three distinct outputs for `set_active_adapter("a") / "b" / None`. Disabling produces bit-exact base output.
2. **Algorithmic sanity**: at init both LoRAs are identity-like (`B = 0`), so `v_fake == v_real == v_base` and `L_dmd ≈ 0`. After warmup, `L_dmd` should drift positive then trend down across many gen steps.
3. **Resume**: stop and resume training; verify generator/fake checkpoints round-trip and the loss trajectory is continuous.
4. **End-to-end smoke**: run inference with `--dmd2-steps 4` against the trained generator LoRA, eyeball a sample on `examples/case1`, then compare to 50-step base on the same input.
5. **Ablation hooks**: `--dmd2-weight-mode uniform` and varying `--dmd2-fake-updates-per-gen` ∈ {1, 5, 10} to confirm sensitivity matches the paper's claims.
