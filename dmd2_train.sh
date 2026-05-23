#!/usr/bin/env bash
# DMD2 + LoRA distillation launcher. See docs/dmd2_lora_implementation.md.
#
# Two LoRA adapters are attached on top of the frozen base DiT:
#   "gen"  -> the few-step generator being distilled (this is what ships)
#   "fake" -> the fake-score critic (discarded after training)
#
# Constraints enforced by the code:
#   --flow-reverse                       (DMD2 formulas assume linear-reverse path)
#   --train-lora                         (gen/fake are both LoRA adapters)
#   gradient_accumulation_steps == 1     (set in your ds_config.json)

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
    --epochs 3000 \
    --save-every 32 \
    --backup-every 5000 \
    --flow-reverse \
    --flow-shift 7.0 \
    --lr 1e-4 \
    --model HYVideo-T/2 \
    --model-base ckpts \
    --gradient-checkpoint \
    --num-workers-render 8 \
    --num-workers 4 \
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
    --dmd2-min-tp 0.02 \
    --dmd2-max-tp 0.98 \
    # Resume:
    # --resume-lora training_outputs/run_XXXXX/lora_gen_last.pt \
    # --resume-dmd2-fake-lora training_outputs/run_XXXXX/lora_fake_last.pt
