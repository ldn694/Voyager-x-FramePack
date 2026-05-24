#!/usr/bin/env bash
# DMD2 inference: loads the gen LoRA from a training run and samples in N steps.
#
# Edit the variables below before running:
#   LORA_PATH  – path to lora_gen_last.pt (or a periodic lora_gen_step_XXXXX.pt)
#   INPUT_PATH – case folder with rgb/, depth/, mask/ subfolders
#   DMD2_STEPS – must match --dmd2-steps used at training time (default 4)
# LoRA rank/alpha are read directly from inside the checkpoint; no need to specify them here.

CUDA_DEVICE=1
LORA_PATH="training_outputs/run_00003/lora_gen_last.pt"
INPUT_PATH="examples/case1"
DMD2_STEPS=4
SAVE_PATH="./results_dmd2"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 sample_image2video.py \
    --model HYVideo-T/2 \
    --model-base "ckpts" \
    --input-path "${INPUT_PATH}" \
    --prompt "A smooth, forward-moving camera fly-through of a real-world indoor scene." \
    --i2v-stability \
    --infer-steps 50 \
    --flow-reverse \
    --flow-shift 7.0 \
    --seed 0 \
    --use-cpu-offload \
    --video-size 512 768 \
    --video-length 49 \
    --embedded-cfg-scale 1.0 \
    --cfg-scale 1.0 \
    --save-path "${SAVE_PATH}" \
    --use-lora \
    --lora-path "${LORA_PATH}" \
    --dmd2-steps ${DMD2_STEPS}
