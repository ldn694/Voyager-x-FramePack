#!/usr/bin/env bash
# MeanFlow few-step inference: loads the trained r_in (+ optional LoRA) and samples
# in N steps via z_r = z_t - (t-r)*u_theta(z_t, r, t).
#
# Edit the variables below before running:
#   MEANFLOW_PATH  – path to meanflow_last.pt from a --train-meanflow run
#   LORA_PATH      – path to lora_last.pt from the same run (omit the two --use-lora
#                    / --lora-path lines below if the student was full-fine-tuned)
#   INPUT_PATH     – case folder with rgb/, depth/, mask/ subfolders
#   MEANFLOW_STEPS – number of sampling intervals (1 = single NFE)

CUDA_DEVICE=0
MEANFLOW_PATH="training_outputs/run_00000/meanflow_last.pt"
LORA_PATH="training_outputs/run_00000/lora_last.pt"
INPUT_PATH="/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/dataset/RealEstate10K/train_debug_768x512/0a2779d52af40db3/input"
MEANFLOW_STEPS=1
SAVE_PATH="./results_meanflow"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 sample_image2video_meanflow.py \
    --model HYVideo-T/2 \
    --model-base "ckpts" \
    --input-path "${INPUT_PATH}" \
    --prompt "a bedroom with a desk and chair in it" \
    --i2v-stability \
    --flow-reverse \
    --flow-shift 7.0 \
    --seed 0 \
    --use-cpu-offload \
    --video-size 512 768 \
    --video-length 49 \
    --embedded-cfg-scale 6.0 \
    --cfg-scale 1.0 \
    --save-path "${SAVE_PATH}" \
    --meanflow-steps ${MEANFLOW_STEPS} \
    --meanflow-path "${MEANFLOW_PATH}" \
    --use-lora \
    --lora-path "${LORA_PATH}"
