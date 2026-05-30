#!/bin/bash
#
# DMD2 evaluation on RealEstate10K (auto-restart on OOM).
# Wraps sample_image2video_realestate10K_dmd2.py: loads the gen LoRA from a DMD2
# training run and samples in a few steps, then computes metrics over the dataset.
#
# Edit the variables below before running:
#   CUDA_DEVICE  – GPU id
#   LORA_PATH    – path to lora_gen_last.pt (or a periodic lora_gen_step_XXXXX.pt)
#   DMD2_STEPS   – must match --dmd2-steps used at training time
#   DATASET_PATH / OUTPUT_PATH – eval dataset and where to write results+metrics
# LoRA rank/alpha are read from inside the checkpoint; no need to specify them here.

CUDA_DEVICE=3

LORA_PATH="training_outputs/run_00008/lora_gen_last.pt"
DMD2_STEPS=10
DATASET_PATH=/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/dataset/RealEstate10K/refined_test_150_768x512
OUTPUT_PATH=/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/evaluation/RealEstate10K/test_dmd2_${DMD2_STEPS}-steps

# 1. Trap Ctrl+C (SIGINT) and exit the shell script completely
trap "echo -e '\n[!] Manual stop (Ctrl+C) detected. Exiting the loop...'; exit 1" SIGINT

while true; do
    echo "[*] Starting/Resuming the DMD2 image2video generation job..."

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 sample_image2video_realestate10K_dmd2.py \
        --model HYVideo-T/2 \
        --prompt "." \
        --i2v-stability \
        --infer-steps 4 \
        --flow-reverse \
        --flow-shift 7.0 \
        --seed 0 \
        --embedded-cfg-scale 1.0 \
        --cfg-scale 1.0 \
        --use-cpu-offload \
        --video-size 512 768 \
        --video-length 49 \
        --model-base "ckpts" \
        --save-path ./results_dmd2 \
        --dataset-path "${DATASET_PATH}" \
        --output-path "${OUTPUT_PATH}" \
        --use-lora \
        --lora-path "${LORA_PATH}" \
        --dmd2-steps ${DMD2_STEPS} \
        # --first-clean-frame \
    # Capture the exit code of the Python script
    EXIT_CODE=$?

    # 3. Check if the script finished perfectly
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[✓] Job finished successfully!"
        break
    fi

    # 4. If the exit code is 130 (Ctrl+C) or 143 (SIGTERM), stop the loop
    if [ $EXIT_CODE -eq 130 ] || [ $EXIT_CODE -eq 143 ]; then
        echo "[!] Manual exit confirmed."
        exit 0
    fi

    # 5. Handle OOM crash
    echo "[x] Script crashed with exit code $EXIT_CODE (likely OOM). Restarting in 30 seconds..."
    sleep 30
done
