#!/bin/bash
#
# MeanFlow evaluation on RealEstate10K (auto-restart on OOM).
# Wraps sample_image2video_realestate10K_meanflow.py: loads the trained r_in
# (meanflow_last.pt) plus the LoRA it rides on, samples in N few-step intervals via
# z_r = z_t - (t-r)*u_theta(z_t, r, t), then computes metrics over the dataset.
#
# Edit the variables below before running:
#   CUDA_DEVICE     – GPU id
#   MEANFLOW_PATH   – path to meanflow_last.pt (or a periodic meanflow_step_XXXXX.pt)
#   LORA_PATH       – path to lora_last.pt from the SAME run (omit the two --use-lora
#                     / --lora-path lines below if the student was full-fine-tuned)
#   MEANFLOW_STEPS  – number of sampling intervals (1 = single NFE)
#   DATASET_PATH / OUTPUT_PATH – eval dataset and where to write results+metrics
# LoRA rank/alpha are read from inside the checkpoint; no need to specify them here.

CUDA_DEVICE=3

MEANFLOW_PATH="training_outputs/run_00000/meanflow_last.pt"
LORA_PATH="training_outputs/run_00000/lora_last.pt"
MEANFLOW_STEPS=1
DATASET_PATH=/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/dataset/RealEstate10K/refined_test_150_768x512
OUTPUT_PATH=/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/evaluation/RealEstate10K/test_meanflow_${MEANFLOW_STEPS}-steps

# Persist Triton's compiled-kernel cache so the JVP attention autotune (48-config
# compile, ~49 min cold for a new shape) is paid once per machine, not per run.
# Shared with meanflow_sample.sh / meanflow_train.sh / deepspeed_train.sh.
export TRITON_CACHE_DIR=/raid/hvtham/ldnhuan/.triton_cache

# 1. Trap Ctrl+C (SIGINT) and exit the shell script completely
trap "echo -e '\n[!] Manual stop (Ctrl+C) detected. Exiting the loop...'; exit 1" SIGINT

while true; do
    echo "[*] Starting/Resuming the MeanFlow image2video generation job..."

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 sample_image2video_realestate10K_meanflow.py \
        --model HYVideo-T/2 \
        --prompt "." \
        --i2v-stability \
        --infer-steps 50 \
        --flow-reverse \
        --flow-shift 7.0 \
        --seed 0 \
        --embedded-cfg-scale 1.0 \
        --cfg-scale 1.0 \
        --use-cpu-offload \
        --video-size 512 768 \
        --video-length 49 \
        --model-base "ckpts" \
        --save-path ./results_meanflow \
        --dataset-path "${DATASET_PATH}" \
        --output-path "${OUTPUT_PATH}" \
        --meanflow-steps ${MEANFLOW_STEPS} \
        --meanflow-path "${MEANFLOW_PATH}" \
        --use-lora \
        --lora-path "${LORA_PATH}" \
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
