#!/bin/bash

CUDA_DEVICE=3

# 1. Trap Ctrl+C (SIGINT) and exit the shell script completely
trap "echo -e '\n[!] Manual stop (Ctrl+C) detected. Exiting the loop...'; exit 1" SIGINT

while true; do
    echo "[*] Starting/Resuming the Hunyuan image2video generation job..."

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 sample_image2video_realestate10K.py \
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
        --model-base "ckpts" \
        --save-path ./results \
        --dataset-path /raid/hvtham/ldnhuan/HunyuanWorld-Voyager/dataset/RealEstate10K/refined_test_150_768x512 \
        --output-path /raid/hvtham/ldnhuan/HunyuanWorld-Voyager/evaluation/RealEstate10K/test_dual-branch_2x2_rerun_50-steps_ratio-60 \
        --use-lora \
        --lora-path "training_outputs/run_00160/lora_last.pt" \
        --use-double-branch \
        --double-branch-path training_outputs/run_00160/double_branch_last.pt \
        --use-multiple-kernels \
        --multiple-kernels-path training_outputs/run_00160/multi_kernel_last.pt \
        --use-kernel-indices 0 6 12 \
        --use-kernel-indices 1 2 3 4 5 7 8 9 10 11 \
        --ratio 0.6 \
        # --use-patch-adapter \
        # --patch-adapter-path "training_outputs/run_00154/patch_adapter_last.pt" \
        # --model-with-double-branch HYVideo-T/2-2branch-no_cross_attn
        # --model-with-double-branch HYVideo-T/2-2branch-cross_attn-unidirectional-q_second
        # --use-context-block
        # --use-context-block
        # --use-context-block \ 
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

# python3 sample_image2video.py \
#     --model HYVideo-T/2 \
#     --input-path "examples/case1" \
#     --prompt "An old-fashioned European village with thatched roofs on the houses." \
#     --i2v-stability \
#     --infer-steps 50 \
#     --flow-reverse \
#     --flow-shift 7.0 \
#     --seed 0 \
#     --embedded-cfg-scale 6.0 \
#     --use-cpu-offload \
#     --save-path ./results


