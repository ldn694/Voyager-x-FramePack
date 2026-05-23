export TMPDIR=/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/tmp
export TMP=/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/tmp
export TEMP=/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/tmp
export DS_SKIP_CUDA_CHECK=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
deepspeed --include localhost:3 --master_port=29505 deepspeed_train_render.py \
    --dataset-root /raid/hvtham/ldnhuan/data/RealEstate10K \
    --output-dir training_outputs \
    --sample-dir sample_outputs \
    --task-flag refined_train_10pct \
    --global-batch-size 1 \
    --width 768 \
    --height 512 \
    --use-cpu-offload \
    --epochs 3000 \
    --save-every 16 \
    --use-cache-text-encoder \
    --flow-reverse \
    --flow-shift 7.0 \
    --lr 1e-4 \
    --model HYVideo-T/2 \
    --gradient-checkpoint \
    --num-workers-render 8 \
    --num-workers 4 \
    --model-input-cache-name model_input_refined_train_10pct_768x512 \
    --use-model-input-cache \
    --backup-every 5000 \
    --model-base "ckpts" \
    --train-lora \
    --lora-rank 640 \
    --patch_adapter_size 1 4 4 \
    --resume-lora training_outputs/run_00159/lora_last.pt \
    --resume-patch-adapter training_outputs/run_00159/patch_adapter_last.pt \
    # --flow-predict-type velocity_flexidit \
    # --lora-rank 64 \
    # --train-multiple-kernels \
    # --kernel-sizes 1 2 2 \
    # --kernel-indices 0 6 12 \
    # --kernel-sizes 1 8 8 \
    # --kernel-indices 1 2 3 4 5 7 8 9 10 11 \
    # --use-double-branch \
    # --model-with-double-branch HYVideo-T/2-2branch-cross_attn-unidirectional-q_second
    # --train-from-scratch
    # --resume-double-branch training_outputs/run_00151/double_branch_last.pt \
    # --resume-multi-kernel training_outputs/run_00151/multi_kernel_last.pt \
    # --sample-time-range 0.36 1.0 \
    # --kernel-indices 0 3 6 9 12 \
    # --kernel-indices 1 2 4 5 7 8 10 11 \
    # --model-with-double-branch HYVideo-T/2-2branch-cross_attn
    # --use-gt-as-cond \
    # --resume-lora /raid/hvtham/ldnhuan/HunyuanWorld-Voyager/training_outputs/run_00089/lora_last.pt \
    # --resume-step-in-ckpt
    # --early-stop-training-loss 0.002 \
    # --kernel-indices 0 6 12 \
    # --resume /raid/hvtham/ldnhuan/HunyuanWorld-Voyager/training_outputs/run_00075/model_HYVideo-S_2_last.pt \
    # --cache-model-input \
    # --kernel-sizes 1 8 8 \
    # --kernel-indices 1 2 4 5 7 8 10 11 \
    # --kernel-size 1 4 4 \
    # --patch_adapter_size 1 8 8 \
