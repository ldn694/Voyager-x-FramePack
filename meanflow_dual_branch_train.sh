export TMPDIR=/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/tmp
export TMP=/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/tmp
export TEMP=/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/tmp
export DS_SKIP_CUDA_CHECK=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Persist Triton's compiled-kernel cache so the JVP attention autotune is paid once.
export TRITON_CACHE_DIR=/raid/hvtham/ldnhuan/.triton_cache
# MeanFlow distillation on the DUAL-BRANCH model (multi-kernel patchify + second
# branch + cross-attention). The JVP `dudt` is threaded through the FIRST branch only
# in --meanflow-db-tangent frozen (Option B): the second branch + cross-attention give
# an exact primal but a frozen (zero) tangent. Validate the dual-branch primal first:
#   MODEL_BASE=ckpts python -m voyager.modules.jvp.check_primal_parity_double_branch \
#       --model HYVideo-T/2 --vae 884-16c-hy --i2v-condition-type latent_concat \
#       --embedded-cfg-scale 6.0 --flow-reverse --train-multiple-kernels \
#       --kernel-sizes 1 2 2 --kernel-sizes 1 4 4 \
#       --kernel-indices 0 --kernel-indices 1 2 3 4 5 6 7 8 9 10 11 12 \
#       --use-double-branch --model-with-double-branch HYVideo-T/2-2branch-cross_attn
deepspeed --include localhost:3 --master_port=29506 deepspeed_train_render.py \
    --dataset-root /raid/hvtham/ldnhuan/data/RealEstate10K \
    --output-dir training_outputs \
    --sample-dir sample_outputs \
    --task-flag meanflow_dual_branch_frozen \
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
    --train-meanflow \
    --meanflow-flow-ratio 0.5 \
    --meanflow-loss-c 1e-3 \
    --meanflow-loss-p 1.0 \
    --meanflow-db-tangent frozen \
    --train-multiple-kernels \
    --kernel-sizes 1 2 2 --kernel-sizes 1 4 4 \
    --kernel-indices 0 --kernel-indices 1 2 3 4 5 6 7 8 9 10 11 12 \
    --use-double-branch --model-with-double-branch HYVideo-T/2-2branch-cross_attn \
    --train-lora \
    --lora-rank 640 \
    # ----- Resume the TRAINED dual-branch teacher (recommended for distillation) -----
    # Point these at your trained dual-branch run so MeanFlow distills the real model,
    # not a fresh-init second branch:
    # --resume-double-branch training_outputs/run_NNNNN/double_branch_last.pt \
    # --resume-multi-kernel  training_outputs/run_NNNNN/multi_kernel_last.pt \
    # ----- Resume the MeanFlow adapters themselves (r_in + r_in_second_branch + LoRA) -----
    # --resume-meanflow training_outputs/run_NNNNN/meanflow_last.pt \
    # --resume-lora     training_outputs/run_NNNNN/lora_last.pt \
    # ----- Option A (exact second-branch/cross-attn tangent) once WP1 lands: -----
    # --meanflow-db-tangent full
