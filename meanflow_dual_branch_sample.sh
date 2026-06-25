#!/usr/bin/env bash
# Few-step MeanFlow inference for the DUAL-BRANCH student.
#
# Same as meanflow_sample.sh, but additionally rebuilds the multi-kernel +
# double-branch backbone (loaded by HunyuanVideoSampler.from_pretrained) so the
# trained dual-branch student is reconstructed before sampling. The sampler runs
# the plain dual-branch model.forward (no JVP — JVP is training-only); the r-time
# is injected via extra_vec=r_in(r) on the main branch and
# extra_vec_second=r_in_second_branch(r) on the second branch.
#
# Edit the variables below before running:
#   MEANFLOW_PATH        – meanflow_last.pt from the dual-branch --train-meanflow run
#   LORA_PATH            – lora_last.pt from the same run (drop the --use-lora /
#                          --lora-path lines if the student was full-fine-tuned)
#   MULTI_KERNEL_PATH    – multi_kernel_last.pt of the dual-branch teacher/student
#   DOUBLE_BRANCH_PATH   – double_branch_last.pt of the same model
#   --use-kernel-indices – MUST match the kernel-index groups the model was trained
#                          with (one group per kernel; kernel SIZES are read from the
#                          multi_kernel ckpt's args, no flag needed).
#   INPUT_PATH           – case folder with rgb/, depth/, mask/ subfolders

CUDA_DEVICE=0
MEANFLOW_PATH="training_outputs/run_00000/meanflow_last.pt"
LORA_PATH="training_outputs/run_00000/lora_last.pt"
MULTI_KERNEL_PATH="training_outputs/run_00000/multi_kernel_last.pt"
DOUBLE_BRANCH_PATH="training_outputs/run_00000/double_branch_last.pt"
INPUT_PATH="/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/dataset/RealEstate10K/train_debug_768x512/0a2779d52af40db3/input"
MEANFLOW_STEPS=1
SAVE_PATH="./results_meanflow_dual_branch"

# Persist Triton's compiled-kernel cache (shared with deepspeed_train.sh). At
# inference the dual-branch path uses plain flash-attn, not the JVP kernel, so the
# ~49 min cold JVP autotune is not paid here — kept only for parity with training.
export TRITON_CACHE_DIR=/raid/hvtham/ldnhuan/.triton_cache

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
    --lora-path "${LORA_PATH}" \
    --use-double-branch \
    --double-branch-path "${DOUBLE_BRANCH_PATH}" \
    --model-with-double-branch HYVideo-T/2-2branch-cross_attn \
    --use-multiple-kernels \
    --multiple-kernels-path "${MULTI_KERNEL_PATH}" \
    --use-kernel-indices 0 6 12 \
    --use-kernel-indices 1 2 3 4 5 7 8 9 10 11
