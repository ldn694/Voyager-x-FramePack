CUDA_DEVICE=1
# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 sample_image2video.py \
#     --model HYVideo-T/2 \
#     --input-path "vggt_output" \
#     --prompt "." \
#     --i2v-stability \
#     --infer-steps 50 \
#     --flow-reverse \
#     --flow-shift 7.0 \
#     --seed 0 \
#     --embedded-cfg-scale 6.0 \
#     --use-cpu-offload \
#     --video-size 256 384 \
#     --save-path ./results_vggt

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 sample_image2video.py \
    --model HYVideo-T/2 \
    --input-path "/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/dataset/RealEstate10K/train_debug_768x512/0a2779d52af40db3/input" \
    --prompt "a bedroom with a desk and chair in it" \
    --i2v-stability \
    --infer-steps 50 \
    --flow-shift 7.0 \
    --seed 0 \
    --use-cpu-offload \
    --video-size 512 768 \
    --save-path ./results_folders \
    --video-length 49 \
    --flow-reverse \
    --embedded-cfg-scale 1.0 \
    --cfg-scale 1.0 \
    --model-base "ckpts" \
    --use-lora \
    --lora-path "/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/training_outputs/run_00157/lora_last.pt" \
    --use-double-branch \
    --double-branch-path /raid/hvtham/ldnhuan/HunyuanWorld-Voyager/training_outputs/run_00157/double_branch_last.pt \
    --use-multiple-kernels \
    --multiple-kernels-path /raid/hvtham/ldnhuan/HunyuanWorld-Voyager/training_outputs/run_00157/multi_kernel_last.pt \
    --use-kernel-indices 0 6 12 \
    --use-kernel-indices 1 2 3 4 5 7 8 9 10 11 \
    --ratio 0.6 \
    --model-with-double-branch HYVideo-T/2-2branch-cross_attn-unidirectional-q_second
    # --model-with-double-branch HYVideo-T/2-2branch-no_cross_attn
    # --use-patch-adapter \
    # --patch-adapter-path "/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/training_outputs/run_00143/patch_adapter_last.pt" \
    # --model HYVideo-T/2 \
    # --input-path "/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/dataset/RealEstate10K/refined_test_150_768x512/5c69737728c7645f/input" \
    # --step-sample 10
    # --ratio 0.0 \
    # --step-sample 10
    # --model-base /raid/hvtham/ldnhuan/HunyuanWorld-Voyager/training_outputs/run_00156/model_HYVideo-B_2_last.pt \

    # --mode-scheduler-name "alternate_scheduler" \
    # --start 0 \
    # --end 30 \
    # --step-interval 1 \
    # --video-size 128 192 \
    # --video-size 512 768 \
    # --load-all \

    # --input-path "/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/dataset/RealEstate10K/train_debug/0a2779d52af40db3/input" \
    # --prompt "a bedroom with a desk and chair in it" \
    # --use-patch-adapter \
    # --patch-adapter-path "/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/training_outputs/run_00025/patch_adapter_last.pt" \
    # --use-kernel-indices 3 9 \
    # --use-kernel-indices 1 2 4 5 7 8 10 11 \
    # --input-path "/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/dataset/RealEstate10K/refined_test_150_768x512/8e143c65e5b4d26a/input" \
    