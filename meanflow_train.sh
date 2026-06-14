export TMPDIR=/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/tmp
export TMP=/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/tmp
export TEMP=/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/tmp
export DS_SKIP_CUDA_CHECK=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# MeanFlow distillation baseline (average-velocity, JVP target; no teacher/DMD).
# The student conditions on two times u_theta(z, r, t) via a zero-init r_in embedder
# attached on top of the base model, and is fine-tuned with a LoRA adapter.
# --flow-reverse is REQUIRED (rectified-flow path convention). MeanFlow needs
# i2v latent_concat mode (the default). Recommended: validate the JVP primal first
#   MODEL_BASE=ckpts python -m voyager.modules.jvp.check_primal_parity \
#       --model HYVideo-T/2 --vae 884-16c-hy \
#       --i2v-condition-type latent_concat --embedded-cfg-scale 6.0 --flow-reverse
deepspeed --include localhost:3 --master_port=29506 deepspeed_train_render.py \
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
    --train-meanflow \
    --meanflow-flow-ratio 0.5 \
    --meanflow-loss-c 1e-3 \
    --meanflow-loss-p 1.0 \
    --train-lora \
    --lora-rank 640 \
    # Resume (per adapter): MeanFlow r_in + the LoRA it rides on.
    # --resume-meanflow training_outputs/run_NNNNN/meanflow_last.pt \
    # --resume-lora training_outputs/run_NNNNN/lora_last.pt \
    # Alternative: full fine-tune instead of LoRA (heavier):
    # --train-from-scratch
