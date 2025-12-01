from loguru import logger
from pathlib import Path
from tqdm import tqdm
import json
import os

import torch
from torch.utils.data import DataLoader

from dataset.RealEstate10K import RealEstate10K
from voyager.config import *
from voyager.utils.train_utils import set_reproducibility, numpy_to_pil
from voyager.inference import load_models
from voyager.constants import PRECISION_TO_TYPE

def parse_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=str, required=True, help='Path to RealEstate10K dataset root')
    parser.add_argument('--width', type=int, default=256, help='Width of images to load')
    parser.add_argument('--height', type=int, default=384, help='Height of images to load')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser = add_inference_args(parser)
    parser = add_training_args(parser)
    parser = add_optimizer_args(parser)
    parser = add_deepspeed_args(parser)
    parser = add_data_args(parser)
    parser = add_train_denoise_schedule_args(parser)
    parser = add_network_args(parser)
    parser = add_i2v_args(parser)
    parser = add_extra_models_args(parser)
    parser = add_denoise_schedule_args(parser)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arg()
    print(args)
    dtype = PRECISION_TO_TYPE[args.precision]
    set_reproducibility(True, args.global_seed)
    dataset_root = args.dataset_root
    dataset = RealEstate10K(dataset_root, set_name=args.task_flag, width=args.width, height=args.height, return_inverse_depth=True)
    dataloader = DataLoader(dataset, batch_size=args.global_batch_size[0], shuffle=True, num_workers=args.num_workers)
    logger.info("Building model...")
    model, vae, text_encoder, text_encoder_2, vae_kwargs = \
        load_models(args, args.device, logger, Path(args.model_base))
    text_encoder.eval()
    text_encoder_2.eval()
    print(f"{text_encoder.device=}")
    print(f"{text_encoder_2.device=}")

    os.makedirs(args.output_dir, exist_ok=True)
    
    llm_i2v_text_state_path = Path(args.output_dir) / "llm_i2v_text_states.pt"
    llm_i2v_text_mask_path = Path(args.output_dir) / "llm_i2v_text_mask.pt"

    clipl_text_state_path = Path(args.output_dir) / "clipl_text_states.pt"

    json_path = Path(args.output_dir) / "sample_id.json"

    all_llm_i2v_text_states = []
    all_llm_i2v_text_masks = []
    all_clipl_text_states = []
    sample_id_dict = {}
    for batch_idx, data in tqdm(enumerate(dataloader), desc="Iterating over dataset"):
        rgbs = data['rgb']  # [B, 3, T, H, W]
        prompt = data['prompt']  # [str] of length B
        sample_id = data['sample_id']  # [str] of length B
        # print(prompt)
        text_inputs_1 = text_encoder.text2tokens(prompt)
        text_ids_1 = text_inputs_1['input_ids']
        text_mask_1 = text_inputs_1['attention_mask']
        text_inputs_2 = text_encoder_2.text2tokens(prompt)
        text_ids_2 = text_inputs_2['input_ids']
        text_mask_2 = text_inputs_2['attention_mask']
        with torch.no_grad():
            text_outputs = text_encoder.encode(
                {"input_ids": text_ids_1, "attention_mask": text_mask_1},
                data_type="video",
                semantic_images=numpy_to_pil(rgbs[:, :, 0, ...].permute(0, 2, 3, 1).cpu().numpy()),
            )
            text_states = text_outputs.hidden_state
            text_mask = text_outputs.attention_mask

            text_states_2 = (
                text_encoder_2.encode(
                    {"input_ids": text_ids_2, "attention_mask": text_mask_2},
                    data_type="video",
                ).hidden_state
                if text_encoder_2 is not None
                else None
            )
        for i in range(len(prompt)):
            all_llm_i2v_text_states.append(text_states[i].cpu())
            all_llm_i2v_text_masks.append(text_mask[i].cpu())
            all_clipl_text_states.append(text_states_2[i].cpu())
            sample_id_dict[sample_id[i]] = {
                "row": len(all_clipl_text_states)-1,
                "text_mask_len": text_mask[i].shape[0],
            }
    
    max_text_len = max([mask.shape[0] for mask in all_llm_i2v_text_masks])
    for i in range(len(all_llm_i2v_text_masks)):
        mask = all_llm_i2v_text_masks[i]
        pad_len = max_text_len - mask.shape[0]
        if pad_len > 0:
            pad_mask = torch.zeros((pad_len,), dtype=mask.dtype)
            all_llm_i2v_text_masks[i] = torch.cat([mask, pad_mask], dim=0)
            pad_state = torch.zeros((pad_len, all_llm_i2v_text_states[i].shape[1]), dtype=all_llm_i2v_text_states[i].dtype)
            all_llm_i2v_text_states[i] = torch.cat([all_llm_i2v_text_states[i], pad_state], dim=0)
    all_llm_i2v_text_states = torch.stack(all_llm_i2v_text_states, dim=0)
    all_llm_i2v_text_masks = torch.stack(all_llm_i2v_text_masks, dim=0)
    all_clipl_text_states = torch.stack(all_clipl_text_states, dim=0)

    torch.save(all_llm_i2v_text_states, llm_i2v_text_state_path)
    torch.save(all_llm_i2v_text_masks, llm_i2v_text_mask_path)
    torch.save(all_clipl_text_states, clipl_text_state_path)
    
    with open(json_path, 'w') as f:
        json.dump(sample_id_dict, f)

