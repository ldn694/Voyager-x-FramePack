from typing import Any, List, Tuple, Optional, Union, Dict
from einops import rearrange
from loguru import logger
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
import torch.utils
import torch.utils.checkpoint

from .activation_layers import get_activation_layer
from .norm_layers import get_norm_layer
from .embed_layers import TimestepEmbedder, PatchEmbed, TextProjection, MultiPatchEmbed
from .attenion import attention, parallel_attention, get_cu_seqlens
from .posemb_layers import apply_rotary_emb
from .mlp_layers import MLP, MLPEmbedder, FinalLayer, MultiFinalLayer
from .modulate_layers import ModulateDiT, modulate, apply_gate, ckpt_wrapper
from .token_refiner import SingleTokenRefiner
from ..utils.helpers import to_3tuple
from .transformer_branch_config import TransformerBranchConfig


class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal dit block with seperate modulation for
    text and image/video, see more details (SD3): https://arxiv.org/abs/2403.03206
                                     (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.img_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.img_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
        )
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True,
                          eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True,
                          eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.img_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.txt_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.txt_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
        )
        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True,
                          eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True,
                          eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.txt_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: tuple = None,
        condition_type: str = None,
        token_replace_vec: torch.Tensor = None,
        frist_frame_token_num: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if condition_type == "token_replace":
            img_mod1, token_replace_img_mod1 = self.img_mod(vec, condition_type=condition_type,
                                                            token_replace_vec=token_replace_vec)
            (img_mod1_shift,
             img_mod1_scale,
             img_mod1_gate,
             img_mod2_shift,
             img_mod2_scale,
             img_mod2_gate) = img_mod1.chunk(6, dim=-1)
            (tr_img_mod1_shift,
             tr_img_mod1_scale,
             tr_img_mod1_gate,
             tr_img_mod2_shift,
             tr_img_mod2_scale,
             tr_img_mod2_gate) = token_replace_img_mod1.chunk(6, dim=-1)
        else:
            (
                img_mod1_shift,
                img_mod1_scale,
                img_mod1_gate,
                img_mod2_shift,
                img_mod2_scale,
                img_mod2_gate,
            ) = self.img_mod(vec).chunk(6, dim=-1)

        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, dim=-1)

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        if condition_type == "token_replace":
            img_modulated = modulate(
                img_modulated, shift=img_mod1_shift, scale=img_mod1_scale, condition_type=condition_type,
                tr_shift=tr_img_mod1_shift, tr_scale=tr_img_mod1_scale,
                frist_frame_token_num=frist_frame_token_num
            )
        else:
            img_modulated = modulate(
                img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
            )
        img_qkv = self.img_attn_qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(
                img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
        )
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # Run actual attention.
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)
        assert (
            cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"

        # attention computation start
        if not self.hybrid_seq_parallel_attn:
            attn = attention(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=img_k.shape[0],
            )
        else:
            attn = parallel_attention(
                self.hybrid_seq_parallel_attn,
                q,
                k,
                v,
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv
            )

        # attention computation end

        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1]:]

        # Calculate the img bloks.
        if condition_type == "token_replace":
            img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate, condition_type=condition_type,
                                   tr_gate=tr_img_mod1_gate, frist_frame_token_num=frist_frame_token_num)
            img = img + apply_gate(
                self.img_mlp(
                    modulate(
                        self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale,
                        condition_type=condition_type, tr_shift=tr_img_mod2_shift,
                        tr_scale=tr_img_mod2_scale, frist_frame_token_num=frist_frame_token_num
                    )
                ),
                gate=img_mod2_gate, condition_type=condition_type,
                tr_gate=tr_img_mod2_gate, frist_frame_token_num=frist_frame_token_num
            )
        else:
            img = img + \
                apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
            img = img + apply_gate(
                self.img_mlp(
                    modulate(
                        self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                    )
                ),
                gate=img_mod2_gate,
            )

        # Calculate the txt bloks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn),
                               gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(
                modulate(
                    self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
                )
            ),
            gate=txt_mod2_gate,
        )

        return img, txt


class MMSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    Also refer to (SD3): https://arxiv.org/abs/2403.03206
                  (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        step_cfg: Optional[Tuple] = None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim ** -0.5

        # qkv and mlp_in
        self.linear1 = nn.Linear(
            hidden_size, hidden_size * 3 + mlp_hidden_dim, **factory_kwargs
        )
        # proj and mlp_out
        self.linear2 = nn.Linear(
            hidden_size + mlp_hidden_dim, hidden_size, **factory_kwargs
        )

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True,
                          eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True,
                          eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )

        self.pre_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(
            hidden_size,
            factor=3,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None
        self.step_cfg = step_cfg

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        condition_type: str = None,
        token_replace_vec: torch.Tensor = None,
        frist_frame_token_num: int = None,
    ) -> torch.Tensor:
        if condition_type == "token_replace":
            mod, tr_mod = self.modulation(vec,
                                          condition_type=condition_type,
                                          token_replace_vec=token_replace_vec)
            (mod_shift,
             mod_scale,
             mod_gate) = mod.chunk(3, dim=-1)
            (tr_mod_shift,
             tr_mod_scale,
             tr_mod_gate) = tr_mod.chunk(3, dim=-1)
        else:
            mod_shift, mod_scale, mod_gate = self.modulation(
                vec).chunk(3, dim=-1)
        if condition_type == "token_replace":
            x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale, condition_type=condition_type,
                             tr_shift=tr_mod_shift, tr_scale=tr_mod_scale, frist_frame_token_num=frist_frame_token_num)
        else:
            x_mod = modulate(self.pre_norm(
                x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D",
                            K=3, H=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            if txt_len > 0:
                img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
                img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            else:
                img_q, txt_q = q, None
                img_k, txt_k = k, None
            img_qq, img_kk = apply_rotary_emb(
                img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk
            

            # =================================================================
            # Compute image to image attention map (MEMORY EFFICIENT & FRAME-WISE):
            if hasattr(self, 'step_cfg') and self.step_cfg is not None:
                import matplotlib.pyplot as plt
                import os
                
                # Properly unpack the tuple from step_cfg
                layer_idx, current_step, stride = self.step_cfg
                
                # Only execute visualization on specific strides
                if layer_idx % stride == 0:
                    # Transpose Q and K to [Batch, Heads, L_img, HeadDim]
                    q_img_trans = img_q.transpose(1, 2)
                    k_img_trans = img_k.transpose(1, 2)
                    
                    B, Heads, L_img, HeadDim = q_img_trans.shape
                    
                    num_frames = 13
                    tokens_per_frame = L_img // num_frames  # Should calculate to 3264
                    chunk_size = 1024 
                    
                    # Allocate accumulator
                    frame_attn_accum = torch.zeros(
                        (B, Heads, L_img, num_frames), 
                        device=q_img_trans.device, 
                        dtype=q_img_trans.dtype
                    )
                    
                    # Ensure scaling factor exists (default is 1 / sqrt(dim))
                    scale = self.scale if hasattr(self, 'scale') else (1.0 / (HeadDim ** 0.5))
                    
                    for i in range(0, L_img, chunk_size):
                        end_i = min(i + chunk_size, L_img)
                        chunk_len = end_i - i
                        q_chunk = q_img_trans[:, :, i:end_i, :]  
                        
                        # Calculate scores and weights for the chunk
                        scores_chunk = torch.matmul(q_chunk, k_img_trans.transpose(-2, -1)) * scale
                        weights_chunk = torch.softmax(scores_chunk, dim=-1)
                        
                        # Group Key tokens and sum
                        weights_chunk_grouped = weights_chunk.view(B, Heads, chunk_len, num_frames, tokens_per_frame)
                        chunk_sum_keys = weights_chunk_grouped.sum(dim=-1)  
                        
                        frame_attn_accum[:, :, i:end_i, :] = chunk_sum_keys
                    
                    # Group Query tokens and take the mean
                    frame_attn_accum = frame_attn_accum.view(B, Heads, num_frames, tokens_per_frame, num_frames)
                    frame_wise_attn = frame_attn_accum.mean(dim=3).mean(dim=1)
                    
                    # Save outputs
                    file_basename = f"layer_{layer_idx:02d}_step_{current_step:04d}"
                    print(f"Saving frame-wise attention map: {file_basename} | Shape: {frame_wise_attn.shape}")
                    
                    # Define the base directory
                    base_dir = "/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/minh_attention_ckpt_full"
                    
                    # Create separate subfolders for 'tensor' and 'frame' inside the layer directory
                    save_dir_tensor = os.path.join(base_dir, f"layer_{layer_idx:02d}", "tensor")
                    save_dir_frame = os.path.join(base_dir, f"layer_{layer_idx:02d}", "frame")
                    
                    # Make sure both directories exist
                    os.makedirs(save_dir_tensor, exist_ok=True)
                    os.makedirs(save_dir_frame, exist_ok=True)
                    
                    # Save the raw tensor (.pt) into the 'tensor' folder
                    save_path_pt = os.path.join(save_dir_tensor, f"frame_wise_attn_{file_basename}.pt")
                    torch.save(frame_wise_attn.detach().cpu(), save_path_pt)
                    
                    # Generate and save the heatmap (.png) into the 'frame' folder
                    attn_numpy = frame_wise_attn[0].detach().cpu().float().numpy() 
                    
                    plt.figure(figsize=(8, 6))
                    plt.imshow(attn_numpy, cmap='viridis', aspect='auto')
                    plt.colorbar(label='Attention Probability')
                    plt.title(f"Frame-to-Frame Transition (Layer {layer_idx}, Step {current_step})")
                    plt.xlabel("Key Frame (Attended To)")
                    plt.ylabel("Query Frame (Attending From)")
                    plt.xticks(range(num_frames))
                    plt.yticks(range(num_frames))
                    
                    save_path_png = os.path.join(save_dir_frame, f"heatmap_{file_basename}.png")
                    plt.savefig(save_path_png, bbox_inches='tight', dpi=300)
                    plt.close()
            # =================================================================


            if txt_q is not None:
                q = torch.cat((img_q, txt_q), dim=1)
            else:
                q = img_q
            if txt_k is not None:
                k = torch.cat((img_k, txt_k), dim=1)
            else:
                k = img_k

        # Compute attention.
        assert (
            cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"

        # attention computation start
        if not self.hybrid_seq_parallel_attn:
            attn = attention(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=x.shape[0],
            )
        else:
            attn = parallel_attention(
                self.hybrid_seq_parallel_attn,
                q,
                k,
                v,
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv
            )
        # attention computation end

        # Compute activation in mlp stream, cat again and run second linear layer.
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))

        if condition_type == "token_replace":
            output = x + apply_gate(output, gate=mod_gate, condition_type=condition_type,
                                    tr_gate=tr_mod_gate, frist_frame_token_num=frist_frame_token_num)
            return output
        else:
            return x + apply_gate(output, gate=mod_gate)

class MMCrossStreamBlock(nn.Module):
    """
    Cross-attention block: update x_q by attending to x_kv

    Now supports different hidden sizes:
      - q_hidden_size for query stream (and attention/model width)
      - kv_hidden_size for key/value input stream

    Projections:
      - linear_q_mlp: q_hidden_size -> (q_hidden_size + mlp_hidden_dim)
      - linear_kv:    kv_hidden_size -> (2 * q_hidden_size)  # produces K,V in q-width
    """

    def __init__(
        self,
        q_hidden_size: int,
        kv_hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.q_hidden_size = q_hidden_size
        self.kv_hidden_size = kv_hidden_size
        self.heads_num = heads_num

        assert q_hidden_size % heads_num == 0, "q_hidden_size must be divisible by heads_num"
        head_dim = q_hidden_size // heads_num

        self.mlp_hidden_dim = int(q_hidden_size * mlp_width_ratio)

        # Query-side packed projection: Q + MLP-in
        self.linear_q_mlp = nn.Linear(
            q_hidden_size,
            q_hidden_size + self.mlp_hidden_dim,
            bias=True,
            **factory_kwargs,
        )

        # KV-side projection: K,V (project KV stream into q-width)
        self.linear_kv = nn.Linear(
            kv_hidden_size,
            2 * q_hidden_size,
            bias=qkv_bias,
            **factory_kwargs,
        )

        # Output projection from (attn_out + mlp_out) -> q_hidden_size
        self.linear_out = nn.Linear(
            q_hidden_size + self.mlp_hidden_dim,
            q_hidden_size,
            bias=True,
            **factory_kwargs,
        )

        # QK norm (operates on head_dim, after reshaping to [B, L, H, head_dim])
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )

        # Pre norms (match each stream width)
        self.pre_norm_q = nn.LayerNorm(
            q_hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.pre_norm_kv = nn.LayerNorm(
            kv_hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.mlp_act = get_activation_layer(mlp_act_type)()

        # Modulation:
        # - query: shift, scale, gate (q_hidden_size)
        self.mod_q = ModulateDiT(
            q_hidden_size, factor=3, act_layer=get_activation_layer("silu"), **factory_kwargs
        )
        # - kv: shift, scale (kv_hidden_size) since it modulates x_kv before projection
        self.mod_kv = ModulateDiT(
            kv_hidden_size, factor=2, act_layer=get_activation_layer("silu"), **factory_kwargs
        )

        self.hybrid_seq_parallel_attn = None

    def forward(
        self,
        x_q: torch.Tensor,      # [B, Lq, q_hidden_size]
        x_kv: torch.Tensor,     # [B, Lkv, kv_hidden_size]
        vec_q: torch.Tensor,
        vec_kv: torch.Tensor,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        freqs_cis_q=None,
        freqs_cis_kv=None,
        condition_type: str = None,
        token_replace_vec: torch.Tensor = None,
        frist_frame_token_num: int = None,
    ) -> torch.Tensor:

        # ------------------ get modulation params ------------------
        if condition_type == "token_replace":
            q_mod, tr_q_mod = self.mod_q(
                vec_q, condition_type=condition_type, token_replace_vec=token_replace_vec
            )
            q_shift, q_scale, q_gate = q_mod.chunk(3, dim=-1)
            tr_q_shift, tr_q_scale, tr_q_gate = tr_q_mod.chunk(3, dim=-1)

            kv_mod, tr_kv_mod = self.mod_kv(
                vec_kv, condition_type=condition_type, token_replace_vec=token_replace_vec
            )
            kv_shift, kv_scale = kv_mod.chunk(2, dim=-1)
            tr_kv_shift, tr_kv_scale = tr_kv_mod.chunk(2, dim=-1)
        else:
            q_shift, q_scale, q_gate = self.mod_q(vec_q).chunk(3, dim=-1)
            kv_shift, kv_scale = self.mod_kv(vec_kv).chunk(2, dim=-1)

            tr_q_shift = tr_q_scale = tr_q_gate = None
            tr_kv_shift = tr_kv_scale = None

        # ------------------ apply modulation ------------------
        if condition_type == "token_replace":
            xq_mod = modulate(
                self.pre_norm_q(x_q),
                shift=q_shift,
                scale=q_scale,
                condition_type=condition_type,
                tr_shift=tr_q_shift,
                tr_scale=tr_q_scale,
                frist_frame_token_num=frist_frame_token_num,
            )
            xkv_mod = modulate(
                self.pre_norm_kv(x_kv),
                shift=kv_shift,
                scale=kv_scale,
                condition_type=condition_type,
                tr_shift=tr_kv_shift,
                tr_scale=tr_kv_scale,
                frist_frame_token_num=frist_frame_token_num,
            )
        else:
            xq_mod = modulate(self.pre_norm_q(x_q), shift=q_shift, scale=q_scale)
            xkv_mod = modulate(self.pre_norm_kv(x_kv), shift=kv_shift, scale=kv_scale)

        # ------------------ projections ------------------
        q_and_mlp = self.linear_q_mlp(xq_mod)
        q, mlp_in = torch.split(
            q_and_mlp, [self.q_hidden_size, self.mlp_hidden_dim], dim=-1
        )

        k, v = torch.split(self.linear_kv(xkv_mod), [self.q_hidden_size, self.q_hidden_size], dim=-1)

        q = rearrange(q, "B L (H D) -> B L H D", H=self.heads_num)
        k = rearrange(k, "B L (H D) -> B L H D", H=self.heads_num)
        v = rearrange(v, "B L (H D) -> B L H D", H=self.heads_num)

        # QK norm
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # ------------------ RoPE (optional) ------------------
        if freqs_cis_q is not None:
            q, _ = apply_rotary_emb(q, q, freqs_cis_q, head_first=False)
        if freqs_cis_kv is not None:
            k, _ = apply_rotary_emb(k, k, freqs_cis_kv, head_first=False)

        # ------------------ attention ------------------
        if self.hybrid_seq_parallel_attn is None:
            attn = attention(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=x_q.shape[0],
            )
        else:
            attn = parallel_attention(
                self.hybrid_seq_parallel_attn,
                q, k, v,
                img_q_len=q.shape[1],
                img_kv_len=k.shape[1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
            )

        # ------------------ output + gated residual ------------------
        out = self.linear_out(torch.cat((attn, self.mlp_act(mlp_in)), dim=2))

        if condition_type == "token_replace":
            return x_q + apply_gate(
                out,
                gate=q_gate,
                condition_type=condition_type,
                tr_gate=tr_q_gate,
                frist_frame_token_num=frist_frame_token_num,
            )
        return x_q + apply_gate(out, gate=q_gate)




class HYVideoDiffusionTransformer(ModelMixin, ConfigMixin):
    """
    HunyuanVideo Transformer backbone

    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.

    Reference:
    [1] Flux.1: https://github.com/black-forest-labs/flux
    [2] MMDiT: http://arxiv.org/abs/2403.03206

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    patch_size: list
        The size of the patch.
    in_channels: int
        The number of input channels.
    out_channels: int
        The number of output channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    heads_num: int
        The number of attention heads.
    mlp_width_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    mlp_act_type: str
        The activation function of the MLP in the transformer block.
    depth_double_blocks: int
        The number of transformer blocks in the double blocks.
    depth_single_blocks: int
        The number of transformer blocks in the single blocks.
    rope_dim_list: list
        The dimension of the rotary embedding for t, h, w.
    qkv_bias: bool
        Whether to use bias in the qkv linear layer.
    qk_norm: bool
        Whether to use qk norm.
    qk_norm_type: str
        The type of qk norm.
    guidance_embed: bool
        Whether to use guidance embedding for distillation.
    text_projection: str
        The type of the text projection, default is single_refiner.
    use_attention_mask: bool
        Whether to use attention mask for text encoder.
    dtype: torch.dtype
        The dtype of the model.
    device: torch.device
        The device of the model.
    """

    @register_to_config
    def __init__(
        self,
        args: Any,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,  # Should be VAE.config.latent_channels.
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,  # For modulation.
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        patch_sizes: Optional[List[List[int]]] = None,
        second_branch_mm_blocks_depth: int = 0,
        second_branch_transformer_config: Optional[TransformerBranchConfig] = None,
        step_cfg: Optional[Tuple] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.patch_size = patch_size
        self.patch_sizes = patch_sizes
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.i2v_condition_type = args.i2v_condition_type

        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection

        self.text_states_dim = args.text_states_dim
        self.text_states_dim_2 = args.text_states_dim_2

        # Gradient checkpoint.
        self.gradient_checkpoint = args.gradient_checkpoint
        self.gradient_checkpoint_layers = args.gradient_checkpoint_layers
        if self.gradient_checkpoint:
            assert self.gradient_checkpoint_layers <= mm_double_blocks_depth + mm_single_blocks_depth, \
                f"Gradient checkpoint layers must be less or equal than the depth of the model. " \
                f"Got gradient_checkpoint_layers={self.gradient_checkpoint_layers} and " \
                f"depth={mm_double_blocks_depth + mm_single_blocks_depth}."

        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(
                f"Got {rope_dim_list} but expected positional dim {pe_dim}"
            )
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        # image projection
        self.img_in = PatchEmbed(
            self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs
        )

        # multi image projection
        if patch_sizes is not None:
            self.img_in = MultiPatchEmbed(
                patch_sizes, self.in_channels, self.hidden_size, **factory_kwargs
            )

        # text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                self.text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                self.text_states_dim, hidden_size, heads_num, depth=2, **factory_kwargs
            )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        # time modulation
        self.time_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu"), **factory_kwargs
        )

        # text modulation
        self.vector_in = MLPEmbedder(
            self.text_states_dim_2, self.hidden_size, **factory_kwargs
        )

        # guidance modulation
        self.guidance_in = (
            TimestepEmbedder(
                self.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )
            if guidance_embed
            else None
        )

        # double blocks
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        # single blocks
        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    step_cfg = step_cfg,  #(layer, step_denoise, stride_to_save_attn_map)
                    **factory_kwargs,
                )
                for _ in range(mm_single_blocks_depth)
            ]
        )

        if second_branch_transformer_config is not None:
            if second_branch_transformer_config.scheduler is not None:
                self.cross_attn_blocks = nn.ModuleList()
                self.double_branch_scheduler = second_branch_transformer_config.scheduler
                #assert if scheduler is unique
                assert len(self.double_branch_scheduler) == len(set(self.double_branch_scheduler)), \
                    f"Scheduler {self.double_branch_scheduler} contains duplicate indices."
                prev_key_block = -1
                prev_non_key_block = -1
                for edge in self.double_branch_scheduler:
                    u, v = edge
                    # there are 2 categories, > 0 and <= 0
                    # u, v must be in differrent categories
                    # assert their category
                    assert (u > 0 and v <= 0) or (u <= 0 and v > 0), \
                        f"Scheduler {self.double_branch_scheduler} contains invalid edge ({u}, {v}). "\
                        f"u and v must be in different categories (>0 and <=0)."
                    assert abs(max(u, v)) < mm_double_blocks_depth + mm_single_blocks_depth, \
                        f"Scheduler {self.double_branch_scheduler} is invalid. "\
                        f"Block index of first branch: {max(u, v)} exceeds the total number of blocks {mm_double_blocks_depth + mm_single_blocks_depth}."
                    assert abs(min(u, v)) < second_branch_mm_blocks_depth, \
                        f"Scheduler {self.double_branch_scheduler} is invalid. "\
                        f"Block index of second branch: {min(u, v)} exceeds the total number of blocks {second_branch_mm_blocks_depth}."

                    if u > 0:
                        assert u >= prev_key_block and -v >= prev_non_key_block, \
                            f"Scheduler {self.double_branch_scheduler} is invalid. "\
                            f"Blocks must be in non decreasing order. Got u: {u}, prev_key_block: {prev_key_block}, v: {v}, prev_non_key_block: {prev_non_key_block}."
                        prev_key_block = u
                        prev_non_key_block = -v
                    else:
                        assert -u >= prev_non_key_block and v >= prev_key_block, \
                            f"Scheduler {self.double_branch_scheduler} is invalid. "\
                            f"Blocks must be in non decreasing order. Got u: {u}, prev_non_key_block: {prev_non_key_block}, u: {u}, prev_key_block: {prev_key_block}: {prev_key_block}."
                        prev_non_key_block = -u
                        prev_key_block = v
                    
                for edge in self.double_branch_scheduler:
                    u, v = edge
                    if u > 0: # cross-attention between key after block u (query) and non-key after block v (key/value)
                        self.cross_attn_blocks.append(
                            MMCrossStreamBlock(
                                q_hidden_size=self.hidden_size,
                                kv_hidden_size=second_branch_transformer_config.hidden_size,
                                heads_num=self.heads_num,
                                mlp_width_ratio=mlp_width_ratio,
                                mlp_act_type=mlp_act_type,
                                qk_norm=qk_norm,
                                qk_norm_type=qk_norm_type,
                                qkv_bias=qkv_bias,
                                **factory_kwargs,
                            )
                        )
                    else: # cross-attention between non-key after block u (query) and key after block v (key/value)
                        self.cross_attn_blocks.append(
                            MMCrossStreamBlock(
                                q_hidden_size=second_branch_transformer_config.hidden_size,
                                kv_hidden_size=self.hidden_size,
                                heads_num=second_branch_transformer_config.heads_num,
                                mlp_width_ratio=second_branch_transformer_config.mlp_width_ratio,
                                mlp_act_type=second_branch_transformer_config.mlp_act_type,
                                qk_norm=second_branch_transformer_config.qk_norm,
                                qk_norm_type=second_branch_transformer_config.qk_norm_type,
                                qkv_bias=second_branch_transformer_config.qkv_bias,
                                **factory_kwargs,
                            )
                        )
                
                self.double_branch_scheduler.append((-1, -1)) # add a last layer to avoid missing the last blocks when no cross-attention at the end
            
            self.proj_to_second_branch = nn.Linear(
                self.hidden_size,
                second_branch_transformer_config.hidden_size,
                **factory_kwargs,
            )
            self.unproj_from_second_branch = nn.Linear(
                second_branch_transformer_config.hidden_size,
                self.hidden_size,
                **factory_kwargs,
            )
            self.time_in_second_branch = TimestepEmbedder(
                second_branch_transformer_config.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )

            self.second_branch_blocks = nn.ModuleList(
                [
                    MMSingleStreamBlock(
                        second_branch_transformer_config.hidden_size,
                        second_branch_transformer_config.heads_num,
                        mlp_width_ratio=second_branch_transformer_config.mlp_width_ratio,
                        mlp_act_type=second_branch_transformer_config.mlp_act_type,
                        qk_norm=second_branch_transformer_config.qk_norm,
                        qk_norm_type=second_branch_transformer_config.qk_norm_type,
                        **factory_kwargs,
                    )
                    for _ in range(second_branch_mm_blocks_depth)
                ]
            )
            self.use_second_branch = True
        else:
            self.use_second_branch = False

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )

        if patch_sizes is not None:
            self.final_layer = MultiFinalLayer(
                patch_sizes,
                self.hidden_size,
                self.out_channels,
                get_activation_layer("silu"),
                **factory_kwargs,
            )

        # context block
        self.use_context_block = args.use_context_block
        if self.use_context_block:
            self.condition_in = PatchEmbed(
                self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs
            )

            self.context_block1 = MMDoubleStreamBlock(
                self.hidden_size,
                self.heads_num,
                mlp_width_ratio=mlp_width_ratio,
                mlp_act_type=mlp_act_type,
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type,
                qkv_bias=qkv_bias,
                **factory_kwargs,
            )

            self.context_block2 = MMSingleStreamBlock(
                self.hidden_size,
                self.heads_num,
                mlp_width_ratio=mlp_width_ratio,
                mlp_act_type=mlp_act_type,
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type,
                **factory_kwargs,
            )

            self.zero_linear1 = nn.Linear(self.hidden_size, self.hidden_size)
            self.zero_linear2 = nn.Linear(self.hidden_size, self.hidden_size)

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,  # Now we don't use it.
        # Text embedding for modulation.
        text_states_2: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        freqs_cos_cond: Optional[torch.Tensor] = None,
        freqs_sin_cond: Optional[torch.Tensor] = None,
        # Guidance for modulation, should be cfg_scale x 1000.
        guidance: torch.Tensor = None,
        return_dict: bool = True,
        indices: Optional[List[int]] = None,
        use_default_only: bool = False,
        freqs_cos_full: Optional[torch.Tensor] = None,
        freqs_sin_full: Optional[torch.Tensor] = None,
        extra_vec: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        logger.debug(f"{x.shape=} {x.min()=} {x.max()=} {x.mean()=}")
        logger.debug(f"latent {x[:, :16, :, :, :].shape=} {x[:, :16, :, :, :].min()=} {x[:, :16, :, :, :].max()=} {x[:, :16, :, :, :].mean()=}")
        logger.debug(f"img_latent {x[:, 16:32, :, :, :].shape=} {x[:, 16:32, :, :, :].min()=} {x[:, 16:32, :, :, :].max()=} {x[:, 16:32, :, :, :].mean()=}")
        logger.debug(f"mask_concat {x[:, 32:33, :, :, :].shape=} {x[:, 32:33, :, :, :].min()=} {x[:, 32:33, :, :, :].max()=} {x[:, 32:33, :, :, :].mean()=}")
        logger.debug(f"partial_cond {x[:, 33:49, :, :, :].shape=} {x[:, 33:49, :, :, :].min()=} {x[:, 33:49, :, :, :].max()=} {x[:, 33:49, :, :, :].mean()=}")
        logger.debug(f"partial_mask {x[:, 49:, :, :, :].shape=} {x[:, 49:, :, :, :].min()=} {x[:, 49:, :, :, :].max()=} {x[:, 49:, :, :, :].mean()=}")
        logger.debug(f"{t.shape=} {t=}")
        logger.debug(f"{text_states.shape=} {text_states.min()=} {text_states.max()} {text_states.mean()=}")
        logger.debug(f"{text_states_2.shape=} {text_states_2.min()=} {text_states_2.max()=} {text_states_2.mean()=}")
        logger.debug(f"{freqs_cos.shape=} {freqs_cos.min()=} {freqs_cos.max()=} {freqs_cos.mean()=}")
        logger.debug(f"{freqs_sin.shape=} {freqs_sin.min()=} {freqs_sin.max()=} {freqs_sin.mean()=}")

        out = {}
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        if hasattr(self, "old_patch_size") and use_default_only:
            tt, th, tw = (
                ot // self.old_patch_size[0],
                oh // self.old_patch_size[1],
                ow // self.old_patch_size[2],
            )
        else:
            tt, th, tw = (
                ot // self.patch_size[0],
                oh // self.patch_size[1],
                ow // self.patch_size[2],
            )

        # Prepare modulation vectors.
        vec = self.time_in(t)

        if self.i2v_condition_type == "token_replace":
            token_replace_t = torch.zeros_like(t)
            token_replace_vec = self.time_in(token_replace_t)
            frist_frame_token_num = th * tw
        else:
            token_replace_vec = None
            frist_frame_token_num = None

        # text modulation
        vec_2 = self.vector_in(text_states_2)
        vec = vec + vec_2
        if self.i2v_condition_type == "token_replace":
            token_replace_vec = token_replace_vec + vec_2

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )

            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec = vec + self.guidance_in(guidance)

        # MeanFlow second-time (r) embedding: constant additive modulation term
        # (mirrors model_forward_jvp's `extra_vec`). Default None -> no-op for all
        # existing callers. Lets the r==t fast path reuse the real forward.
        if extra_vec is not None:
            vec = vec + extra_vec

        # Embed image, condition and text.
        if self.use_context_block:
            condition = img.clone()
            height = (condition.shape[-2] - 2) // 2
            condition = condition[..., -height:, :]  # depth
            condition = self.condition_in(condition)

        if self.patch_sizes is not None and not use_default_only:
            img, patch_indices = self.img_in(img, indices=indices)
            assert torch.all(patch_indices == patch_indices[0:1]), \
                "patch_indices differ across batch: cannot index shared RoPE [N,*] with batch-dependent masks."
            if self.use_second_branch:
                B, N, dim = img.shape

                non_keyframe_mask = (patch_indices != 0) # first patch size corresponds to keyframe, [B, N]
                K = int(non_keyframe_mask.sum(dim=1)[0].item())  # It is guaranteed that for each batch, there is exactly K tokens belong to non-keyframes.
                assert torch.all(non_keyframe_mask.sum(dim=1) == K)
                img_second_branch = img[non_keyframe_mask].reshape(B, K, dim)
                img_second_branch = self.proj_to_second_branch(img_second_branch)
                freqs_cos_second_branch = freqs_cos[non_keyframe_mask[0].cpu()]
                freqs_sin_second_branch = freqs_sin[non_keyframe_mask[0].cpu()]

                keyframe_mask = (patch_indices == 0).to(img.device) # first patch size corresponds to keyframe
                K = int(keyframe_mask.sum(dim=1)[0].item()) # It is guaranteed that for each batch, there is exactly K tokens belong to keyframes.
                assert torch.all(keyframe_mask.sum(dim=1) == K)
                img = img[keyframe_mask].reshape(B, K, dim)
                freqs_cos = freqs_cos[keyframe_mask[0].cpu()]
                freqs_sin = freqs_sin[keyframe_mask[0].cpu()]

                vec_second_branch = self.time_in_second_branch(t)
        else:
            if use_default_only and hasattr(self, "old_img_in"):
                img = self.old_img_in(img)
            else:
                img = self.img_in(img)
        
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(
                txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )
        
        if self.patch_sizes is not None and self.use_second_branch and not use_default_only:
            prev_first_branch_index = -1
            prev_second_branch_index = -1
            no_text_mask = torch.zeros(
                (img.shape[0], 0),
                dtype=text_mask.dtype,
                device=text_mask.device,
            )
            first_branch_cu_seqlens = get_cu_seqlens(no_text_mask, img.shape[1])
            first_branch_text_cu_seqlens = get_cu_seqlens(text_mask, img.shape[1])
            second_branch_cu_seqlens = get_cu_seqlens(no_text_mask, img_second_branch.shape[1])

            first_branch_max_seqlen = img.shape[1]
            first_branch_text_max_seqlen = img.shape[1] + txt.shape[1]
            second_branch_max_seqlen = img_second_branch.shape[1]
            txt_seq_len = txt.shape[1]

            last_img, last_txt, last_x = None, None, None
            last_img_second_branch = None

            for cross_attn_id, edge in enumerate(self.double_branch_scheduler):
                u, v = edge
                if u == -1 and v == -1:
                    u = len(self.double_blocks) + len(self.single_blocks) - 1
                    v = -len(self.second_branch_blocks) + 1
                    last_layer = True
                else:
                    last_layer = False
                if u > 0:
                    first_branch_index = u
                    second_branch_index = -v
                else:
                    first_branch_index = v
                    second_branch_index = -u
                #-----------------------First branch blocks------------------------#
                for layer_num in range(prev_first_branch_index + 1, first_branch_index + 1):
                    if last_img is None:
                        last_img = img
                    if last_txt is None:
                        last_txt = txt
                    if last_x is None and layer_num >= len(self.double_blocks):
                        last_x = torch.cat((last_img, last_txt), 1)
                    if layer_num < len(self.double_blocks):
                        block = self.double_blocks[layer_num]
                        double_block_args = [
                            last_img,
                            last_txt,
                            vec,
                            first_branch_text_cu_seqlens,
                            first_branch_text_cu_seqlens,
                            first_branch_text_max_seqlen,
                            first_branch_text_max_seqlen,
                            (freqs_cos, freqs_sin),
                            self.i2v_condition_type,
                            token_replace_vec,
                            frist_frame_token_num,
                        ]
                        logger.debug(f"First branch double block {layer_num} processing")
                        if self.training and self.gradient_checkpoint and \
                                (self.gradient_checkpoint_layers == -1 or layer_num < self.gradient_checkpoint_layers):
                            # print(f'gradient checkpointing...')
                            img, txt = torch.utils.checkpoint.checkpoint(
                                ckpt_wrapper(block), *double_block_args, use_reentrant=False)
                        else:
                            img, txt = block(*double_block_args)
                        last_img = img
                        last_txt = txt
                    else:
                        block = self.single_blocks[layer_num - len(self.double_blocks)]
                        single_block_args = [
                            last_x,
                            vec,
                            txt_seq_len,
                            first_branch_text_cu_seqlens,
                            first_branch_text_cu_seqlens,
                            first_branch_text_max_seqlen,
                            first_branch_text_max_seqlen,
                            (freqs_cos, freqs_sin),
                            self.i2v_condition_type,
                            token_replace_vec,
                            frist_frame_token_num,
                        ]
                        logger.debug(f"First branch single block {layer_num - len(self.double_blocks)} processing")
                        if self.training and self.gradient_checkpoint and \
                                (self.gradient_checkpoint_layers == -1 or layer_num < self.gradient_checkpoint_layers):
                            # print(f'gradient checkpointing...')
                            x = torch.utils.checkpoint.checkpoint(
                                ckpt_wrapper(block), *single_block_args, use_reentrant=False)
                        else:
                            x = block(*single_block_args)
                        last_x = x
                        last_img = last_x[:, :img.shape[1], ...]
                #-----------------------Second branch blocks------------------------#
                for layer_num in range(prev_second_branch_index + 1, second_branch_index + 1):
                    if last_img_second_branch is None:
                        last_img_second_branch = img_second_branch
                    block = self.second_branch_blocks[layer_num]
                    single_block_args = [
                        last_img_second_branch,
                        vec_second_branch,
                        0,
                        second_branch_cu_seqlens,
                        second_branch_cu_seqlens,
                        second_branch_max_seqlen,
                        second_branch_max_seqlen,
                        (freqs_cos_second_branch, freqs_sin_second_branch),
                        self.i2v_condition_type,
                        token_replace_vec,
                        frist_frame_token_num,
                    ]
                    logger.debug(f"Second branch single block {layer_num} processing")
                    if self.training and self.gradient_checkpoint and \
                            (self.gradient_checkpoint_layers == -1 or layer_num < self.gradient_checkpoint_layers):
                        # print(f'gradient checkpointing...')
                        img_second_branch = torch.utils.checkpoint.checkpoint(
                            ckpt_wrapper(block), *single_block_args, use_reentrant=False)
                    else:
                        img_second_branch = block(*single_block_args)
                    last_img_second_branch = img_second_branch
                prev_first_branch_index = first_branch_index
                prev_second_branch_index = second_branch_index
                #-----------------------Cross attention------------------------#
                if not last_layer:
                    if u > 0:
                        # cross-attention between key after block u (query) and non-key after block v (key/value)
                        logger.debug(f"Cross attention block {cross_attn_id} processing: keyframe {u} as query, non-keyframe {v} as key/value")
                        last_img = self.cross_attn_blocks[cross_attn_id](
                            last_img,
                            last_img_second_branch,
                            vec,
                            vec_second_branch,
                            cu_seqlens_q=first_branch_cu_seqlens,
                            cu_seqlens_kv=second_branch_cu_seqlens,
                            max_seqlen_q=first_branch_max_seqlen,
                            max_seqlen_kv=second_branch_max_seqlen,
                            freqs_cis_q=(freqs_cos, freqs_sin),
                            freqs_cis_kv=(freqs_cos_second_branch, freqs_sin_second_branch),
                            condition_type=self.i2v_condition_type,
                            token_replace_vec=token_replace_vec,
                            frist_frame_token_num=frist_frame_token_num,
                        )
                    else:
                        # cross-attention between non-key after block u (query) and key after block v (key/value)
                        logger.debug(f"Cross attention block {cross_attn_id} processing: non-keyframe {u} as query, keyframe {v} as key/value")
                        last_img_second_branch = self.cross_attn_blocks[cross_attn_id](
                            last_img_second_branch,
                            last_img,
                            vec_second_branch,
                            vec,
                            cu_seqlens_q=second_branch_cu_seqlens,
                            cu_seqlens_kv=first_branch_cu_seqlens,
                            max_seqlen_q=second_branch_max_seqlen,
                            max_seqlen_kv=first_branch_max_seqlen,
                            freqs_cis_q=(freqs_cos_second_branch, freqs_sin_second_branch),
                            freqs_cis_kv=(freqs_cos, freqs_sin),
                            condition_type=self.i2v_condition_type,
                            token_replace_vec=token_replace_vec,
                            frist_frame_token_num=frist_frame_token_num,
                        )
                    if last_x is not None:
                        # last_x is concatenation of [img, txt]. We need to keep the updated text 
                        # (which resides in last_x) and combine it with the cross-attended last_img.
                        current_txt = last_x[:, last_img.shape[1]:, ...]
                        last_x = torch.cat((last_img, current_txt), dim=1)
            last_img_second_branch = self.unproj_from_second_branch(last_img_second_branch)
            img = self.merge_back(
                last_img,
                last_img_second_branch,
                patch_indices
            )
        else:
            txt_seq_len = txt.shape[1]
            img_seq_len = img.shape[1]

            # Compute cu_squlens and max_seqlen for flash attention
            cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
            cu_seqlens_kv = cu_seqlens_q
            max_seqlen_q = img_seq_len + txt_seq_len
            max_seqlen_kv = max_seqlen_q

            if self.use_context_block:
                cond_seq_len = condition.shape[1]
                cu_seqlens_q_cond = get_cu_seqlens(text_mask, cond_seq_len)
                cu_seqlens_kv_cond = cu_seqlens_q_cond
                max_seqlen_q_cond = cond_seq_len + txt_seq_len
                max_seqlen_kv_cond = max_seqlen_q_cond

                # ---------------------------- Context Block ------------------------------
                context_block_args = [
                    condition,
                    txt,
                    vec,
                    cu_seqlens_q_cond,
                    cu_seqlens_kv_cond,
                    max_seqlen_q_cond,
                    max_seqlen_kv_cond,
                    (freqs_cos_cond, freqs_sin_cond),
                    # (freqs_cos, freqs_sin),
                    self.i2v_condition_type,
                    token_replace_vec,
                    frist_frame_token_num,
                ]
                condition1, txt1 = self.context_block1(*context_block_args)

                condition2 = torch.cat((condition1, txt1), 1)
                context_block_args = [
                    condition2,
                    vec,
                    txt_seq_len,
                    cu_seqlens_q_cond,
                    cu_seqlens_kv_cond,
                    max_seqlen_q_cond,
                    max_seqlen_kv_cond,
                    (freqs_cos_cond, freqs_sin_cond),
                    # (freqs_cos, freqs_sin),
                    self.i2v_condition_type,
                    token_replace_vec,
                    frist_frame_token_num,
                ]
                condition2 = self.context_block2(*context_block_args)

                condition1 = self.zero_linear1(condition1)
                condition2 = self.zero_linear2(condition2)

                condition2 = torch.cat(
                    (torch.zeros_like(img)[:, :-condition1.shape[1]], condition2), dim=1)
                condition1 = torch.cat(
                    (torch.zeros_like(img)[:, :-condition1.shape[1]], condition1), dim=1)

            freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
            # --------------------- Pass through DiT blocks ------------------------
            for layer_num, block in enumerate(self.double_blocks):
                double_block_args = [
                    img,
                    txt,
                    vec,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    freqs_cis,
                    self.i2v_condition_type,
                    token_replace_vec,
                    frist_frame_token_num,
                ]

                if self.training and self.gradient_checkpoint and \
                        (self.gradient_checkpoint_layers == -1 or layer_num < self.gradient_checkpoint_layers):
                    # print(f'gradient checkpointing...')
                    img, txt = torch.utils.checkpoint.checkpoint(
                        ckpt_wrapper(block), *double_block_args, use_reentrant=False)
                    if self.use_context_block:
                        img += condition1
                else:
                    img, txt = block(*double_block_args)
                    if self.use_context_block:
                        img += condition1

            # Merge txt and img to pass through single stream blocks.
            x = torch.cat((img, txt), 1)

            if len(self.single_blocks) > 0:
                for _, block in enumerate(self.single_blocks):
                    single_block_args = [
                        x,
                        vec,
                        txt_seq_len,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        (freqs_cos, freqs_sin),
                        self.i2v_condition_type,
                        token_replace_vec,
                        frist_frame_token_num,
                    ]

                    if self.training and self.gradient_checkpoint and \
                            (self.gradient_checkpoint_layers == -1 or \
                            layer_num + len(self.double_blocks) < self.gradient_checkpoint_layers):
                        x = torch.utils.checkpoint.checkpoint(ckpt_wrapper(
                            block), *single_block_args, use_reentrant=False)
                        if self.use_context_block:
                            x += condition2
                    else:
                        x = block(*single_block_args)
                        if self.use_context_block:
                            x += condition2

            img = x[:, :img_seq_len, ...]

        # ---------------------------- Final layer ------------------------------
        # (N, T, patch_size ** 2 * out_channels)
        if self.patch_sizes is None or use_default_only:
            if use_default_only and hasattr(self, "old_final_layer"):
                img = self.old_final_layer(img, vec)
            else:
                img = self.final_layer(img, vec)
            img = self.unpatchify(img, tt, th, tw, use_default_only=use_default_only)
        else:
            img = self.final_layer(img, vec, indices=indices, patch_indices=patch_indices)
            img = self.unpatchify_multi(
                img,
                T=ot,
                H=oh,
                W=ow,
                patch_sizes=self.patch_sizes,
                indices=indices,
            )
        if return_dict:
            out["x"] = img
            return out
        return img
    
    def merge_back(
        self,
        img_key: torch.Tensor,          # [B, K_key, D]
        img_nonkey: torch.Tensor,       # [B, K_nonkey, D]
        patch_indices_full: torch.Tensor # [B, N]
    ) -> torch.Tensor:
        B, N = patch_indices_full.shape
        D = img_key.shape[-1]

        key_mask = (patch_indices_full == 0)       # [B, N]
        nonkey_mask = ~key_mask                    # [B, N]

        out = img_key.new_empty((B, N, D))         # [B, N, D]

        # boolean assignment expects flattened RHS: (#true, D)
        out[key_mask] = img_key.reshape(-1, D)
        out[nonkey_mask] = img_nonkey.reshape(-1, D)

        return out

    def unpatchify(self, x, t, h, w, use_default_only=False):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        if use_default_only and hasattr(self, "old_patch_size"):
            pt, ph, pw = self.old_patch_size
        else:
            pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs
    
    def unpatchify_multi(
        self,
        xs,
        T: int,
        H: int,
        W: int,
        patch_sizes,
        indices,
    ) -> torch.Tensor:
        """
        Inverse of MultiPatchEmbed + MultiFinalLayer.

        Args:
            xs (List[Tensor]):
                List of tensors from MultiFinalLayer. For kernel k,
                xs[k] has shape (B, N_k, C * pt * ph * pw), where
                patch_sizes[k] == (pt, ph, pw) (after to_3tuple).
            T, H, W (int):
                Original temporal and spatial resolution of the latent.
                The returned tensor will have shape (B, C, T, H, W).
            patch_sizes (Sequence[int or Sequence[int]]):
                Per-kernel patch sizes. Each element is either an int
                or a (pt, ph, pw)-like sequence.
            indices (Sequence[Tensor]):
                Per-kernel 1D LongTensor listing the frame indices
                where the corresponding kernel was applied (same
                semantics as in MultiPatchEmbed).

        Returns:
            torch.Tensor:
                Reconstructed latent of shape (B, C, T, H, W).
        """
        # Basic checks
        if not xs:
            raise ValueError("unpatchify_multi: 'xs' is empty.")
        x0 = xs[0]
        B = x0.shape[0]
        C = self.unpatchify_channels
        device = x0.device
        dtype = x0.dtype

        # Output volume and weight map for averaging overlaps
        out = torch.zeros(B, C, T, H, W, device=device, dtype=dtype)
        weight = torch.zeros(1, 1, T, H, W, device=device, dtype=dtype)

        for x_k, patch_size, idxs in zip(xs, patch_sizes, indices):
            # Normalize patch size to (pt, ph, pw)
            pt, ph, pw = to_3tuple(patch_size)

            # Number of frames this kernel was applied to
            T_k = len(idxs)
            if T_k == 0:
                continue

            # Spatial patch grid for this kernel
            if H % ph != 0 or W % pw != 0:
                raise ValueError(
                    f"unpatchify_multi: H={H}, W={W} not divisible by patch "
                    f"size (ph={ph}, pw={pw})."
                )
            H_k = H // ph
            W_k = W // pw

            expected_tokens = T_k * H_k * W_k
            if x_k.shape[1] != expected_tokens:
                raise ValueError(
                    "unpatchify_multi: token count mismatch for patch size "
                    f"{patch_size}. Got {x_k.shape[1]} tokens, expected "
                    f"{expected_tokens} (= T_k*H_k*W_k)."
                )

            # x_k: (B, N_k, C * pt * ph * pw)
            x_k = x_k.view(B, T_k, H_k, W_k, C, pt, ph, pw)
            # -> (B, C, T_k * pt, H_k * ph, W_k * pw)
            x_k = torch.einsum("nthwcopq->nctohpwq", x_k)
            x_k = x_k.reshape(B, C, T_k * pt, H_k * ph, W_k * pw)

            # Scatter into global volume along temporal dimension
            # Each local time slice corresponds to global frame idxs[local_t]
            for local_t, global_t in enumerate(idxs):
                t_local_slice = slice(local_t * pt, (local_t + 1) * pt)
                t_global_slice = slice(global_t * pt, (global_t + 1) * pt)

                if t_global_slice.stop > T * pt:
                    raise ValueError(
                        "unpatchify_multi: temporal slice exceeds target size: "
                        f"global_t={global_t}, pt={pt}, T={T}"
                    )

                out[:, :, t_global_slice, :, :] += x_k[:, :, t_local_slice, :, :]
                weight[:, :, t_global_slice, :, :] += 1

        # Make sure all pixels were written at least once
        if (weight == 0).any():
            raise ValueError(
                "unpatchify_multi: some positions in the output volume were "
                "never written to. Check 'patch_sizes' and 'indices'."
            )

        # Average overlapping contributions
        out = out / weight

        return out


    def params_count(self):
        counts = {
            "double": sum(
                [
                    sum(p.numel() for p in block.img_attn_qkv.parameters())
                    + sum(p.numel() for p in block.img_attn_proj.parameters())
                    + sum(p.numel() for p in block.img_mlp.parameters())
                    + sum(p.numel() for p in block.txt_attn_qkv.parameters())
                    + sum(p.numel() for p in block.txt_attn_proj.parameters())
                    + sum(p.numel() for p in block.txt_mlp.parameters())
                    for block in self.double_blocks
                ]
            ),
            "single": sum(
                [
                    sum(p.numel() for p in block.linear1.parameters())
                    + sum(p.numel() for p in block.linear2.parameters())
                    for block in self.single_blocks
                ]
            ),
            "total": sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"] + counts["single"]
        return counts

    def set_input_tensor(self, input_tensor):
        pass

#################################################################################
#                             HunyuanVideo Configs                              #
#################################################################################


HUNYUAN_VIDEO_CONFIG = {
    "HYVideo-T/2": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
    },
    "HYVideo-T/2-cfgdistill": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "guidance_embed": True,
    },
    "HYVideo-M/2": {
        "mm_double_blocks_depth": 8,
        "mm_single_blocks_depth": 16,
        "rope_dim_list": [24, 84, 84],
        "hidden_size": 960,
        "heads_num": 5,
        "mlp_width_ratio": 4,
    },
    "HYVideo-S/2": {
        "mm_double_blocks_depth": 6,
        "mm_single_blocks_depth": 12,
        "rope_dim_list": [12, 42, 42],
        "hidden_size": 480,
        "heads_num": 5,
        "mlp_width_ratio": 4,
    },
    "HYVideo-S/2-2branch": {
        "mm_double_blocks_depth": 6,
        "mm_single_blocks_depth": 12,
        "rope_dim_list": [12, 42, 42],
        "hidden_size": 480,
        "heads_num": 5,
        "mlp_width_ratio": 4,
        "second_branch_mm_blocks_depth": 4,
        "second_branch_transformer_config": TransformerBranchConfig(
            hidden_size=480,
            heads_num=5,
            mlp_width_ratio=4,
            scheduler=[],
        ),
    },
    "HYVideo-T/2-2branch-cross_attn": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "second_branch_mm_blocks_depth": 18,
        "second_branch_transformer_config": TransformerBranchConfig(
            hidden_size=512,
            heads_num=4,
            mlp_width_ratio=4,
            scheduler=[(0, 3), (-3, 11), (15, -4), (-5, 19), (-6, 23), (-9, 31), (35, -10), (-11, 39), (-12, 43), (-15, 51), (55, -16), (-17, 59)],
        ),
    },
    "HYVideo-T/2-2branch-no_cross_attn": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "second_branch_mm_blocks_depth": 18,
        "second_branch_transformer_config": TransformerBranchConfig(
            hidden_size=512,
            heads_num=4,
            mlp_width_ratio=4,
            scheduler=[],
        ),
    },
    "HYVideo-T/2-2branch-cross_attn-unidirectional-q_second": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "second_branch_mm_blocks_depth": 18,
        "second_branch_transformer_config": TransformerBranchConfig(
            hidden_size=512,
            heads_num=4,
            mlp_width_ratio=4,
            scheduler=[(0, 3), (-3, 11), (-5, 19), (-6, 23), (-9, 31), (-11, 39), (-12, 43), (-15, 51), (-17, 59)],
        ),
    },
    "HYVideo-B/2": {
        "mm_double_blocks_depth": 6,
        "mm_single_blocks_depth": 12,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 1536,
        "heads_num": 12,
        "mlp_width_ratio": 4,
    },
    "HYVideo-T/2-2branch-cross_attn-0001": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "second_branch_mm_blocks_depth": 18,
        "second_branch_transformer_config": TransformerBranchConfig(
            hidden_size=512,
            heads_num=4,
            mlp_width_ratio=4,
            scheduler=[(0, 3), (7, -1), (-3, 11), (15, -4), (-5, 19), (23, -6), (-7, 27), (31, -9), (-10, 35), (39, -11), (-12, 43), (47, -13), (-15, 51), (55, -16), (-17, 59)],
        ),
    },
    "HYVideo-S/2-2branch-cross_attn": {
        "mm_double_blocks_depth": 6,
        "mm_single_blocks_depth": 12,
        "rope_dim_list": [12, 42, 42],
        "hidden_size": 480,
        "heads_num": 5,
        "mlp_width_ratio": 4,
        "second_branch_mm_blocks_depth": 6,
        "second_branch_transformer_config": TransformerBranchConfig(
            hidden_size=480,
            heads_num=5,
            mlp_width_ratio=4,
            scheduler=[(0, 2), (-1, 5), (-2, 8), (-3, 11), (-4, 14), (-5, 17)],
        ),
    },
    "HYVideo-S/2-2branch-cross_attn-full": {
        "mm_double_blocks_depth": 6,
        "mm_single_blocks_depth": 12,
        "rope_dim_list": [12, 42, 42],
        "hidden_size": 480,
        "heads_num": 5,
        "mlp_width_ratio": 4,
        "second_branch_mm_blocks_depth": 18,
        "second_branch_transformer_config": TransformerBranchConfig(
            hidden_size=480,
            heads_num=5,
            mlp_width_ratio=4,
            scheduler=[(0, 2), (2, -4), (-4, 6), (6, -8), (-8, 10), (10, -12), (-12, 14), (14, -16), (-16, 17)],
        ),
    },
    "HYVideo-S/2-4x4": {
        "mm_double_blocks_depth": 6,
        "mm_single_blocks_depth": 12,
        "rope_dim_list": [12, 42, 42],
        "hidden_size": 480,
        "heads_num": 5,
        "mlp_width_ratio": 4,
        "patch_size": [1, 4, 4],
    },
}
