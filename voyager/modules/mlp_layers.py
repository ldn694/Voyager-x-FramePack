# Modified from timm library:
# https://github.com/huggingface/pytorch-image-models/blob/648aaa41233ba83eb38faf5ba9d415d574823241/timm/layers/mlp.py#L13

from functools import partial

import torch
import torch.nn as nn

from .modulate_layers import modulate
from ..utils.helpers import to_2tuple, to_3tuple


class MLP(nn.Module):
    """Multi-Layer Perceptron (MLP) as used in Vision Transformer, MLP-Mixer and related networks
    
    This is a flexible MLP implementation that can be used as a building block
    in various transformer-based architectures. It supports both linear and
    convolutional layers, with optional normalization and dropout.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
        device=None,
        dtype=None,
    ):
        """Initialize MLP layer
        
        Args:
            in_channels (int): Number of input channels/features
            hidden_channels (int, optional): Number of hidden channels. Defaults to in_channels
            out_features (int, optional): Number of output features. Defaults to in_channels
            act_layer: Activation function class (default: nn.GELU)
            norm_layer: Normalization layer class (default: None)
            bias (bool or tuple): Whether to use bias in linear layers
            drop (float or tuple): Dropout probability for each layer
            use_conv (bool): Whether to use Conv2d instead of Linear layers
            device: Device to place the module on
            dtype: Data type for the module parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        # Set default values for output and hidden dimensions
        out_features = out_features or in_channels
        hidden_channels = hidden_channels or in_channels
        
        # Convert single values to tuples for consistency
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        
        # Choose between linear and convolutional layers
        linear_layer = partial(
            nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        # First fully connected layer
        self.fc1 = linear_layer(
            in_channels, hidden_channels, bias=bias[0], **factory_kwargs
        )
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        
        # Optional normalization layer
        self.norm = (
            norm_layer(hidden_channels, **factory_kwargs)
            if norm_layer is not None
            else nn.Identity()
        )
        
        # Second fully connected layer
        self.fc2 = linear_layer(
            hidden_channels, out_features, bias=bias[1], **factory_kwargs
        )
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        """Forward pass through the MLP
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after MLP transformation
        """
        x = self.fc1(x)      # First linear transformation
        x = self.act(x)      # Apply activation function
        x = self.drop1(x)    # Apply dropout
        x = self.norm(x)     # Apply normalization (if any)
        x = self.fc2(x)      # Second linear transformation
        x = self.drop2(x)    # Apply dropout
        return x


class MLPEmbedder(nn.Module):
    """MLP-based embedding layer
    
    A simple MLP used for embedding transformations, copied from the Flux library.
    This is typically used for processing conditional information or embeddings.
    """

    def __init__(self, in_dim: int, hidden_dim: int, device=None, dtype=None):
        """Initialize MLP embedder
        
        Args:
            in_dim (int): Input dimension
            hidden_dim (int): Hidden dimension for the MLP
            device: Device to place the module on
            dtype: Data type for the module parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        # Input projection layer
        self.in_layer = nn.Linear(
            in_dim, hidden_dim, bias=True, **factory_kwargs)
        self.silu = nn.SiLU()  # Swish/SiLU activation function
        
        # Output projection layer
        self.out_layer = nn.Linear(
            hidden_dim, hidden_dim, bias=True, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP embedder
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Embedded tensor with same dimension as hidden_dim
        """
        return self.out_layer(self.silu(self.in_layer(x)))


class FinalLayer(nn.Module):
    """The final layer of DiT (Diffusion Transformer)
    
    This layer is responsible for the final output projection in diffusion models.
    It includes adaptive layer normalization (AdaLN) modulation and projects
    the hidden representations to the output space (e.g., pixel values).
    """

    def __init__(
        self, hidden_size, patch_size, out_channels, act_layer, device=None, dtype=None
    ):
        """Initialize the final layer
        
        Args:
            hidden_size (int): Size of the hidden representations
            patch_size (int or tuple): Size of patches (H, W) or single value
            out_channels (int): Number of output channels
            act_layer: Activation function for the modulation network
            device: Device to place the module on
            dtype: Data type for the module parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # Layer normalization without learnable parameters
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        
        # Output projection layer
        if isinstance(patch_size, int):
            # For 2D patches (H=W=patch_size)
            self.linear = nn.Linear(
                hidden_size,
                patch_size * patch_size * out_channels,
                bias=True,
                **factory_kwargs
            )
        else:
            # For 3D patches (H, W, T)
            self.linear = nn.Linear(
                hidden_size,
                patch_size[0] * patch_size[1] * patch_size[2] * out_channels,
                bias=True,
            )
        
        # Zero-initialize the output projection for stable training
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        # Adaptive Layer Normalization (AdaLN) modulation network
        # This network generates scale and shift parameters based on conditioning
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size,
                      bias=True, **factory_kwargs),
        )
        
        # Zero-initialize the modulation network for stable training
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        """Forward pass through the final layer
        
        Args:
            x (torch.Tensor): Input hidden representations
            c (torch.Tensor): Conditioning information for AdaLN modulation
            
        Returns:
            torch.Tensor: Final output projections
        """
        # Generate scale and shift parameters from conditioning
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        
        # Apply adaptive layer normalization with modulation
        x = modulate(self.norm_final(x), shift=shift, scale=scale)
        
        # Project to output space
        x = self.linear(x)
        return x

class MultiFinalLayer(nn.Module):
    """
    The final layer of DiT (Diffusion Transformer) with multiple kernels.

    This layer is responsible for the final output projection in diffusion models.
    It includes adaptive layer normalization (AdaLN) modulation and projects the
    hidden representations to the output space (e.g., pixel values) using multiple
    kernels for different patch sizes.
    """

    def __init__(
        self,
        hidden_size,
        patch_sizes,
        out_channels,
        act_layer,
        device=None,
        dtype=None,
    ):
        """
        Initialize the multi-kernel final layer.

        Args:
            hidden_size (int): Size of the hidden representations
            patch_sizes (list of int or tuple): List of patch sizes (H, W) or single values
            out_channels (int): Number of output channels
            act_layer: Activation function for the modulation network
            device: Device to place the module on
            dtype: Data type for the module parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # Layer normalization without learnable parameters
        self.norm_final = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
            **factory_kwargs,
        )

        self.patch_sizes = patch_sizes

        # Output projection layers for each patch size
        self.linears = nn.ModuleList()
        for patch_size in patch_sizes:
            if isinstance(patch_size, int):
                out_dim = patch_size * patch_size * out_channels
            else:
                out_dim = (
                    patch_size[0]
                    * patch_size[1]
                    * patch_size[2]
                    * out_channels
                )

            linear = nn.Linear(
                hidden_size,
                out_dim,
                bias=True,
                **factory_kwargs,
            )

            # Zero-initialize the output projection for stable training
            nn.init.zeros_(linear.weight)
            nn.init.zeros_(linear.bias)

            self.linears.append(linear)

        # Adaptive Layer Normalization (AdaLN) modulation network
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(
                hidden_size,
                2 * hidden_size,
                bias=True,
                **factory_kwargs,
            ),
        )

        # Zero-initialize the modulation network for stable training
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c, indices, patch_indices):
        """
        Forward pass through the multi-kernel final layer.

        Args:
            x (torch.Tensor): Input hidden representations (B, N, C)
            c (torch.Tensor): Conditioning information for AdaLN modulation
            indices (list): Frame indices for each patch size
            patch_indices (torch.Tensor): Shape (B, N), indicating which patch size
                each token corresponds to

        Returns:
            list[torch.Tensor]: Final output projections for each patch size
        """
        # Generate scale and shift parameters from conditioning
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)

        # Apply adaptive layer normalization with modulation
        x = modulate(self.norm_final(x), shift=shift, scale=scale)

        outputs = []
        B, N, C = x.shape

        # Project to output space using the appropriate linear layer
        for k, linear in enumerate(self.linears):
            mask = (patch_indices == k)  # (B, N)
            if not mask.any():
                continue

            counts = mask.sum(dim=1)  # (B,)
            assert torch.all(counts == counts[0]), (
                f"Rows do not have equal True counts: {counts}"
            )

            M = counts[0].item()

            # Gather tokens belonging to this kernel
            x_k = x[mask].view(B, M, C)  # (B, M, C)
            out_k = linear(x_k)          # (B, M, out_dim_k)

            outputs.append(out_k)

        return outputs

class MultiFinalLayerTranspose(nn.Module):
    """The final layer of DiT (Diffusion Transformer) with multiple kernels
    
    This layer is responsible for the final output projection in diffusion models.
    It includes adaptive layer normalization (AdaLN) modulation and projects
    the hidden representations to the output space (e.g., pixel values) using
    multiple kernels for different patch sizes.
    """

    def __init__(
        self, hidden_size, patch_sizes, out_channels, act_layer, device=None, dtype=None
    ):
        """Initialize the multi-kernel final layer
        
        Args:
            hidden_size (int): Size of the hidden representations
            patch_sizes (list of int or tuple): List of patch sizes (H, W) or single values
            out_channels (int): Number of output channels
            act_layer: Activation function for the modulation network
            device: Device to place the module on
            dtype: Data type for the module parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.patch_sizes_3d = [to_3tuple(ps) for ps in patch_sizes]
        self.hidden_size = hidden_size
        self.out_channels = out_channels

        # --- Layer norm + AdaLN (same as before) ---
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True, **factory_kwargs),
        )
        # Zero-init modulation linear for stable training (as in DiT)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        # --- Single linear for smallest patch size ---
        base_pt, base_ph, base_pw = self.patch_sizes_3d[0]
        self.base_pt = base_pt
        self.base_ph = base_ph
        self.base_pw = base_pw

        base_out_dim = out_channels * base_pt * base_ph * base_pw  # C * pt * ph * pw

        self.unproj = nn.Linear(
            hidden_size,
            base_out_dim,
            bias=True,
            **factory_kwargs,
        )
        # Optional: zero-init to mimic DiT final layer behavior
        nn.init.zeros_(self.unproj.weight)
        nn.init.zeros_(self.unproj.bias)

        # --- Progressive ConvTranspose3d upsamplers between scales ---
        self.upsamplers = nn.ModuleList()
        for i in range(len(self.patch_sizes_3d) - 1):
            pt_in, ph_in, pw_in = self.patch_sizes_3d[i]
            pt_out, ph_out, pw_out = self.patch_sizes_3d[i + 1]

            assert pt_out % pt_in == 0 and ph_out % ph_in == 0 and pw_out % pw_in == 0, \
                f"patch_sizes[{i+1}] must be divisible by patch_sizes[{i}]"

            rt = pt_out // pt_in
            rh = ph_out // ph_in
            rw = pw_out // pw_in

            # We keep channels constant: out_channels -> out_channels
            self.upsamplers.append(
                nn.ConvTranspose3d(
                    out_channels,
                    out_channels,
                    kernel_size=(rt, rh, rw),
                    stride=(rt, rh, rw),
                    padding=0,
                    bias=True,
                    **factory_kwargs,
                )
            )
            # NOTE: do NOT zero-init these, otherwise gradient flow may be blocked.

    def forward(self, x, c, indices, patch_indices):
        """Forward pass through the multi-kernel final layer.

        Args:
            x (torch.Tensor): Hidden representations, shape (B, N, hidden_size).
            c (torch.Tensor): Conditioning for AdaLN, shape (B, hidden_size).
            indices (list): Not used here; can be used downstream to arrange frames.
            patch_indices (torch.Tensor): (B, N) integer tensor indicating which
                patch size each token corresponds to (0..len(patch_sizes)-1).

        Returns:
            list[torch.Tensor]: List of outputs, one per *used* patch size.
                Each element has shape (B, N_k, C_out * pt_k * ph_k * pw_k),
                where (pt_k, ph_k, pw_k) is the k-th patch size in 3D form.
        """
        B, N, hidden = x.shape
        assert hidden == self.hidden_size, \
            f"Expected hidden_size={self.hidden_size}, got {hidden}"

        # AdaLN
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift=shift, scale=scale)  # (B, N, hidden)

        outputs = []

        # For each patch scale k
        for k, ps_3d in enumerate(self.patch_sizes_3d):
            mask = (patch_indices == k)  # (B, N) bool
            if not mask.any():
                # No tokens at this scale; skip to keep behavior similar to original code
                continue

            counts = mask.sum(dim=1)  # (B,)
            assert torch.all(counts == counts[0]), \
                f"Rows do not have equal True counts for scale {k}: {counts}"
            M = counts[0].item()  # number of tokens per batch at this scale

            # Gather tokens belonging to this scale: (B, M, hidden)
            x_k = x[mask].view(B, M, hidden)

            # --- Base unprojection with shared linear ---
            # Treat (B, M) as a big batch for the linear and ConvTranspose3d
            x_k_flat = x_k.reshape(B * M, hidden)  # (B*M, hidden)
            patch = self.unproj(x_k_flat)          # (B*M, C * base_pt * base_ph * base_pw)
            patch = patch.view(
                B * M,
                self.out_channels,
                self.base_pt,
                self.base_ph,
                self.base_pw,
            )  # (B*M, C_out, pt_base, ph_base, pw_base)

            # --- Progressive upsampling from base to scale k ---
            for j in range(k):  # 0 steps for smallest scale, 1 step for next, etc.
                patch = self.upsamplers[j](patch)

            # At this point, patch should have spatial size equal to patch_sizes_3d[k]
            pt_k, ph_k, pw_k = ps_3d
            _, C_out, pt_cur, ph_cur, pw_cur = patch.shape
            # Sanity check (can be removed for speed)
            assert (pt_cur, ph_cur, pw_cur) == (pt_k, ph_k, pw_k), \
                f"Scale {k}: expected patch size {(pt_k, ph_k, pw_k)}, got {(pt_cur, ph_cur, pw_cur)}"

            # --- Flatten back to (B, M, C * pt * ph * pw) ---
            # Order is (C, pt, ph, pw) inside the last dimension,
            # so downstream you can safely do:
            # x_k.view(B, T_k, H_k, W_k, C, pt, ph, pw)
            out_k = patch.view(B, M, -1)  # (B, M, C_out * pt_k * ph_k * pw_k)

            outputs.append(out_k)
            
        return outputs