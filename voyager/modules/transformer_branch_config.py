from dataclasses import dataclass, asdict
from typing import List

@dataclass
class TransformerBranchConfig:
    hidden_size: int = 512
    heads_num: int = 8
    mlp_width_ratio: float = 4.0
    mlp_act_type: str = "gelu_tanh"
    qk_norm: bool = False
    qk_norm_type: str = "rms"
    qkv_bias: bool = True
    scheduler: List[List[int]] = None # positive: q = first branch, negative or zero: q = second branch
    
    def to_dict(self):
        return asdict(self)
    
def get_transformer_branch_config_from_args(args) -> TransformerBranchConfig:
    return TransformerBranchConfig(
        hidden_size=getattr(args, "hidden_size", 512),
        heads_num=getattr(args, "heads_num", 8),
        mlp_width_ratio=getattr(args, "mlp_width_ratio", 4.0),
        mlp_act_type=getattr(args, "mlp_act_type", "gelu_tanh"),
        qk_norm=getattr(args, "qk_norm", False),
        qk_norm_type=getattr(args, "qk_norm_type", "rms"),
        qkv_bias=getattr(args, "qkv_bias", True),
        scheduler=getattr(args, "scheduler", None),
    )