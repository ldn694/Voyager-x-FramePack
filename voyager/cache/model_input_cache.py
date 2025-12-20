import torch
from pathlib import Path
import json
from typing import List, Tuple

class ModelInputCache:
    def __init__(self, folder_path):
        self.root = Path(folder_path)
        self.latents = torch.load(self.root / "latents.pt")
        self.cond_latents = torch.load(self.root / "cond_latents.pt")
        self.partial_cond = torch.load(self.root / "partial_cond.pt")
        self.partial_mask = torch.load(self.root / "partial_mask.pt")
        with open(self.root / "sample_id_to_index.json", "r") as f:
            self.sample_id_dict = json.load(f)
        print(f"Loaded ModelInputCache from {folder_path}:")
        print(f"   Latents: {self.latents.shape}")
        print(f"   Cond latents: {self.cond_latents.shape}")
        print(f"   Partial cond: {self.partial_cond.shape}")
        print(f"   Partial mask: {self.partial_mask.shape}")
    
    def get_model_input(self, sample_id: str | List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(sample_id, str):
            row = self.sample_id_dict.get(sample_id, None)
            if row is None:
                return None
            return (self.latents[row],
                    self.cond_latents[row],
                    self.partial_cond[row],
                    self.partial_mask[row])
        elif isinstance(sample_id, list):
            ids = []
            for sid in sample_id:
                row = self.sample_id_dict.get(sid, None)
                if row is None:
                    return None
                ids.append(row)
            ids = torch.tensor(ids, dtype=torch.long)
            return (self.latents[ids],
                    self.cond_latents[ids],
                    self.partial_cond[ids],
                    self.partial_mask[ids])
        else:
            raise ValueError("sample_id must be str or List[str]")