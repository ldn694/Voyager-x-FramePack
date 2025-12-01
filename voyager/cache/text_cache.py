import torch
from pathlib import Path
import json
from typing import List, Tuple

class TextEncoderCache:
    def __init__(self, folder_path):
        self.root = Path(folder_path)
        self.clipl_text_states = torch.load(self.root / "clipl_text_states.pt")
        self.llm_i2v_text_states = torch.load(self.root / "llm_i2v_text_states.pt")
        self.llm_i2v_text_masks = torch.load(self.root / "llm_i2v_text_mask.pt")
        with open(self.root / "sample_id.json", "r") as f:
            self.sample_id_dict = json.load(f)
        print(f"Loaded TextEncoderCache from {folder_path}:")
        print(f"  CLIPL text states: {self.clipl_text_states.shape}")
        print(f"  LLM I2V text states: {self.llm_i2v_text_states.shape}")
        print(f"  LLM I2V text masks: {self.llm_i2v_text_masks.shape}")
    
    def get_clipl_text_state(self, sample_id: str | List[str]) -> torch.Tensor:
        if isinstance(sample_id, str):
            row = self.sample_id_dict[sample_id]["row"]
            return self.clipl_text_states[row]
        elif isinstance(sample_id, list):
            ids = []
            for sid in sample_id:
                row = self.sample_id_dict[sid]["row"]
                ids.append(row)
            ids = torch.tensor(ids, dtype=torch.long)
            return self.clipl_text_states[ids]
        else:
            raise ValueError("sample_id must be str or List[str]")
    
    # def _get_clipl_zero_row(self) -> torch.Tensor:
    #     """Return a single zero row with same feature dim & device as clipl_text_states."""
    #     feature_shape = self.clipl_text_states.shape[1:]
    #     return torch.zeros(
    #         (1, *feature_shape),
    #         dtype=self.clipl_text_states.dtype,
    #         device=self.clipl_text_states.device,
    #     )
    
    # def get_clipl_text_state(self, sample_id: str | List[str]) -> torch.Tensor:
    #     if isinstance(sample_id, str):
    #         info = self.sample_id_dict.get(sample_id)
    #         if info is None:
    #             # Unknown ID → return a single zero row
    #             return self._get_clipl_zero_row().squeeze(0)
    #         row = info["row"]
    #         return self.clipl_text_states[row]

    #     elif isinstance(sample_id, list):
    #         batch_size = len(sample_id)
    #         feature_shape = self.clipl_text_states.shape[1:]
    #         # Preallocate (B, *feat) and fill row by row
    #         out = torch.zeros(
    #             (batch_size, *feature_shape),
    #             dtype=self.clipl_text_states.dtype,
    #             device=self.clipl_text_states.device,
    #         )
    #         for i, sid in enumerate(sample_id):
    #             info = self.sample_id_dict.get(sid)
    #             if info is None:
    #                 # Leave zero row for unknown sid
    #                 continue
    #             row = info["row"]
    #             out[i] = self.clipl_text_states[row]
    #         return out

    #     else:
    #         raise ValueError("sample_id must be str or List[str]")
    
    def get_llm_i2v_text_state_and_mask(self, sample_id: str | List[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(sample_id, str):
            if sample_id not in self.sample_id_dict:
                return (torch.zeros(1,1, dtype=self.llm_i2v_text_states.dtype), torch.zeros(1,1, dtype=self.llm_i2v_text_masks.dtype))
            row = self.sample_id_dict[sample_id]["row"]
            text_mask_len = self.sample_id_dict[sample_id]["text_mask_len"]
            return (self.llm_i2v_text_states[row][:text_mask_len],
                    self.llm_i2v_text_masks[row][:text_mask_len])
        elif isinstance(sample_id, list):
            state_list = []
            mask_list = []
            max_text_mask_len = max([self.sample_id_dict[sid]["text_mask_len"] for sid in sample_id])
            for sid in sample_id:
                row = self.sample_id_dict[sid]["row"]
                state_list.append(self.llm_i2v_text_states[row][:max_text_mask_len])
                mask_list.append(self.llm_i2v_text_masks[row][:max_text_mask_len])
            return torch.stack(state_list, dim=0), torch.stack(mask_list, dim=0)
        else:
            raise ValueError("sample_id must be str or List[str]")

    # def _get_llm_zero_state_and_mask(
    #     self,
    #     seq_len: int = 1,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Return (state, mask) zeros with shape:
    #       state: (seq_len, *state_tail_shape)
    #       mask:  (seq_len, *mask_tail_shape)

    #     where original shapes are:
    #       llm_i2v_text_states: (N, T, *state_tail_shape)
    #       llm_i2v_text_masks:  (N, T, *mask_tail_shape)
    #     """
    #     state_tail_shape = self.llm_i2v_text_states.shape[2:]
    #     mask_tail_shape = self.llm_i2v_text_masks.shape[2:]

    #     device = self.llm_i2v_text_states.device
    #     dtype_states = self.llm_i2v_text_states.dtype
    #     dtype_masks = self.llm_i2v_text_masks.dtype

    #     state = torch.zeros(
    #         (seq_len, *state_tail_shape),
    #         dtype=dtype_states,
    #         device=device,
    #     )
    #     mask = torch.zeros(
    #         (seq_len, *mask_tail_shape),
    #         dtype=dtype_masks,
    #         device=device,
    #     )
    #     return state, mask

    # def get_llm_i2v_text_state_and_mask(
    #     self, sample_id: str | List[str]
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     # -------- single ID --------
    #     if isinstance(sample_id, str):
    #         info = self.sample_id_dict.get(sample_id)
    #         if info is None:
    #             # Unknown → minimal zero sequence
    #             return self._get_llm_zero_state_and_mask(seq_len=1)

    #         row = info["row"]
    #         text_mask_len = info["text_mask_len"]
    #         # shapes: (T, *tail)
    #         state = self.llm_i2v_text_states[row][:text_mask_len]
    #         mask = self.llm_i2v_text_masks[row][:text_mask_len]
    #         return state, mask

    #     # -------- list of IDs --------
    #     elif isinstance(sample_id, list):
    #         if len(sample_id) == 0:
    #             raise ValueError("sample_id list must not be empty")

    #         state_tail_shape = self.llm_i2v_text_states.shape[2:]
    #         mask_tail_shape = self.llm_i2v_text_masks.shape[2:]

    #         device = self.llm_i2v_text_states.device
    #         dtype_states = self.llm_i2v_text_states.dtype
    #         dtype_masks = self.llm_i2v_text_masks.dtype

    #         # Per-sample lengths, default 1 for unknown
    #         lengths: List[int] = []
    #         for sid in sample_id:
    #             info = self.sample_id_dict.get(sid)
    #             if info is None:
    #                 lengths.append(1)
    #             else:
    #                 lengths.append(info["text_mask_len"])

    #         max_text_mask_len = max(lengths)
    #         batch_size = len(sample_id)

    #         # (B, T, *tail)
    #         state_out = torch.zeros(
    #             (batch_size, max_text_mask_len, *state_tail_shape),
    #             dtype=dtype_states,
    #             device=device,
    #         )
    #         mask_out = torch.zeros(
    #             (batch_size, max_text_mask_len, *mask_tail_shape),
    #             dtype=dtype_masks,
    #             device=device,
    #         )

    #         for i, sid in enumerate(sample_id):
    #             info = self.sample_id_dict.get(sid)
    #             if info is None:
    #                 # keep zeros
    #                 continue
    #             row = info["row"]
    #             text_mask_len = info["text_mask_len"]

    #             state_slice = self.llm_i2v_text_states[row][:text_mask_len]
    #             mask_slice = self.llm_i2v_text_masks[row][:text_mask_len]

    #             state_out[i, :text_mask_len] = state_slice
    #             mask_out[i, :text_mask_len] = mask_slice

    #         return state_out, mask_out

    #     else:
    #         raise ValueError("sample_id must be str or List[str]")