"""PyTorch Dataset + collate for byte-level multimodal diffusion.

Samples are raw byte arrays (0-255), padded/truncated to ``max_seq_len``.
"""

import json
import torch
from torch.utils.data import Dataset


class ByteMultimodalDataset(Dataset):
    def __init__(self, jsonl_path, max_len=128, pad_byte=0):
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))
        self.max_len = max_len
        self.pad_byte = pad_byte

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        byte_ids = list(s["bytes"])[: self.max_len]
        if len(byte_ids) < self.max_len:
            byte_ids = byte_ids + [self.pad_byte] * (self.max_len - len(byte_ids))
        return {
            "input_ids": torch.tensor(byte_ids, dtype=torch.long),  # (L,)
            "modality": s["modality"],
            "concept": s["concept"],
            "real_len": min(len(list(s["bytes"])), self.max_len),
        }


def collate(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])  # (B, L)
    return {
        "input_ids": input_ids,
        "modalities": [b["modality"] for b in batch],
        "concepts": [b["concept"] for b in batch],
        "real_lens": [b["real_len"] for b in batch],
    }
