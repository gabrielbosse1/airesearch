"""ByteCore-DualMemory (Phase 24): byte foundation + gated 8192-dim memory.

* :mod:`models.foundation` -- causal byte Transformer with the Gated Skill
  Adapter (1024 <-> 8192 GLU bridge) in every layer, the confidence-headed
  adaptive refinement loop, and token-id tool generation.
* :mod:`models.memory_model` -- DualMemory: episodic + procedural banks,
  1024-dim keys / 8192-dim values, gradient-free writes, differentiable reads.
* :mod:`models.encoders` -- frozen multimodal encoders + trainable skill
  projectors + the uint8 matrix codec for bidirectional byte prediction.

No tokenizer, no regex tool parsing, no host-side text hacks.
"""

from .foundation import (BOS, EOS, PAD, TOOL_SEARCH, TOOL_SEARCH_END,
                         OBS_START, OBS_END, EDIT, EDIT_END, VOCAB_SIZE,
                         CausalByteTransformer, FoundationConfig,
                         GatedSkillAdapter, IterativeRefinement, RMSNorm,
                         SwiGLUFFN, decode_ids, encode_bytes, load_foundation,
                         make_norm, pad_batch, save_foundation)
from .memory_model import (KEY_DIM, VALUE_DIM, MODALITIES, DualMemory,
                           EpisodicEventBank, ProceduralSkillBank,
                           SkillBankView, stack_views)
from .encoders import (FrozenEncoder, MultimodalIngestor, SkillKeyProjector,
                       SkillProjector, TextSkillProjector, build_encoder,
                       bytes_to_matrix, matrix_to_bytes,
                       pooled_text_embedding, preprocess_image,
                       sample_video_frames)

__all__ = [
    "BOS", "EOS", "PAD", "TOOL_SEARCH", "TOOL_SEARCH_END", "OBS_START",
    "OBS_END", "EDIT", "EDIT_END", "VOCAB_SIZE",
    "CausalByteTransformer", "FoundationConfig", "GatedSkillAdapter",
    "IterativeRefinement", "RMSNorm", "SwiGLUFFN", "decode_ids", "encode_bytes",
    "load_foundation", "make_norm", "pad_batch", "save_foundation",
    "KEY_DIM", "VALUE_DIM", "MODALITIES", "DualMemory", "EpisodicEventBank",
    "ProceduralSkillBank", "SkillBankView", "stack_views",
    "FrozenEncoder", "MultimodalIngestor", "SkillKeyProjector", "SkillProjector",
    "TextSkillProjector", "build_encoder", "bytes_to_matrix", "matrix_to_bytes",
    "pooled_text_embedding", "preprocess_image", "sample_video_frames",
]
