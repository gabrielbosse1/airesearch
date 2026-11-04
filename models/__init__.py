"""ByteCore-DualMemory: byte-level foundation + hierarchical external memory.

Phase 22 / 23 surface only. Everything importable here is on the runtime path
of the current architecture:

* :mod:`models.foundation` -- the causal byte-level Transformer (RMSNorm,
  SwiGLU, RoPE, GQA, weight-tied iterative refinement).
* :mod:`models.memory_model` -- episodic / semantic-exemplar / procedural
  hierarchical memory with differentiable top-K reads.
* :mod:`models.foundation_memory` -- memory injected into the byte stream as a
  ``[MEM] topic = value`` prefix.
* :mod:`models.react` and :mod:`models.sandbox` -- ``Thought/Action/Observation``
  tool use as ordinary bytes, executed in a sandbox.
* :mod:`models.holdout`, :mod:`models.procedural`, :mod:`models.contradiction_graph`
  -- provenance guard, byte-exact procedure replay and temporal fact graphs.

No tokenizers, no diffusion core, no pointer-generator copy head and no
conversation stack remain in the tree; see ``ARCHITECTURE_AUDIT.md``.
"""

from .contradiction_graph import Fact, ContradictionReport
from .foundation import (BOS, EOS, PAD, REF_END, REF_START, VOCAB_SIZE,
                         CausalByteTransformer, FoundationConfig,
                         IterativeRefinement, RMSNorm, SwiGLUFFN, apply_rotary,
                         decode_ids, encode_bytes, load_foundation, make_norm,
                         pad_batch, repeat_kv, rotary_tables, rotate_half,
                         save_foundation)
from .foundation_memory import MemoryEntry, PrefixMemory, ScriptedRetriever
from .holdout import HoldoutGuard, HoldoutViolation, content_hash, pair_hash
from .memory_model import (ALL_TIERS, TIER_EPISODIC, TIER_PROCEDURAL,
                           TIER_SEMANTIC, EpisodicMemoryBank, ImportanceScorer,
                           MemoryModel, ProceduralSkillBank, SemanticExemplarBank)
from .procedural import Procedure, ProcedureExecution
from .react import (ActionCall, ReactLoop, ReactRun, format_action,
                    format_observation, last_action, observation_line,
                    parse_action, render_prompt, render_trajectory,
                    sandbox_observation)
from .sandbox import ByteSandbox, SandboxViolation

__all__ = [
    "Fact", "ContradictionReport",
    "BOS", "EOS", "PAD", "REF_START", "REF_END", "VOCAB_SIZE",
    "CausalByteTransformer", "FoundationConfig", "IterativeRefinement",
    "RMSNorm", "SwiGLUFFN", "apply_rotary", "decode_ids", "encode_bytes",
    "load_foundation", "make_norm", "pad_batch", "repeat_kv", "rotary_tables",
    "rotate_half", "save_foundation",
    "MemoryEntry", "PrefixMemory", "ScriptedRetriever",
    "HoldoutGuard", "HoldoutViolation", "content_hash", "pair_hash",
    "ALL_TIERS", "TIER_EPISODIC", "TIER_PROCEDURAL", "TIER_SEMANTIC",
    "EpisodicMemoryBank", "ImportanceScorer", "MemoryModel",
    "ProceduralSkillBank", "SemanticExemplarBank",
    "Procedure", "ProcedureExecution",
    "ActionCall", "ReactLoop", "ReactRun", "format_action",
    "format_observation", "last_action", "observation_line", "parse_action",
    "render_prompt", "render_trajectory", "sandbox_observation",
    "ByteSandbox", "SandboxViolation",
]
