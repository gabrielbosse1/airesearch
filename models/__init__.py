"""ByteCore-Diffusion: a byte-level diffusion model with a concept bank.

Multimodal (image / text / audio) generative model that operates on raw
bytes (0-255) -- no external tokenizers. A concept bank stores learned
concepts as memory entries; new concepts are written (not trained), so the
model learns new concepts without retraining.
"""

from .bytecore_diffusion import ByteCoreDiffusion
from .concept_bank import ConceptBank
from .diffusion_core import DiffusionCore
from .memory_model import MemoryModel
from .generator_conditioned import GeneratorConditioned
from .cognitive import CognitiveAction, CognitiveObservation, CognitiveResult
from .contradiction_graph import Fact, ContradictionReport
from .procedural import Procedure, ProcedureExecution
from .alignment import soft_dtw_alignment
from .holdout import HoldoutGuard, HoldoutViolation, content_hash, pair_hash
from .edit_router import EditRequest, EditOutcome, EditRouter, route_edit
from .conversation_intents import ConversationIntent, INTENT_NAMES, NUM_INTENTS
from .dialogue_response import ByteDialogueDecoder, load_dialogue_decoder
from .conversation import ConversationMemory, ConversationResult, ConversationSession, ConversationTurn
from .contextual_response import ContextualByteDecoder, load_contextual_decoder, pack_response_context
from .episodic_conversation import EpisodicConversationMemory, RetrievedEpisodes
from .conversation_v2 import ContextualConversationSession
from .vector_memory import VectorFactCodec, VectorMemoryReader, VectorConversationMemory, load_vector_reader
from .diverse_response import DiverseByteDecoder, distribution_diversity_loss, load_diverse_decoder
from .conversation_v3 import VectorConversationSession
from .sandbox import ByteSandbox, SandboxViolation
from .tool_use import (MAX_ARGUMENT_BYTES, MAX_RESULT_BYTES, ROUTE_NAMES, TOOL_NAMES, ToolCall,
                       ToolCallPolicy, ToolFrameError, ToolFrameTruncated, ToolGeneration,
                       ToolArgumentDecoder, ToolRouteHead, encode_tool_call, encode_tool_result,
                       find_tool_call, load_tool_policy, pack_tool_request, parse_tool_call,
                       sanitize_result)
# Phase 16 / ByteCore v2: the scalable foundation.
from .foundation import (BOS, EOS, PAD, VOCAB_SIZE, CausalByteTransformer, FoundationConfig,
                         IterativeRefinement, RMSNorm, SwiGLUFFN, apply_rotary, make_norm,
                         repeat_kv, rotary_tables, rotate_half,
                         decode_ids, encode_bytes, load_foundation, save_foundation)
from .react import (ReactLoop, ReactRun, ActionCall, format_action, format_observation,
                    last_action, observation_line, parse_action, render_prompt,
                    render_trajectory, sandbox_observation)
from .foundation_memory import (MemoryEntry, PrefixMemory, ScriptedRetriever,
                                VectorFactRetriever, decode_vectors)

__all__ = [
    "ByteCoreDiffusion", "ConceptBank", "DiffusionCore", "MemoryModel",
    "GeneratorConditioned", "CognitiveAction", "CognitiveObservation",
    "CognitiveResult", "Fact", "ContradictionReport", "Procedure", "ProcedureExecution",
    "soft_dtw_alignment", "HoldoutGuard", "HoldoutViolation", "content_hash", "pair_hash",
    "EditRequest", "EditOutcome", "EditRouter", "route_edit",
    "ConversationIntent", "INTENT_NAMES", "NUM_INTENTS", "ByteDialogueDecoder", "load_dialogue_decoder",
    "ConversationMemory", "ConversationResult", "ConversationSession", "ConversationTurn",
    "ContextualByteDecoder", "load_contextual_decoder", "pack_response_context",
    "EpisodicConversationMemory", "RetrievedEpisodes", "ContextualConversationSession",
    "VectorFactCodec", "VectorMemoryReader", "VectorConversationMemory", "load_vector_reader",
    "DiverseByteDecoder", "distribution_diversity_loss", "load_diverse_decoder", "VectorConversationSession",
    "ByteSandbox", "SandboxViolation", "MAX_ARGUMENT_BYTES", "MAX_RESULT_BYTES", "ROUTE_NAMES", "TOOL_NAMES",
    "ToolCall", "ToolCallPolicy", "ToolFrameError", "ToolFrameTruncated", "ToolGeneration",
    "ToolArgumentDecoder", "ToolRouteHead", "encode_tool_call", "encode_tool_result", "find_tool_call",
    "load_tool_policy", "pack_tool_request", "parse_tool_call", "sanitize_result",
    "BOS", "EOS", "PAD", "VOCAB_SIZE", "CausalByteTransformer", "FoundationConfig",
    "IterativeRefinement", "RMSNorm", "SwiGLUFFN", "apply_rotary", "make_norm",
    "repeat_kv", "rotary_tables", "rotate_half",
    "decode_ids", "encode_bytes", "load_foundation", "save_foundation",
    "ReactLoop", "ReactRun", "ActionCall", "format_action", "format_observation", "last_action",
    "observation_line", "parse_action", "render_prompt", "render_trajectory", "sandbox_observation",
    "MemoryEntry", "PrefixMemory", "ScriptedRetriever", "VectorFactRetriever", "decode_vectors",
]
