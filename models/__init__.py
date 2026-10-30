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
]
