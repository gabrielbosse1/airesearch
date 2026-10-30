"""Interactive raw-byte conversation prototype; not a general chatbot.

By default, recent turns live only in this process. With --memory PATH, Phase 12
AUTOMATICALLY persists complete user/assistant turns and reversible fact vectors
to that unencrypted file. The vector codec is not encryption.
Use --no-episode-storage to disable implicit transcript storage/retrieval; local
history and explicit remember commands still work. A turn saying "do not save
this" is not recorded. User IDs are namespaces, not authentication. Forgetting
a fact also clears that user's stored transcript, not backups or disk remnants.
Independent processes must not concurrently write the same memory checkpoint.
"""
from __future__ import annotations

import argparse

import torch

from models.contextual_response import load_contextual_decoder
from models.conversation import ConversationMemory, ConversationSession
from models.conversation_v2 import ContextualConversationSession
from models.conversation_v3 import VectorConversationSession
from models.diverse_response import load_diverse_decoder
from models.vector_memory import VectorConversationMemory, load_vector_reader
from models.dialogue_response import load_dialogue_decoder
from models.episodic_conversation import EpisodicConversationMemory
from models.memory_model import MemoryModel
from training.train_confidence import load_generator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generator", help="generator checkpoint (defaults to selected version)")
    parser.add_argument("--response", help="contextual response checkpoint bound to the generator")
    parser.add_argument("--vector-reader", default="runs/phase12/vector_reader.pt", help="Phase 12 vector reader checkpoint")
    parser.add_argument("--diverse-response", default="runs/phase12/diverse_response.pt", help="Phase 12 diverse CHAT checkpoint")
    parser.add_argument("--memory", help="opt-in UNENCRYPTED persistent memory checkpoint")
    parser.add_argument("--user", default="local", help="caller-authenticated user namespace (not authentication)")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--temperature", type=float, default=None, help="response byte sampling; 0 for greedy bytes")
    parser.add_argument("--episode-limit", type=int, default=5, help="number of most recent scoped episodes, 1..5")
    parser.add_argument("--no-episode-storage", action="store_true", help="disable automatic episode storage and retrieval")
    legacy = parser.add_mutually_exclusive_group()
    legacy.add_argument("--legacy-phase10", action="store_true", help="use explicit-memory-only Phase 10 checkpoints")
    legacy.add_argument("--legacy-phase11", action="store_true", help="use the Phase 11 transcript-grounded checkpoints")
    parser.add_argument("--show-intent", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    phase = "phase10" if args.legacy_phase10 else "phase11" if args.legacy_phase11 else "phase12"
    generator_path = args.generator or f"runs/{phase}/generator.pt"
    response_path = args.response or f"runs/{phase}/response.pt"
    model = load_generator(generator_path, "cpu", args.heads)
    bank = MemoryModel(d_model=model.d_model, capacity=192)
    if args.legacy_phase10:
        decoder = load_dialogue_decoder(response_path, generator_path=generator_path)
        memory = ConversationMemory(bank, args.memory)
        session = ConversationSession(model, decoder, memory, args.user.encode("utf-8"), min_confidence=args.threshold)
    elif args.legacy_phase11:
        decoder = load_contextual_decoder(response_path, generator_path=generator_path)
        memory = EpisodicConversationMemory(bank, args.memory)
        session = ContextualConversationSession(model, decoder, memory, args.user.encode("utf-8"),
                                                min_confidence=args.threshold, episode_limit=args.episode_limit,
                                                record_episodes=not args.no_episode_storage,
                                                temperature=0.0 if args.temperature is None else args.temperature)
    else:
        decoder = load_contextual_decoder(response_path, generator_path=generator_path)
        codec, reader = load_vector_reader(args.vector_reader, generator_path=generator_path)
        diverse = load_diverse_decoder(args.diverse_response, generator_path=generator_path)
        memory = VectorConversationMemory(bank, codec, args.memory)
        requested_temperature = 0.15 if args.temperature is None else args.temperature
        if not 0 <= requested_temperature <= 2:
            parser.error("temperature must be between 0 and 2")
        session = VectorConversationSession(model, decoder, memory, reader, diverse, args.user.encode("utf-8"),
                                            min_confidence=args.threshold, episode_limit=args.episode_limit,
                                            record_episodes=not args.no_episode_storage,
                                            temperature=requested_temperature or 0.15,
                                            sample_responses=requested_temperature > 0)
    document: bytes | None = None
    print("Byte conversation prototype. /quit exits; /document TEXT or /document-hex HEX loads a byte document.")
    if args.legacy_phase10:
        print("Phase 10: only explicit remember commands save facts; unencrypted persistence is " + ("ON." if args.memory else "OFF."))
    elif args.memory and not args.no_episode_storage:
        print("PRIVACY: full conversation transcripts and any fact vectors are PERSISTED UNENCRYPTED to " + args.memory + ".")
        print("Use --no-episode-storage to turn implicit transcript storage off. User IDs are not authentication.")
        print("Forget my <key> also clears your stored transcript; it cannot erase backups. Do not share concurrent writers.")
    elif args.memory:
        print("Implicit episode storage/retrieval is OFF. Local history remains in process; explicit remembered facts persist UNENCRYPTED.")
    else:
        print("Persistent storage is OFF; recent turns and explicit facts stay in this process only.")
    while True:
        try:
            line = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print(); break
        if line == "/quit":
            break
        if line.startswith("/document "):
            document = line[len("/document "):].encode("utf-8")
            print(f"Document loaded: {len(document)} bytes.")
            continue
        if line.startswith("/document-hex "):
            try:
                document = bytes.fromhex(line[len("/document-hex "):])
            except ValueError:
                print("Invalid hexadecimal bytes."); continue
            print(f"Document loaded: {len(document)} bytes.")
            continue
        if line == "/show-document":
            print("Document:", repr(document))
            continue
        result = session.respond(line.encode("utf-8"), document=document)
        print("Assistant:", result.response.decode("utf-8"))
        if result.edited_document is not None:
            document = result.edited_document
            print("Edited bytes:", repr(document))
        if args.show_intent:
            confidence = "n/a" if result.confidence is None else f"{result.confidence:.3f}"
            print(f"  [intent={result.intent}, confidence={confidence}, action={result.action}, source={result.response_source}, "
                  f"routing={result.routing_reason}, retrieved={result.retrieved_episode_count}, "
                  f"episode_recorded={result.episode_recorded}, persisted={result.episode_persisted}, "
                  f"vectors={result.retrieved_vector_count}, vector_only={result.vector_only}]")


if __name__ == "__main__":
    main()
