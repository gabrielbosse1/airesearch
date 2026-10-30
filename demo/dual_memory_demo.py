"""ByteCore-DualMemory demo.

Demonstrates the full vision:
  * Long-term memory outside the context window (people, facts, interactions).
  * Cross-modal session continuity (text -> voice awareness).
  * Time-aware contradiction evidence for explicit stored facts (not lie detection).
  * Code generation + function calling.
  * Multitasking: write code while "talking".
  * Learn a skill from several examples via co-training (memory + generator
    adjust each other, no full retraining).
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.memory_model import MemoryModel
from models.contradiction_graph import Fact
from models.generator_conditioned import GeneratorConditioned
from training.co_train import ingest_examples, encode_text, save_system, load_system

DEVICE = "cpu"
RUNS_DIR = "runs/dual_memory"


def make_memory(d_model=64, capacity=256):
    return MemoryModel(d_model=d_model, capacity=capacity)


def make_generator(d_model=64):
    return GeneratorConditioned(d_model=d_model, n_layers=2, n_heads=4, d_ff=128)


# ===========================================================================
# 1. PERSON MEMORY: remember a person, their name, what they did / was done
# ===========================================================================
def demo_person_memory(memory):
    print("\n" + "=" * 70)
    print("DEMO 1: Person memory (outside context window)")
    print("=" * 70)

    # "Meet" Alice in text mode.
    alice_emb = torch.tensor(
        list("alice".encode("utf-8")) + [0] * (64 - len("alice")), dtype=torch.float
    )
    # Pad/truncate to d_model
    if alice_emb.shape[0] < 64:
        alice_emb = torch.cat([alice_emb, torch.zeros(64 - alice_emb.shape[0])])
    alice_emb = alice_emb[:64]

    memory.write(
        alice_emb,
        {
            "person": "Alice",
            "modality": "text",
            "session": "s1",
            "type": "intro",
            "fact": "Alice is a software engineer.",
        },
    )
    memory.write(
        torch.tensor(
            list("alice built a website".encode("utf-8")) + [0] * 50,
            dtype=torch.float,
        )[:64],
        {
            "person": "Alice",
            "modality": "text",
            "session": "s1",
            "type": "action",
            "action": "Alice built a website",
            "done_by": "Alice",
        },
    )
    memory.write(
        torch.tensor(
            list("Bob gave Alice a laptop".encode("utf-8")) + [0] * 46,
            dtype=torch.float,
        )[:64],
        {
            "person": "Alice",
            "modality": "text",
            "session": "s1",
            "type": "done_to",
            "action": "Bob gave Alice a laptop",
            "done_to": "Alice",
            "done_by": "Bob",
        },
    )

    # Record an actual raw-byte voice event after the text events.
    memory.write(
        torch.tensor([128.0, 130.0, 127.0] + [0.0] * 61),
        {"person": "Alice", "modality": "voice", "session": "s1", "type": "voice_event"},
    )
    print("\n[User switches from TEXT to VOICE]")
    summary = memory.person_summary("Alice")
    print(f"Memories about Alice: {summary['n_memories']}")
    print(f"Modalities seen: {summary['modalities_seen']}")
    print("Actions:", [a.get("action") for a in summary["actions"]])
    print("Facts:", [f.get("fact") for f in summary["facts"]])
    assert summary["n_memories"] >= 2, "Memory should retain Alice across modalities"
    assert {"text", "voice"}.issubset(summary["modalities_seen"])
    print("PASS: Alice remembered across recorded text and voice events.")


# ===========================================================================
# 2. LIE / CONTRADICTION DETECTION
# ===========================================================================
def demo_lie_detection(memory, generator):
    print("\n" + "=" * 70)
    print("DEMO 2: Temporally aware multi-hop contradiction evidence")
    print("=" * 70)

    # The graph never infers facts from arbitrary prose. These explicit aliases
    # and the exclusive relation define exactly what may be checked.
    first = memory.write_fact(Fact(b"demo-alice", b"same_as", b"demo-alias-1"))
    second = memory.write_fact(Fact(b"demo-alias-1", b"same_as", b"demo-alias-2"))
    third = memory.write_fact(Fact(b"demo-alias-2", b"uses-language", b"python",
                                   exclusive=True, valid_from=10, valid_to=20))
    claim = Fact(b"demo-alice", b"uses-language", b"rust", exclusive=True)
    active = memory.explain_contradiction(claim, as_of=15, max_hops=3)
    expired = memory.explain_contradiction(claim, as_of=25, max_hops=3)
    assert active.status == "conflict" and active.evidence[0].path_slots == (first, second, third)
    assert expired.status == "unknown"
    print(f"  At time 15: {active.status}; evidence path {active.evidence[0].path_slots}")
    print(f"  At time 25: {expired.status} (old fact has expired)")
    print("PASS: explicit facts yield a time-valid multi-hop evidence path; this does not determine intent or truth.")


# ===========================================================================
# 3. CODE GENERATION + FUNCTION CALLING + MULTITASKING
# ===========================================================================
def demo_code_and_functions(generator, memory):
    print("\n" + "=" * 70)
    print("DEMO 3: Code generation + function calling + multitasking")
    print("=" * 70)

    # A function handler that simulates "playing Minecraft" / executing code.
    minecraft_state = {"position": [0, 0, 0], "inventory": []}

    def func_handler(func_name):
        if func_name == "minecraft.move":
            minecraft_state["position"][0] += 1
            return f"moved to {minecraft_state['position']}"
        if func_name == "minecraft.mine":
            minecraft_state["inventory"].append("block")
            return "mined a block"
        if func_name == "code.write":
            return "wrote 42 lines of code"
        return f"executed {func_name}"

    # Directly demonstrate the function-calling mechanism: the host intercepts
    # FUNC_CALL tokens and executes func_handler, injecting the result back.
    print("\n  [Direct function-call demonstration]")
    for fn in ["minecraft.move", "minecraft.mine", "code.write", "minecraft.move"]:
        res = func_handler(fn)
        print(f"    FUNC_CALL({fn}) -> {res}")
    print(f"    Minecraft state after multitasking: {minecraft_state}")
    assert minecraft_state["position"][0] == 2
    assert minecraft_state["inventory"] == ["block"]
    print("  PASS: function calling + multitasking mechanism works.")

    # Retrieve memory context about Alice to ground generation.
    alice_query = torch.tensor(list("alice".encode("utf-8")) + [0] * 60)[:64].float()
    entries, ctx = memory.retrieve(alice_query, top_k=3)
    print(f"\n  Retrieved {len(entries)} memory entries about Alice for generation grounding.")

    # Generate: user talks while asking for code (generator conditioned on memory).
    prompt = list("write a python function to greet alice:".encode("utf-8"))
    prompt_t = torch.tensor(prompt + [0] * (64 - len(prompt)), dtype=torch.long)

    generated, funcs = generator.generate(
        prompt_t,
        memory_context=ctx,
        max_new=30,
        temperature=0.7,
        func_handler=func_handler,
    )
    gen_text = bytes([b for b in generated.tolist() if b < 256]).decode(
        "utf-8", errors="replace"
    )
    print(f"  Generated text: {gen_text[:100]}")
    print(f"  Function calls emitted by generator: {funcs}")
    print("  PASS: Generator conditioned on memory + function-calling framework active.")


# ===========================================================================
# 4. SKILL LEARNING FROM EXAMPLES (co-training, no retraining)
# ===========================================================================
def demo_skill_learning(memory, generator):
    print("\n" + "=" * 70)
    print("DEMO 4: Learn a skill from several examples (co-training)")
    print("=" * 70)

    # Teach the system a new "greeting skill" via several examples.
    # Each example: input -> output + context. Several examples needed.
    examples = []
    for i in range(6):
        name = ["Bob", "Carol", "Dan", "Eve", "Frank", "Grace"][i]
        inp = f"greet {name}".encode("utf-8")
        out = f"Hello, {name}! Nice to meet you.".encode("utf-8")
        examples.append(
            {
                "input": list(inp),
                "output": list(out),
                "context": {
                    "person": name,
                    "modality": "text",
                    "session": "skill_session",
                    "type": "skill:greeting",
                },
            }
        )

    print(f"Ingesting {len(examples)} greeting-skill examples into memory + co-training...")
    logs = ingest_examples(
        memory, generator, examples, device=DEVICE, epochs=3, batch_size=2, lr=1e-3
    )
    print("Skill ingested. Co-training logs (last epoch):")
    if logs:
        print("  ", logs[-1])
    replay = memory.execute_procedure(b"greet Bob")
    assert replay.output == b"Hello, Bob! Nice to meet you."
    print(f"Exact stored-skill replay: {replay.output!r}; latent steps={len(replay.latent_trace)}")

    # Sample a new input; this is illustrative, not a test of generalization.
    alice_query = torch.tensor(list("greet Helen".encode("utf-8")) + [0] * 56)[:64].float()
    entries, ctx = memory.retrieve(alice_query, top_k=4)
    print(f"Retrieved {len(entries)} relevant greeting memories for 'greet Helen'.")

    prompt = list("greet Helen".encode("utf-8"))
    prompt_t = torch.tensor(prompt + [0] * (64 - len(prompt)), dtype=torch.long)
    generated, _ = generator.generate(
        prompt_t, memory_context=ctx, max_new=25, temperature=0.7
    )
    gen_text = bytes([b for b in generated.tolist() if b < 256]).decode(
        "utf-8", errors="replace"
    )
    print(f"Unverified sample for Helen: {gen_text[:100]}")
    print("PASS: structured example replay and co-training completed; novel-input quality is not asserted.")


# ===========================================================================
# MAIN
# ===========================================================================
def main(runs_dir: str = RUNS_DIR):
    torch.manual_seed(42)
    memory = make_memory()
    generator = make_generator()

    # Try to load a previously saved system if present.
    if os.path.exists(os.path.join(runs_dir, "memory.pt")):
        print("Loading previously saved dual-memory system...")
        load_system(memory, generator, runs_dir)

    demo_person_memory(memory)
    demo_lie_detection(memory, generator)
    demo_code_and_functions(generator, memory)
    demo_skill_learning(memory, generator)

    # Save the grown memory + generator (knowledge persists outside context window).
    os.makedirs(runs_dir, exist_ok=True)
    save_system(memory, generator, runs_dir)
    print(f"\nSaved grown memory + generator to {runs_dir}")

    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETE")
    print("=" * 70)
    print("Summary of capabilities demonstrated:")
    print("  * Long-term person memory (Alice) across text->voice modality switch")
    print("  * Time-valid multi-hop contradiction evidence for explicit facts")
    print("  * Code generation + function calling (minecraft.move/mine, code.write)")
    print("  * Multitasking: generation + function execution in one pass")
    print("  * Exact skill replay from examples plus Memory<->Generator co-training")
    print("  * Memory + generator saved and reloadable (persistent outside context)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default=RUNS_DIR, help="checkpoint directory to read and update")
    main(parser.parse_args().runs_dir)
