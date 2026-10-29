# ByteCore-DualMemory

A **diffusion-based, byte-level, multimodal cognitive architecture** with a
**dual-model long-term memory system** that learns new concepts, skills, and
people *without full retraining* — via a Memory<->Generator co-training loop
and direct memory writes.

## Philosophy
"Tokens are a crutch." Everything runs on raw bytes (0–255): text (UTF-8),
grayscale images, and 8-bit PCM audio are all fed as byte arrays. No external
tokenizers (`tiktoken`, `transformers.AutoTokenizer`, …) are ever used.

## The two models (co-trained)

| Model | Role |
|-------|------|
| **MemoryModel** (`models/memory_model.py`) | Long-term episodic + semantic memory outside the context window. Remembers people (names, what they did, what was done to them), tracks cross-modal sessions (text → voice), detects lies/contradictions, and stores skills from examples. |
| **GeneratorConditioned** (`models/generator_conditioned.py`) | Byte-level generator conditioned on retrieved memory context. Produces text/code and emits **function-call tokens** (tool use, e.g. Minecraft actions, code execution). |

## Core capabilities (demonstrated in `demo/dual_memory_demo.py`)

1. **Long-term person memory** — remembers "Alice" (name, facts, actions) across a
   TEXT → VOICE modality switch; memory lives outside the context window.
2. **Lie / contradiction detection** — new statements are scored against stored
   facts; contradictions are flagged.
3. **Code generation + function calling** — generates code and emits
   `FUNC_CALL`/`FUNC_ARG`/`FUNC_END` tokens that a host intercepts (e.g.
   `minecraft.move`, `minecraft.mine`, `code.write`), enabling multitasking
   (write code while talking).
4. **Skill learning from examples** — teach a new greeting skill from 6 examples;
   Memory and Generator co-train so the skill becomes stable knowledge with no
   full retraining.
5. **Persistent memory** — `save_system` / `load_system` persist memory + generator
   to disk (`runs/dual_memory/`).

## Co-training (the "models train each other" loop)

The two models apply pressure on each other:

* **Generator → Memory:** the generator must emit output consistent with what
  memory knows (consistency loss). Chaotic output fails memory reconstruction
  and the generator is pushed to be more consistent / "memorable."
* **Memory → Generator:** the memory's retrieved context must actually help the
  generator predict the right bytes (retrieval loss), so memory learns to be a
  better context provider.

## External example ingestion

`training.co_train.ingest_examples()` writes (input, output, context) triples
to memory (fast write, no gradient) then co-trains both models on them. A skill
learned from several examples becomes stable knowledge — the diffusion-style
memory adjustment happens as you give it more external examples.

## Project layout
```
configs/default.yaml          hyperparameters
data/                         synthetic multimodal byte dataset generator
models/
  memory_model.py             long-term memory (people, sessions, lie detection)
  generator_conditioned.py    byte-level generator + function calling
  bytecore_diffusion.py       byte-level diffusion backbone
  concept_bank.py             write-without-retraining concept bank
  byte_embed.py               raw-byte + sinusoidal embeddings
training/
  train.py                    diffusion denoising training
  co_train.py                 Memory <-> Generator co-training loop
evaluation/
  evaluate.py                 accuracy, noise robustness, timing report
demo/
  dual_memory_demo.py         end-to-end capability demo
runs/
  dual_memory/                saved memory + generator checkpoint
  report.md                   evaluation report
```

## Run
```bash
pip install -r requirements.txt
python demo/dual_memory_demo.py     # person memory, lie detection, code+func
                                    # calling, skill learning, co-training
python training/train.py            # diffusion denoising training
python evaluation/evaluate.py       # accuracy + noise report
```

## Results (CPU, from `demo/dual_memory_demo.py`)
- Person memory retained across text→voice switch: **PASS**
- Lie detection (consistent vs contradictory): **PASS**
- Code + function-calling framework: **PASS**
- Co-training mutual improvement: retrieval_sim 0.54→0.96, consistency_sim 0.16→0.88
- Skill from 6 examples without retraining: **PASS**

## Notes
- Raw bytes (0–255) throughout; no external tokenizers.
- Minecraft / voice / real-time chat integration are *host-level* concerns:
  this repo provides the function-calling + memory foundation that makes them
  possible (the `func_handler` callback is the integration point).
- Models are small (CPU-friendly) demonstrations of the architecture; scale
  d_model / n_layers for production use.
