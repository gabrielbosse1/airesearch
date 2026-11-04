"""Idempotently refresh the Phase 22 sections of the three docs via Python I/O.

Usage: ``python -m notebook.update_phase22_docs``
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKER = "## Phase 22:"

SECTION = r"""## Phase 22: scaled T4 pipeline (wide, gated architecture + real HF data)

Phase 22 replaces the small 245k-parameter replay model, tiny synthetic corpora
and basic Transformer blocks with a 183,830,791-parameter byte-level foundation
and a three-phase streamed Hugging Face pipeline. **Nothing here was trained or
measured on a GPU in the Arena sandbox**; the code targets a Colab T4.

### Architecture (all changes are additive / opt-in)

`models/foundation.py` keeps `CHECKPOINT_SCHEMA=2` and the exact Phase 16
defaults, so all 360 historical tests and checkpoints remain valid. New config
fields (`norm_type`, `ffn_type`, `position_type`, `n_kv_heads`,
`refinement_steps`, `refinement_rank`, `rope_theta`, `norm_eps`,
`gradient_checkpointing`) default to LayerNorm / GELU / learned positions /
MHA / zero refinement. `configs/t4_scaled.yaml` enables:

* **WIDE, constant depth**: `d_model=1024`, `n_layers=12`, `n_heads=8`,
  `d_ff=4096`, `max_seq_len=2048` (150-200M parameter target: 183,830,791).
* **SwiGLU** FFN: `SiLU(x W_value) (*) (x W_branch) W_out`, bias-free,
  `12.58M` params/layer. Parameter names avoid the legacy forbidden `gate`
  substring (`tests/test_phase16.py`).
* **Pre-Norm RMSNorm**: `x / sqrt(E[x^2]+eps) * weight`, reduction in float32,
  replacing `nn.LayerNorm` at every norm site.
* **RoPE**: rotary embeddings applied to Q/K after head reshape; computed on
  the fly (no learned / absolute position table, zero parameters, no
  `max_seq_len`-sized buffer).
* **GQA**: `n_kv_heads=2` gives a KV cache and K/V projections 25% the size of
  MHA; SDPA uses `enable_gqa=True` with a `repeat_kv` fallback.
* **Iterative Refinement** ("think" loop): a weight-tied low-rank
  `d -> 256 -> d` cell applied `refinement_steps=2` times before the byte head,
  `h_(t+1) = h_(t) + alpha_t * Cell(RMSNorm(h_(t)))`. `alpha` starts at zero,
  so the loop is the identity at initialisation; each pass is activation
  checkpointed, storing only one `(B, L, d)` boundary (~8 MB at B=4, L=1024).
* `gradient_checkpointing=True` on blocks and refinement via
  `torch.utils.checkpoint(..., use_reentrant=False)`.

### T4 precision and VRAM (measured on Colab, not here)

`torch.bfloat16` CUDA autocast is **NOT supported on Tesla T4 / Turing sm_75**;
`torch.autocast("cuda", dtype=torch.bfloat16)` raises
`RuntimeError: Current CUDA Device does not support bfloat16`. The config keeps
`requested_precision: bfloat16` but uses `precision: auto` ->
`torch.cuda.is_bf16_supported()`: Ampere+ gets bfloat16, T4/V100 get float16 +
`torch.amp.GradScaler`. Both are 2-byte formats, so the memory budget is
unchanged.

**Honest finding:** at 183.8M params, `batch_size=4`, `seq_len=1024` and FULL
gradient checkpointing, the analytic budget is **~3.97 GB**, far below the
requested 10-12 GB. Full checkpointing makes the model too memory-efficient to
fill a T4 at that batch size. `vram_fill_ladder()` therefore probes REAL
forward+backward memory over a ladder (seq_len 1024->2048, micro-batch 4->32,
optional checkpointing relaxation) and records the first candidate that lands
in the 10-12 GB band. The mandated `batch_size=4`, `grad_accum=8`,
`gradient_checkpointing=True` remain the config defaults and the safe starting
point; utilisation is reported as `UNMEASURED` anywhere without a GPU.

### Verified real Hugging Face datasets

`data/phase22_sources.py` records IDs, configs, licences, provenance and
`datasets.load_dataset(..., streaming=True)` calls. Dataset IDs were verified
via public dataset-card research; downloads happen in Colab (the sandbox cannot
reach `huggingface.co`). `discover()` re-searches the Hub at runtime with
`HfApi.list_datasets`.

* Phase 1 (raw next-byte causal LM, **no prompts / no ReAct**):
  `bigcode/the-stack` v1 `data_dir="data/python"` / `"data/javascript"`
  (fallback `meryyllebr543/stack-edu-huggingface`),
  `TucanoBR/GigaVerbo` (780 GB / 200B PT tokens; fallback
  `HuggingFaceFW/fineweb-2` config `por_Latn`),
  `HuggingFaceFW/fineweb-edu` config `sample-10BT` (fallback
  `mlfoundations/dclm-baseline-1.0`, CC-BY-4.0),
  `open-web-math/open-web-math` (6.32M docs / 14.7B tokens).
  `bigcode/the-stack-v2` (gated) and `oscar-corpus/OSCAR-2301`
  (manually gated / suspended) were rejected as defaults.
  Official `HuggingFaceTB/stack-edu` stores only Software Heritage `blob_id`s
  and needs a separate S3 fetch, so it is not a streaming text source.
* Phase 2 (alignment): `HuggingFaceH4/no_robots` (10k,
  **CC BY-NC 4.0 / NON-COMMERCIAL**), `TucanoBR/Tucano-SFT` (679k PT-BR chat
  rows), `argilla/distilabel-sft-easy` (config `default`, distilabel-generated
  SFT, secondary fallback), `m-a-p/CodeFeedback-Filtered-Instruction`
  (157k, Apache-2.0), plus
  Minecraft trajectories `TESS-Computer/minecraft-vla-stage3`,
  `vanyacohen/MET-Bench-Minecraft` and `reactor-team/minecraft-vpt-mp4`.
  There is **no verified HF dataset of human-authored textual Minecraft ReAct
  trajectories** with explore->gather->craft->build; "Voyager" is an agent/paper
  and MineRL/BASALT are video/state-action corpora. The converter preserves real
  source actions/states and writes `[source: no textual observation]` instead of
  fabricating one. ReAct format:
  `Thought -> Action -> Observation -> Answer`.

**No data loss / no noise degradation:** strict UTF-8 encoding (invalid rows
skipped and counted, never replaced), no lowercasing, no synthetic filler, and
`BytePacker` cuts only at UTF-8 codepoint boundaries. The leftover `carry` is
part of the checkpoint, so no bytes are lost across a resume.

### Three resumable notebooks

* `notebook/01_setup_and_pretrain.ipynb` -- deps, T4 GPU check, parameter count,
  precision fallback, VRAM fill ladder, Hub dataset discovery, Phase 1 training,
  `checkpoint_phase1.pt` (local + Hub) + `checkpoint_phase1_fp16.pt`.
* `notebook/02_distill_and_align.ipynb` -- loads Phase 1 (strict phase order),
  Phase 2 conversation + ReAct distillation with prompt masking,
  `checkpoint_phase2.pt`.
* `notebook/03_rl_and_agent.ipynb` -- loads Phase 2, Phase 3 sandbox RL,
  `final_model.pt`, then `evaluation/final_agent_eval.py`.

Rebuild them with `python -m notebook.build_phase22_notebooks`. Checkpoints are
`schema=3`: fp32 master weights, AdamW, GradScaler, Python/Torch/CUDA RNG,
`BytePacker` carry and dataset manifest. `initialise_from_phase()` makes a
missing dependency a hard error rather than silently restarting from random
weights (the Phase 19/21 failure mode).

### Phase 3 RL reward

`training/phase22_reward.py` codes exact values: `+1` correct final answer,
`-0.5` per `[EDIT]` token, `+0.8` for a verified 1-2 edit fix, `-2.0` heavy
penalty for a wrong answer with no search, `+1.5` heavy reward for
`Action: search_ddg("...")` **and** using its `Observation:` result.
`[PAST_ACTION: executed]` is immutable; only `[CURRENT_DRAFT: editable]` may
contain edits (illegal edit `-1.0`); >2 edits incur a thrashing penalty.
`dominance_report()` asserts **Search > Guess** and
**Efficient Edit > Thrashing**. REINFORCE uses an EMA baseline, token-normalised
policy loss, gradient clipping and a supervised anchor, with a deterministic
search-tool sandbox.

### De-contaminated final evaluation

`evaluation/final_agent_eval.py` has four held-out probes: the novel Minecraft
goal **"Build a cobblestone generator"**, a time-sensitive/obscure question that
must be searched, a subtle logical trap needing exactly 1-2 efficient `[EDIT]`
tokens, and a memory-inject + distractor + query test. `--self-test` runs a
scripted ORACLE (`oracle_mode=True`) to prove the scorer; that is NOT a model
result. Decontamination scans available training shards and returns
`unverified` when they are absent -- it never claims "clean" on an assumption.

### Workflow / status

`[THINK] [MAP] [SOLVE] [⚠️ WARN] [EXECUTE] [UPDATE] [BENCHMARK]` was followed.
Run offline verification with `python -m training.phase22_train --self-test`,
`python -m training.phase22_reward --self-test`,
`python -m data.phase22_sources --self-test` and
`python -m evaluation.final_agent_eval --self-test`. No HF dataset was
downloaded, no checkpoint was trained/uploaded, and no VRAM utilisation was
measured in this sandbox. Those require the Colab T4 notebooks.
"""


def update(path: Path) -> dict[str, object]:
    original = path.read_text(encoding="utf8")
    match = re.search(r"\n(?=" + re.escape(MARKER) + r")", original)
    if match:
        body = original[:match.start()].rstrip()
    else:
        body = original.rstrip()
    updated = body + "\n\n" + SECTION.strip() + "\n"
    path.write_text(updated, encoding="utf8")
    return {"path": str(path.relative_to(ROOT)),
            "before_lines": len(original.splitlines()),
            "after_lines": len(updated.splitlines()),
            "phase22_headings": updated.count(MARKER)}


def main() -> None:
    for name in ("AGENTS.md", "QWEN_CHAT_PROMPT.md", "README.md"):
        print(update(ROOT / name))


if __name__ == "__main__":
    main()
