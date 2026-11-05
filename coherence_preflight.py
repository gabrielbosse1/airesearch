"""Phase 25.2 coherence pre-flight: 500 steps, PT-BR + code, strict assertions.

Sandbox default is a CPU-feasible scale of the SAME Phase 25.1 architecture
path: sliding-window attention (window_size=1024), memory cross-attention in
all 8 layers, refinement/thought loop, delayed memory commits. It uses a
deterministic offline 1000-example PT-BR/code slice because the sandbox cannot
stream from Hugging Face. On Colab/T4, pass --production to use the full preset.

    python coherence_preflight.py                 # 500-step sandbox pre-flight
    python coherence_preflight.py --production    # full T4 preset (GPU)

The script never writes training data to disk. Checkpoints/metrics live under
ignored runs/. Exit code 0 means every coherence assertion passed.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Iterator

import torch

from models.encoders import pooled_text_embedding
from models.foundation import PAD, CausalByteTransformer, FoundationConfig, decode_ids, encode_bytes
from training.train import Trainer, load_config

PTBR_PROMPT = "Olá, eu sou um modelo de linguagem e"
CODE_PROMPT = "def fibonacci(n):"
PTBR_MARKERS = ("modelo", "linguagem", "português", "brasil", "ajudar", "texto", "código")
CODE_MARKERS = ("return", "if", "for", "def", "fibonacci", "n")


def offline_slice(count: int = 1000, *, seed: int = 252) -> list[tuple[str, dict[str, Any]]]:
    """Deterministic high-quality PT-BR/code fixture slice for offline pre-flight.

    This is NOT a wholesale dataset download. It is a small in-memory slice used
    only because the sandbox cannot reach the Hub. Production runs stream.
    """
    if count < 2:
        raise ValueError("count must be >= 2")
    rng = random.Random(seed)
    ptbr_templates = [
        PTBR_PROMPT + " estou aqui para ajudar. Eu escrevo texto em português do Brasil e também posso gerar código Python simples.",
        PTBR_PROMPT + " sou um modelo de linguagem treinado para continuar texto e código com clareza.",
        "O Brasil é um país da América do Sul. A língua oficial é o português. Nossa cultura mistura música, literatura e tecnologia.",
        "Um modelo de linguagem lê bytes e aprende padrões. A atenção deslizante mantém o contexto local, enquanto a memória recupera informação distante.",
        "Olá, eu sou um modelo de linguagem e posso ajudar com texto, código e raciocínio passo a passo. A coerência vem de padrões aprendidos, não de sorte.",
        "A memória persistente guarda habilidades e eventos. Quando o contexto local não basta, a atenção cruzada consulta a memória comprometida.",
        "Programar em Python exige indentação correta, nomes claros e funções pequenas. Um teste de coerência deve verificar sintaxe e palavras reconhecíveis.",
        "Eu gosto de café, música brasileira e código limpo. A língua portuguesa tem acentos e bytes UTF-8; o modelo aprende byte a byte.",
    ]
    code_templates = [
        CODE_PROMPT + "\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
        "def soma(a, b):\n    return a + b\n\nprint(soma(2, 3))\n",
        "def fatorial(n):\n    if n <= 1:\n        return 1\n    return n * fatorial(n - 1)\n",
        "def eh_par(n):\n    return n % 2 == 0\n\nprint(eh_par(10))\n",
        "def media(valores):\n    if not valores:\n        return 0.0\n    return sum(valores) / len(valores)\n",
        "def contar_palavras(texto):\n    palavras = texto.split()\n    return len(palavras)\n",
        "def fibonacci_lista(n):\n    seq = [0, 1]\n    for _ in range(2, n):\n        seq.append(seq[-1] + seq[-2])\n    return seq[:n]\n",
        "class Contador:\n    def __init__(self):\n        self.valor = 0\n\n    def incrementar(self):\n        self.valor += 1\n        return self.valor\n",
    ]
    examples: list[tuple[str, dict[str, Any]]] = []
    half = count // 2
    for i in range(half):
        # Repeat the exact prompt completions often enough for a 500-step
        # overfit-style pre-flight, while keeping varied natural sentences.
        if i % 3 == 0:
            text = ptbr_templates[0]
        elif i % 3 == 1:
            text = ptbr_templates[1]
        else:
            text = ptbr_templates[2 + (i % (len(ptbr_templates) - 2))]
        if i % 5 == 0:
            text += f" Exemplo {i}."
        examples.append(("ptbr", {"kind": "text", "data": text.encode("utf8")}))
    for i in range(count - half):
        if i % 3 == 0:
            text = code_templates[0]
        else:
            text = code_templates[1 + (i % (len(code_templates) - 1))]
        if i % 7 == 0:
            text += f"# exemplo {i}\n"
        examples.append(("code", {"kind": "text", "data": text.encode("utf8")}))
    rng.shuffle(examples)
    return examples


def stream_examples(examples: list[tuple[str, dict[str, Any]]], *, seed: int = 252
                    ) -> Iterator[tuple[str, dict[str, Any]]]:
    """Len-less cycling iterator: Trainer rejects sized epoch-style datasets."""
    rng = random.Random(seed)
    pool = list(examples)
    while True:
        rng.shuffle(pool)
        for item in pool:
            yield item


def preflight_config(*, steps: int = 500, workdir: str | Path = "runs/coherence_preflight",
                     production: bool = False) -> dict[str, Any]:
    config = load_config("configs/t4_scaled.yaml")
    config["training"] = dict(config.get("training", {}))
    config["training"].update({
        "max_steps": int(steps), "micro_batch_size": 1 if production else 2,
        "grad_accum": 32 if production else 1,
        "seq_len": 16384 if production else 512,
        "learning_rate": 3e-4 if production else 2e-3,
        "warmup_steps": 50 if production else 10,
        "log_every": 50, "checkpoint_every": 0, "checkpoint_minutes": 1e9,
        "tool_trajectory_rate": 0.0, "seed": 252,
        "ingest_to_memory": True,
    })
    config["encoders"] = {"image": None, "audio": None, "video": None}
    config["checkpoints"] = {"local_dir": str(Path(workdir) / "checkpoints"),
                             "hf_repo": None}
    config["output"] = {"run_dir": str(workdir),
                        "metrics_jsonl": str(Path(workdir) / "metrics.jsonl"),
                        "evaluate_final": False}
    if not production:
        # Same architecture path, CPU-feasible width. Window stays 1024.
        config["model"] = dict(config["model"])
        config["model"].update({
            "d_model": 128, "n_layers": 8, "n_heads": 4, "n_kv_heads": 2,
            "d_ff": 256, "max_seq_len": 2048, "dropout": 0.0,
            "attention_type": "sliding_window", "window_size": 1024,
            "sliding_chunk_size": 256, "memory_cross_attention": True,
            "refinement_steps": 4, "refinement_rank": 64, "max_thought_steps": 4,
            "memory_dim": 256, "gradient_checkpointing": True,
        })
        config["memory"] = {"key_dim": 128, "value_dim": 256, "capacity": 64,
                            "decay_margin": 0.25, "max_eviction_events": 256}
        config["vram"] = dict(config.get("vram", {}))
        config["vram"]["frozen_encoder_dtype"] = "float32"
    FoundationConfig.from_mapping(config["model"])  # fail fast
    return config


def seed_memory_from_slice(trainer: Trainer, examples: list[tuple[str, dict[str, Any]]],
                           *, slots: int = 32) -> int:
    """Commit detached text skills so text/code batches exercise cross-attention.

    The production delayed-commit rule still applies during training; this is
    an explicit pre-flight bootstrap of the persistent timeline, not same-batch
    leakage.
    """
    trainer.model.eval()
    written = 0
    with torch.no_grad():
        for group, sample in examples:
            if written >= slots or sample.get("kind") != "text":
                continue
            data = bytes(sample["data"])[: trainer.seq_len]
            ids = encode_bytes(data, bos=True, eos=False).unsqueeze(0).to(trainer.device)
            pooled = pooled_text_embedding(trainer.model, ids, PAD)
            key, value = trainer.text_projector(pooled)
            trainer.memory.write_skill(key[0].detach(), value[0].detach(),
                                       modality="text", metadata={"preflight": group})
            written += 1
    trainer.model.train()
    return written


def _fixed_probe_batch(examples: list[tuple[str, dict[str, Any]]], *, device: torch.device
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    docs = [PTBR_PROMPT.encode("utf8") + b" estou aqui para ajudar.",
            CODE_PROMPT.encode("utf8") + b"\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n"]
    docs.extend(bytes(sample["data"]) for _, sample in examples[:6] if sample.get("kind") == "text")
    sequences = [encode_bytes(doc, bos=True, eos=True) for doc in docs]
    length = max(int(seq.numel()) for seq in sequences) - 1
    ids = torch.full((len(sequences), length), PAD, dtype=torch.long)
    targets = torch.full((len(sequences), length), PAD, dtype=torch.long)
    for row, seq in enumerate(sequences):
        size = int(seq.numel()) - 1
        ids[row, :size] = seq[:-1]
        targets[row, :size] = seq[1:]
    return ids.to(device), targets.to(device)


def probe_loss(trainer: Trainer, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
    ids, targets = batch
    trainer.model.eval()
    with torch.no_grad():
        loss = trainer.model.loss(ids, targets, trainer.memory.view(),
                                  confidence_weight=0.0, return_metrics=False)
    trainer.model.train()
    return float(loss.detach().item())


def generate_snapshot(trainer: Trainer, prompt: str, *, max_new_bytes: int = 100) -> dict[str, Any]:
    trainer.model.eval()
    prefix = encode_bytes(prompt.encode("utf8"), bos=True, eos=False)
    with torch.no_grad():
        ids = trainer.model.generate(prefix, max_new_bytes=max_new_bytes, temperature=0.0,
                                     memory=trainer.memory.view())
    continuation = decode_ids(ids).decode("utf8", "replace")
    trainer.model.train()
    return {"prompt": prompt, "continuation": continuation,
            "full_text": prompt + continuation,
            "requested_bytes": max_new_bytes, "generated_bytes": int(ids.numel())}


def _printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    good = sum(1 for ch in text if ch.isprintable() or ch in "\n\t")
    return good / len(text)


def evaluate_snapshots(snapshots: dict[int, dict[str, Any]], *, strict: bool = True) -> dict[str, Any]:
    s0, s250, s500 = snapshots[0], snapshots[250], snapshots[500]
    loss0, loss250, loss500 = s0["loss"], s250["loss"], s500["loss"]
    pt0, pt500 = s0["ptbr"], s500["ptbr"]
    code0, code500 = s0["code"], s500["code"]
    pt_markers = sum(1 for word in PTBR_MARKERS if word in pt500["continuation"].lower())
    code_markers = sum(1 for word in CODE_MARKERS if word in code500["continuation"])
    code_compiles = False
    code_error = None
    try:
        compile(code500["full_text"], "<coherence_preflight>", "exec")
        code_compiles = True
    except SyntaxError as exc:
        code_error = str(exc)
    pt0_markers = sum(1 for word in PTBR_MARKERS if word in pt0["continuation"].lower())
    code0_compiles = False
    try:
        compile(code0["full_text"], "<coherence_preflight_step0>", "exec")
        code0_compiles = True
    except SyntaxError:
        code0_compiles = False
    step0_gibberish = (pt0_markers < 2 or not code0_compiles
                       or _printable_ratio(pt0["continuation"]) < 0.95)
    metrics = {
        "loss_step_0": loss0, "loss_step_250": loss250, "loss_step_500": loss500,
        "ptbr_markers_step_500": pt_markers, "code_markers_step_500": code_markers,
        "ptbr_printable_ratio_step_500": _printable_ratio(pt500["continuation"]),
        "code_printable_ratio_step_500": _printable_ratio(code500["continuation"]),
        "code_compiles_step_500": code_compiles, "code_syntax_error": code_error,
        "step0_gibberish": step0_gibberish,
    }
    if strict:
        assert loss500 < loss0, f"model did not learn: loss {loss0:.4f} -> {loss500:.4f}"
        assert "Olá" in pt500["full_text"], "PT-BR prompt marker missing"
        assert pt_markers >= 2, f"PT-BR continuation lacks recognizable words: {pt500['continuation']!r}"
        assert _printable_ratio(pt500["continuation"]) >= 0.95, "PT-BR continuation is mostly non-printable bytes"
        assert "def" in code500["full_text"], "code prompt marker missing"
        assert code_markers >= 2, f"code continuation lacks Python markers: {code500['continuation']!r}"
        assert code_compiles, f"generated code is not valid Python syntax: {code_error}"
        assert step0_gibberish, "step 0 was already coherent; pre-flight is not proving learning"
    return metrics


def run_preflight(*, steps: int = 500, workdir: str | Path = "runs/coherence_preflight",
                  production: bool = False, strict: bool = True) -> dict[str, Any]:
    if steps != 500:
        raise ValueError("Phase 25.2 pre-flight requires exactly 500 steps")
    torch.manual_seed(252)
    random.seed(252)
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    config = preflight_config(steps=steps, workdir=workdir, production=production)
    examples = offline_slice(1000)
    trainer = Trainer(config, device="auto", samples=stream_examples(examples), resume=False)
    memory_slots = seed_memory_from_slice(trainer, examples)
    batch = _fixed_probe_batch(examples, device=trainer.device)
    snapshots: dict[int, dict[str, Any]] = {}

    def snapshot(step: int) -> None:
        snapshots[step] = {"loss": probe_loss(trainer, batch),
                           "ptbr": generate_snapshot(trainer, PTBR_PROMPT),
                           "code": generate_snapshot(trainer, CODE_PROMPT)}
        print(f"== step {step} == loss={snapshots[step]['loss']:.4f}")
        print("PT-BR:", repr(snapshots[step]["ptbr"]["continuation"]))
        print("CODE :", repr(snapshots[step]["code"]["continuation"]))

    start = time.monotonic()
    snapshot(0)
    trainer.run(max_steps=250)
    snapshot(250)
    trainer.run(max_steps=500)
    snapshot(500)
    assert trainer.step == 500, f"expected exactly 500 optimizer steps, got {trainer.step}"
    metrics = evaluate_snapshots(snapshots, strict=strict)
    cross_scales = [float(block.skill_adapter.cross_scale.detach().abs().item())
                    for block in trainer.model.blocks if block.skill_adapter.cross_attention_enabled]
    report = {
        "phase": "25.2", "steps": trainer.step, "production": production,
        "device": str(trainer.device), "elapsed_s": round(time.monotonic() - start, 2),
        "architecture": {"attention_type": trainer.model_config.attention_type,
                         "window_size": trainer.model_config.window_size,
                         "sliding_chunk_size": trainer.model_config.sliding_chunk_size,
                         "memory_cross_attention": trainer.model_config.memory_cross_attention,
                         "n_layers": trainer.model_config.n_layers,
                         "d_model": trainer.model_config.d_model,
                         "parameters": trainer.model.parameters_count},
        "data": {"examples": len(examples), "offline_slice": not production,
                 "memory_slots_seeded": memory_slots},
        "snapshots": snapshots, "metrics": metrics,
        "cross_scale_abs": cross_scales,
        "assertions_passed": bool(strict),
    }
    path = Path(workdir) / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"OK: coherence pre-flight passed; report={path}")
    return report


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Phase 25.2 500-step coherence pre-flight")
    parser.add_argument("--workdir", default="runs/coherence_preflight")
    parser.add_argument("--production", action="store_true", help="use full T4 preset (GPU)")
    parser.add_argument("--report", default=None, help="optional copy of the JSON report")
    args = parser.parse_args(argv)
    report = run_preflight(workdir=args.workdir, production=args.production)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    return report


if __name__ == "__main__":
    main()
