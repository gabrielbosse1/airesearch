# ByteCore-Diffusion Evaluation Report

## Overview
- **Model**: ByteCore-Diffusion (byte-level diffusion, raw bytes 0-255)
- **Parameters**: 195,905  (0.78 MB, < 50 MB)
- **Reconstruction (first batch, padded)**: 0.9987
- **Reconstruction (all real bytes)**: 0.9973
- **Corrupted-input reconstruction ratio** (not recovery): 0.9984
- **Original-byte recovery ratio at maximum corruption**: 0.5560
- **Inference time**: 1.624 ms/sample
- **Exact test rows also present in train**: 139/500 (synthetic data; not a fully novel test set)
- **Exact-byte-novel test subset**: 361 rows, reconstruction 0.9987 (near duplicates not excluded)

## Per-modality reconstruction (real bytes, no padding)

| Modality | Accuracy |
|----------|----------|
| audio | 0.9966 |
| text | 1.0000 |
| image | 0.9977 |

## Corrupted-input reconstruction (includes padding)
This is a self-reconstruction metric, not recovery of the original.

### Per noise mode
| Mode | clean | 0.02 | 0.05 | 0.10 |
|------|-------|------|------|------|
| substitute | 0.9987 | 0.9984 | 0.9973 | 0.9959 |
| insert | 0.9987 | 0.9985 | 0.9975 | 0.9961 |
| delete | 0.9987 | 0.9990 | 0.9989 | 0.9991 |

## Recovery against the original (real bytes, position-wise)

| Mode | 0.02 | 0.05 | 0.10 |
|------|------|------|------|
| substitute | 0.9774 | 0.9519 | 0.9128 |
| insert | 0.6314 | 0.4566 | 0.3693 |
| delete | 0.6404 | 0.4664 | 0.3812 |

## Concept storage — learn new concepts without retraining

New concepts are written directly to the concept bank (a memory write, no gradient step). The model can then generate conditioned on them.

| Modality | Concept written | Match on real bytes | Match including padding |
|----------|-----------------|---------------------|-------------------------|
| text | new:text | 0.0833 | 0.5234 |
| image | new:image | 0.0312 | 0.5156 |
| audio | new:audio | 0.0156 | 0.5078 |

## Notes
- Raw bytes (0-255) throughout; no external tokenizers.
- Reconstruction = exact byte match after one denoising step (light noise, t=2/50).
- Corrupted-input reconstruction compares to the corrupted input, not the clean target.
- Original-byte recovery excludes padding and compares to clean bytes at the same positions; insertions/deletions are not realigned.
- Ratios divide maximum-corruption accuracy by the corresponding clean accuracy.
- Concept examples come from the training split and pass an exact-byte holdout guard; no validation/test examples are written to memory.
- Exact split-overlap counts do not detect near duplicates or prove checkpoint training provenance.
- Concept bank: new concepts stored as memory entries (write, not train); instantly available at inference — no retraining required.
- Inference time measured on CPU, one denoising step.