# ByteCore-Diffusion Evaluation Report

## Overview
- **Model**: ByteCore-Diffusion (byte-level diffusion, raw bytes 0-255)
- **Parameters**: 187,328  (0.75 MB, < 50 MB)
- **Reconstruction accuracy (clean)**: 0.9982
- **Robustness score** (mean max-noise acc / clean): 0.9989
- **Inference time**: 0.210 ms/sample

## Per-modality reconstruction accuracy

| Modality | Accuracy |
|----------|----------|
| audio | 0.9982 |
| text | 1.0000 |
| image | 0.9976 |

## Byte-level noise robustness (denoising accuracy after corruption)


### Per noise mode
| Mode | clean | 0.02 | 0.05 | 0.10 |
|------|-------|------|------|------|
| substitute | 0.9982 | 0.9982 | 0.9972 | 0.9957 |
| insert | 0.9982 | 0.9986 | 0.9981 | 0.9971 |
| delete | 0.9982 | 0.9985 | 0.9987 | 0.9986 |

## Concept storage — learn new concepts without retraining

New concepts are written directly to the concept bank (a memory write, no gradient step). The model can then generate conditioned on them.

| Modality | Concept written | Generated-vs-example byte match |
|----------|-----------------|---------------------------------|
| text | new:text | 0.5781 |
| image | new:image | 0.5234 |
| audio | new:audio | 0.5000 |

## Notes
- Raw bytes (0-255) throughout; no external tokenizers.
- Reconstruction accuracy = exact byte match after one denoising step (light noise, t=2/50).
- Robustness score = mean accuracy at max noise level / clean accuracy, averaged over substitute/insert/delete modes.
- Concept bank: new concepts stored as memory entries (write, not train); instantly available at inference — no retraining required.
- Inference time measured on CPU, one denoising step.