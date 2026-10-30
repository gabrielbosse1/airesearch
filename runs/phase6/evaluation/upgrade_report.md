# ByteCore-DualMemory Upgrade Verification

## Cognitive loop (narrow synthetic task)
- Held-out exact results: 1.0000 on 256 tasks; DONE fraction: 1.0000.
- Mean cpu wall time: 20.27 ms/task.
- Commands I/D/R/M operate only on a three-byte a–h payload; this is not general editing accuracy.

## Procedural and graph memory
- Exact byte-pattern replay and ordered latent trace round-trip: PASS.
- Multi-hop path (16, 17, 18): conflict at time 10; unknown at time 20.

## Held-out confidence measurements
- Diffusion q-sampled denoising: 3360 real-byte predictions, correctness 0.5827, Brier 0.1050, ECE-10 0.0274.
- Generator next byte without memory: 3296 real-byte predictions, correctness 0.0067, Brier 0.0065, ECE-10 0.0008.
- Confidence was trained on separate training data and temperature-fitted on validation; the test set and metrics above do not validate scratchpad or ancestral-sampling halting.

## Adaptive computation mechanism (not a quality benchmark)
- Eight ancestral samples at threshold 0.9 used [5, 9, 5, 6, 7, 5, 5, 10] steps (full schedule: 50).
- These sampling states differ from the calibration data. A scope-mismatch warning is expected; no correctness or speed/quality claim is implied.
