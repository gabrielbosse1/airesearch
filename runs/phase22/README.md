# Phase 22 offline reports

`benchmark.json` -- CPU-measured parameter counts, forward/backward timings, analytic VRAM estimates, precision fallback, reward ordering and the pipeline smoke test.

`vram_estimate.json` -- the full fill ladder (all `measured=false` here).

`reward_dominance.json` -- Search>Guess / Efficient Edit>Thrashing proof.

`final_eval.json` -- scripted-oracle scorer proof (`oracle_mode=true`).

`architecture_report.json` -- component inventory and exact parameter breakdown.

`dataset_inventory.json` -- verified HF IDs, configs, licences and gating rejects.

**No GPU VRAM utilisation, HF download, checkpoint training or model quality is claimed here.** Run the three notebooks on a Colab T4.
