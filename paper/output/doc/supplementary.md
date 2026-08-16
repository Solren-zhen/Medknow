# Supplementary Material

## Table S1. Run-to-run stability of the internal referral strategy across three training seeds

The referral analysis in the main text uses the primary model (seed 42). To confirm that the internal referral results were not specific to a single training run, we repeated the analysis with the two additional training seeds (2024 and 2026) under the identical protocol: explicit model weights, 30 MC Dropout passes, temperature T = 1.67, and a fixed 0.5 threshold on the MC-mean probability. Retained-case error and retained-case missed-pneumonia rate (FNR) are reported at fixed referral rates, together with the random-referral control at matched rates.

| Referral rate | Seed 42 | Seed 2024 | Seed 2026 |
|---|---|---|---|
| 0% (baseline) — error / FNR | 4.0% / 3.7% | 2.8% / 1.8% | 4.2% / 4.4% |
| 10% — retained error | 1.4% | 0.7% | 1.1% |
| 10% — retained FNR | 1.0% | 0.2% | 0.8% |
| 10% — random referral error | 3.9% | 2.6% | 3.7% |
| 25% — retained error | 0.3% | 0.3% | 0.1% |
| 25% — retained FNR | 0.0% | 0.0% | 0.2% |
| 25% — random referral error | 3.4% | 2.1% | 3.4% |

At a 25% referral rate, the retained-case error ranged from 0.1% to 0.3% across seeds (random referral: 2.1–3.4%) and the retained-case missed-pneumonia rate ranged from 0.0% to 0.2%: zero missed cases for seeds 42 and 2024 and a single missed case for seed 2026, out of 896 internal test images (677 pneumonia). At a 10% referral rate, the retained-case error ranged from 0.7% to 1.4% across seeds (random referral: 2.6–3.9%). The internal referral benefit therefore replicates across training seeds, with the "no missed cases" property holding at a 25% referral rate for two of three seeds and degrading only minimally (one case) for the third.
