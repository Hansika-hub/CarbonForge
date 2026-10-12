# CarbonGlobe: Age-for-Time Anchoring with Shift-Aligned Residual Learning

Solution for the **IEEE BigData Cup 2026 AI Emulation Challenge (CarbonGlobe, Kaggle)**.
Public leaderboard score: **0.066** (relative MSE, lower is better), down from 0.153 for our neural baseline.

**Team:** Hansika N M, Jayasree B, Janani S P
Department of Computer Science and Engineering, SSN College of Engineering, Chennai, India

---

## Task

Emulate the Ecosystem Demography (ED) model. For each of 6,171 test sites, 15 seed ages and 2 climate scenarios (SSP126, SSP585), predict year-40 **canopy height** and **above-ground biomass (AGB)** from the site's initial state and 40 years of monthly climate drivers (136 variables). That gives 185,130 rows.

Metric: `S = ½ (MSE_height + MSE_agb)`, with height and AGB divided by their global means (10.90645 and 3.39567).

The main difficulty is **distribution shift**: test initial states and future climates differ from the training data, so random-site validation is optimistic.

## Method (V3_C)

1. **Align initial states.** Quantile-map each test site's initial state onto the 54,152-site global distribution, per age and variable, with random tie-breaking.
2. **Age-for-time anchor.** A site's own initial-state curve over seed age is read at `age + 40` (clipped to 500) and recalibrated with a per-age linear fit on global sites. This anchor alone scores **0.093**.
3. **Climate residual.** Eight summary statistics for each of the 136 drivers give 1,088 features. They are aligned to the global distribution with pooled-scenario mean/std matching, then the top 80 are selected. XGBoost (depth 6, ~2,750 rounds, 2 seeds) predicts the residual `target − anchor` on 812,280 site-age rows.
4. **Shrunk blend.** `final = max(anchor + Σ w·residual, 0)`. The weights are fitted per target and age group with non-negative least squares on directional extrapolation folds and capped at a total of 0.7.

## Results (public leaderboard)

| Submission | Score |
|---|---|
| v8 CNN + BiGRU + GRU decoder | 0.156 |
| v9 + per-age baseline residual | 0.153 |
| XGBoost residual on global anchor | 0.074 |
| Filtered climate, residual weight 1.0 / 0.5 | 0.089 / 0.068 |
| Anchor only (V3_A) | 0.093 |
| Filtered climate, capped weights (V3_B) | 0.073 |
| **All climate aligned, capped weights (V3_C)** | **0.066** |

Probe submissions show the gain is entirely AGB at ages ≥ 50. The height residual alone scores 0.094, and a uniform AGB uplift hurts. Random-site validation (0.03–0.06) did not predict the leaderboard, and the paper discusses this gap.

## Repository layout

```
.
├── carbonglobe_v3.ipynb       # final notebook (writes V3_A..V3_D submissions)
├── carbonglobe_v3.py          # same code as a script
├── carbonglobe_v2.py          # earlier pipeline
├── blend_probes.py            # leaderboard probes: scaled / height-only / ages>=50
├── uplift_probes.py           # uniform AGB uplift probes
├── y0_variants.py             # affine initial-state correction variants (E0-E2, unscored)
├── inspect_data.py            # data layout diagnostics
├── inspect_data2.py
├── paper/
│   ├── carbonglobe_paper.tex  # IEEE conference paper
│   └── carbonglobe_paper.pdf
└── README.md
```

## Running

Data is not included. Download the competition files from Kaggle and set the data paths at the top of `carbonglobe_v3.py`.

```bash
pip install numpy pandas scipy scikit-learn xgboost
python carbonglobe_v3.py
```

- A GPU (we used one NVIDIA T4) runs the full pipeline in under 40 minutes.
- Caches (`glob_targets_v3.npz`, `clim_feats_v3.npz`) are written on the first run.
- Outputs are `V3_A_anchor_only.csv`, `V3_B_main.csv`, `V3_C_clim_aligned.csv` and `V3_D_M2M3_avg.csv`. The submitted file is **V3_C**.

## Caveats

- Scores are public-leaderboard only. The residual cap and the final candidate were chosen using the public leaderboard.
- Single run per configuration, so there are no confidence intervals.
- The alignment assumes test sites are a random sample of the global population at each age, which cannot be verified.
- Test initial states keep constant offsets that quantile mapping does not explain. The affine correction variants in `y0_variants.py` have not been scored.

## Future work

Fix the test initial state, train on full 41-year trajectories, build a validation that tracks the leaderboard, ensemble a neural sequence model with the XGBoost residual, and break the remaining error down by age and target.

## Paper

See `paper/carbonglobe_paper.pdf` for the full description and the novelty analysis.

## License

Add a license of your choice (for example MIT) before publishing.
