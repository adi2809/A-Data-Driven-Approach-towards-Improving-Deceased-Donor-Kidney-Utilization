# Reproducibility

Runs are deterministic conditional on the licensed inputs, pinned package versions, requested year subset, and random seed in `kidney_utilization/config.py`.

Use a clean Python 3.12 environment, install with `python -m pip install -e .`, and run:

```bash
python -m unittest discover tests
python scripts/run_pipeline.py \
  --match-source-dir /path/to/match_runs \
  --saf-source-dir /path/to/standard_analysis_files \
  --kdpi-do-file /path/to/kdpi_mapping.do \
  --threads 8
```

The synthetic test verifies the build/train path without licensed data. It also verifies that:

- OfferPred scores every row in each validation/test match run;
- `B/Z` rows are retained for inference but excluded from OfferPred loss;
- DiscardPred is a binary discard classifier at threshold `0.5`; and
- LocationPred probabilities sum to one across all rows on the localization branch and to zero on the discard branch.

Archive the generated run manifests and the feature/hyperparameter audits with reported results.
