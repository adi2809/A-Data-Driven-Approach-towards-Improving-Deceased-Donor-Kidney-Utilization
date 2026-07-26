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
