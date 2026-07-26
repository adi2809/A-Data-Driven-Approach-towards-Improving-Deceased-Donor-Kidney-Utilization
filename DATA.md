# Data

The licensed study data cannot be redistributed. Full reproduction requires authorized access to:

- kidney match-run `.dta` files;
- Standard Analysis Files in `.sas7bdat`/`.sas7bcat` form; and
- the KDPI/KDRI mapping `.do` file used for the study period.

Pass all three locations explicitly to `scripts/run_pipeline.py`. Source-schema field names are preserved where required to read the licensed files.

Derived tables, model artifacts, plots, and manifests are written under:

```text
warehouse/
  match_runs/
  saf/
  match_offer_features/
```

The entire `warehouse/` tree is ignored by git and must not be committed.
