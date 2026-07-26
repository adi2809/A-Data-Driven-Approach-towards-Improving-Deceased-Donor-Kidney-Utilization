# A Data-Driven Approach towards Improving Deceased-Donor Kidney Utilization

Reproducibility code for the accepted Machine Learning for Healthcare paper.

The release contains the paper pipeline only: construction of ordered kidney match runs, temporally split benchmark data, OfferPred, DiscardPred, segment-hazard LocationPred, evaluation artifacts, and provenance audits. Restricted source data and generated artifacts are not included.

## Paper Inference

For every validation or test match run:

1. OfferPred scores every supplied row in order, including rows with `Y`, `N`, `B`, and `Z` response codes.
2. The complete score sequence is summarized for DiscardPred.
3. If `discard_probability >= 0.5`, the run is assigned to the discard branch.
4. Otherwise, LocationPred assigns segment hazards, redistributes each segment's mass using the OfferPred scores, and returns the modal row as the predicted first acceptance.

OfferPred loss and offer-level metrics use only observed `Y/N` outcomes, with training rows ending at the second observed acceptance. That label restriction is not applied when scoring validation or test runs.

## Reproduction

Python 3.12 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover tests
```

Run the paper pipeline from the repository root:

```bash
python scripts/run_pipeline.py \
  --match-source-dir /path/to/match_runs \
  --saf-source-dir /path/to/standard_analysis_files \
  --kdpi-do-file /path/to/kdpi_mapping.do \
  --threads 8
```

The required licensed inputs and generated layout are summarized in [DATA.md](DATA.md). Reproducibility checks are listed in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Outputs are written below the gitignored `warehouse/` directory.

This repository is distributed under the terms in [LICENSE](LICENSE). For reuse beyond reproducing the paper, contact the authors through the repository.

## Citation

Garg, A. S., Feigenbaum, I., Yu, M. E., Mohan, S., & Sethuraman, J. (2026). A Data-Driven Approach towards Improving Deceased-Donor Kidney Utilization. *Proceedings of Machine Learning Research*, 340, 1-29. Machine Learning for Healthcare.

```bibtex
@inproceedings{garg2026kidneyutilization,
  title     = {A Data-Driven Approach towards Improving Deceased-Donor Kidney Utilization},
  author    = {Garg, Aditya Shankar and Feigenbaum, Itai and Yu, Miko E. and Mohan, Sumit and Sethuraman, Jay},
  booktitle = {Machine Learning for Healthcare},
  series    = {Proceedings of Machine Learning Research},
  volume    = {340},
  pages     = {1--29},
  year      = {2026}
}
```

Machine-readable metadata are provided in [CITATION.cff](CITATION.cff).
