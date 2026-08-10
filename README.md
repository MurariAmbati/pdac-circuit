# PDAC Chromatin-Circuit

Computational design and adversarial validation of synthetic gene circuits targeting
pancreatic-ductal-adenocarcinoma transcription factors.

Live site: https://murariambati.github.io/pdac-circuit

This repository carries both the project website and a mirror of the pipeline that produced the
results shown on it. Every figure and table on the site is generated from the result files committed
here, so a published number cannot drift away from the artefact it came from.

## Layout

```
src/pdac_circuit/   installable package, Modules I-VII plus the training and provenance harness
pipeline/           analysis entrypoints, one script per experiment
results/            result files behind every published number
models/             model manifests with metrics and weight SHA-256, plus versioned tree models
data/manifests/     provenance manifests, one entry per file with source URL and SHA-256
tests/              test suite
docs/               long-form technical writing rendered into the Addenda and Reports pages
scripts/            site tooling: figures, page and circuit-page generation, pipeline mirror
_pages/  index.md   site content
images/             figures, 300 dpi PNG for the web and vector PDF for print
```

Raw sequencing data and trained neural weights are not versioned. They are reconstructed from the
manifests, which carry hashes rather than merely paths.

## Reproducing a result

```
pip install -e .
pdac fetch-data --all-open
pdac verify-data
```

Verification recomputes the SHA-256 of every downloaded file against its manifest entry and fails on a
mismatch. Each experiment is then a single script.

```
python pipeline/grna_cnn_kim_retrain.py
python pipeline/promoter_scaleup.py
python pipeline/enhancer_maxdata.py
python pipeline/promoter_gan_scaleup.py
```

Each writes into `results/` and, where a model is deployed, refreshes its manifest and frozen
prediction fixture together.

```
pytest -q
```

## Rebuilding the site

```
python scripts/make_figures.py
python scripts/build_pages.py
bundle exec jekyll serve
```

`make_figures.py` reads `results/` and writes `images/`. `build_pages.py` reads `docs/`, the report
files at the root, and `results/`, and writes the generated pages under `_pages/`, including the
Evaluation, Figures and Circuits pages. None of those is written by hand, so a published number is
corrected by correcting the result file behind it.

## Relationship to the pipeline repository

The pipeline is developed in
[pdac-chromatin-circuit](https://github.com/MurariAmbati/pdac-chromatin-circuit), which is
authoritative. The mirror here is produced by a script, not by hand.

```
python scripts/sync_from_pipeline.py
```

It records the source commit in `PIPELINE_SOURCE.json`, so divergence between the two copies is
detectable rather than silent.
