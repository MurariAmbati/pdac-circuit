# PDAC Chromatin-Circuit — project site

Source for the project website: computational design and adversarial validation of synthetic gene
circuits targeting pancreatic-ductal-adenocarcinoma transcription factors.

Live site: https://murariambati.github.io/pdac-circuit

## Structure

```
_config.yml        site configuration and navigation
_layouts/          page template
_includes/         head, navigation, footer
_pages/            results, methods, validation, data
assets/css/        stylesheet
images/            figures (PNG for the web, PDF for print)
scripts/           figure generation
index.md           overview
```

## Figures

Figures are generated from the pipeline's result files rather than transcribed, so the published
numbers cannot drift from the underlying data:

```
python scripts/make_figures.py
```

Each figure is written as a 300 dpi PNG and a vector PDF.

## Local preview

```
bundle install
bundle exec jekyll serve
```
