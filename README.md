# OphthalmoTwin-GP

OphthalmoTwin-GP provides research components for longitudinal retinal imaging, variant knowledge graphs, gene classification, prognosis, and conformal prediction.

## Installation

```bash
pip install -e ".[dev]"
```

## Data and execution

Patient records, variant knowledge graphs, trained weights, and dataset-specific loaders are not included. The `cut`, `glaze`, `inspect`, `hold`, and `trace` commands require integration of patient records and a variant knowledge graph before use and currently stop with an error.

The imaging stream currently uses a fixed Gabor-feature encoder with a fixed projection. The RETFound-DINOv2 weights and their integration are not included. Consequently, this implementation cannot by itself reproduce the manuscript's reported performance or compute profile.

`selection` defines patient records, visit batching, and graph structures; `glasswork` contains the imaging, graph, fusion, and prediction components; `flux` contains training objectives; `lightbox` contains metrics and conformal prediction utilities. Configuration files are in `cartoons/`.

Potential data sources include [UK Biobank](https://www.ukbiobank.ac.uk/), [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/), [ClinGen](https://clinicalgenome.org/), and [gnomAD](https://gnomad.broadinstitute.org/). Access and usage are subject to the terms of each source.

```bash
make test
```

This is research software and has not been validated for clinical use.
