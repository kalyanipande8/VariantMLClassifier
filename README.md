# VariantMLClassifier

A real small PyTorch MLP classifier for variant pathogenicity prediction
across 1,220 simulated annotated genomic variants, with a data-integration
module harmonizing 18 feature types from simulated VCF and clinical
annotation sources.

> Developed deep learning classifier for variant pathogenicity prediction
> on 1,200+ annotated genomic records. Built data integration module
> harmonizing 18 feature types from VCF and clinical annotation sources.

## What's real vs. synthetic — read this first

**Real:** `src/classifier.py` trains a genuine 3-layer PyTorch MLP
(32 -> 16 -> 2 units, ReLU, Adam optimizer, cross-entropy loss) on a
held-out train/test split and measures real accuracy and AUC.
`src/feature_harmonization.py` performs a real schema-reconciliation check
confirming all 18 feature types are present across the simulated
VCF-sourced and clinical-annotation-sourced inputs with no gaps or
duplicates.

**Synthetic:** there is no real VCF or clinical annotation dataset in this
environment. `src/simulate_variants.py` generates 1,220 variants with 18
features, where pathogenicity status is driven by exactly 6 of the 18
features (CADD, SIFT, PolyPhen, conservation, SpliceAI, REVEL scores) with
added noise — independent of the classifier and harmonization code.

## Results (real, from `run_all.py`)

```
1,220 variants, 18 feature types

Feature harmonization: 18/18 features harmonized, 0 missing, 0 duplicates
Classifier: accuracy 0.9279, AUC 0.9787
```

`tests/test_pipeline.py` (5 tests, all passing) checks variant count
>=1200, exactly 18 feature types, complete harmonization, and classifier
AUC/accuracy above chance-level thresholds.

## Usage

```bash
pip install -r requirements.txt
python3 run_all.py
python3 -m pytest tests/ -v
```

## Project structure

```
src/
  simulate_variants.py       # 1,220-variant, 18-feature simulation
  feature_harmonization.py     # VCF + clinical source schema reconciliation
  classifier.py                  # PyTorch MLP training + evaluation
run_all.py
tests/test_pipeline.py
```

## Tech stack

Python 3.9+, PyTorch, scikit-learn, pytest.

## Limitations

- No real VCF files or clinical annotation sources are used.
- The true pathogenicity model is a linear combination of 6 features plus
  noise; real variant-effect prediction involves far more complex,
  nonlinear, and structurally-informed signal.
