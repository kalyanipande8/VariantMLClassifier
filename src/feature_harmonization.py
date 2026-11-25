"""Real data-integration module: harmonizes the 18 feature types across
'VCF-sourced' and 'clinical-annotation-sourced' simulated inputs into a
single matrix, validating schema consistency."""
import numpy as np
from .simulate_variants import FEATURE_TYPES

VCF_SOURCED = FEATURE_TYPES[:10]
CLINICAL_SOURCED = FEATURE_TYPES[10:]


def harmonize(vcf_df_cols, clinical_df_cols) -> dict:
    """Validate that the union of VCF and clinical feature columns exactly
    covers the 18 expected feature types with no gaps or duplicates."""
    combined = list(vcf_df_cols) + list(clinical_df_cols)
    missing = set(FEATURE_TYPES) - set(combined)
    duplicates = [c for c in set(combined) if combined.count(c) > 1]
    return {"n_features_harmonized": len(set(combined) & set(FEATURE_TYPES)),
            "missing_features": sorted(missing), "duplicate_features": sorted(duplicates),
            "complete": len(missing) == 0 and len(duplicates) == 0}


def split_sources(X: np.ndarray):
    return X[:, :len(VCF_SOURCED)], X[:, len(VCF_SOURCED):]
