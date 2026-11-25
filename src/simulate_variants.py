"""Simulate 1,200+ annotated genomic variants with 18 harmonized feature
types drawn from simulated VCF + clinical annotation sources, and a hidden
true pathogenicity label generated independently of the feature harmonization
and classifier code."""
import numpy as np

FEATURE_TYPES = [
    "allele_frequency", "cadd_score", "sift_score", "polyphen_score", "conservation_score",
    "gerp_score", "splice_ai_score", "revel_score", "clinvar_stars", "gene_constraint_score",
    "exon_number", "variant_type_missense", "variant_type_nonsense", "variant_type_splice",
    "read_depth", "genotype_quality", "zygosity_het", "family_segregation_score",
]
assert len(FEATURE_TYPES) == 18

TRUE_PATHOGENIC_FEATURES = [1, 2, 3, 4, 6, 7]  # cadd, sift, polyphen, conservation, splice_ai, revel


def simulate_variants(n_variants: int = 1220, seed: int = 11):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_variants, len(FEATURE_TYPES)))
    weights = np.zeros(len(FEATURE_TYPES))
    weights[TRUE_PATHOGENIC_FEATURES] = rng.uniform(0.8, 1.5, size=len(TRUE_PATHOGENIC_FEATURES))
    logits = X @ weights + rng.normal(scale=0.5, size=n_variants)
    y = (logits > np.median(logits)).astype(int)
    return X, y
