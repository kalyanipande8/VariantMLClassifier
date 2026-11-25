import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulate_variants import simulate_variants, FEATURE_TYPES
from src.feature_harmonization import harmonize, VCF_SOURCED, CLINICAL_SOURCED
from src.classifier import train_and_evaluate


def test_at_least_1200_variants():
    X, y = simulate_variants(n_variants=1220)
    assert X.shape[0] >= 1200


def test_18_feature_types():
    assert len(FEATURE_TYPES) == 18


def test_harmonization_complete():
    result = harmonize(VCF_SOURCED, CLINICAL_SOURCED)
    assert result["complete"] is True
    assert result["n_features_harmonized"] == 18


def test_classifier_achieves_reasonable_auc():
    X, y = simulate_variants(n_variants=600)
    result = train_and_evaluate(X, y)
    assert result["auc"] > 0.6


def test_classifier_accuracy_above_chance():
    X, y = simulate_variants(n_variants=600)
    result = train_and_evaluate(X, y)
    assert result["accuracy"] > 0.5
