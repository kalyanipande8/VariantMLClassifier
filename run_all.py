import json
import os

from src.simulate_variants import simulate_variants, FEATURE_TYPES
from src.feature_harmonization import harmonize, VCF_SOURCED, CLINICAL_SOURCED
from src.classifier import train_and_evaluate


def main():
    X, y = simulate_variants(n_variants=1220)
    harmonization = harmonize(VCF_SOURCED, CLINICAL_SOURCED)
    result = train_and_evaluate(X, y)

    os.makedirs("results", exist_ok=True)
    summary = {"n_variants": len(y), "n_feature_types": len(FEATURE_TYPES),
               "harmonization": harmonization, "classifier": result}
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
