import json
import sys

from custody.stats import mcnemar_test
from scripts.run_real_data_eval import build_real_data_eval_matrix, paired_correctness


def run_significance_analysis(vision_root: str, n_attack_instances: int = 30, seed: int = 0) -> dict:
    result = build_real_data_eval_matrix(
        vision_root, n_attack_instances=n_attack_instances, seed=seed, return_raw=True
    )
    report = {}
    for transform_type, transform_result in result.items():
        report[transform_type] = {}
        for tier_name, raw in transform_result["_raw"].items():
            baseline_scores, labels, baseline_threshold = raw["snapshot_baseline"]
            full_scores, _, full_threshold = raw["full_system"]

            correct_baseline = paired_correctness(baseline_scores, labels, baseline_threshold)
            correct_full = paired_correctness(full_scores, labels, full_threshold)

            report[transform_type][tier_name] = mcnemar_test(correct_full, correct_baseline)
    return report


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else r"D:\DF paper\data\vision"
    print(json.dumps(run_significance_analysis(root), indent=2))
