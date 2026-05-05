"""Analyze the distribution of answer_judge results from a judged parquet file."""
import json
import sys
from collections import Counter

import pyarrow.parquet as pq


def _load_v(raw):
    if not raw:
        return {}
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return raw if isinstance(raw, dict) else {}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/rollout_slices/slice_00_judged.parquet"
    print(f"Reading {path} ...")
    table = pq.read_table(path)
    n_total = table.num_rows
    print(f"Total records: {n_total}\n")

    verification_col = table.column("verification").to_pylist()
    training_phase_col = table.column("training_phase").to_pylist()
    filter_tag_col = (
        table.column("filter_tag").to_pylist()
        if "filter_tag" in table.schema.names
        else [None] * n_total
    )

    phase_counter = Counter(training_phase_col)
    filter_tag_counter = Counter(filter_tag_col)

    majority_match_gt = Counter()
    n_flagged = 0
    n_with_samples = 0

    has_judge = 0
    judge_gt_quality = Counter()
    judge_equivalence = Counter()
    judge_combined = Counter()
    llm_majority_count_dist = Counter()
    judge_attempts_dist = Counter()

    for i in range(n_total):
        v = _load_v(verification_col[i])
        mmgt = v.get("majority_matches_gt")
        if mmgt is not None:
            majority_match_gt[str(mmgt)] += 1
        if mmgt is False:
            n_flagged += 1
        if v.get("samples"):
            n_with_samples += 1

        gt_q = v.get("llm_judge_gt_quality")
        equiv = v.get("llm_judge_equivalence")
        if gt_q:
            has_judge += 1
            judge_gt_quality[gt_q] += 1
            judge_equivalence[equiv or "(none)"] += 1
            judge_combined[f"{gt_q} / {equiv}"] += 1

            mc = v.get("llm_judge_majority_count")
            if mc is not None:
                llm_majority_count_dist[int(mc)] += 1

            att = v.get("llm_judge_attempts")
            if att is not None:
                judge_attempts_dist[int(att)] += 1

    n_equivalent = judge_equivalence.get("equivalent", 0)
    n_not_equivalent = judge_equivalence.get("not_equivalent", 0)
    n_not_applicable = judge_equivalence.get("not_applicable", 0)

    print("=" * 65)
    print("1. TRAINING PHASE DISTRIBUTION")
    print("=" * 65)
    for phase, cnt in phase_counter.most_common():
        print(f"  {phase or '(empty)':<20s} {cnt:>8d}  ({cnt/n_total*100:.2f}%)")

    print(f"\n{'=' * 65}")
    print("2. FILTER TAG DISTRIBUTION (top 20)")
    print("=" * 65)
    for tag, cnt in filter_tag_counter.most_common(20):
        print(f"  {str(tag or '(none)'):<35s} {cnt:>8d}  ({cnt/n_total*100:.2f}%)")

    print(f"\n{'=' * 65}")
    print("3. MAJORITY_MATCHES_GT (original rollout verdict)")
    print("=" * 65)
    for k, cnt in majority_match_gt.most_common():
        print(f"  {k:<10s} {cnt:>8d}  ({cnt/n_total*100:.2f}%)")
    print(f"  (no rollout) {n_total - sum(majority_match_gt.values()):>8d}")

    print(f"\n{'=' * 65}")
    print("4. LLM JUDGE RESULTS")
    print("=" * 65)
    print(f"  Total flagged (majority_matches_gt=False): {n_flagged}")
    print(f"  Records with LLM judge result:             {has_judge}")
    if has_judge:
        print()
        print("  4a. GT Quality:")
        for k, cnt in judge_gt_quality.most_common():
            print(f"      {k:<20s} {cnt:>8d}  ({cnt/has_judge*100:.2f}%)")

        print("\n  4b. Equivalence Judgement:")
        for k, cnt in judge_equivalence.most_common():
            print(f"      {k:<20s} {cnt:>8d}  ({cnt/has_judge*100:.2f}%)")

        print("\n  4c. Combined (gt_quality / equivalence):")
        for k, cnt in judge_combined.most_common():
            print(f"      {k:<40s} {cnt:>8d}  ({cnt/has_judge*100:.2f}%)")

        print("\n  4d. LLM Majority Count (from LLM's own grouping):")
        for k in sorted(llm_majority_count_dist.keys()):
            cnt = llm_majority_count_dist[k]
            print(f"      count={k:<3d} {cnt:>8d}  ({cnt/has_judge*100:.2f}%)")

        print("\n  4e. API Attempts:")
        for k in sorted(judge_attempts_dist.keys()):
            cnt = judge_attempts_dist[k]
            print(f"      attempts={k:<3d} {cnt:>8d}  ({cnt/has_judge*100:.2f}%)")

    print(f"\n{'=' * 65}")
    print("5. IMPACT SUMMARY")
    print("=" * 65)
    print(f"  Total records:                        {n_total:>8d}")
    print(f"  With rollout samples:                 {n_with_samples:>8d}  ({n_with_samples/n_total*100:.2f}%)")
    print(f"  Flagged (majority!=gt):               {n_flagged:>8d}  ({n_flagged/n_total*100:.2f}%)")
    print(f"  LLM judged:                           {has_judge:>8d}")
    print()
    print(f"  → equivalent (rescued to posttrain):  {n_equivalent:>8d}  ({n_equivalent/max(has_judge,1)*100:.2f}% of judged)")
    print(f"  → not_equivalent (confirmed mismatch):{n_not_equivalent:>8d}  ({n_not_equivalent/max(has_judge,1)*100:.2f}% of judged)")
    print(f"  → not_applicable (gt unclear):        {n_not_applicable:>8d}  ({n_not_applicable/max(has_judge,1)*100:.2f}% of judged)")
    no_result = n_flagged - has_judge
    if no_result > 0:
        print(f"  → no LLM result (api error/no sample):{no_result:>8d}")

    print()
    original_posttrain = phase_counter.get("posttrain", 0)
    print(f"  Training phase after judge:")
    print(f"    posttrain:  {original_posttrain:>8d}")
    print(f"    midtrain:   {phase_counter.get('midtrain', 0):>8d}")
    print(f"    drop:       {phase_counter.get('drop', 0):>8d}")
    print()


if __name__ == "__main__":
    main()
