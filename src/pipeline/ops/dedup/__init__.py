from .exact import ExactDedupProcessor, exact_deduplicate, postprocess_duplicates
from .fuzzy import FuzzyDedupProcessor, minhash_candidate_pairs
from .judge import DedupJudgeProcessor, judge_candidate_pairs

__all__ = [
    "DedupJudgeProcessor",
    "ExactDedupProcessor",
    "FuzzyDedupProcessor",
    "exact_deduplicate",
    "judge_candidate_pairs",
    "minhash_candidate_pairs",
    "postprocess_duplicates",
]
