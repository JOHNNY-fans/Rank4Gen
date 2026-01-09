import json
import re
import string
import argparse
from collections import Counter, defaultdict
from tabulate import tabulate


# =========================
# Text normalization & metrics
# =========================

def normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    if not text:
        return ""

    def lower(t: str) -> str:
        return t.lower()

    def remove_punc(t: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in t if ch not in exclude)

    def remove_articles(t: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t: str) -> str:
        return " ".join(t.split())

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Exact match after normalization."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score_token(prediction: str, ground_truth: str) -> float:
    """Token-level F1 for non-CJK languages (English-like)."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def char_f1_score(prediction: str, ground_truth: str) -> float:
    """Character-level F1, commonly used for Chinese-like evaluation."""
    prediction = prediction or ""
    ground_truth = ground_truth or ""

    pred = "".join(prediction.split())
    gt = "".join(ground_truth.split())

    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0

    pred_chars = list(pred)
    gt_chars = list(gt)

    common = Counter(pred_chars) & Counter(gt_chars)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_chars)
    recall = num_same / len(gt_chars)
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def mean_over_ground_truths(metric_fn, prediction: str, ground_truths) -> float:
    """Average metric over multiple ground truths."""
    if not ground_truths:
        return 0.0
    return sum(metric_fn(prediction, gt) for gt in ground_truths) / len(ground_truths)


# =========================
# Auto-detect CJK for choosing F1 type
# =========================

def contains_cjk(text: str) -> bool:
    """Return True if text contains any CJK character."""
    if not text:
        return False
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            return True
    return False


def auto_f1_over_ground_truths(prediction: str, ground_truths) -> float:
    """
    Auto-select F1:
      - If ground truths contain CJK => char-level F1
      - Else => token-level F1
    """
    joined = "".join(ground_truths or [])
    if contains_cjk(joined):
        return mean_over_ground_truths(char_f1_score, prediction, ground_truths)
    return mean_over_ground_truths(f1_score_token, prediction, ground_truths)


# =========================
# Source-field detection
# =========================

def detect_has_source(file_path: str) -> bool:
    """
    Return True if there exists at least one successful sample with a non-empty 'source' field.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if data.get("status") != "success":
                continue

            src = data.get("source", None)
            if src is not None and src != "":
                return True

    return False


# =========================
# Core evaluation (single file)
# =========================

def evaluate_single_file(file_path: str, dataset_name: str = "dataset"):
    """
    Rules:
      - If any successful sample has non-empty 'source' => group by 'source'
      - Otherwise => treat the entire file as one dataset named `dataset_name`

    Returns:
      dict[key] = {"em_sum": float, "f1_sum": float, "count": int}
    """
    split_by_source = detect_has_source(file_path)
    agg = defaultdict(lambda: {"em_sum": 0.0, "f1_sum": 0.0, "count": 0})

    def get_key(sample: dict) -> str:
        if split_by_source:
            key = sample.get("source", "unknown")
            return key if key not in (None, "") else "unknown"
        return dataset_name

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = get_key(data)
            status = data.get("status")

            # Failures count as 0 score, but still counted.
            if status != "success":
                agg[key]["count"] += 1
                continue

            prediction = data.get("answer", "")
            ground_truths = data.get("std_answer", [])
            if isinstance(ground_truths, str):
                ground_truths = [ground_truths]
            elif not isinstance(ground_truths, list):
                ground_truths = []

            em_val = mean_over_ground_truths(exact_match_score, prediction, ground_truths)
            f1_val = auto_f1_over_ground_truths(prediction, ground_truths)

            agg[key]["em_sum"] += em_val
            agg[key]["f1_sum"] += f1_val
            agg[key]["count"] += 1

    return agg


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a single jsonl result file.\n"
            "- If 'source' exists (at least one successful sample): aggregate by source\n"
            "- Otherwise: aggregate the whole file as one dataset"
        )
    )
    parser.add_argument("--result_file", required=True, help="Path to the jsonl result file")
    parser.add_argument("--ranker", required=True, help="Ranker name (e.g., Rank4Gen)")
    parser.add_argument("--method", required=True, help="Method name (e.g., index / snapshot)")
    parser.add_argument("--downstream_model", required=True, help="Downstream model name")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="dataset",
        help="Dataset name used when file has no 'source' field",
    )

    args = parser.parse_args()

    agg = evaluate_single_file(args.result_file, dataset_name=args.dataset_name)

    # Output columns as requested
    headers = ["Ranker", "Method", "Downstream_Model", "Dataset", "EM", "F1", "Count"]
    table = []

    for dataset_key, vals in sorted(agg.items(), key=lambda x: x[0]):
        count = vals["count"]
        if count == 0:
            continue

        em = (vals["em_sum"] / count) * 100.0
        f1 = (vals["f1_sum"] / count) * 100.0

        table.append([
            args.ranker,
            args.method,
            args.downstream_model,
            dataset_key,
            f"{em:.2f}",
            f"{f1:.2f}",
            count,
        ])

    print(tabulate(table, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()
