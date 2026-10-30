
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval" / "classification_results"


def safe_divide(numerator: float, denominator: float) -> float:
    """Return zero when a metric has a zero denominator."""
    return numerator / denominator if denominator else 0.0


def get_relevant_chunk_ids(result: dict[str, Any]) -> set[str]:
    """Read the annotated relevant passage IDs for one question."""
    relevant = set()

    for field in ("gold_evidence", "acceptable_evidence"):
        for evidence in result.get(field, []):
            chunk_id = evidence.get("chunk_id")
            if chunk_id:
                relevant.add(str(chunk_id))

    return relevant


def confusion_matrix_metrics(
    actual: list[int],
    predicted: list[int],
) -> dict[str, Any]:
    """Calculate binary classification metrics."""
    tp = sum(a == 1 and p == 1 for a, p in zip(actual, predicted))
    tn = sum(a == 0 and p == 0 for a, p in zip(actual, predicted))
    fp = sum(a == 0 and p == 1 for a, p in zip(actual, predicted))
    fn = sum(a == 1 and p == 0 for a, p in zip(actual, predicted))

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    specificity = safe_divide(tn, tn + fp)

    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "specificity": specificity,
        "evaluated_passages": len(actual),
    }


def calculate_roc_auc(
    labels: list[int],
    scores: list[float],
) -> float | None:
    """
    Calculate ROC-AUC using rank statistics.

    Returns None if the sample does not contain both classes.
    Higher scores must indicate greater relevance.
    """
    positives = sum(label == 1 for label in labels)
    negatives = sum(label == 0 for label in labels)

    if positives == 0 or negatives == 0:
        return None

    ordered = sorted(
        zip(scores, labels),
        key=lambda item: item[0],
    )

    # Assign average ranks to tied scores.
    rank_sum_positive = 0.0
    i = 0

    while i < len(ordered):
        j = i + 1

        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1

        average_rank = ((i + 1) + j) / 2

        for _, label in ordered[i:j]:
            if label == 1:
                rank_sum_positive += average_rank

        i = j

    auc = (
        rank_sum_positive - positives * (positives + 1) / 2
    ) / (positives * negatives)

    return float(auc)


def percentile(values: list[float], p: float) -> float | None:
    """Calculate a percentile using linear interpolation."""
    if not values:
        return None

    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    return (
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def evaluate_result_file(
    result_path: Path,
    top_k: int = 5,
) -> dict[str, Any]:
    """Evaluate one saved baseline or reranking experiment."""
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    questions = payload.get("results", [])

    if not questions:
        raise ValueError(f"No question results found in {result_path}")

    actual_labels: list[int] = []
    predicted_labels: list[int] = []
    relevance_scores: list[float] = []

    per_question: list[dict[str, Any]] = []
    question_precisions = []
    question_recalls = []
    question_f1s = []
    reciprocal_ranks = []
    recalls_at_k = []
    ndcgs_at_k = []
    latencies = []
    generation_latencies = []

    for question in questions:
        relevant_ids = get_relevant_chunk_ids(question)

        # Do not count unanswerable questions as passage-classification
        # examples: they have no positive evidence by definition.
        if not question.get("answerable", True):
            continue

        if not relevant_ids:
            # No validated positive labels means classification metrics
            # cannot be meaningfully calculated for this question.
            continue

        # Use the candidate set as the shared evaluation universe.
        candidates = question.get("candidate_chunks", [])
        retrieved = question.get("retrieved_chunks", [])

        if not candidates:
            candidates = retrieved

        if not candidates:
            continue

        retrieved_ids = [
            str(chunk.get("chunk_id", ""))
            for chunk in retrieved[:top_k]
        ]
        predicted_positive_ids = set(retrieved_ids)

        # Binary relevance labels are based on the supplied gold evidence.
        # This assumes the gold annotation is exhaustive for this candidate
        # set. Unannotated passages are treated as non-relevant.
        question_actual = []
        question_predicted = []

        for chunk in candidates:
            chunk_id = str(chunk.get("chunk_id", ""))
            label = int(chunk_id in relevant_ids)
            prediction = int(chunk_id in predicted_positive_ids)

            question_actual.append(label)
            question_predicted.append(prediction)

            # Prefer cross-encoder scores when available; otherwise use
            # the stored retrieval score.
            raw_score = chunk.get("reranker_score")
            if raw_score is None:
                raw_score = chunk.get("score")

            if raw_score is not None:
                try:
                    score = float(raw_score)
                    if math.isfinite(score):
                        actual_labels.append(label)
                        predicted_labels.append(prediction)
                        relevance_scores.append(score)
                except (TypeError, ValueError):
                    pass

        # Classification metrics should use the same passages and labels.
        # Keep all candidates, even if an individual score is missing.
        metrics = confusion_matrix_metrics(
            question_actual,
            question_predicted,
        )

        question_precisions.append(metrics["precision"])
        question_recalls.append(metrics["recall"])
        question_f1s.append(metrics["f1_score"])

        # Calculate rank-based metrics using the final ordered results.
        ranked_ids = [
            str(chunk.get("chunk_id", ""))
            for chunk in retrieved
        ]

        first_relevant_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(ranked_ids, start=1)
                if chunk_id in relevant_ids
            ),
            None,
        )

        reciprocal_ranks.append(
            1.0 / first_relevant_rank if first_relevant_rank else 0.0
        )

        recall = safe_divide(
            len(set(ranked_ids[:top_k]) & relevant_ids),
            len(relevant_ids),
        )
        recalls_at_k.append(recall)

        # Binary nDCG@k for the annotated relevance labels.
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, chunk_id in enumerate(ranked_ids[:top_k], start=1)
            if chunk_id in relevant_ids
        )

        ideal_relevant_count = min(len(relevant_ids), top_k)
        ideal_dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_relevant_count + 1)
        )

        ndcgs_at_k.append(safe_divide(dcg, ideal_dcg))

        latency = question.get("latency_ms")
        if isinstance(latency, (int, float)) and latency >= 0:
            latencies.append(float(latency))

        generation_latency = question.get("generation_latency_ms")
        if (
            isinstance(generation_latency, (int, float))
            and generation_latency >= 0
        ):
            generation_latencies.append(float(generation_latency))

        per_question.append(
            {
                "question_id": question.get("question_id"),
                "question": question.get("question"),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                f"recall_at_{top_k}": recall,
                "mrr": (
                    1.0 / first_relevant_rank
                    if first_relevant_rank
                    else 0.0
                ),
                f"ndcg_at_{top_k}": ndcgs_at_k[-1],
                "latency_ms": latency,
            }
        )

    classification = confusion_matrix_metrics(
        actual_labels,
        predicted_labels,
    )

    auc = calculate_roc_auc(actual_labels, relevance_scores)

    summary = {
        "experiment": result_path.stem,
        "source_file": result_path.name,
        "questions_in_file": len(questions),
        "questions_evaluated": len(per_question),
        "top_k": top_k,
        "classification": classification,
        "roc_auc": auc,
        "macro_precision": mean(question_precisions) if question_precisions else None,
        "macro_recall": mean(question_recalls) if question_recalls else None,
        "macro_f1_score": mean(question_f1s) if question_f1s else None,
        f"recall_at_{top_k}": mean(recalls_at_k) if recalls_at_k else None,
        "mrr": mean(reciprocal_ranks) if reciprocal_ranks else None,
        f"ndcg_at_{top_k}": mean(ndcgs_at_k) if ndcgs_at_k else None,
        "latency_ms": {
            "mean": mean(latencies) if latencies else None,
            "median": median(latencies) if latencies else None,
            "p95": percentile(latencies, 0.95),
            "generation_mean": (
                mean(generation_latencies) if generation_latencies else None
            ),
            "measured_questions": len(latencies),
        },
        "per_question": per_question,
        "interpretation_notes": [
            "Precision, recall, F1 and confusion matrix use top-k selection as the positive prediction rule.",
            "Unannotated candidate passages are assumed non-relevant; validate that gold evidence annotations are sufficiently complete.",
            "ROC-AUC is None when both positive and negative labels are not present.",
            "Latency is copied from the saved experiment and is not remeasured by this script.",
            "Saved results are only valid for the dataset and model configuration used to produce them.",
        ],
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate classification, ranking and latency metrics "
                    "from saved RagEval experiments."
    )

    parser.add_argument(
        "--files",
        nargs="*",
        help="Specific result JSON files to evaluate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of final passages treated as selected (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated metric reports.",
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    if args.files:
        files = [Path(name).resolve() for name in args.files]
    else:
        files = sorted(DEFAULT_RESULTS_DIR.glob("*.json"))

    if not files:
        raise SystemExit(
            f"No experiment JSON files found. Checked: {DEFAULT_RESULTS_DIR}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    for result_path in files:
        if not result_path.is_file():
            print(f"SKIP: file not found: {result_path}")
            continue

        try:
            summary = evaluate_result_file(result_path, args.top_k)
            summaries.append(summary)

            output_path = (
                args.output_dir / f"{result_path.stem}-metrics.json"
            )
            output_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            print(f"\nExperiment: {summary['experiment']}")
            print(f"Questions evaluated: {summary['questions_evaluated']}")
            print(
                "Precision:",
                summary["classification"]["precision"],
            )
            print(
                "Recall:",
                summary["classification"]["recall"],
            )
            print(
                "F1-score:",
                summary["classification"]["f1_score"],
            )
            print("ROC-AUC:", summary["roc_auc"])
            print(
                f"Recall@{args.top_k}:",
                summary[f"recall_at_{args.top_k}"],
            )
            print("MRR:", summary["mrr"])
            print(
                f"nDCG@{args.top_k}:",
                summary[f"ndcg_at_{args.top_k}"],
            )
            print(
                "Mean latency (ms):",
                summary["latency_ms"]["mean"],
            )
            print("Saved:", output_path)

        except (ValueError, OSError, json.JSONDecodeError) as exc:
            print(f"ERROR evaluating {result_path.name}: {exc}")

    if summaries:
        comparison_path = args.output_dir / "comparison.csv"

        columns = [
            "experiment",
            "questions_evaluated",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            f"recall_at_{args.top_k}",
            "mrr",
            f"ndcg_at_{args.top_k}",
            "mean_latency_ms",
            "median_latency_ms",
            "p95_latency_ms",
        ]

        with comparison_path.open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()

            for summary in summaries:
                classification = summary["classification"]
                latency = summary["latency_ms"]

                writer.writerow(
                    {
                        "experiment": summary["experiment"],
                        "questions_evaluated": summary["questions_evaluated"],
                        "precision": classification["precision"],
                        "recall": classification["recall"],
                        "f1_score": classification["f1_score"],
                        "roc_auc": summary["roc_auc"],
                        f"recall_at_{args.top_k}": summary[
                            f"recall_at_{args.top_k}"
                        ],
                        "mrr": summary["mrr"],
                        f"ndcg_at_{args.top_k}": summary[
                            f"ndcg_at_{args.top_k}"
                        ],
                        "mean_latency_ms": latency["mean"],
                        "median_latency_ms": latency["median"],
                        "p95_latency_ms": latency["p95"],
                    }
                )

        print(f"\nComparison CSV saved: {comparison_path}")


if __name__ == "__main__":
    main()
