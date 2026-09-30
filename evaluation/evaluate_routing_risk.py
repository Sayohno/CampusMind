#!/usr/bin/env python3
"""CampusMind Routing / Risk evaluation.

支持两种模式：

rules
    仅使用确定性的 IntentAnalyzer
    与 RiskAnalyzer。

hybrid
    使用 Rule Fast Path，
    在需要时调用 LLM SemanticAnalyzer。

该评测只覆盖：

- Intent classification
- Risk classification
- Policy / Agent routing

不会真正执行 Agent、Tool Calling、
RAG 或最终回答生成，因此不会产生
与路由评测无关的额外 LLM 调用。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from core.config import (
    Settings,
    build_llm_client,
)
from core.intent_analyzer import (
    IntentAnalyzer,
)
from core.risk_analyzer import (
    RiskAnalyzer,
)
from core.risk_policy import (
    RiskPolicyEngine,
)
from core.schemas import (
    RiskAssessment,
    RiskLevel,
)
from core.semantic_analyzer import (
    SemanticAnalyzer,
)


DATASETS = {
    "diagnostic": (
        PROJECT_ROOT
        / "evaluation"
        / "datasets"
        / "routing_risk_diagnostic.jsonl"
    ),
    "holdout": (
        PROJECT_ROOT
        / "evaluation"
        / "datasets"
        / "routing_risk_holdout.jsonl"
    ),
}


def load_cases(
    dataset: str,
    split: str,
) -> list[dict]:
    dataset_path = DATASETS[
        dataset
    ]

    rows: list[dict] = []

    with dataset_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if not line.strip():
                continue

            row = json.loads(
                line
            )

            if (
                split == "all"
                or row.get("split")
                == split
            ):
                rows.append(
                    row
                )

    return rows


def macro_f1(
    y_true: list[str],
    y_pred: list[str],
) -> tuple[
    float,
    dict[str, dict[str, float]],
]:
    labels = sorted(
        set(y_true)
        | set(y_pred)
    )

    scores: dict[
        str,
        dict[str, float],
    ] = {}

    for label in labels:
        tp = sum(
            true == label
            and pred == label
            for true, pred
            in zip(
                y_true,
                y_pred,
            )
        )

        fp = sum(
            true != label
            and pred == label
            for true, pred
            in zip(
                y_true,
                y_pred,
            )
        )

        fn = sum(
            true == label
            and pred != label
            for true, pred
            in zip(
                y_true,
                y_pred,
            )
        )

        precision = (
            tp / (tp + fp)
            if tp + fp
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn
            else 0.0
        )

        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
            if (
                precision
                + recall
            )
            else 0.0
        )

        scores[label] = {
            "precision":
                precision,
            "recall":
                recall,
            "f1":
                f1,
        }

    macro = (
        sum(
            item["f1"]
            for item
            in scores.values()
        )
        / len(scores)
        if scores
        else 0.0
    )

    return (
        macro,
        scores,
    )


def binary_high_metrics(
    y_true: list[str],
    y_pred: list[str],
) -> tuple[
    float,
    float,
    float,
    int,
    int,
    int,
]:
    tp = sum(
        true == "high"
        and pred == "high"
        for true, pred
        in zip(
            y_true,
            y_pred,
        )
    )

    fp = sum(
        true != "high"
        and pred == "high"
        for true, pred
        in zip(
            y_true,
            y_pred,
        )
    )

    fn = sum(
        true == "high"
        and pred != "high"
        for true, pred
        in zip(
            y_true,
            y_pred,
        )
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        )
        else 0.0
    )

    return (
        precision,
        recall,
        f1,
        tp,
        fp,
        fn,
    )


def pct(
    value: float,
) -> str:
    return (
        f"{value * 100:.1f}%"
    )


async def evaluate_case_rules(
    row: dict,
    *,
    intent_analyzer:
        IntentAnalyzer,
    risk_analyzer:
        RiskAnalyzer,
    policy_engine:
        RiskPolicyEngine,
) -> dict:
    message = row[
        "message"
    ]

    intent = (
        intent_analyzer.analyze(
            message
        )
    )

    risk = (
        risk_analyzer.analyze(
            message
        )
    )

    policy = (
        policy_engine.decide(
            intent,
            risk,
        )
    )

    return {
        **row,
        "pred_intent":
            intent.intent.value,
        "pred_risk":
            risk.risk_level.value,
        "pred_agent":
            policy.allowed_agent.value,
        "semantic_fallback":
            False,
        "semantic_error":
            None,
    }


async def evaluate_case_hybrid(
    row: dict,
    *,
    intent_analyzer:
        IntentAnalyzer,
    risk_analyzer:
        RiskAnalyzer,
    semantic_analyzer:
        SemanticAnalyzer,
    policy_engine:
        RiskPolicyEngine,
) -> dict:
    message = row[
        "message"
    ]

    intent = (
        intent_analyzer.analyze(
            message
        )
    )

    risk = (
        risk_analyzer.analyze(
            message
        )
    )

    intent_needs_semantic = (
        intent_analyzer
        .needs_semantic_fallback(
            message,
            intent,
        )
    )

    risk_needs_semantic = (
        risk_analyzer
        .needs_semantic_fallback(
            message,
            risk,
        )
    )

    semantic_used = False
    semantic_error = None

    if (
        intent_needs_semantic
        or risk_needs_semantic
    ):
        semantic_used = True

        try:
            semantic = (
                await
                semantic_analyzer.analyze(
                    message
                )
            )

        except Exception as exc:
            semantic_error = (
                f"{exc.__class__.__name__}: "
                f"{str(exc)[:200]}"
            )

            if (
                risk_needs_semantic
                and risk_analyzer
                .has_potential_high_signal(
                    message
                )
            ):
                risk = (
                    RiskAssessment(
                        risk_level=(
                            RiskLevel.HIGH
                        ),
                        signals=[
                            "semantic_fallback_failed",
                            "potential_high_risk_signal",
                        ],
                        confidence=0.80,
                    )
                )

        else:
            if (
                intent_needs_semantic
                and semantic
                .intent_result
                .confidence
                >= 0.70
            ):
                intent = (
                    semantic.intent_result
                )

            if (
                risk_needs_semantic
                and semantic
                .risk_assessment
                .confidence
                >= 0.75
            ):
                semantic_risk = (
                    semantic
                    .risk_assessment
                )

                if not (
                    risk.risk_level
                    == RiskLevel.HIGH
                    and (
                        semantic_risk
                        .risk_level
                        != RiskLevel.HIGH
                    )
                ):
                    risk = (
                        semantic_risk
                    )

    policy = (
        policy_engine.decide(
            intent,
            risk,
        )
    )

    return {
        **row,
        "pred_intent":
            intent.intent.value,
        "pred_risk":
            risk.risk_level.value,
        "pred_agent":
            policy.allowed_agent.value,
        "semantic_fallback":
            semantic_used,
        "semantic_error":
            semantic_error,
    }


async def evaluate(
    dataset: str,
    split: str,
    mode: str,
) -> None:
    rows = load_cases(
        dataset,
        split,
    )

    if not rows:
        raise SystemExit(
            "No samples for "
            f"dataset={dataset}, "
            f"split={split}"
        )

    intent_analyzer = (
        IntentAnalyzer()
    )

    risk_analyzer = (
        RiskAnalyzer()
    )

    policy_engine = (
        RiskPolicyEngine()
    )

    semantic_analyzer = None

    if mode == "hybrid":
        settings = (
            Settings.from_env()
        )

        if (
            settings.llm_mode
            != "real"
        ):
            raise SystemExit(
                "Hybrid evaluation "
                "requires LLM_MODE=real."
            )

        llm_client = (
            build_llm_client(
                settings
            )
        )

        semantic_analyzer = (
            SemanticAnalyzer(
                llm_client
            )
        )

    records: list[dict] = []

    for index, row in enumerate(
        rows,
        start=1,
    ):
        print(
            f"[{index:03d}/"
            f"{len(rows):03d}] "
            f"{row['id']} ... ",
            end="",
            flush=True,
        )

        if mode == "rules":
            record = (
                await
                evaluate_case_rules(
                    row,
                    intent_analyzer=(
                        intent_analyzer
                    ),
                    risk_analyzer=(
                        risk_analyzer
                    ),
                    policy_engine=(
                        policy_engine
                    ),
                )
            )

        else:
            assert (
                semantic_analyzer
                is not None
            )

            record = (
                await
                evaluate_case_hybrid(
                    row,
                    intent_analyzer=(
                        intent_analyzer
                    ),
                    risk_analyzer=(
                        risk_analyzer
                    ),
                    semantic_analyzer=(
                        semantic_analyzer
                    ),
                    policy_engine=(
                        policy_engine
                    ),
                )
            )

        records.append(
            record
        )

        fallback_text = (
            "semantic"
            if record[
                "semantic_fallback"
            ]
            else "rules"
        )

        print(
            fallback_text
        )

    sample_count = len(
        records
    )

    intent_accuracy = (
        sum(
            row["pred_intent"]
            == row[
                "expected_intent"
            ]
            for row
            in records
        )
        / sample_count
    )

    risk_accuracy = (
        sum(
            row["pred_risk"]
            == row[
                "expected_risk"
            ]
            for row
            in records
        )
        / sample_count
    )

    routing_accuracy = (
        sum(
            row["pred_agent"]
            == row[
                "expected_agent"
            ]
            for row
            in records
        )
        / sample_count
    )

    expected_intents = [
        row["expected_intent"]
        for row in records
    ]

    predicted_intents = [
        row["pred_intent"]
        for row in records
    ]

    expected_risks = [
        row["expected_risk"]
        for row in records
    ]

    predicted_risks = [
        row["pred_risk"]
        for row in records
    ]

    intent_macro_f1, _ = (
        macro_f1(
            expected_intents,
            predicted_intents,
        )
    )

    risk_macro_f1, _ = (
        macro_f1(
            expected_risks,
            predicted_risks,
        )
    )

    (
        high_precision,
        high_recall,
        high_f1,
        high_tp,
        high_fp,
        high_fn,
    ) = binary_high_metrics(
        expected_risks,
        predicted_risks,
    )

    semantic_calls = sum(
        bool(
            row[
                "semantic_fallback"
            ]
        )
        for row in records
    )

    semantic_errors = sum(
        row["semantic_error"]
        is not None
        for row in records
    )

    print()
    print("=" * 70)
    print(
        "CampusMind Routing & "
        "Risk Benchmark"
    )
    print("=" * 70)

    print(
        f"Dataset                  : "
        f"{dataset}"
    )

    print(
        f"Mode                     : "
        f"{mode}"
    )

    print(
        f"Split                    : "
        f"{split}"
    )

    print(
        f"Samples                  : "
        f"{sample_count}"
    )

    print(
        f"Intent Accuracy          : "
        f"{pct(intent_accuracy)}"
    )

    print(
        f"Intent Macro-F1          : "
        f"{pct(intent_macro_f1)}"
    )

    print(
        f"Risk Accuracy            : "
        f"{pct(risk_accuracy)}"
    )

    print(
        f"Risk Macro-F1            : "
        f"{pct(risk_macro_f1)}"
    )

    print(
        f"High-Risk Precision      : "
        f"{pct(high_precision)}"
    )

    print(
        f"High-Risk Recall         : "
        f"{pct(high_recall)}"
    )

    print(
        f"High-Risk F1             : "
        f"{pct(high_f1)}"
    )

    print(
        f"High-Risk TP/FP/FN       : "
        f"{high_tp}/"
        f"{high_fp}/"
        f"{high_fn}"
    )

    print(
        f"Agent Routing Accuracy   : "
        f"{pct(routing_accuracy)}"
    )

    print(
        f"Semantic Fallback Calls  : "
        f"{semantic_calls}/"
        f"{sample_count} "
        f"("
        f"{pct(semantic_calls / sample_count)}"
        f")"
    )

    print(
        f"Semantic Errors          : "
        f"{semantic_errors}"
    )

    print(
        "\nPer-category:"
    )

    by_category: dict[
        str,
        list[dict],
    ] = defaultdict(list)

    for row in records:
        by_category[
            row["category"]
        ].append(
            row
        )

    for category in sorted(
        by_category
    ):
        items = (
            by_category[
                category
            ]
        )

        category_intent = (
            sum(
                row["pred_intent"]
                == row[
                    "expected_intent"
                ]
                for row
                in items
            )
            / len(items)
        )

        category_risk = (
            sum(
                row["pred_risk"]
                == row[
                    "expected_risk"
                ]
                for row
                in items
            )
            / len(items)
        )

        category_route = (
            sum(
                row["pred_agent"]
                == row[
                    "expected_agent"
                ]
                for row
                in items
            )
            / len(items)
        )

        print(
            f"  {category:14s} "
            f"n={len(items):2d} "
            f"intent="
            f"{pct(category_intent):>6s} "
            f"risk="
            f"{pct(category_risk):>6s} "
            f"route="
            f"{pct(category_route):>6s}"
        )

    errors = [
        row
        for row in records
        if (
            row["pred_intent"]
            != row[
                "expected_intent"
            ]
            or row["pred_risk"]
            != row[
                "expected_risk"
            ]
            or row["pred_agent"]
            != row[
                "expected_agent"
            ]
        )
    ]

    print(
        f"\nError samples: "
        f"{len(errors)}"
    )

    for row in errors:
        print(
            f"\n[{row['id']}] "
            f"{row['message']}\n"
            f"  category: "
            f"{row['category']}\n"
            f"  fallback: "
            f"{row['semantic_fallback']}\n"
            f"  intent  : "
            f"{row['expected_intent']} "
            f"-> "
            f"{row['pred_intent']}\n"
            f"  risk    : "
            f"{row['expected_risk']} "
            f"-> "
            f"{row['pred_risk']}\n"
            f"  agent   : "
            f"{row['expected_agent']} "
            f"-> "
            f"{row['pred_agent']}"
        )

        if row["semantic_error"]:
            print(
                "  semantic_error: "
                f"{row['semantic_error']}"
            )

    report = {
        "dataset":
            dataset,
        "mode":
            mode,
        "split":
            split,
        "samples":
            sample_count,
        "intent_accuracy":
            intent_accuracy,
        "intent_macro_f1":
            intent_macro_f1,
        "risk_accuracy":
            risk_accuracy,
        "risk_macro_f1":
            risk_macro_f1,
        "high_risk_precision":
            high_precision,
        "high_risk_recall":
            high_recall,
        "high_risk_f1":
            high_f1,
        "high_risk_tp":
            high_tp,
        "high_risk_fp":
            high_fp,
        "high_risk_fn":
            high_fn,
        "agent_routing_accuracy":
            routing_accuracy,
        "semantic_fallback_calls":
            semantic_calls,
        "semantic_errors":
            semantic_errors,
        "errors": [
            {
                "id":
                    row["id"],
                "message":
                    row["message"],
                "category":
                    row["category"],
                "expected_intent":
                    row[
                        "expected_intent"
                    ],
                "pred_intent":
                    row[
                        "pred_intent"
                    ],
                "expected_risk":
                    row[
                        "expected_risk"
                    ],
                "pred_risk":
                    row[
                        "pred_risk"
                    ],
                "expected_agent":
                    row[
                        "expected_agent"
                    ],
                "pred_agent":
                    row[
                        "pred_agent"
                    ],
                "semantic_fallback":
                    row[
                        "semantic_fallback"
                    ],
                "semantic_error":
                    row[
                        "semantic_error"
                    ],
            }
            for row in errors
        ],
    }

    report_path = (
        PROJECT_ROOT
        / "evaluation"
        / (
            f"report_"
            f"{dataset}_"
            f"{mode}_"
            f"{split}.json"
        )
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nReport saved to: "
        f"{report_path}"
    )


if __name__ == "__main__":
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--dataset",
        choices=[
            "diagnostic",
            "holdout",
        ],
        default="diagnostic",
    )

    parser.add_argument(
        "--split",
        choices=[
            "dev",
            "test",
            "holdout",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "rules",
            "hybrid",
        ],
        default="rules",
    )

    args = (
        parser.parse_args()
    )

    asyncio.run(
        evaluate(
            dataset=args.dataset,
            split=args.split,
            mode=args.mode,
        )
    )