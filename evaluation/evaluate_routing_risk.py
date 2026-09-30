#!/usr/bin/env python3
"""CampusMind Routing / Risk evaluation.

Supports two evaluation modes:

rules:
    Only deterministic IntentAnalyzer + RiskAnalyzer.

hybrid:
    Deterministic fast path + LLM SemanticAnalyzer fallback.

This script evaluates only:
- Intent
- Risk
- Policy / Agent routing

It does NOT run the selected Agent, Tool Calling Loop, RAG,
or response generation, so evaluation does not waste extra LLM calls.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import Settings, build_llm_client
from core.intent_analyzer import IntentAnalyzer
from core.risk_analyzer import RiskAnalyzer
from core.risk_policy import RiskPolicyEngine
from core.schemas import RiskAssessment, RiskLevel
from core.semantic_analyzer import SemanticAnalyzer


DATASET = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "routing_risk_v1.jsonl"
)


def load_cases(split: str):
    rows = []

    with DATASET.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if not line.strip():
                continue

            row = json.loads(line)

            if (
                split == "all"
                or row["split"] == split
            ):
                rows.append(row)

    return rows


def macro_f1(
    y_true,
    y_pred,
):
    labels = sorted(
        set(y_true)
        | set(y_pred)
    )

    scores = {}

    for label in labels:
        tp = sum(
            t == label
            and p == label
            for t, p in zip(
                y_true,
                y_pred,
            )
        )

        fp = sum(
            t != label
            and p == label
            for t, p in zip(
                y_true,
                y_pred,
            )
        )

        fn = sum(
            t == label
            and p != label
            for t, p in zip(
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
            / (precision + recall)
            if precision + recall
            else 0.0
        )

        scores[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    macro = (
        sum(
            item["f1"]
            for item in scores.values()
        )
        / len(scores)
    )

    return macro, scores


def binary_high_metrics(
    y_true,
    y_pred,
):
    tp = sum(
        t == "high"
        and p == "high"
        for t, p in zip(
            y_true,
            y_pred,
        )
    )

    fp = sum(
        t != "high"
        and p == "high"
        for t, p in zip(
            y_true,
            y_pred,
        )
    )

    fn = sum(
        t == "high"
        and p != "high"
        for t, p in zip(
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
        / (precision + recall)
        if precision + recall
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


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


async def evaluate_case_rules(
    row,
    *,
    intent_analyzer,
    risk_analyzer,
    policy_engine,
):
    intent = intent_analyzer.analyze(
        row["message"]
    )

    risk = risk_analyzer.analyze(
        row["message"]
    )

    policy = policy_engine.decide(
        intent,
        risk,
    )

    return {
        **row,
        "pred_intent":
            intent.intent.value,
        "pred_risk":
            risk.risk_level.value,
        "pred_agent":
            policy.allowed_agent.value,
        "semantic_fallback": False,
        "semantic_error": None,
    }


async def evaluate_case_hybrid(
    row,
    *,
    intent_analyzer,
    risk_analyzer,
    semantic_analyzer,
    policy_engine,
):
    message = row["message"]

    # 1) Deterministic fast path.
    intent = intent_analyzer.analyze(
        message
    )

    risk = risk_analyzer.analyze(
        message
    )

    intent_needs_semantic = (
        intent_analyzer.needs_semantic_fallback(
            message,
            intent,
        )
    )

    risk_needs_semantic = (
        risk_analyzer.needs_semantic_fallback(
            message,
            risk,
        )
    )

    semantic_used = False
    semantic_error = None

    # 2) Only ambiguous/conflicting cases
    #    enter LLM semantic fallback.
    if (
        intent_needs_semantic
        or risk_needs_semantic
    ):
        semantic_used = True

        try:
            semantic = (
                await semantic_analyzer.analyze(
                    message
                )
            )

        except Exception as exc:
            semantic_error = (
                f"{exc.__class__.__name__}: "
                f"{str(exc)[:200]}"
            )

            # Same safety behavior as Orchestrator:
            # if semantic disambiguation fails while
            # potential high-risk language exists,
            # fail closed.
            if (
                risk_needs_semantic
                and risk_analyzer
                .has_potential_high_signal(
                    message
                )
            ):
                risk = RiskAssessment(
                    risk_level=RiskLevel.HIGH,
                    signals=[
                        "semantic_fallback_failed",
                        "potential_high_risk_signal",
                    ],
                    confidence=0.80,
                )

        else:
            # Intent may be replaced only when
            # the deterministic result was ambiguous.
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

            # Risk may be semantically corrected,
            # but deterministic HIGH cannot be
            # downgraded by the LLM.
            if (
                risk_needs_semantic
                and semantic
                .risk_assessment
                .confidence
                >= 0.75
            ):
                semantic_risk = (
                    semantic.risk_assessment
                )

                if (
                    risk.risk_level
                    == RiskLevel.HIGH
                    and semantic_risk.risk_level
                    != RiskLevel.HIGH
                ):
                    pass
                else:
                    risk = semantic_risk

    # 3) Final deterministic policy decision.
    policy = policy_engine.decide(
        intent,
        risk,
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
    split: str,
    mode: str,
):
    rows = load_cases(split)

    if not rows:
        raise SystemExit(
            f"No samples for split={split}"
        )

    intent_analyzer = IntentAnalyzer()
    risk_analyzer = RiskAnalyzer()
    policy_engine = RiskPolicyEngine()

    semantic_analyzer = None

    if mode == "hybrid":
        settings = Settings.from_env()

        if settings.llm_mode != "real":
            raise SystemExit(
                "Hybrid evaluation requires "
                "LLM_MODE=real."
            )

        llm_client = build_llm_client(
            settings
        )

        semantic_analyzer = (
            SemanticAnalyzer(
                llm_client
            )
        )

    records = []

    for index, row in enumerate(
        rows,
        start=1,
    ):
        print(
            f"[{index:03d}/{len(rows):03d}] "
            f"{row['id']} ... ",
            end="",
            flush=True,
        )

        if mode == "rules":
            record = (
                await evaluate_case_rules(
                    row,
                    intent_analyzer=
                        intent_analyzer,
                    risk_analyzer=
                        risk_analyzer,
                    policy_engine=
                        policy_engine,
                )
            )

        else:
            record = (
                await evaluate_case_hybrid(
                    row,
                    intent_analyzer=
                        intent_analyzer,
                    risk_analyzer=
                        risk_analyzer,
                    semantic_analyzer=
                        semantic_analyzer,
                    policy_engine=
                        policy_engine,
                )
            )

        records.append(record)

        fallback_text = (
            "semantic"
            if record[
                "semantic_fallback"
            ]
            else "rules"
        )

        print(fallback_text)

    n = len(records)

    intent_acc = (
        sum(
            row["pred_intent"]
            == row["expected_intent"]
            for row in records
        )
        / n
    )

    risk_acc = (
        sum(
            row["pred_risk"]
            == row["expected_risk"]
            for row in records
        )
        / n
    )

    route_acc = (
        sum(
            row["pred_agent"]
            == row["expected_agent"]
            for row in records
        )
        / n
    )

    y_intent = [
        row["expected_intent"]
        for row in records
    ]

    p_intent = [
        row["pred_intent"]
        for row in records
    ]

    y_risk = [
        row["expected_risk"]
        for row in records
    ]

    p_risk = [
        row["pred_risk"]
        for row in records
    ]

    intent_macro, _ = macro_f1(
        y_intent,
        p_intent,
    )

    risk_macro, _ = macro_f1(
        y_risk,
        p_risk,
    )

    (
        high_precision,
        high_recall,
        high_f1,
        high_tp,
        high_fp,
        high_fn,
    ) = binary_high_metrics(
        y_risk,
        p_risk,
    )

    semantic_calls = sum(
        row["semantic_fallback"]
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
        "CampusMind Routing & Risk Benchmark"
    )
    print("=" * 70)

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
        f"{n}"
    )

    print(
        f"Intent Accuracy          : "
        f"{pct(intent_acc)}"
    )

    print(
        f"Intent Macro-F1          : "
        f"{pct(intent_macro)}"
    )

    print(
        f"Risk Accuracy            : "
        f"{pct(risk_acc)}"
    )

    print(
        f"Risk Macro-F1            : "
        f"{pct(risk_macro)}"
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
        f"{pct(route_acc)}"
    )

    print(
        f"Semantic Fallback Calls  : "
        f"{semantic_calls}/{n} "
        f"({pct(semantic_calls / n)})"
    )

    print(
        f"Semantic Errors          : "
        f"{semantic_errors}"
    )

    print("\nPer-category:")

    by_category = defaultdict(list)

    for row in records:
        by_category[
            row["category"]
        ].append(row)

    for category in sorted(
        by_category
    ):
        items = (
            by_category[category]
        )

        category_intent = (
            sum(
                row["pred_intent"]
                == row["expected_intent"]
                for row in items
            )
            / len(items)
        )

        category_risk = (
            sum(
                row["pred_risk"]
                == row["expected_risk"]
                for row in items
            )
            / len(items)
        )

        category_route = (
            sum(
                row["pred_agent"]
                == row["expected_agent"]
                for row in items
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
            != row["expected_intent"]
            or row["pred_risk"]
            != row["expected_risk"]
            or row["pred_agent"]
            != row["expected_agent"]
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
                f"  semantic_error: "
                f"{row['semantic_error']}"
            )

    report = {
        "mode": mode,
        "split": split,
        "samples": n,
        "intent_accuracy":
            intent_acc,
        "intent_macro_f1":
            intent_macro,
        "risk_accuracy":
            risk_acc,
        "risk_macro_f1":
            risk_macro,
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
            route_acc,
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
        f"\nReport saved to: "
        f"{report_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=[
            "dev",
            "test",
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

    args = parser.parse_args()

    asyncio.run(
        evaluate(
            split=args.split,
            mode=args.mode,
        )
    )