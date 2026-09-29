#!/usr/bin/env python3
"""CampusMind Routing / Risk Benchmark V1.

Dev:
    python evaluation/evaluate_routing_risk.py --split dev

Frozen test (requires explicit acknowledgement):
    python evaluation/evaluate_routing_risk.py --split test --confirm-test
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.intent_analyzer import IntentAnalyzer
from core.risk_analyzer import RiskAnalyzer
from core.risk_policy import RiskPolicyEngine

DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "routing_risk_v1.jsonl"


def load_cases(split: str):
    rows = []
    with DATASET.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if split == "all" or row["split"] == split:
                rows.append(row)
    return rows


def macro_f1(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    scores = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores[label] = {"precision": precision, "recall": recall, "f1": f1}
    return sum(v["f1"] for v in scores.values()) / len(scores), scores


def binary_high_metrics(y_true, y_pred):
    tp = sum(t == "high" and p == "high" for t, p in zip(y_true, y_pred))
    fp = sum(t != "high" and p == "high" for t, p in zip(y_true, y_pred))
    fn = sum(t == "high" and p != "high" for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, tp, fp, fn


def pct(x):
    return f"{x * 100:.1f}%"


def evaluate(split: str):
    rows = load_cases(split)
    intent_analyzer = IntentAnalyzer()
    risk_analyzer = RiskAnalyzer()
    policy_engine = RiskPolicyEngine()

    records = []
    for row in rows:
        intent = intent_analyzer.analyze(row["message"])
        risk = risk_analyzer.analyze(row["message"])
        policy = policy_engine.decide(intent, risk)
        records.append({
            **row,
            "pred_intent": intent.intent.value,
            "pred_risk": risk.risk_level.value,
            "pred_agent": policy.allowed_agent.value,
        })

    n = len(records)
    if not n:
        raise SystemExit(f"No samples for split={split}")

    intent_acc = sum(r["pred_intent"] == r["expected_intent"] for r in records) / n
    risk_acc = sum(r["pred_risk"] == r["expected_risk"] for r in records) / n
    route_acc = sum(r["pred_agent"] == r["expected_agent"] for r in records) / n

    y_intent = [r["expected_intent"] for r in records]
    p_intent = [r["pred_intent"] for r in records]
    y_risk = [r["expected_risk"] for r in records]
    p_risk = [r["pred_risk"] for r in records]

    intent_macro, _ = macro_f1(y_intent, p_intent)
    risk_macro, _ = macro_f1(y_risk, p_risk)
    hp, hr, hf1, htp, hfp, hfn = binary_high_metrics(y_risk, p_risk)

    print("=" * 70)
    print("CampusMind Routing & Risk Benchmark V1")
    print("=" * 70)
    print(f"Split                    : {split}")
    print(f"Samples                  : {n}")
    print(f"Intent Accuracy          : {pct(intent_acc)}")
    print(f"Intent Macro-F1          : {pct(intent_macro)}")
    print(f"Risk Accuracy            : {pct(risk_acc)}")
    print(f"Risk Macro-F1            : {pct(risk_macro)}")
    print(f"High-Risk Precision      : {pct(hp)}")
    print(f"High-Risk Recall         : {pct(hr)}")
    print(f"High-Risk F1             : {pct(hf1)}")
    print(f"High-Risk TP/FP/FN       : {htp}/{hfp}/{hfn}")
    print(f"Agent Routing Accuracy   : {pct(route_acc)}")

    print("\nPer-category:")
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)
    for cat in sorted(by_cat):
        items = by_cat[cat]
        ia = sum(x["pred_intent"] == x["expected_intent"] for x in items) / len(items)
        ra = sum(x["pred_risk"] == x["expected_risk"] for x in items) / len(items)
        aa = sum(x["pred_agent"] == x["expected_agent"] for x in items) / len(items)
        print(f"  {cat:14s} n={len(items):2d} intent={pct(ia):>6s} risk={pct(ra):>6s} route={pct(aa):>6s}")

    errors = [
        r for r in records
        if (
            r["pred_intent"] != r["expected_intent"]
            or r["pred_risk"] != r["expected_risk"]
            or r["pred_agent"] != r["expected_agent"]
        )
    ]
    print(f"\nError samples: {len(errors)}")
    for r in errors:
        print(
            f"\n[{r['id']}] {r['message']}\n"
            f"  category: {r['category']}\n"
            f"  intent  : {r['expected_intent']} -> {r['pred_intent']}\n"
            f"  risk    : {r['expected_risk']} -> {r['pred_risk']}\n"
            f"  agent   : {r['expected_agent']} -> {r['pred_agent']}"
        )

    report = {
        "benchmark": "routing_risk_v1",
        "split": split,
        "samples": n,
        "intent_accuracy": intent_acc,
        "intent_macro_f1": intent_macro,
        "risk_accuracy": risk_acc,
        "risk_macro_f1": risk_macro,
        "high_risk_precision": hp,
        "high_risk_recall": hr,
        "high_risk_f1": hf1,
        "high_risk_tp": htp,
        "high_risk_fp": hfp,
        "high_risk_fn": hfn,
        "agent_routing_accuracy": route_acc,
        "errors": [{
            "id": r["id"],
            "message": r["message"],
            "category": r["category"],
            "expected_intent": r["expected_intent"],
            "pred_intent": r["pred_intent"],
            "expected_risk": r["expected_risk"],
            "pred_risk": r["pred_risk"],
            "expected_agent": r["expected_agent"],
            "pred_agent": r["pred_agent"],
        } for r in errors],
    }
    path = PROJECT_ROOT / "evaluation" / f"report_{split}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved to: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument(
        "--confirm-test",
        action="store_true",
        help="Required when running the frozen test split.",
    )
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_test:
        raise SystemExit(
            "Frozen test split is intentionally protected. "
            "Use --confirm-test only after dev-side changes are finished."
        )
    evaluate(args.split)
