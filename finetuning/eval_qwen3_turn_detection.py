# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from datasets import load_dataset

from qwen_asr.turn_detection import (
    DEFAULT_TURN_DETECTION_PROMPT,
    Qwen3GenerativeTurnDetector,
    Qwen3TurnDetector,
)
from qwen_asr.turn_detection.metrics import (
    canonical_prediction_label,
    finite_or_none,
    latency_summary,
    normalize_binary_label,
    summarize_binary_classification,
    summarize_slices,
    threshold_metrics,
)


def parse_args():
    p = argparse.ArgumentParser("Evaluate Qwen3-ASR turn detection checkpoints")
    p.add_argument("--mode", type=str, choices=["classifier", "generative"], required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--test_file", type=str, default="")
    p.add_argument("--output_json", type=str, default="")
    p.add_argument("--predictions_jsonl", type=str, default="")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--target_complete_precision", type=float, default=0.97)
    p.add_argument("--default_left_context_ms", type=float, default=2000.0)
    p.add_argument("--default_right_context_ms", type=float, default=600.0)
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--max_new_tokens", type=int, default=4)
    p.add_argument("--constrained_decode", type=int, default=1)
    p.add_argument("--device_map", type=str, default="auto")
    return p.parse_args()


def load_json_records(path: str) -> List[Dict[str, Any]]:
    ds = load_dataset("json", data_files={"data": path})["data"]
    return [dict(row) for row in ds]


def batched(items: Sequence[Dict[str, Any]], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def resolve_prompt(model_path: str, prompt: str) -> str:
    prompt = (prompt or "").strip()
    if prompt:
        return prompt
    config_path = os.path.join(model_path, "turn_detection_generative_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("prompt", DEFAULT_TURN_DETECTION_PROMPT)
    return DEFAULT_TURN_DETECTION_PROMPT


def load_model(args):
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    if args.mode == "classifier":
        return Qwen3TurnDetector.from_pretrained(
            args.model_path,
            dtype=dtype,
            device_map=args.device_map,
        )
    return Qwen3GenerativeTurnDetector.from_pretrained(
        args.model_path,
        prompt=resolve_prompt(args.model_path, args.prompt),
        max_new_tokens=args.max_new_tokens,
        dtype=dtype,
        device_map=args.device_map,
    )


def evaluate_split(model, records: List[Dict[str, Any]], args, split_name: str) -> Dict[str, Any]:
    y_true: List[int] = []
    complete_prob: List[float] = []
    latencies: List[float] = []
    ttfts: List[float] = []
    full_inference_latencies: List[float] = []
    first_token_margin: List[float] = []
    exact_match: List[float] = []
    enriched_records: List[Dict[str, Any]] = []

    for batch in batched(records, max(1, int(args.batch_size))):
        audios = [row["audio"] for row in batch]
        cut_time_ms = [row.get("cut_time_ms") for row in batch]
        left_context_ms = [row.get("left_context_ms", args.default_left_context_ms) for row in batch]
        right_context_ms = [row.get("right_context_ms", args.default_right_context_ms) for row in batch]
        prompts = [row.get("prompt", None) for row in batch]

        if args.mode == "classifier":
            predictions = model.predict_batch(
                audio=audios,
                cut_time_ms=cut_time_ms,
                left_context_ms=left_context_ms,
                right_context_ms=right_context_ms,
                prompt=prompts,
                threshold=0.5,
                measure_timings_per_sample=True,
            )
        else:
            predictions = model.predict_batch(
                audio=audios,
                cut_time_ms=cut_time_ms,
                left_context_ms=left_context_ms,
                right_context_ms=right_context_ms,
                prompt=prompts,
                constrained_decode=bool(args.constrained_decode),
                measure_timings_per_sample=True,
            )

        for row, pred in zip(batch, predictions):
            label_id = normalize_binary_label(row.get("label"))
            if label_id is None:
                raise ValueError(f"Unsupported label in evaluation row: {row.get('label')!r}")
            y_true.append(int(label_id))
            complete_prob.append(float(pred.complete_prob))
            if pred.latency_ms is not None:
                latencies.append(float(pred.latency_ms))
            if getattr(pred, "ttft_ms", None) is not None:
                ttfts.append(float(pred.ttft_ms))
            if getattr(pred, "full_inference_ms", None) is not None:
                full_inference_latencies.append(float(pred.full_inference_ms))
            if getattr(pred, "first_token_margin", None) is not None:
                first_token_margin.append(float(pred.first_token_margin))
            if getattr(pred, "exact_match", None) is not None:
                exact_match.append(1.0 if pred.exact_match else 0.0)

            enriched = dict(row)
            enriched.update(
                {
                    "split": split_name,
                    "gold_label_id": int(label_id),
                    "gold_label": "complete" if int(label_id) == 1 else "incomplete",
                    "pred_label": canonical_prediction_label(pred.label),
                    "complete_prob": float(pred.complete_prob),
                    "incomplete_prob": float(pred.incomplete_prob),
                    "latency_ms": finite_or_none(pred.latency_ms),
                    "ttft_ms": finite_or_none(getattr(pred, "ttft_ms", None)),
                    "full_inference_ms": finite_or_none(getattr(pred, "full_inference_ms", None)),
                }
            )
            if getattr(pred, "raw_text", None) is not None:
                enriched["raw_text"] = pred.raw_text
            if getattr(pred, "complete_logprob", None) is not None:
                enriched["complete_logprob"] = finite_or_none(pred.complete_logprob)
            if getattr(pred, "incomplete_logprob", None) is not None:
                enriched["incomplete_logprob"] = finite_or_none(pred.incomplete_logprob)
            if getattr(pred, "first_token_margin", None) is not None:
                enriched["first_token_margin"] = finite_or_none(pred.first_token_margin)
            if getattr(pred, "exact_match", None) is not None:
                enriched["exact_match"] = bool(pred.exact_match)
            enriched_records.append(enriched)

    summary = summarize_binary_classification(
        y_true,
        complete_prob,
        target_precision=float(args.target_complete_precision),
    )
    summary["latency"] = latency_summary(latencies)
    summary["ttft"] = latency_summary(ttfts)
    summary["full_inference"] = latency_summary(full_inference_latencies)
    if first_token_margin:
        summary["first_token_margin"] = {
            "mean": float(np.mean(first_token_margin)),
            "p50": float(np.percentile(first_token_margin, 50)),
            "p95": float(np.percentile(first_token_margin, 95)),
        }
    if exact_match:
        summary["exact_match_rate"] = float(np.mean(exact_match))
    summary["slice_metrics"] = summarize_slices(
        enriched_records,
        y_true,
        complete_prob,
        target_precision=float(args.target_complete_precision),
    )
    return {
        "summary": summary,
        "records": enriched_records,
        "y_true": y_true,
        "complete_prob": complete_prob,
    }


def apply_dev_thresholds_to_test(dev_summary: Dict[str, Any], y_true: Sequence[int], y_prob: Sequence[float]) -> Dict[str, Any]:
    high_precision_thr = float(dev_summary["threshold_high_precision"]["threshold"])
    balanced_f1_thr = float(dev_summary["threshold_balanced_f1"]["threshold"])
    return {
        "threshold_default_0_5": threshold_metrics(y_true, y_prob, threshold=0.5),
        "threshold_high_precision_from_dev": threshold_metrics(y_true, y_prob, threshold=high_precision_thr),
        "threshold_balanced_f1_from_dev": threshold_metrics(y_true, y_prob, threshold=balanced_f1_thr),
    }


def attach_threshold_flags(records: List[Dict[str, Any]], threshold_block: Dict[str, Any]) -> None:
    thr_default = float(threshold_block["threshold_default_0_5"]["threshold"])
    thr_hp = float(
        threshold_block.get("threshold_high_precision_from_dev", threshold_block.get("threshold_high_precision", {})).get(
            "threshold",
            0.5,
        )
    )
    thr_bal = float(
        threshold_block.get("threshold_balanced_f1_from_dev", threshold_block.get("threshold_balanced_f1", {})).get(
            "threshold",
            0.5,
        )
    )
    for record in records:
        gold = int(record["gold_label_id"])
        prob = float(record["complete_prob"])
        record["pred_default_0_5"] = "complete" if prob >= thr_default else "incomplete"
        record["pred_high_precision"] = "complete" if prob >= thr_hp else "incomplete"
        record["pred_balanced_f1"] = "complete" if prob >= thr_bal else "incomplete"
        record["false_complete_default_0_5"] = bool(prob >= thr_default and gold == 0)
        record["false_complete_high_precision"] = bool(prob >= thr_hp and gold == 0)
        record["false_complete_balanced_f1"] = bool(prob >= thr_bal and gold == 0)


def maybe_write_json(path: str, payload: Dict[str, Any]) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def maybe_write_predictions(path: str, records: List[Dict[str, Any]]) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    if not args.eval_file and not args.test_file:
        raise ValueError("At least one of --eval_file or --test_file must be provided.")

    model = load_model(args)
    report: Dict[str, Any] = {
        "mode": args.mode,
        "model_path": args.model_path,
        "target_complete_precision": float(args.target_complete_precision),
    }
    records_for_dump: List[Dict[str, Any]] = []

    eval_result = None
    if args.eval_file:
        eval_records = load_json_records(args.eval_file)
        eval_result = evaluate_split(model, eval_records, args, split_name="eval")
        attach_threshold_flags(eval_result["records"], eval_result["summary"])
        report["eval"] = {
            "file": args.eval_file,
            "summary": eval_result["summary"],
        }
        records_for_dump.extend(eval_result["records"])

    if args.test_file:
        test_records = load_json_records(args.test_file)
        test_result = evaluate_split(model, test_records, args, split_name="test")
        if eval_result is not None:
            test_result["summary"]["thresholds_applied_from_dev"] = apply_dev_thresholds_to_test(
                eval_result["summary"],
                test_result["y_true"],
                test_result["complete_prob"],
            )
            attach_threshold_flags(test_result["records"], test_result["summary"]["thresholds_applied_from_dev"])
        else:
            attach_threshold_flags(test_result["records"], test_result["summary"])
        report["test"] = {
            "file": args.test_file,
            "summary": test_result["summary"],
        }
        records_for_dump.extend(test_result["records"])

    maybe_write_json(args.output_json, report)
    maybe_write_predictions(args.predictions_jsonl, records_for_dump)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
