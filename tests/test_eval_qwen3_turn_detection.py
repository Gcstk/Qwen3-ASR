# coding=utf-8
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
import unittest


_ROOT = Path(__file__).resolve().parents[1]

_METRICS_PATH = _ROOT / "qwen_asr" / "turn_detection" / "metrics.py"
_METRICS_SPEC = importlib.util.spec_from_file_location("qwen_asr.turn_detection.metrics", _METRICS_PATH)
_METRICS_MODULE = importlib.util.module_from_spec(_METRICS_SPEC)
assert _METRICS_SPEC is not None and _METRICS_SPEC.loader is not None
_METRICS_SPEC.loader.exec_module(_METRICS_MODULE)

_TORCH_STUB = types.ModuleType("torch")
_TORCH_STUB.cuda = types.SimpleNamespace(
    is_available=lambda: False,
    get_device_capability=lambda *_args, **_kwargs: (0, 0),
)
_TORCH_STUB.bfloat16 = "bfloat16"
_TORCH_STUB.float16 = "float16"
sys.modules.setdefault("torch", _TORCH_STUB)

_DATASETS_STUB = types.ModuleType("datasets")
_DATASETS_STUB.load_dataset = lambda *args, **kwargs: None
sys.modules.setdefault("datasets", _DATASETS_STUB)

_TURN_DETECTION_STUB = types.ModuleType("qwen_asr.turn_detection")
_TURN_DETECTION_STUB.DEFAULT_TURN_DETECTION_PROMPT = "prompt"
_TURN_DETECTION_STUB.Qwen3GenerativeTurnDetector = object
_TURN_DETECTION_STUB.Qwen3TurnDetector = object

_QWEN_ASR_STUB = types.ModuleType("qwen_asr")
_QWEN_ASR_STUB.turn_detection = _TURN_DETECTION_STUB

sys.modules["qwen_asr"] = _QWEN_ASR_STUB
sys.modules["qwen_asr.turn_detection"] = _TURN_DETECTION_STUB
sys.modules["qwen_asr.turn_detection.metrics"] = _METRICS_MODULE

_EVAL_PATH = _ROOT / "finetuning" / "eval_qwen3_turn_detection.py"
_EVAL_SPEC = importlib.util.spec_from_file_location("eval_qwen3_turn_detection", _EVAL_PATH)
_EVAL_MODULE = importlib.util.module_from_spec(_EVAL_SPEC)
assert _EVAL_SPEC is not None and _EVAL_SPEC.loader is not None
_EVAL_SPEC.loader.exec_module(_EVAL_MODULE)

evaluate_split = _EVAL_MODULE.evaluate_split


class _FakePrediction:
    def __init__(self, label, complete_prob, incomplete_prob, latency_ms, ttft_ms, full_inference_ms):
        self.label = label
        self.complete_prob = complete_prob
        self.incomplete_prob = incomplete_prob
        self.latency_ms = latency_ms
        self.ttft_ms = ttft_ms
        self.full_inference_ms = full_inference_ms


class _FakeModel:
    def __init__(self):
        self.calls = []

    def predict_batch(self, **kwargs):
        self.calls.append(kwargs)
        return [
            _FakePrediction(
                label="complete",
                complete_prob=0.9,
                incomplete_prob=0.1,
                latency_ms=12.345,
                ttft_ms=23.456,
                full_inference_ms=34.567,
            ),
            _FakePrediction(
                label="incomplete",
                complete_prob=0.2,
                incomplete_prob=0.8,
                latency_ms=22.345,
                ttft_ms=33.456,
                full_inference_ms=44.567,
            ),
        ]


class EvalQwen3TurnDetectionTest(unittest.TestCase):
    def test_evaluate_split_records_ttft_and_full_inference(self):
        args = SimpleNamespace(
            batch_size=8,
            default_left_context_ms=2000.0,
            default_right_context_ms=600.0,
            mode="classifier",
            constrained_decode=1,
            target_complete_precision=0.97,
        )
        records = [
            {"audio": "a.wav", "label": "complete"},
            {"audio": "b.wav", "label": "incomplete"},
        ]

        model = _FakeModel()
        result = evaluate_split(model, records, args, split_name="eval")

        self.assertEqual(len(model.calls), 1)
        self.assertTrue(model.calls[0]["measure_timings_per_sample"])
        self.assertEqual(result["records"][0]["ttft_ms"], 23.456)
        self.assertEqual(result["records"][0]["full_inference_ms"], 34.567)
        self.assertEqual(result["records"][1]["ttft_ms"], 33.456)
        self.assertEqual(result["records"][1]["full_inference_ms"], 44.567)
        self.assertEqual(result["summary"]["ttft"]["count"], 2.0)
        self.assertEqual(result["summary"]["full_inference"]["count"], 2.0)


if __name__ == "__main__":
    unittest.main()
