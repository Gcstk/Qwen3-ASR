# coding=utf-8
import importlib.util
import unittest
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    path = _ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


_TRAIN_CONVERTER = _load_module(
    "convert_easy_turn_to_qwen_asr_jsonl",
    "finetuning/convert_easy_turn_to_qwen_asr_jsonl.py",
)
_POSTPROCESS = _load_module(
    "convert_qwen_asr_jsonl_remove_language",
    "finetuning/convert_qwen_asr_jsonl_remove_language.py",
)


class EasyTurnLanguageCleanupTest(unittest.TestCase):
    def test_build_target_without_language_prefix(self):
        text = _TRAIN_CONVERTER.build_qwen_asr_target(
            lang_name="Chinese",
            transcript="闭嘴",
            label="wait",
            output_format="label_first",
            predict_language=False,
        )
        self.assertEqual(text, "<turn_state><wait><asr_text>闭嘴")

    def test_build_prompt_without_language_instruction(self):
        prompt = _TRAIN_CONVERTER.build_prompt(
            task_text="<TRANSCRIBE> <WAIT>",
            prompt_mode="fixed",
            fixed_prompt="",
            predict_language=False,
        )
        self.assertNotIn("language ", prompt.lower())
        self.assertIn("<turn_state><标签><asr_text>", prompt)

    def test_postprocess_strips_language_from_text_and_prompt(self):
        record = {
            "text": "language Chinese<turn_state><wait><asr_text>闭嘴",
            "prompt": "请转录音频内容，并严格使用格式：language 语种<turn_state><标签><asr_text>转写文本。",
            "easy_turn_lang_qwen": "Chinese",
        }
        updated, changed = _POSTPROCESS.transform_record(
            record,
            prompt_strategy="rewrite",
            fixed_prompt=_POSTPROCESS.DEFAULT_PROMPT_NO_LANGUAGE,
        )
        self.assertTrue(changed)
        self.assertEqual(updated["text"], "<turn_state><wait><asr_text>闭嘴")
        self.assertEqual(updated["predict_language"], 0)
        self.assertTrue(updated["easy_turn_lang_prediction_removed"])
        self.assertNotIn("language ", updated["prompt"].lower())

    def test_postprocess_plain_asr_text_falls_back_to_asr_marker(self):
        text, changed = _POSTPROCESS.strip_language_prefix("language Chinese<asr_text>你好")
        self.assertTrue(changed)
        self.assertEqual(text, "<asr_text>你好")


if __name__ == "__main__":
    unittest.main()
