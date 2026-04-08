# coding=utf-8
import importlib.util
import unittest
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _ROOT / "finetuning" / "rewrite_qwen_asr_jsonl_audio_paths.py"
_SPEC = importlib.util.spec_from_file_location("rewrite_qwen_asr_jsonl_audio_paths", _SCRIPT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


class RewriteQwenAsrJsonlAudioPathsTest(unittest.TestCase):
    def test_rewrite_posix_prefix(self):
        new_path, changed = _MODULE.rewrite_audio_path(
            "/data/liuke_data/datasets/asr_datasets/Easy_turn/Easy-Turn-Testset/testset/complete/real/a.wav",
            "/data/liuke_data/datasets/asr_datasets/Easy_turn/Easy-Turn-Testset/testset",
            "/mnt/newdisk/Easy-Turn-Testset/testset",
        )
        self.assertTrue(changed)
        self.assertEqual(new_path, "/mnt/newdisk/Easy-Turn-Testset/testset/complete/real/a.wav")

    def test_rewrite_windows_prefix(self):
        new_path, changed = _MODULE.rewrite_audio_path(
            "D:\\datasets\\easy_turn\\audio\\a.wav",
            "D:\\datasets\\easy_turn",
            "E:\\mirror\\easy_turn",
        )
        self.assertTrue(changed)
        self.assertEqual(new_path, "E:\\mirror\\easy_turn\\audio\\a.wav")

    def test_keep_unmatched_path(self):
        new_path, changed = _MODULE.rewrite_audio_path(
            "/some/other/root/a.wav",
            "/data/old",
            "/data/new",
        )
        self.assertFalse(changed)
        self.assertEqual(new_path, "/some/other/root/a.wav")


if __name__ == "__main__":
    unittest.main()
