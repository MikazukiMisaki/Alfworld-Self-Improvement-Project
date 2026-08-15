from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_result_provenance import audit_results


class ResultProvenanceAuditTests(unittest.TestCase):
    def test_content_overrides_indexed_looking_directory_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "indexed-looking-name"
            run.mkdir()
            manifest = {
                "run_id": "misleading-name",
                "model_version": "Qwen/Qwen3-8B",
                "resolved_config": {
                    "collection": {
                        "episodes": 1,
                        "model_config": "configs/model/qwen3_8b.yaml",
                    },
                    "environment": {"split": "valid_seen"},
                    "model": {
                        "enable_thinking": False,
                        "generation": {"max_new_tokens": 32},
                    },
                },
                "seed_schedule": [42],
            }
            (run / "run_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (run / "metrics.json").write_text(
                json.dumps({"success_rate": 0.0}), encoding="utf-8"
            )
            (run / "trajectory.jsonl").write_text(
                json.dumps(
                    {
                        "steps": [
                            {
                                "model_output": "Action: look",
                                "done": False,
                                "reward": 0.0,
                                "metadata": {},
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = audit_results(root)
        self.assertEqual(report["runs"][0]["pipeline_version"], "free_form_v1")
        self.assertEqual(
            report["runs"][0]["action_selection_mode"], "free_form_validated"
        )
