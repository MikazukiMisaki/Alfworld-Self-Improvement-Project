from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trajectory.provenance import (
    ProvenanceError,
    ProvenanceRequirement,
    discover_run_directories,
    load_run_trajectories,
    validate_manifest_provenance,
)


class ProvenanceTests(unittest.TestCase):
    def test_explicit_consistent_manifest_passes(self) -> None:
        validate_manifest_provenance(self._manifest())

    def test_missing_pipeline_version_fails_closed(self) -> None:
        manifest = self._manifest()
        del manifest["pipeline_version"]
        with self.assertRaisesRegex(ProvenanceError, "pipeline_version"):
            validate_manifest_provenance(manifest)

    def test_inconsistent_mode_and_pipeline_fails_closed(self) -> None:
        manifest = self._manifest()
        manifest["pipeline_version"] = "legacy_v1"
        with self.assertRaisesRegex(ProvenanceError, "does not match"):
            validate_manifest_provenance(manifest)

    def test_loader_prevents_mixed_pipeline_consumption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._write_run(Path(directory), self._manifest())
            indexed_only = ProvenanceRequirement.one(
                pipeline_version="indexed_v1",
                action_selection_mode="indexed_admissible",
                split="valid_seen",
            )
            with self.assertRaisesRegex(ProvenanceError, "not allowed"):
                load_run_trajectories(run, requirement=indexed_only)

    def test_discovery_returns_only_explicit_matching_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            accepted = self._write_run(root / "accepted", self._manifest())
            historical = self._manifest()
            del historical["pipeline_version"]
            self._write_run(root / "historical", historical)
            requirement = ProvenanceRequirement.one(
                pipeline_version="free_form_v1",
                action_selection_mode="free_form_validated",
                split="valid_seen",
            )
            self.assertEqual(
                discover_run_directories((root,), requirement=requirement),
                (accepted,),
            )

    @staticmethod
    def _manifest() -> dict:
        return {
            "pipeline_version": "free_form_v1",
            "action_selection_mode": "free_form_validated",
            "split": "valid_seen",
            "resolved_config": {
                "model": {
                    "pipeline_version": "free_form_v1",
                    "action_selection": {"mode": "free_form_validated"},
                },
                "environment": {"split": "valid_seen"},
            },
        }

    @staticmethod
    def _write_run(root: Path, manifest: dict) -> Path:
        run = root / "run"
        run.mkdir(parents=True)
        (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run / "trajectory.jsonl").write_text(
            json.dumps({"trajectory_id": "synthetic"}) + "\n", encoding="utf-8"
        )
        return run
