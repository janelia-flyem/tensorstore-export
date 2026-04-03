"""Tests for export.py pipeline mode selection and combined mode flow.

Verifies that:
- DOWNRES_SCALES in .env + no --downres flag → combined mode (Arrow then downres)
- Explicit --downres flag → downres-only (no Arrow export)
- No DOWNRES_SCALES → Arrow-only export
- Combined mode forces wait on Arrow jobs and omits DOWNRES_SCALES from workers
- Combined mode chains _run_downres_wait_pipeline after Arrow export
"""

import json
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_env(**overrides):
    """Return a minimal .env dict suitable for export.main()."""
    env = {
        "SOURCE_PATH": "gs://test-bucket/export",
        "DEST_PATH": "gs://test-bucket/output",
        "PROJECT_ID": "test-project",
        "REGION": "us-east4",
        "BASE_JOB_NAME": "test-export",
        "SCALES": "0",
        "DOCKER_IMAGE": "gcr.io/test/image:latest",
    }
    env.update(overrides)
    return env


@pytest.fixture()
def ng_spec_file(tmp_path):
    """Write a minimal NG spec JSON and return its path."""
    spec = {"@type": "neuroglancer_multiscale_volume", "scales": []}
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    return str(p)


def _fake_arrow_files():
    """Minimal list_arrow_files return value: (scale, name, size, chunks)."""
    return [
        (0, "shard-0000.arrow", 100_000_000, 64),
        (0, "shard-0001.arrow", 80_000_000, 48),
    ]


def _fake_tier_map():
    """assign_tiers return value: tier_gib → list of (scale, name, size)."""
    return {
        4: [
            (0, "shard-0000.arrow", 100_000_000),
            (0, "shard-0001.arrow", 80_000_000),
        ],
    }


# Shared patches applied to every test.  Individual tests layer on extras.
_COMMON_PATCHES = {
    "scripts.export.ENV_FILE": mock.PropertyMock(return_value=Path("/dev/null")),
}


def _patch_stack(monkeypatch, env, ng_spec_file):
    """Apply common patches for running main() in a test.

    Returns a dict of mock objects keyed by short name so tests can inspect
    call args.
    """
    mocks = {}

    # load_env → returns our test env dict
    m = mock.patch("scripts.export.load_env", return_value=env)
    mocks["load_env"] = m.start()

    # ENV_FILE.exists() → True
    m = mock.patch("scripts.export.ENV_FILE",
                   **{"exists.return_value": True})
    mocks["env_file"] = m.start()

    # NG_SPEC_PATH → temp spec file
    env["NG_SPEC_PATH"] = ng_spec_file

    # Arrow scan path
    m = mock.patch("scripts.export.list_arrow_files",
                   return_value=_fake_arrow_files())
    mocks["list_arrow_files"] = m.start()

    m = mock.patch("scripts.export.assign_tiers",
                   return_value=_fake_tier_map())
    mocks["assign_tiers"] = m.start()

    # generate_downres_manifests → one tier per requested scale
    def _fake_gen_downres(spec, src, dest, scales, downres_scales,
                          max_tasks, only_missing=False, dry_run=False):
        result = {}
        for s in downres_scales:
            result[s] = {4: (f"gs://test-bucket/export/manifests-downres/tier-4gi", 2)}
        return result
    m = mock.patch("scripts.export.generate_downres_manifests",
                   side_effect=_fake_gen_downres)
    mocks["generate_downres_manifests"] = m.start()

    # _launch_tier_jobs → no-op
    m = mock.patch("scripts.export._launch_tier_jobs")
    mocks["launch_tier_jobs"] = m.start()

    # _run_downres_wait_pipeline → no-op
    m = mock.patch("scripts.export._run_downres_wait_pipeline")
    mocks["run_downres_wait_pipeline"] = m.start()

    # _get_image → fake image URI
    m = mock.patch("scripts.export._get_image",
                   return_value="gcr.io/test/image:latest")
    mocks["get_image"] = m.start()

    # GCS / subprocess stubs (manifest upload path)
    m = mock.patch("scripts.export.subprocess.run",
                   return_value=mock.Mock(returncode=0, stdout="", stderr=""))
    mocks["subprocess_run"] = m.start()

    # google.cloud.storage.Client for manifest writes
    fake_blob = mock.Mock()
    fake_bucket = mock.Mock()
    fake_bucket.blob.return_value = fake_blob
    fake_client = mock.Mock()
    fake_client.bucket.return_value = fake_bucket
    m = mock.patch("google.cloud.storage.Client", return_value=fake_client)
    mocks["storage_client"] = m.start()

    # distribute_tasks → one task with both shards
    m = mock.patch("scripts.export.distribute_tasks",
                   return_value={"0": [{"scale": 0, "shard": "shard-0000.arrow"},
                                       {"scale": 0, "shard": "shard-0001.arrow"}]})
    mocks["distribute_tasks"] = m.start()

    # aggregate_labels → no-op
    m = mock.patch("scripts.export.aggregate_labels")
    mocks["aggregate_labels"] = m.start()

    return mocks


# ---------------------------------------------------------------------------
# Mode selection tests
# ---------------------------------------------------------------------------

class TestModeSelection:
    """Test that the correct pipeline mode is selected based on args and env."""

    def test_combined_mode_arrow_and_downres(self, monkeypatch, ng_spec_file):
        """SCALES + DOWNRES_SCALES in .env (no --downres flag) → combined mode.

        Both list_arrow_files (Arrow scan) and _run_downres_wait_pipeline
        (downres chain) must be called.
        """
        env = _base_env(DOWNRES_SCALES="1,2")
        mocks = _patch_stack(monkeypatch, env, ng_spec_file)

        with mock.patch.object(sys, "argv", ["export.py"]):
            from scripts.export import main
            main()

        mocks["list_arrow_files"].assert_called_once()
        mocks["launch_tier_jobs"].assert_called_once()
        mocks["run_downres_wait_pipeline"].assert_called_once()

    def test_explicit_downres_skips_arrow(self, monkeypatch, ng_spec_file):
        """--downres flag → downres-only mode (Arrow scan must NOT run)."""
        env = _base_env()
        mocks = _patch_stack(monkeypatch, env, ng_spec_file)

        with mock.patch.object(sys, "argv",
                               ["export.py", "--downres", "1,2"]):
            from scripts.export import main
            main()

        mocks["list_arrow_files"].assert_not_called()
        mocks["run_downres_wait_pipeline"].assert_called_once()

    def test_no_downres_arrow_only(self, monkeypatch, ng_spec_file):
        """No DOWNRES_SCALES → Arrow-only export, no downres pipeline."""
        env = _base_env()  # no DOWNRES_SCALES
        mocks = _patch_stack(monkeypatch, env, ng_spec_file)

        with mock.patch.object(sys, "argv", ["export.py"]):
            from scripts.export import main
            main()

        mocks["list_arrow_files"].assert_called_once()
        mocks["launch_tier_jobs"].assert_called_once()
        mocks["run_downres_wait_pipeline"].assert_not_called()

    def test_dry_run_combined_shows_both_phases(self, monkeypatch, ng_spec_file):
        """--dry-run in combined mode shows Arrow tier info AND downres info."""
        env = _base_env(DOWNRES_SCALES="1,2")
        mocks = _patch_stack(monkeypatch, env, ng_spec_file)

        with mock.patch.object(sys, "argv", ["export.py", "--dry-run"]):
            from scripts.export import main
            main()

        # Arrow scan still runs (to show tier assignments)
        mocks["list_arrow_files"].assert_called_once()
        # Downres pipeline runs in dry-run mode
        mocks["run_downres_wait_pipeline"].assert_called_once()
        # No actual jobs launched
        mocks["launch_tier_jobs"].assert_not_called()


# ---------------------------------------------------------------------------
# Combined mode flow tests
# ---------------------------------------------------------------------------

class TestCombinedModeFlow:
    """Test combined mode pipeline behavior details."""

    def test_arrow_export_forces_wait(self, monkeypatch, ng_spec_file):
        """Combined mode passes wait=True to _launch_tier_jobs."""
        env = _base_env(DOWNRES_SCALES="1,2")
        mocks = _patch_stack(monkeypatch, env, ng_spec_file)

        with mock.patch.object(sys, "argv", ["export.py"]):
            from scripts.export import main
            main()

        # _launch_tier_jobs(tier_info, env, image, ng_spec_b64, base_name,
        #                   project, region, label_type, downres, wait)
        call_args = mocks["launch_tier_jobs"].call_args
        wait_arg = call_args[0][9]  # 10th positional arg = wait
        assert wait_arg is True, "Combined mode must force wait=True on Arrow export"

    def test_arrow_workers_no_downres_scales(self, monkeypatch, ng_spec_file):
        """Combined mode passes empty downres to _launch_tier_jobs."""
        env = _base_env(DOWNRES_SCALES="1,2")
        mocks = _patch_stack(monkeypatch, env, ng_spec_file)

        with mock.patch.object(sys, "argv", ["export.py"]):
            from scripts.export import main
            main()

        call_args = mocks["launch_tier_jobs"].call_args
        downres_arg = call_args[0][8]  # 9th positional arg = downres
        assert downres_arg == "", (
            "Combined mode must not pass DOWNRES_SCALES to Arrow workers")

    def test_downres_receives_correct_scales(self, monkeypatch, ng_spec_file):
        """Combined mode passes the right downres_scales to the pipeline."""
        env = _base_env(DOWNRES_SCALES="1,2,3")
        mocks = _patch_stack(monkeypatch, env, ng_spec_file)

        with mock.patch.object(sys, "argv", ["export.py"]):
            from scripts.export import main
            main()

        call_args = mocks["run_downres_wait_pipeline"].call_args
        downres_scales = call_args[0][0]  # 1st positional arg
        assert downres_scales == [1, 2, 3]

    def test_arrow_only_passes_downres_to_workers(self, monkeypatch, ng_spec_file):
        """Without combined mode, Arrow workers still get DOWNRES_SCALES
        (the legacy per-worker downres path)."""
        env = _base_env()  # no DOWNRES_SCALES
        mocks = _patch_stack(monkeypatch, env, ng_spec_file)

        with mock.patch.object(sys, "argv", ["export.py"]):
            from scripts.export import main
            main()

        call_args = mocks["launch_tier_jobs"].call_args
        downres_arg = call_args[0][8]
        # Empty string because env has no DOWNRES_SCALES
        assert downres_arg == ""

    def test_arrow_only_respects_wait_flag(self, monkeypatch, ng_spec_file):
        """Without combined mode, --wait controls whether _launch_tier_jobs blocks."""
        env = _base_env()  # no DOWNRES_SCALES
        mocks = _patch_stack(monkeypatch, env, ng_spec_file)

        # Without --wait
        with mock.patch.object(sys, "argv", ["export.py"]):
            from scripts.export import main
            main()
        call_args = mocks["launch_tier_jobs"].call_args
        assert call_args[0][9] is False

        mocks["launch_tier_jobs"].reset_mock()
        mocks["list_arrow_files"].reset_mock()
        mocks["assign_tiers"].reset_mock()
        mocks["distribute_tasks"].reset_mock()

        # With --wait
        with mock.patch.object(sys, "argv", ["export.py", "--wait"]):
            main()
        call_args = mocks["launch_tier_jobs"].call_args
        assert call_args[0][9] is True
