"""Tests for the --supervoxels CLI flag and LABEL_TYPE .env resolution."""

import pytest

from scripts.export import resolve_label_type


class TestResolveLabelType:
    """Test resolve_label_type(supervoxels_flag, env)."""

    def test_default_is_labels(self):
        assert resolve_label_type(False, {}) == "labels"

    def test_flag_returns_supervoxels(self):
        assert resolve_label_type(True, {}) == "supervoxels"

    def test_flag_overrides_env(self):
        assert resolve_label_type(True, {"LABEL_TYPE": "labels"}) == "supervoxels"

    def test_env_supervoxels(self):
        assert resolve_label_type(False, {"LABEL_TYPE": "supervoxels"}) == "supervoxels"

    def test_env_labels(self):
        assert resolve_label_type(False, {"LABEL_TYPE": "labels"}) == "labels"
