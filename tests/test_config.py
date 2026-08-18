"""Tests for the medknow configuration system."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medknow.config import PROJECT_ROOT, get, load_config


def test_default_config_has_required_sections():
    cfg = load_config(None)
    for section in (
        "paths",
        "data",
        "model",
        "training",
        "uncertainty",
        "calibration",
        "referral",
        "evaluation",
        "api",
    ):
        assert section in cfg, f"missing config section: {section}"


def test_defaults_match_manuscript_settings():
    cfg = load_config(None)
    assert get(cfg, "model.name") == "resnet18"
    assert get(cfg, "model.freeze_backbone") is True
    assert get(cfg, "uncertainty.mc_samples") == 30
    assert get(cfg, "data.internal.split.train") == 0.70
    assert get(cfg, "data.internal.seed") == 42
    assert get(cfg, "calibration.ece_bins") == 15
    assert get(cfg, "referral.decision_threshold") == 0.5
    assert 0.10 in get(cfg, "referral.rates")
    assert 0.25 in get(cfg, "referral.rates")
    assert get(cfg, "api.max_mc_samples") == 50


def test_relative_paths_resolved_against_project_root():
    cfg = load_config(None)
    out = Path(get(cfg, "paths.output_dir"))
    assert out.is_absolute()
    assert PROJECT_ROOT in out.parents
    ext = Path(get(cfg, "data.external.rsna.root"))
    assert ext.is_absolute()


def test_config_merge_from_yaml(tmp_path):
    f = tmp_path / "custom.yaml"
    f.write_text(
        "model:\n  name: resnet50\ntraining:\n  epochs: 5\n",
        encoding="utf-8",
    )
    cfg = load_config(str(f))
    assert get(cfg, "model.name") == "resnet50"
    assert get(cfg, "training.epochs") == 5
    # untouched defaults are preserved
    assert get(cfg, "uncertainty.mc_samples") == 30


def test_repo_configs_load():
    # The three shipped configs must load and merge cleanly.
    for name in ("baseline.yaml", "mc_dropout.yaml", "temperature_scaling.yaml"):
        cfg = load_config(str(PROJECT_ROOT / "configs" / name))
        assert get(cfg, "model.name") == "resnet18"
        assert get(cfg, "uncertainty.mc_samples") == 30


def test_checkpoint_reproducibility_defaults():
    cfg = load_config(None)
    assert get(cfg, "evaluation.bootstrap_iters") == 2000
    assert get(cfg, "evaluation.bootstrap_seed") == 42


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")


def test_get_returns_default_for_missing_key():
    cfg = load_config(None)
    assert get(cfg, "nope.missing", "fallback") == "fallback"
