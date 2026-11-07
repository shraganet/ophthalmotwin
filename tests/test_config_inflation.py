from __future__ import annotations

from pathlib import Path

import pytest

from ophthalmotwin_gp.cartoon import Run, load_run, structure, to_mapping

ROOT = Path(__file__).resolve().parents[1]
CARTOONS = sorted((ROOT / "cartoons" / "panel").glob("*.yaml"))


@pytest.mark.parametrize("path", CARTOONS, ids=lambda p: p.name)
def test_cartoon_loads_and_is_consistent(path: Path) -> None:
    run = load_run(path)
    assert isinstance(run, Run)
    assert run.cohort.n_genes == run.panel.n_genes
    assert len(run.firing.quantile_levels) == run.panel.n_quantiles
    assert run.panel.fusion_mode in {"cross", "concat"}
    assert run.panel.twin_dim % run.panel.fusion_heads == 0


def test_roundtrip_mapping() -> None:
    run = load_run(ROOT / "cartoons" / "panel" / "main.yaml")
    restored = structure(to_mapping(run))
    assert restored == run


def test_override_applies() -> None:
    run = load_run(CARTOONS[0], ["firing.lr=0.5", "cohort.n_patients=33"])
    assert run.firing.lr == 0.5
    assert run.cohort.n_patients == 33
