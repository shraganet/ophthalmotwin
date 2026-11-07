from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Cohort:
    n_patients: int = 480
    n_genes: int = 63
    n_variants: int = 320
    n_inheritance: int = 3
    max_visits: int = 5
    img_channels: int = 1
    img_size: int = 32
    latent_dim: int = 24
    val_fraction: float = 0.2
    cal_fraction: float = 0.2
    rare_gene_cutoff: int = 50
    seed: int = 1234


@dataclass(frozen=True)
class Panel:
    img_embed_dim: int = 1024
    gen_embed_dim: int = 256
    twin_dim: int = 1024
    node_dim: int = 64
    graph_hidden: int = 256
    hgt_layers: int = 3
    hgt_heads: int = 4
    fusion_heads: int = 8
    tpe_freqs: int = 8
    tpe_width: int = 256
    n_genes: int = 63
    n_quantiles: int = 3
    dropout: float = 0.1
    vcep_weight: float = 1.0
    vcep_only_weight: float = 0.5
    use_longitudinal: bool = True
    use_genotype: bool = True
    fusion_mode: str = "cross"


@dataclass(frozen=True)
class Firing:
    lr: float = 1e-4
    epochs: int = 5
    batch_size: int = 8
    weight_decay: float = 1e-4
    warmup_steps: int = 50
    max_steps: int = 0
    amp: bool = False
    grad_clip: float = 1.0
    lambda_prog: float = 0.5
    lambda_early: float = 0.3
    lambda_reg: float = 1e-4
    alpha: float = 0.10
    raps_lambda: float = 0.05
    raps_kreg: int = 5
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9)
    device: str = "auto"
    out_dir: str = "runs/main"
    log_every: int = 10
    seed: int = 1234


@dataclass(frozen=True)
class Run:
    name: str = "main"
    cohort: Cohort = field(default_factory=Cohort)
    panel: Panel = field(default_factory=Panel)
    firing: Firing = field(default_factory=Firing)


_SECTIONS: dict[str, type[Any]] = {"cohort": Cohort, "panel": Panel, "firing": Firing}


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config {path} must be a mapping")
    return loaded


def _resolve_chain(path: Path) -> dict[str, Any]:
    raw = _read_yaml(path)
    parents = raw.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    merged: dict[str, Any] = {}
    for parent in parents:
        merged = _deep_merge(merged, _resolve_chain((path.parent / parent).resolve()))
    return _deep_merge(merged, raw)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(target: type[Any], values: dict[str, Any]) -> Any:
    fields = {f.name: f for f in dataclasses.fields(target)}
    unknown = set(values) - set(fields)
    if unknown:
        raise ValueError(f"{target.__name__} got unknown keys {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, raw in values.items():
        annotation = fields[name].type
        if annotation in (tuple, "tuple[float, ...]") or name == "quantile_levels":
            kwargs[name] = tuple(float(x) for x in raw)
        else:
            kwargs[name] = raw
    return target(**kwargs)


def structure(mapping: dict[str, Any]) -> Run:
    payload = dict(mapping)
    name = str(payload.pop("name", "main"))
    sections: dict[str, Any] = {}
    for key, target in _SECTIONS.items():
        sections[key] = _coerce(target, dict(payload.pop(key, {})))
    if payload:
        raise ValueError(f"unexpected top-level config keys {sorted(payload)}")
    return Run(name=name, **sections)


def apply_overrides(mapping: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    out = mapping
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must be key=value")
        dotted, value = item.split("=", 1)
        out = _deep_merge(out, _explode(dotted.strip(), _parse_scalar(value.strip())))
    return out


def _explode(dotted: str, value: Any) -> dict[str, Any]:
    parts = dotted.split(".")
    nested: dict[str, Any] = {parts[-1]: value}
    for part in reversed(parts[:-1]):
        nested = {part: nested}
    return nested


def _parse_scalar(text: str) -> Any:
    parsed = yaml.safe_load(text)
    return parsed


def load_run(config_path: str | Path, overrides: list[str] | None = None) -> Run:
    mapping = _resolve_chain(Path(config_path).resolve())
    if overrides:
        mapping = apply_overrides(mapping, overrides)
    return structure(mapping)


def to_mapping(run: Run) -> dict[str, Any]:
    return {
        "name": run.name,
        "cohort": dataclasses.asdict(run.cohort),
        "panel": dataclasses.asdict(run.panel),
        "firing": dataclasses.asdict(run.firing),
    }
