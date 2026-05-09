"""Configuration management via dataclasses with YAML serialization.

Provides structured configs for data, models, training, and the full experiment.
All fields carry defaults matching the IBM AML LI-Small / HI-Small setup.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    """Paths and parameters for the IBM AML dataset."""

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    dataset_variant: str = "LI-Small"  # LI-Small, HI-Small, LI-Medium, HI-Medium

    # Graph construction -------------------------------------------------
    num_time_snapshots: int = 12  # number of temporal snapshots for EvolveGCN
    snapshot_strategy: str = "quantile"  # "quantile" (equal edges/window) or "fixed" (equal time/window)
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Feature engineering ------------------------------------------------
    log_transform_amount: bool = True
    encode_cyclic_time: bool = True  # sin/cos encoding for hour, day-of-week

    # Edge / node feature dimensions (set after construction) ------------
    node_feat_dim: int = 0
    edge_feat_dim: int = 0

    def resolve_raw_dir(self) -> Path:
        """Return resolved absolute path to the raw data directory."""
        p = Path(self.raw_dir)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p

    def resolve_processed_dir(self) -> Path:
        p = Path(self.processed_dir)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p


@dataclass
class ModelConfig:
    """Architecture-specific hyperparameters."""

    # Shared --------------------------------------------------------------
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    learning_rate: float = 0.001
    weight_decay: float = 5e-4

    # GAT -----------------------------------------------------------------
    gat_heads: int = 4

    # GraphSAGE -----------------------------------------------------------
    sage_aggregator: str = "mean"  # mean, max, lstm, pool

    # EvolveGCN -----------------------------------------------------------
    evolvegcn_variant: str = "EvolveGCN-H"  # EvolveGCN-H (GRU on weights) or EvolveGCN-O (LSTM)
    evolvegcn_hidden_dim: int = 64  # hidden dim of the RNN that evolves weights


@dataclass
class TrainingConfig:
    """Training hyperparameters shared across models."""

    num_epochs: int = 200
    batch_size: int = 1024  # edge-level batch size for mini-batch training
    early_stopping_patience: int = 25

    # Class imbalance ----------------------------------------------------
    pos_weight: Optional[float] = None  # if None, auto-computed from training set

    # Optimizer ----------------------------------------------------------
    optimizer: str = "adam"  # adam, adamw
    lr_scheduler: str = "reduce_on_plateau"  # reduce_on_plateau, cosine, none
    lr_scheduler_patience: int = 10
    lr_scheduler_factor: float = 0.5

    # Hardware -----------------------------------------------------------
    device: str = "cuda"  # cuda, cpu
    num_workers: int = 0


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration aggregating all sub-configs."""

    seed: int = 42
    experiment_name: str = "aml-gnn-baseline"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Logging ------------------------------------------------------------
    use_wandb: bool = False
    wandb_project: str = "aml-gnn-deep"
    log_interval: int = 10  # log metrics every N epochs

    def to_yaml(self, path: Path) -> None:
        """Serialize config to a YAML file, converting nested dataclasses."""
        d = _dataclass_to_dict(self)
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """Deserialize config from a YAML file."""
        with open(path) as f:
            d = yaml.safe_load(f)
        return _dict_to_dataclass(cls, d)


def _dataclass_to_dict(obj) -> dict:
    """Recursively convert a dataclass instance (or nested structure) to a plain dict."""
    from dataclasses import asdict, is_dataclass

    if is_dataclass(obj):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


def _dict_to_dataclass(cls, d: dict):
    """Recursively instantiate a nested dataclass from a plain dict."""
    from dataclasses import fields, is_dataclass

    kwargs = {}
    for fld in fields(cls):
        if fld.name not in d:
            continue
        val = d[fld.name]
        if is_dataclass(fld.type):
            kwargs[fld.name] = _dict_to_dataclass(fld.type, val)
        else:
            kwargs[fld.name] = val
    return cls(**kwargs)
