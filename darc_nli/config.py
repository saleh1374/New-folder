"""Configuration objects and experiment presets (DARC-NLI v1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

#: Folder name used on disk for every track.
TRACK_NAMES: dict[int, str] = {
    1: "english",
    2: "spanish",
    3: "basque",
    4: "mixed",
}

TRACK_LANGUAGES: dict[int, str] = {
    1: "English (monolingual)",
    2: "Spanish (monolingual)",
    3: "Basque (monolingual)",
    4: "Mixed multilingual (EN/ES/EU)",
}


@dataclass
class LossConfig:
    """Weights of the three objectives: L = CE + l_scl * SupCon + l_dc * DC."""

    lambda_supcon: float = 0.0
    lambda_dc: float = 0.0
    temperature: float = 0.1
    ce_class_weights: bool = False

    @property
    def name(self) -> str:
        if self.lambda_supcon > 0 and self.lambda_dc > 0:
            return "ce_supcon_dc"
        if self.lambda_supcon > 0:
            return "ce_supcon"
        if self.lambda_dc > 0:
            return "ce_dc"
        return "baseline"


#: The four mandatory ablations from the technical document.
#:
#: Weights were fixed on Track-1 dev only (never on test) by a small sweep
#: documented in ``docs/hyperparameter_sweep.md``: SupCon 0.1 -> 0.665,
#: 0.05 -> 0.700, 0.03 -> 0.700; DC 0.1 -> 0.683, 0.05 -> 0.687.
EXPERIMENT_PRESETS: dict[str, LossConfig] = {
    "baseline": LossConfig(lambda_supcon=0.0, lambda_dc=0.0),
    "ce_supcon": LossConfig(lambda_supcon=0.05, lambda_dc=0.0),
    "ce_dc": LossConfig(lambda_supcon=0.0, lambda_dc=0.05),
    "ce_supcon_dc": LossConfig(lambda_supcon=0.05, lambda_dc=0.05),
}

EXPERIMENT_NAMES: tuple[str, ...] = tuple(EXPERIMENT_PRESETS)


@dataclass
class ExperimentConfig:
    """Everything needed to reproduce one run."""

    track: int
    experiment: str = "baseline"
    model_name: str = "xlm-roberta-base"
    data_root: Path = Path("data/repo")

    output_dir: Path = Path("outputs")
    seed: int = 42

    # optimisation
    epochs: int = 5  # the document allows 3-5; 5 was still improving on dev
    batch_size: int = 16
    grad_accum: int = 1
    lr: float = 2e-5
    #: 0 -> same as ``lr``. A larger head lr is standard for small datasets.
    head_lr: float = 0.0
    #: ``linear`` (document default) or ``cosine``.
    scheduler: str = "linear"
    #: CE label smoothing; 0 disables it.
    label_smoothing: float = 0.0
    #: Layer-wise lr decay (e.g. 0.95); 0 disables it.
    layerwise_lr_decay: float = 0.0
    #: Freeze the embeddings and the first N encoder layers.
    freeze_layers: int = 0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    dropout: float = 0.1
    max_length: int = 0  # 0 -> use the audit recommendation
    fp16: bool = True

    # sampler
    units_per_class_per_batch: int = 2
    class_balanced: bool = True

    # projection head
    projection_dim: int = 256

    # model selection / evaluation
    select_metric: str = "weighted_f1"  # weighted_f1 | hard_cons | soft_cons
    #: Award each reversible pair the average of both directions before the
    #: argmax (v2 experiment, off by default because the document forbids
    #: constrained decoding in v1).
    pair_consistency: bool = False

    # smoke testing
    max_train_steps: int = 0
    max_train_samples: int = 0
    eval_batch_size: int = 64
    num_workers: int = 2
    log_every: int = 50

    loss: LossConfig = field(default_factory=LossConfig)

    def __post_init__(self) -> None:
        if self.experiment not in EXPERIMENT_PRESETS:
            raise ValueError(f"unknown experiment {self.experiment!r}")
        # Presets are copied so callers can override temperature afterwards.
        preset = EXPERIMENT_PRESETS[self.experiment]
        self.loss = LossConfig(
            lambda_supcon=preset.lambda_supcon,
            lambda_dc=preset.lambda_dc,
            temperature=preset.temperature,
            ce_class_weights=preset.ce_class_weights,
        )
        self.data_root = Path(self.data_root)
        self.output_dir = Path(self.output_dir)

    @property
    def track_name(self) -> str:
        return TRACK_NAMES[self.track]

    @property
    def run_name(self) -> str:
        return self.loss.name

    @property
    def run_dir(self) -> Path:
        return self.output_dir / "checkpoints" / self.track_name / self.run_name

    @property
    def results_dir(self) -> Path:
        return self.output_dir / "results" / self.track_name / self.run_name

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["data_root"] = str(self.data_root)
        payload["output_dir"] = str(self.output_dir)
        return payload
