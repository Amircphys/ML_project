from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LogisticRegressionParams:
    random_state: int = field(default=42)


@dataclass
class RandomForestParams:
    random_state: int = field(default=42)
    n_estimators: Optional[int] = field(default=None)
    max_depth: Optional[int] = field(default=None)


@dataclass
class CatBoostParams:
    iterations: int = field(default=1000)
    learning_rate: float = field(default=0.1)
    depth: int = field(default=6)


@dataclass
class TrainingParams:
    LogisticRegressionArgs: LogisticRegressionParams
    RandomForestArgs: RandomForestParams
    CatBoostArgs: CatBoostParams
    model_type: str = field(default="RandomForest")
