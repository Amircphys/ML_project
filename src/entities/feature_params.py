from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FeatureParams:
    categorical_columns: List[str]
    numerical_columns: List[str]
    drop_columns: Optional[List[str]]
    target_column: str = field()
