from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class DatasetIndex:
    """Índice maestro persistible en dataset/dataset_index.json"""
    version: str = "1.0.0"
    updated_at: str = ""
    characters: Dict[str, Any] = field(default_factory=dict)
