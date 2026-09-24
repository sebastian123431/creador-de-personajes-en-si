from dataclasses import dataclass, field
from typing import List
from models.frame import Frame


@dataclass
class Animation:
    """Representa una fila de animación de 4 frames (ej. walk_down)."""
    name: str
    frames: List[Frame] = field(default_factory=list)
    row_index: int = 0
