from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class Frame:
    """Representa un frame individual de animación normalizado o extraído."""
    image_path: Path
    index: int
    bbox: Tuple[int, int, int, int]  # (min_x, min_y, max_x, max_y) o (x, y, w, h)
    center_x: int
    baseline_y: int
    width: int
    height: int
    alpha_area: int = 0
