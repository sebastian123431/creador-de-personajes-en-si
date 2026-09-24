from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from models.body_part import BodyPart
from models.skeleton import Skeleton


@dataclass
class PoseFrame:
    """
    Representa el estado postural y anatómico completo de un frame:
    esqueleto articular (18 anchors), partes corporales segmentadas,
    baseline de apoyo, centro de masa y dirección de vista.
    """
    character_id: str
    variant: str
    animation: str
    frame_index: int
    skeleton: Skeleton = field(default_factory=Skeleton)
    body_parts: Dict[str, BodyPart] = field(default_factory=dict)
    baseline: int = 0
    center: Tuple[int, int] = (0, 0)
    orientation: str = "down"  # "down", "up", "left", "right"
    estimated_direction: bool = False  # True si no hay vista de esa dirección en la referencia
    source_image_path: Optional[Path] = None

    def to_dict(self) -> dict:
        return {
            "character_id": self.character_id,
            "variant": self.variant,
            "animation": self.animation,
            "frame_index": self.frame_index,
            "skeleton": self.skeleton.to_dict(),
            "body_parts": {k: bp.to_dict() for k, bp in self.body_parts.items()},
            "baseline": self.baseline,
            "center": list(self.center),
            "orientation": self.orientation,
            "estimated_direction": self.estimated_direction,
            "source_image_path": str(self.source_image_path).replace("\\", "/") if self.source_image_path else None,
        }
