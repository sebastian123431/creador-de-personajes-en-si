from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Anchor:
    name: str
    x: int
    y: int
    confidence: float = 1.0
    is_manual: bool = False

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "confidence": round(self.confidence, 3),
            "is_manual": self.is_manual,
        }


# Definición oficial de conexiones de huesos para rendering y cinemática
SKELETON_BONES = [
    ("head", "neck"),
    ("neck", "chest"),
    ("chest", "hip"),
    # Brazo izquierdo
    ("chest", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_hand"),
    # Brazo derecho
    ("chest", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_hand"),
    # Pierna izquierda
    ("hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot"),
    # Pierna derecha
    ("hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot"),
]

OFFICIAL_ANCHOR_NAMES = [
    "head",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
    "chest",
    "hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_foot",
    "right_foot",
]


@dataclass
class Skeleton:
    """
    Estructura articular de 18 anchors adaptada a Pixel Art para Fase 2.
    """
    anchors: Dict[str, Anchor] = field(default_factory=dict)

    def get_anchor(self, name: str) -> Optional[Anchor]:
        return self.anchors.get(name)

    def set_anchor(self, name: str, x: int, y: int, confidence: float = 1.0, is_manual: bool = False):
        self.anchors[name] = Anchor(name=name, x=x, y=y, confidence=confidence, is_manual=is_manual)

    @property
    def anchor_count(self) -> int:
        return len(self.anchors)

    @property
    def average_confidence(self) -> float:
        if not self.anchors:
            return 0.0
        return sum(a.confidence for a in self.anchors.values()) / len(self.anchors)

    @property
    def low_confidence_anchors(self) -> List[str]:
        """Retorna anchors con confidence < 0.60 para marcar 'REVIEW REQUIRED'."""
        return [name for name, a in self.anchors.items() if a.confidence < 0.60]

    def to_dict(self) -> Dict[str, dict]:
        return {name: a.to_dict() for name, a in self.anchors.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "Skeleton":
        skel = cls()
        for name, vals in data.items():
            if isinstance(vals, dict):
                skel.set_anchor(
                    name=name,
                    x=int(vals.get("x", 0)),
                    y=int(vals.get("y", 0)),
                    confidence=float(vals.get("confidence", 1.0)),
                    is_manual=bool(vals.get("is_manual", False)),
                )
        return skel
