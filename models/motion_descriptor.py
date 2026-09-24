import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.guard import SourceDatasetGuard

logger = logging.getLogger("SpriteStudio.MotionDescriptor")


@dataclass
class MotionDescriptor:
    """
    Descriptor cinemático por animación/personaje/variante.
    Almacena exclusivamente datos geométricos de movimiento (traslaciones, razones relativas,
    ángulos, anchors y confianza) sin imágenes, píxeles ni base64.
    """
    character_id: str
    variant: str
    animation: str
    source_path: str
    source_sha256: str
    frame_count: int = 4
    orientation: str = "down"
    character_width: float = 32.0
    character_height: float = 64.0
    root_motion: List[Dict[str, float]] = field(default_factory=list)
    anchors: List[Dict[str, Any]] = field(default_factory=list)
    parts: List[Dict[str, Dict[str, float]]] = field(default_factory=list)
    translations_px: List[Dict[str, Dict[str, float]]] = field(default_factory=list)
    translations_ratio: List[Dict[str, Dict[str, float]]] = field(default_factory=list)
    angles: List[Dict[str, float]] = field(default_factory=list)
    confidence: float = 1.0
    review_required: bool = False
    pose_analyzer_version: str = "2.2"
    template_version: str = "2.2"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "character_id": self.character_id,
            "variant": self.variant,
            "animation": self.animation,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "frame_count": self.frame_count,
            "orientation": self.orientation,
            "character_width": round(self.character_width, 2),
            "character_height": round(self.character_height, 2),
            "root_motion": self.root_motion,
            "anchors": self.anchors,
            "parts": self.parts,
            "translations_px": self.translations_px,
            "translations_ratio": self.translations_ratio,
            "angles": self.angles,
            "confidence": round(self.confidence, 3),
            "review_required": self.review_required,
            "pose_analyzer_version": self.pose_analyzer_version,
            "template_version": self.template_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MotionDescriptor":
        return cls(
            character_id=data["character_id"],
            variant=data["variant"],
            animation=data["animation"],
            source_path=data.get("source_path", ""),
            source_sha256=data.get("source_sha256", ""),
            frame_count=int(data.get("frame_count", 4)),
            orientation=data.get("orientation", "down"),
            character_width=float(data.get("character_width", 32.0)),
            character_height=float(data.get("character_height", 64.0)),
            root_motion=data.get("root_motion", []),
            anchors=data.get("anchors", []),
            parts=data.get("parts", []),
            translations_px=data.get("translations_px", []),
            translations_ratio=data.get("translations_ratio", []),
            angles=data.get("angles", []),
            confidence=float(data.get("confidence", 1.0)),
            review_required=bool(data.get("review_required", False)),
            pose_analyzer_version=data.get("pose_analyzer_version", "2.2"),
            template_version=data.get("template_version", "2.2"),
        )

    def save(self, path: Path, guard: Optional[SourceDatasetGuard] = None) -> Path:
        target_path = Path(path).resolve()
        if guard:
            guard.assert_can_write(target_path, operation_desc="guardado de motion descriptor")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return target_path

    @classmethod
    def load(cls, path: Path) -> "MotionDescriptor":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
