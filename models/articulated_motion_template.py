import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.layer_resolver import LayerResolver


@dataclass
class PartMotion:
    """
    Representa la cinemática articular (traslación en px, traslación relativa a tamaño, rotación y escala)
    de una parte corporal en un frame específico de la animación.
    """
    dx: float = 0.0
    dy: float = 0.0
    dx_ratio: float = 0.0
    dy_ratio: float = 0.0
    angle_deg: float = 0.0
    scale: float = 1.0
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dx": round(float(self.dx), 2),
            "dy": round(float(self.dy), 2),
            "dx_ratio": round(float(self.dx_ratio), 4),
            "dy_ratio": round(float(self.dy_ratio), 4),
            "angle_deg": round(float(self.angle_deg), 2),
            "scale": round(float(self.scale), 3),
            "confidence": round(float(self.confidence), 3),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PartMotion":
        return cls(
            dx=float(data.get("dx", 0.0)),
            dy=float(data.get("dy", 0.0)),
            dx_ratio=float(data.get("dx_ratio", 0.0)),
            dy_ratio=float(data.get("dy_ratio", 0.0)),
            angle_deg=float(data.get("angle_deg", 0.0)),
            scale=float(data.get("scale", 1.0)),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass
class ArticulatedFrameTemplate:
    """
    Cinemática de un frame específico dentro del ciclo de animación (típicamente frames 1, 2, 3, 4).
    """
    frame_index: int
    root_dx: float = 0.0
    root_dy: float = 0.0
    root_dx_ratio: float = 0.0
    root_dy_ratio: float = 0.0
    parts: Dict[str, PartMotion] = field(default_factory=dict)
    anchors_rel: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "root_dx": round(float(self.root_dx), 2),
            "root_dy": round(float(self.root_dy), 2),
            "root_dx_ratio": round(float(self.root_dx_ratio), 4),
            "root_dy_ratio": round(float(self.root_dy_ratio), 4),
            "parts": {name: pm.to_dict() for name, pm in self.parts.items()},
            "anchors_rel": self.anchors_rel,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArticulatedFrameTemplate":
        parts = {k: PartMotion.from_dict(v) for k, v in data.get("parts", {}).items()}
        return cls(
            frame_index=int(data["frame_index"]),
            root_dx=float(data.get("root_dx", 0.0)),
            root_dy=float(data.get("root_dy", 0.0)),
            root_dx_ratio=float(data.get("root_dx_ratio", 0.0)),
            root_dy_ratio=float(data.get("root_dy_ratio", 0.0)),
            parts=parts,
            anchors_rel=data.get("anchors_rel", {}),
        )


@dataclass
class ArticulatedMotionTemplate:
    """
    Plantilla de movimiento articulado cinemático V2 aprendida a partir de los
    personajes maestros aprobados ("personajes al 100%").
    """
    animation_name: str
    orientation: str = "down"
    frame_count: int = 4
    samples_used: int = 0
    outliers_detected: int = 0
    status: str = "active"  # "active" o "review_required"
    frames: List[ArticulatedFrameTemplate] = field(default_factory=list)

    def __post_init__(self):
        self.orientation = LayerResolver.normalize_orientation(self.animation_name or self.orientation)

    def get_frame(self, frame_index: int) -> Optional[ArticulatedFrameTemplate]:
        for f in self.frames:
            if f.frame_index == frame_index:
                return f
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "animation_name": self.animation_name,
            "orientation": self.orientation,
            "frame_count": self.frame_count,
            "samples_used": self.samples_used,
            "outliers_detected": self.outliers_detected,
            "status": self.status,
            "frames": [f.to_dict() for f in self.frames],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArticulatedMotionTemplate":
        frames = [ArticulatedFrameTemplate.from_dict(f) for f in data.get("frames", [])]
        return cls(
            animation_name=data["animation_name"],
            orientation=data.get("orientation", "down"),
            frame_count=int(data.get("frame_count", 4)),
            samples_used=int(data.get("samples_used", 0)),
            outliers_detected=int(data.get("outliers_detected", 0)),
            status=data.get("status", "active"),
            frames=frames,
        )

    def save(self, path: Path) -> Path:
        """Guarda la plantilla en JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: Path) -> "ArticulatedMotionTemplate":
        """Carga una plantilla desde un archivo JSON."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Plantilla no encontrada: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
