from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class BodyPart:
    """
    Representa una parte corporal segmentada para el sistema de movimiento articulado V2.
    Permite traslación, rotación respecto a pivot, ordenamiento por capas (z_index)
    y preservación estricta de píxeles originales.
    """
    name: str
    bbox: Tuple[int, int, int, int]  # (min_x, min_y, max_x, max_y)
    pivot_x: float                   # Coordenada X del pivote (absoluta o relativa al frame)
    pivot_y: float                   # Coordenada Y del pivote (absoluta o relativa al frame)
    parent: Optional[str] = None     # Nombre de la parte padre en la jerarquía esquelética
    z_index: int = 0                 # Profundidad de renderizado (menor = detrás, mayor = delante)
    mask_path: Optional[Path] = None # Ruta al recorte/máscara PNG transparente
    confidence: float = 1.0          # Nivel de confianza de la segmentación (0.0 a 1.0)

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0] + 1)

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1] + 1)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "bbox": list(self.bbox),
            "pivot_x": self.pivot_x,
            "pivot_y": self.pivot_y,
            "parent": self.parent,
            "z_index": self.z_index,
            "mask_path": str(self.mask_path).replace("\\", "/") if self.mask_path else None,
            "confidence": round(self.confidence, 3),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BodyPart":
        mask = Path(data["mask_path"]) if data.get("mask_path") else None
        return cls(
            name=data["name"],
            bbox=tuple(data["bbox"]),
            pivot_x=float(data["pivot_x"]),
            pivot_y=float(data["pivot_y"]),
            parent=data.get("parent"),
            z_index=int(data.get("z_index", 0)),
            mask_path=mask,
            confidence=float(data.get("confidence", 1.0)),
        )

