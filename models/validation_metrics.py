"""
Modelos de datos para métricas de validación V2 (ValidationMetrics y ValidationResult).
Proporciona diagnósticos detallados y desacoplados:
- motion_activity_score
- head_identity_score
- palette_integrity_score
- baseline_stability_score
- anchor_continuity_score
- limb_continuity_score
- silhouette_consistency_score
- diagnostic_identity_score
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ValidationMetrics:
    """
    Métricas desacopladas de fidelidad y estabilidad cinemática.
    Cada valor oscila entre 0.0 (falla total) y 1.0 (óptimo).
    """
    motion_activity_score: float = 0.0        # Movimiento inter-frame (NO llamar quality)
    head_identity_score: float = 1.0          # Conservación de silueta/posición/paleta de cabeza
    palette_integrity_score: float = 1.0      # Ausencia de colores nuevos inesperados
    baseline_stability_score: float = 1.0     # Estabilidad del contacto con el suelo
    anchor_continuity_score: float = 1.0      # Ausencia de saltos abruptos en anchors
    limb_continuity_score: float = 1.0        # Conservación de longitudes de extremidades
    silhouette_consistency_score: float = 1.0 # Proporción y consistencia del área alfa
    diagnostic_identity_score: float = 1.0    # Combinación diagnóstica ponderada

    def to_dict(self) -> Dict[str, float]:
        """Serializa las métricas a diccionario redondeando a 4 decimales."""
        return {k: round(float(v), 4) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationMetrics":
        """Instancia ValidationMetrics desde un diccionario."""
        return cls(
            motion_activity_score=float(data.get("motion_activity_score", 0.0)),
            head_identity_score=float(data.get("head_identity_score", 1.0)),
            palette_integrity_score=float(data.get("palette_integrity_score", 1.0)),
            baseline_stability_score=float(data.get("baseline_stability_score", 1.0)),
            anchor_continuity_score=float(data.get("anchor_continuity_score", 1.0)),
            limb_continuity_score=float(data.get("limb_continuity_score", 1.0)),
            silhouette_consistency_score=float(data.get("silhouette_consistency_score", 1.0)),
            diagnostic_identity_score=float(data.get("diagnostic_identity_score", 1.0)),
        )


@dataclass
class ValidationResult:
    """
    Resultado de la validación técnica y visual de una animación transferida.
    """
    character_id: str
    variant: str
    animation: str
    template_version: str
    metrics: ValidationMetrics = field(default_factory=ValidationMetrics)
    warnings: List[str] = field(default_factory=list)
    review_required: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serializa el resultado a formato JSON-serializable."""
        return {
            "character_id": self.character_id,
            "variant": self.variant,
            "animation": self.animation,
            "template_version": self.template_version,
            "metrics": self.metrics.to_dict(),
            "warnings": list(self.warnings),
            "review_required": bool(self.review_required),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationResult":
        """Carga un ValidationResult desde un diccionario."""
        raw_metrics = data.get("metrics", {})
        metrics = ValidationMetrics.from_dict(raw_metrics) if isinstance(raw_metrics, dict) else ValidationMetrics()
        return cls(
            character_id=str(data.get("character_id", "")),
            variant=str(data.get("variant", "")),
            animation=str(data.get("animation", "")),
            template_version=str(data.get("template_version", "")),
            metrics=metrics,
            warnings=list(data.get("warnings", [])),
            review_required=bool(data.get("review_required", False)),
            created_at=str(data.get("created_at", "")),
        )

    def save(self, path: Path) -> Path:
        """Guarda el resultado en JSON de forma atómica."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "ValidationResult":
        """Carga el resultado desde un archivo JSON."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Archivo de validación no encontrado: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
