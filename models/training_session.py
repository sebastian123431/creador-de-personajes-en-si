import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.guard import SourceDatasetGuard


@dataclass
class TrainingSession:
    """
    Representa el estado y métricas de una sesión de entrenamiento cinemático V2.
    """
    session_id: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    )
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: Optional[str] = None
    status: str = "running"  # running, completed, cancelled, failed
    is_smoke: bool = False

    characters_total: int = 0
    characters_processed: int = 0

    variants_total: int = 0
    variants_processed: int = 0

    animations_total: int = 0
    animations_processed: int = 0

    frames_processed: int = 0
    frames_cached: int = 0

    cache_hits: int = 0
    cache_misses: int = 0

    low_confidence_frames: int = 0
    review_required_count: int = 0
    templates_generated: int = 0

    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    cancel_requested: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "is_smoke": self.is_smoke,
            "characters_total": self.characters_total,
            "characters_processed": self.characters_processed,
            "variants_total": self.variants_total,
            "variants_processed": self.variants_processed,
            "animations_total": self.animations_total,
            "animations_processed": self.animations_processed,
            "frames_processed": self.frames_processed,
            "frames_cached": self.frames_cached,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "low_confidence_frames": self.low_confidence_frames,
            "review_required_count": self.review_required_count,
            "templates_generated": self.templates_generated,
            "warnings_count": len(self.warnings),
            "warnings": self.warnings,
            "errors_count": len(self.errors),
            "errors": self.errors,
            "cancel_requested": self.cancel_requested,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingSession":
        return cls(
            session_id=data.get("session_id", ""),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at"),
            status=data.get("status", "completed"),
            is_smoke=bool(data.get("is_smoke", False)),
            characters_total=int(data.get("characters_total", 0)),
            characters_processed=int(data.get("characters_processed", 0)),
            variants_total=int(data.get("variants_total", 0)),
            variants_processed=int(data.get("variants_processed", 0)),
            animations_total=int(data.get("animations_total", 0)),
            animations_processed=int(data.get("animations_processed", 0)),
            frames_processed=int(data.get("frames_processed", 0)),
            frames_cached=int(data.get("frames_cached", 0)),
            cache_hits=int(data.get("cache_hits", 0)),
            cache_misses=int(data.get("cache_misses", 0)),
            low_confidence_frames=int(data.get("low_confidence_frames", 0)),
            review_required_count=int(data.get("review_required_count", 0)),
            templates_generated=int(data.get("templates_generated", 0)),
            warnings=data.get("warnings", []),
            errors=data.get("errors", []),
            cancel_requested=bool(data.get("cancel_requested", False)),
        )

    def save(self, path: Path, guard: Optional[SourceDatasetGuard] = None) -> Path:
        target_path = Path(path).resolve()
        if guard:
            guard.assert_can_write(target_path, operation_desc="guardado de reporte de sesión de entrenamiento")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return target_path
