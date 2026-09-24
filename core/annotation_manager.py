import json
import logging
from pathlib import Path
from typing import Dict, Optional

from core.guard import SourceDatasetGuard
from models.skeleton import Skeleton, Anchor

logger = logging.getLogger("SpriteStudio.AnnotationManager")


class AnnotationManager:
    """
    Gestiona el almacenamiento y carga de anotaciones manuales de anchors en:
    dataset/annotations/<character_id>/<variant>/<animation>/<frame_index>.json
    Garantiza que las anotaciones se escriban FUERA del dataset original.
    """

    def __init__(self, base_annotations_dir: Optional[Path] = None, guard: Optional[SourceDatasetGuard] = None):
        self.base_dir = Path(base_annotations_dir or "dataset/annotations").resolve()
        self.guard = guard

    def get_annotation_path(self, character_id: str, variant: str, animation: str, frame_index: int) -> Path:
        return self.base_dir / character_id / variant / animation / f"{frame_index:02d}.json"

    def has_annotation(self, character_id: str, variant: str, animation: str, frame_index: int) -> bool:
        return self.get_annotation_path(character_id, variant, animation, frame_index).exists()

    def load_annotation(self, character_id: str, variant: str, animation: str, frame_index: int) -> Optional[Skeleton]:
        path = self.get_annotation_path(character_id, variant, animation, frame_index)
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            anchors_data = data.get("anchors", {})
            return Skeleton.from_dict(anchors_data)
        except Exception as e:
            logger.error(f"Error cargando anotación desde {path}: {e}")
            return None

    def save_annotation(
        self,
        character_id: str,
        variant: str,
        animation: str,
        frame_index: int,
        skeleton: Skeleton,
    ) -> Path:
        target_path = self.get_annotation_path(character_id, variant, animation, frame_index)
        target_path = target_path.resolve()

        if self.guard:
            self.guard.assert_can_write(target_path, operation_desc="guardado de anotaciones")

        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Marcar los anchors guardados manualmente como is_manual=True
        anchors_dict = {}
        for name, a in skeleton.anchors.items():
            anchors_dict[name] = {
                "x": a.x,
                "y": a.y,
                "confidence": 1.0 if a.is_manual else a.confidence,
                "is_manual": True if a.is_manual else False,
            }

        payload = {
            "character_id": character_id,
            "variant": variant,
            "animation": animation,
            "frame": frame_index,
            "anchors": anchors_dict,
        }

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)

        logger.info(f"Anotación manual guardada en: {target_path}")
        return target_path
