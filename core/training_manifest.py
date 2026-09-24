import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.guard import SourceDatasetGuard

logger = logging.getLogger("SpriteStudio.TrainingManifest")


@dataclass
class TrainingManifestEntry:
    character_id: str
    variant: str
    spritesheet_path: str
    spritesheet_sha256: str
    pose_analyzer_version: str = "2.2"
    template_version: str = "2.2"
    processed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    animations_processed: List[str] = field(default_factory=list)
    descriptor_paths: Dict[str, str] = field(default_factory=dict)
    status: str = "pending"  # pending, processing, completed, failed, review_required

    def to_dict(self) -> Dict[str, Any]:
        return {
            "character_id": self.character_id,
            "variant": self.variant,
            "spritesheet_path": self.spritesheet_path,
            "spritesheet_sha256": self.spritesheet_sha256,
            "pose_analyzer_version": self.pose_analyzer_version,
            "template_version": self.template_version,
            "processed_at": self.processed_at,
            "animations_processed": self.animations_processed,
            "descriptor_paths": self.descriptor_paths,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingManifestEntry":
        return cls(
            character_id=data["character_id"],
            variant=data["variant"],
            spritesheet_path=data["spritesheet_path"],
            spritesheet_sha256=data["spritesheet_sha256"],
            pose_analyzer_version=data.get("pose_analyzer_version", "2.2"),
            template_version=data.get("template_version", "2.2"),
            processed_at=data.get("processed_at", datetime.now(timezone.utc).isoformat()),
            animations_processed=data.get("animations_processed", []),
            descriptor_paths=data.get("descriptor_paths", {}),
            status=data.get("status", "pending"),
        )


class TrainingManifest:
    """
    Manifiesto de entrenamiento incremental V2.
    Registra el estado, hashes SHA256 y rutas de descriptores por cada variante procesada.
    Se persiste fuera del dataset maestro mediante escritura atómica (tmp -> replace).
    """

    def __init__(
        self,
        manifest_path: Optional[Path] = None,
        guard: Optional[SourceDatasetGuard] = None,
    ):
        self.manifest_path = Path(
            manifest_path or (Path.cwd() / "dataset" / "training_v2" / "training_manifest.json")
        ).resolve()
        self.guard = guard or SourceDatasetGuard()
        self.entries: Dict[str, TrainingManifestEntry] = {}
        self.load()

    @staticmethod
    def make_key(character_id: str, variant: str) -> str:
        return f"{character_id}::{variant}"

    def get_entry(self, character_id: str, variant: str) -> Optional[TrainingManifestEntry]:
        return self.entries.get(self.make_key(character_id, variant))

    def update_entry(self, entry: TrainingManifestEntry) -> None:
        self.entries[self.make_key(entry.character_id, entry.variant)] = entry

    def is_cache_valid(
        self,
        character_id: str,
        variant: str,
        current_sha256: str,
        pose_version: str = "2.2",
        template_version: str = "2.2",
        required_animations: Optional[List[str]] = None,
    ) -> bool:
        """
        Determina si el análisis previo puede reutilizarse íntegramente (CACHE HIT).
        Requiere:
        1. Entrada existente con status 'completed' o 'review_required'.
        2. Hash SHA256 del spritesheet exactamente idéntico.
        3. Versión de PoseAnalyzerV2 idéntica.
        4. Versión de plantilla idéntica.
        5. Todos los descriptores requeridos existen en disco.
        """
        entry = self.get_entry(character_id, variant)
        if not entry:
            return False

        if entry.status not in ("completed", "review_required"):
            return False

        if entry.spritesheet_sha256 != current_sha256:
            return False

        if entry.pose_analyzer_version != pose_version:
            return False

        if entry.template_version != template_version:
            return False

        if required_animations:
            for anim in required_animations:
                if anim not in entry.animations_processed:
                    return False
                desc_path_str = entry.descriptor_paths.get(anim)
                if not desc_path_str or not Path(desc_path_str).exists():
                    return False

        return True

    def load(self) -> None:
        if not self.manifest_path.exists():
            self.entries = {}
            return

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            entries_dict = {}
            raw_entries = data.get("entries", {})
            for key, val in raw_entries.items():
                entries_dict[key] = TrainingManifestEntry.from_dict(val)

            self.entries = entries_dict
            logger.debug(f"Manifest cargado desde {self.manifest_path} con {len(self.entries)} entradas.")
        except Exception as e:
            logger.warning(f"Error cargando manifest en {self.manifest_path} ({e}). Inicializando vacío.")
            self.entries = {}

    def save(self) -> None:
        """
        Guarda el manifiesto de forma atómica:
        1. Protege la carpeta fuente contra escrituras accidentales.
        2. Escribe en un archivo temporal .tmp.
        3. Reemplaza atómicamente el archivo destino.
        """
        self.guard.assert_can_write(self.manifest_path, operation_desc="guardado de training_manifest.json")
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": "2.2",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_entries": len(self.entries),
            "entries": {k: v.to_dict() for k, v in self.entries.items()},
        }

        tmp_path = self.manifest_path.with_suffix(".tmp")
        self.guard.assert_can_write(tmp_path, operation_desc="escritura temporal de training_manifest")

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        # Reemplazo atómico
        os.replace(tmp_path, self.manifest_path)
        logger.debug(f"Manifest guardado atómicamente en {self.manifest_path}.")
