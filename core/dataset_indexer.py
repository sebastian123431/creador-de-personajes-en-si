import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from core.guard import SourceDatasetGuard
from models.character import Character
from models.dataset import DatasetIndex

logger = logging.getLogger("SpriteStudio.Indexer")


class DatasetIndexer:
    """
    Gestiona la creación, serialización y actualización incremental de dataset_index.json.
    Garantiza que el índice se escriba FUERA del dataset protegido (en dataset/indexed/).
    """

    def __init__(self, index_path: Path, guard: Optional[SourceDatasetGuard] = None):
        self.index_path = Path(index_path).resolve()
        self.guard = guard

    def build_index_data(
        self,
        characters: Dict[str, Character],
        dataset_root: Optional[Path] = None,
        base_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Construye el formato oficial:
        {
            "dataset_root": "dataset/finished_characters/approved/personajes al 100%",
            "updated_at": "...",
            "total_characters": N,
            "characters": { ... }
        }
        """
        def format_path(p: Optional[Path]) -> Optional[str]:
            if not p:
                return None
            try:
                if base_path:
                    return str(p.resolve().relative_to(base_path.resolve())).replace("\\", "/")
                return str(p.resolve()).replace("\\", "/")
            except ValueError:
                return str(p.resolve()).replace("\\", "/")

        chars_dict = {}
        for char_id, char in characters.items():
            variants_dict = {}
            for v_name, v_obj in char.variants.items():
                variants_dict[v_name] = {
                    "reference": format_path(v_obj.reference_image),
                    "spritesheet": format_path(v_obj.spritesheet),
                    "status": v_obj.status,
                    "hashes": v_obj.file_hashes,
                }

            chars_dict[char_id] = {
                "source_folder": format_path(char.source_dir),
                "display_name": char.display_name,
                "is_approved": char.is_approved,
                "variants": variants_dict,
            }

        root_str = format_path(dataset_root) if dataset_root else "dataset/finished_characters/approved/personajes al 100%"

        return {
            "dataset_root": root_str,
            "version": "1.0.0",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_characters": len(chars_dict),
            "characters": chars_dict,
        }

    def save_index(
        self,
        characters: Dict[str, Character],
        dataset_root: Optional[Path] = None,
        base_path: Optional[Path] = None
    ) -> Path:
        """
        Serializa el índice en dataset/indexed/dataset_index.json de forma segura y atómica.
        Lanza excepción si intenta escribirse dentro de la carpeta protegida de personajes.
        """
        if self.guard:
            self.guard.assert_can_write(self.index_path, operation_desc="guardado de índice")

        data = self.build_index_data(characters, dataset_root=dataset_root, base_path=base_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = self.index_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        temp_path.replace(self.index_path)
        logger.info(f"Índice maestro de dataset guardado exitosamente en: {self.index_path}")
        return self.index_path

    def load_index(self) -> Optional[DatasetIndex]:
        """Carga el índice existente."""
        if not self.index_path.exists():
            return None

        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            return DatasetIndex(
                version=raw_data.get("version", "1.0.0"),
                updated_at=raw_data.get("updated_at", ""),
                characters=raw_data.get("characters", {}),
            )
        except Exception as e:
            logger.error(f"Error cargando {self.index_path}: {e}")
            return None

    def detect_changes(self, new_characters: Dict[str, Character]) -> Dict[str, Any]:
        """Compara hashes para detección incremental."""
        saved_index = self.load_index()
        if not saved_index or not saved_index.characters:
            return {
                "added": list(new_characters.keys()),
                "modified": [],
                "unchanged": [],
                "removed": [],
            }

        saved_chars = saved_index.characters
        added, modified, unchanged = [], [], []

        for char_id, char in new_characters.items():
            if char_id not in saved_chars:
                added.append(char_id)
                continue

            saved_char = saved_chars[char_id]
            is_char_modified = False

            for v_name, v_obj in char.variants.items():
                saved_v = saved_char.get("variants", {}).get(v_name, {})
                saved_hashes = saved_v.get("hashes", {})
                if v_obj.file_hashes != saved_hashes:
                    is_char_modified = True
                    break

            if is_char_modified:
                modified.append(char_id)
            else:
                unchanged.append(char_id)

        removed = [cid for cid in saved_chars.keys() if cid not in new_characters]

        return {
            "added": added,
            "modified": modified,
            "unchanged": unchanged,
            "removed": removed,
        }
