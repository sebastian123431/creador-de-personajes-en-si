import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from core.spritesheet_builder import SpritesheetBuilder, ConsistencyMetrics
from models.animation import Animation

logger = logging.getLogger("SpriteStudio.ExportService")


class ExportService:
    """
    Servicio de exportación final para videojuegos (Unity / Godot / Unreal):
    - Spritesheet 4x16 de 64 frames (1024x3072 px o estándar configurado)
    - Secuencia de frames PNG
    - Metadatos JSON de animación
    - Consistency Score
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir or "projects/export").resolve()
        self.builder = SpritesheetBuilder()

    def export_character_variant(
        self,
        character_id: str,
        variant_name: str,
        animations: Dict[str, Animation],
        target_folder: Optional[Path] = None
    ) -> Tuple[Path, Path, ConsistencyMetrics]:
        """
        Exporta el paquete de Unity para la variante indicada.
        """
        dest_dir = (target_folder or self.output_dir) / character_id / variant_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        sheet_path, meta_path, metrics = self.builder.build_spritesheet(
            animations=animations,
            output_dir=dest_dir,
            sheet_filename=f"{character_id}_{variant_name}_spritesheet.png",
            metadata_filename=f"{character_id}_{variant_name}_metadata.json"
        )

        logger.info(f"Exportación para '{character_id}:{variant_name}' finalizada en: {dest_dir}")
        return sheet_path, meta_path, metrics
