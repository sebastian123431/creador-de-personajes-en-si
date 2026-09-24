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

    def export_unity_package_v2(
        self,
        character_id: str,
        variant_name: str,
        animations: Dict[str, Animation],
        target_folder: Optional[Path] = None,
    ) -> Tuple[Path, Path, ConsistencyMetrics]:
        """
        Exportación avanzada V2 para Unity y motores 2D:
        - Spritesheet 4x16 (64 frames)
        - Metadata JSON enriquecida con información de articulaciones, capas Z y clips de animación.
        """
        import json
        dest_dir = (target_folder or self.output_dir) / character_id / variant_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        sheet_path, meta_path, metrics = self.builder.build_spritesheet(
            animations=animations,
            output_dir=dest_dir,
            sheet_filename=f"{character_id}_{variant_name}_spritesheet_v2.png",
            metadata_filename=f"{character_id}_{variant_name}_metadata_v2.json"
        )

        # Enriquecer metadata con especificaciones de motor de juego V2
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)

            meta_data["engine_version"] = "2.0-articulated"
            meta_data["rig"] = {
                "type": "18_anchor_articulated",
                "body_parts": ["head", "torso", "left_arm", "right_arm", "left_leg", "right_leg"],
                "head_identity_locked": True,
            }
            meta_data["unity_importer_hints"] = {
                "spriteMode": 2,  # Multiple
                "pixelsPerUnit": 16,
                "filterMode": "Point",
                "textureCompression": "None",
            }

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=4, ensure_ascii=False)

        except Exception as e:
            logger.warning(f"No se pudo enriquecer metadata de exportación V2: {e}")

        logger.info(f"Exportación V2 para Unity finalizada en: {dest_dir}")
        return sheet_path, meta_path, metrics

