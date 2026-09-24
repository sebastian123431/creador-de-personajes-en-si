import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

from core.alpha_analyzer import AlphaAnalyzer
from core.guard import SourceDatasetGuard
from core.sheet_detector import SheetDetector, SheetDetectionResult
from models.animation import Animation
from models.frame import Frame

logger = logging.getLogger("SpriteStudio.FrameExtractor")

OFFICIAL_ANIMATION_ROWS = [
    "idle_down",
    "walk_down",
    "idle_up",
    "walk_up",
    "idle_left",
    "walk_left",
    "idle_right",
    "walk_right",
    "cook_down",
    "cook_up",
    "cook_left",
    "cook_right",
    "think",
    "pickup",
    "serve",
    "celebrate"
]


class FrameExtractor:
    """
    Extrae los 64 frames de un spritesheet de movimientos y los organiza
    en las 16 filas oficiales de animación (4 frames por fila).
    Garantiza que todos los archivos se escriban FUERA del dataset original.
    """

    def __init__(
        self,
        output_base_dir: Optional[Path] = None,
        detector: Optional[Any] = None,
        guard: Optional[SourceDatasetGuard] = None,
        sheet_detector: Optional[Any] = None,
    ):
        self.output_base_dir = Path(output_base_dir or "dataset/extracted_frames").resolve()
        self.detector = sheet_detector or detector or SheetDetector()
        self.guard = guard

    def extract_from_sheet(
        self,
        sheet_path: Path,
        character_id: str,
        variant: str,
        custom_output_dir: Optional[Path] = None,
        source_character_folder: Optional[Path] = None
    ) -> Dict[str, Animation]:
        """
        Segmenta el spritesheet en 16 animaciones x 4 frames = 64 frames.
        Guarda los frames en dataset/extracted_frames/<char>/<variant>/<anim>/01.png
        y genera metadata de procedencia (source_modified: false).
        """
        animations: Dict[str, Animation] = {}

        if not sheet_path.exists():
            logger.error(f"Spritesheet no encontrado: {sheet_path}")
            return animations

        out_root = (custom_output_dir or self.output_base_dir) / character_id / variant
        out_root = out_root.resolve()

        # Validación estricta: impedir escribir dentro de la carpeta fuente protegida
        if self.guard:
            self.guard.assert_can_write(out_root, operation_desc="extracción de frames")

        out_root.mkdir(parents=True, exist_ok=True)

        detection: SheetDetectionResult = self.detector.detect_sheet(sheet_path)
        if not detection.is_valid:
            logger.error(f"Fallo en la detección del spritesheet {sheet_path.name}")
            return animations

        with Image.open(sheet_path) as full_img:
            if full_img.mode != "RGBA":
                full_img = full_img.convert("RGBA")

            for r_idx, anim_name in enumerate(OFFICIAL_ANIMATION_ROWS):
                anim_dir = out_root / anim_name
                anim_dir.mkdir(parents=True, exist_ok=True)

                row_cells = detection.cells[r_idx]
                anim_frames: List[Frame] = []

                for c_idx, cell in enumerate(row_cells):
                    frame_num = c_idx + 1
                    frame_filename = f"{frame_num:02d}.png"
                    frame_path = anim_dir / frame_filename

                    # Recortar celda pixel-perfect
                    box = cell.box if hasattr(cell, "box") else (cell.x, cell.y, cell.x + cell.w, cell.y + cell.h)
                    frame_crop = full_img.crop(box)
                    frame_crop.save(frame_path, "PNG")

                    # Metadata de procedencia requerida
                    meta_path = anim_dir / f"{frame_num:02d}_meta.json"
                    meta_content = {
                        "character_id": character_id,
                        "variant": variant,
                        "animation": anim_name,
                        "frame": frame_num,
                        "source_character_folder": str(source_character_folder or sheet_path.parent).replace("\\", "/"),
                        "source_spritesheet": str(sheet_path).replace("\\", "/"),
                        "source_modified": False,
                    }
                    with open(meta_path, "w", encoding="utf-8") as mf:
                        json.dump(meta_content, mf, indent=4)

                    # Analizar bbox
                    alpha_np = np.array(frame_crop.getchannel("A"))
                    coords = np.argwhere(alpha_np > 10)

                    if coords.size > 0:
                        min_y, min_x = coords.min(axis=0)
                        max_y, max_x = coords.max(axis=0)
                        bbox = (int(min_x), int(min_y), int(max_x), int(max_y))
                        center_x = int((min_x + max_x) // 2)
                        baseline_y = int(max_y)
                        alpha_area = int(np.sum(alpha_np > 10))
                    else:
                        bbox = (0, 0, cell.w, cell.h)
                        center_x = cell.w // 2
                        baseline_y = cell.h
                        alpha_area = 0

                    frame_obj = Frame(
                        image_path=frame_path,
                        index=frame_num,
                        bbox=bbox,
                        center_x=center_x,
                        baseline_y=baseline_y,
                        width=cell.w,
                        height=cell.h,
                        alpha_area=alpha_area
                    )
                    anim_frames.append(frame_obj)

                animations[anim_name] = Animation(
                    name=anim_name,
                    frames=anim_frames,
                    row_index=r_idx
                )

        logger.info(
            f"Extracción completada para '{character_id}:{variant}'. "
            f"64 frames y metadata guardados en: {out_root}"
        )
        return animations
