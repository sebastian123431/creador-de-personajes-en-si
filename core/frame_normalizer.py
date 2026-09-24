import logging
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image

from core.bbox_detector import BoundingBoxDetector
from core.baseline_detector import BaselineDetector
from models.frame import Frame

logger = logging.getLogger("SpriteStudio.FrameNormalizer")


class FrameNormalizer:
    """
    Normaliza frames individuales a un canvas estándar preservando la fidelidad
    pixel-perfect (NEAREST), centrando horizontalmente y alineando pies a una baseline común.
    """

    def __init__(
        self,
        canvas_width: int = 256,
        canvas_height: int = 192,
        baseline_offset_from_bottom: int = 16,
        output_base_dir: Optional[Path] = None,
    ):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.target_baseline_y = canvas_height - baseline_offset_from_bottom
        self.target_center_x = canvas_width // 2
        self.output_base_dir = Path(output_base_dir or "dataset/normalized")

    def normalize_frame(
        self,
        source_frame: Frame | Path,
        character_id: Optional[str] = None,
        variant: Optional[str] = None,
        animation_name: Optional[str] = None,
        custom_output_path: Optional[Path] = None
    ) -> Frame:
        """
        Aplica los 8 pasos de normalización pixel-perfect a un frame.
        """
        input_path = source_frame.image_path if isinstance(source_frame, Frame) else Path(source_frame)
        if isinstance(source_frame, Frame):
            frame_idx = source_frame.index
        else:
            try:
                frame_idx = int(input_path.stem)
            except ValueError:
                frame_idx = 1

        with Image.open(input_path) as img:
            if img.mode != "RGBA":
                img = img.convert("RGBA")

            # 1. Detectar bbox del personaje
            bbox = BoundingBoxDetector.detect_sprite_bbox(img)
            if not bbox:
                # Frame vacío
                bbox = (0, 0, img.width, img.height)
                sprite_crop = img
                feet_y = img.height - 1
                center_x_sprite = img.width // 2
            else:
                min_x, min_y, max_x, max_y = bbox
                sprite_crop = img.crop((min_x, min_y, max_x + 1, max_y + 1))
                feet_y = max_y - min_y
                center_x_sprite = (max_x - min_x) // 2

            crop_w, crop_h = sprite_crop.size

            # 2. Preservar aspect ratio y verificar si cabe en el canvas
            scale = 1.0
            max_avail_w = self.canvas_width - 8
            max_avail_h = self.target_baseline_y - 8

            if crop_w > max_avail_w or crop_h > max_avail_h:
                scale = min(max_avail_w / crop_w, max_avail_h / crop_h)
                new_w = max(1, int(round(crop_w * scale)))
                new_h = max(1, int(round(crop_h * scale)))
                # Pixel perfect: NUNCA bilineal, SIEMPRE NEAREST
                sprite_crop = sprite_crop.resize((new_w, new_h), resample=Image.Resampling.NEAREST)
                crop_w, crop_h = sprite_crop.size
                center_x_sprite = crop_w // 2
                feet_y = crop_h - 1

            # 3 & 4 & 5 & 6. Crear canvas estándar transparente y situar personaje
            canvas = Image.new("RGBA", (self.canvas_width, self.canvas_height), (0, 0, 0, 0))

            paste_x = int(self.target_center_x - center_x_sprite)
            paste_y = int(self.target_baseline_y - feet_y)

            # 7. Mantener transparencia utilizando el sprite_crop como máscara
            canvas.paste(sprite_crop, (paste_x, paste_y), mask=sprite_crop)

            # Determinar ruta de salida
            if custom_output_path:
                out_file = custom_output_path
            elif character_id and variant and animation_name:
                out_dir = self.output_base_dir / character_id / variant / animation_name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"{frame_idx:02d}.png"
            else:
                out_dir = input_path.parent / "normalized"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / input_path.name

            out_file.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(out_file, "PNG")

            norm_bbox = (paste_x, paste_y, paste_x + crop_w - 1, paste_y + crop_h - 1)

            return Frame(
                image_path=out_file,
                index=frame_idx,
                bbox=norm_bbox,
                center_x=self.target_center_x,
                baseline_y=self.target_baseline_y,
                width=self.canvas_width,
                height=self.canvas_height,
                alpha_area=isinstance(source_frame, Frame) and source_frame.alpha_area or (crop_w * crop_h)
            )
