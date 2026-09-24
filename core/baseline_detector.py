import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image

from core.bbox_detector import BoundingBoxDetector


class BaselineDetector:
    """
    Detector de línea base (pies/suelo) para garantizar un anclaje consistente
    y evitar que el personaje flote o se hunda entre animaciones.
    """

    @staticmethod
    def detect_feet_baseline(image_or_path: Image.Image | Path, alpha_threshold: int = 10) -> int:
        """
        Retorna la coordenada Y del píxel opaco más bajo (pies).
        """
        bbox = BoundingBoxDetector.detect_sprite_bbox(image_or_path, alpha_threshold)
        if bbox is None:
            if isinstance(image_or_path, Image.Image):
                return image_or_path.height - 1
            with Image.open(Path(image_or_path)) as img:
                return img.height - 1

        min_x, min_y, max_x, max_y = bbox
        return max_y
