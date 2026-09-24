import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image

from core.alpha_analyzer import AlphaAnalyzer


class BoundingBoxDetector:
    """
    Detector de bounding boxes para siluetas y regiones de personajes en pixel art.
    """

    @staticmethod
    def detect_sprite_bbox(image_or_path: Image.Image | Path, alpha_threshold: int = 10) -> Optional[Tuple[int, int, int, int]]:
        """
        Retorna (min_x, min_y, max_x, max_y) de los píxeles opacos.
        """
        if isinstance(image_or_path, (str, Path)):
            with Image.open(Path(image_or_path)) as img:
                return BoundingBoxDetector.detect_sprite_bbox(img, alpha_threshold)

        bands = image_or_path.getbands()
        if "A" not in bands:
            return (0, 0, image_or_path.width - 1, image_or_path.height - 1)

        alpha = np.array(image_or_path.getchannel("A"))
        coords = np.argwhere(alpha > alpha_threshold)

        if coords.size == 0:
            return None

        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)

        return int(min_x), int(min_y), int(max_x), int(max_y)

    @staticmethod
    def detect_head_body_split(image: Image.Image, alpha_threshold: int = 10, head_ratio_approx: float = 0.35) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """
        Divide heurísticamente el bounding box en cabeza y torso/cuerpo inferior
        basado en la silueta y proporciones de pixel art.
        """
        bbox = BoundingBoxDetector.detect_sprite_bbox(image, alpha_threshold)
        if not bbox:
            return (0, 0, 0, 0), (0, 0, 0, 0)

        min_x, min_y, max_x, max_y = bbox
        total_h = max_y - min_y + 1

        split_y = int(min_y + total_h * head_ratio_approx)

        head_bbox = (min_x, min_y, max_x, split_y)
        body_bbox = (min_x, split_y + 1, max_x, max_y)

        return head_bbox, body_bbox
