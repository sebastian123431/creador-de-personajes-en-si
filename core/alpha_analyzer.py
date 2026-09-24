import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image


class AlphaAnalyzer:
    """
    Herramientas de análisis de canal alfa para spritesheets y frames pixel art.
    """

    @staticmethod
    def load_alpha_mask(image_path: Path, threshold: int = 10) -> Optional[np.ndarray]:
        """
        Carga una imagen y retorna una máscara booleana 2D (True donde alfa > threshold).
        """
        if not image_path.exists():
            return None
        with Image.open(image_path) as img:
            if "A" not in img.getbands():
                # Si no tiene alfa, asumir todo opaco
                return np.ones((img.height, img.width), dtype=bool)
            alpha = np.array(img.getchannel("A"))
            return alpha > threshold

    @staticmethod
    def get_alpha_bounding_box(mask_or_image: np.ndarray | Path, threshold: int = 10) -> Optional[Tuple[int, int, int, int]]:
        """
        Calcula el bounding box exacto (min_x, min_y, max_x, max_y) de los píxeles visibles.
        """
        if isinstance(mask_or_image, (str, Path)):
            mask = AlphaAnalyzer.load_alpha_mask(Path(mask_or_image), threshold=threshold)
            if mask is None:
                return None
        else:
            if mask_or_image.dtype == bool:
                mask = mask_or_image
            elif len(mask_or_image.shape) == 3 and mask_or_image.shape[2] == 4:
                mask = mask_or_image[:, :, 3] > threshold
            else:
                mask = mask_or_image > threshold

        coords = np.argwhere(mask)
        if coords.size == 0:
            return None

        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)

        # max_x y max_y son inclusivos
        return int(min_x), int(min_y), int(max_x), int(max_y)
