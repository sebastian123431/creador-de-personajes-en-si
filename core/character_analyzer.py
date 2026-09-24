from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from core.bbox_detector import BoundingBoxDetector
from core.baseline_detector import BaselineDetector


@dataclass
class CharacterAnalysisResult:
    sprite_width: int
    sprite_height: int
    head_bbox: Tuple[int, int, int, int]
    body_bbox: Tuple[int, int, int, int]
    center_x: int
    center_y: int
    baseline_y: int
    head_ratio: float
    body_ratio: float
    dominant_colors: List[Tuple[int, int, int]]
    alpha_area: int
    orientation: str  # "down", "up", "left", "right"


class CharacterAnalyzer:
    """
    Analizador estructural y visual de personajes pixel art a partir de referencias o frames.
    """

    def analyze_sprite(
        self,
        image_or_path: Image.Image | Path,
        orientation: str = "down",
        alpha_threshold: int = 10
    ) -> Optional[CharacterAnalysisResult]:
        """
        Extrae proporciones anatómicas, baseline, bounding boxes y paleta dominante.
        """
        if isinstance(image_or_path, (str, Path)):
            with Image.open(Path(image_or_path)) as img:
                return self.analyze_sprite(img, orientation=orientation, alpha_threshold=alpha_threshold)

        img = image_or_path.convert("RGBA")
        w, h = img.size

        bbox = BoundingBoxDetector.detect_sprite_bbox(img, alpha_threshold=alpha_threshold)
        if not bbox:
            return None

        min_x, min_y, max_x, max_y = bbox
        sprite_w = max_x - min_x + 1
        sprite_h = max_y - min_y + 1

        center_x = int((min_x + max_x) // 2)
        center_y = int((min_y + max_y) // 2)
        baseline_y = int(max_y)

        # Segmentar cabeza y cuerpo (heurística pixel art proporciones chibi/semi-chibi ~35% cabeza)
        head_bbox, body_bbox = BoundingBoxDetector.detect_head_body_split(img, alpha_threshold=alpha_threshold)

        head_h = head_bbox[3] - head_bbox[1] + 1 if head_bbox[3] >= head_bbox[1] else 0
        body_h = body_bbox[3] - body_bbox[1] + 1 if body_bbox[3] >= body_bbox[1] else 0

        head_ratio = round(head_h / sprite_h, 3) if sprite_h > 0 else 0.0
        body_ratio = round(body_h / sprite_h, 3) if sprite_h > 0 else 0.0

        # Calcular área alfa
        alpha_arr = np.array(img.getchannel("A"))
        alpha_area = int(np.sum(alpha_arr > alpha_threshold))

        # Extraer colores dominantes (filtrando píxeles transparentes)
        img_np = np.array(img)
        visible_pixels = img_np[alpha_arr > alpha_threshold][:, :3]

        dominant_colors = []
        if len(visible_pixels) > 0:
            # Cuantizar a múltiplos de 16 para agrupar colores similares
            quantized = (visible_pixels // 16) * 16
            unique_colors, counts = np.unique(quantized, axis=0, return_counts=True)
            sorted_indices = np.argsort(-counts)
            for idx in sorted_indices[:5]:
                dominant_colors.append(tuple(int(c) for c in unique_colors[idx]))

        return CharacterAnalysisResult(
            sprite_width=sprite_w,
            sprite_height=sprite_h,
            head_bbox=head_bbox,
            body_bbox=body_bbox,
            center_x=center_x,
            center_y=center_y,
            baseline_y=baseline_y,
            head_ratio=head_ratio,
            body_ratio=body_ratio,
            dominant_colors=dominant_colors,
            alpha_area=alpha_area,
            orientation=orientation
        )
