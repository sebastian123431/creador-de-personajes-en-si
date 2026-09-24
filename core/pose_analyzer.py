from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from core.bbox_detector import BoundingBoxDetector


@dataclass
class AnchorPoint:
    name: str
    x: int
    y: int
    confidence: float = 1.0


@dataclass
class PoseSkeleton:
    anchors: Dict[str, AnchorPoint] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Dict[str, int]]:
        return {name: {"x": pt.x, "y": pt.y} for name, pt in self.anchors.items()}


class PoseAnalyzer:
    """
    Analizador de pose y anclajes articulares adaptado para sprites 2D Pixel Art.
    Combina silueta, componentes conectados y heurísticas geométricas.
    """

    ANCHOR_NAMES = [
        "head", "neck",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_hand", "right_hand",
        "hip",
        "left_knee", "right_knee",
        "left_foot", "right_foot"
    ]

    def estimate_pose_anchors(self, image_or_path: Image.Image | Path, alpha_threshold: int = 10) -> PoseSkeleton:
        """
        Calcula las coordenadas de los anclajes corporales esenciales para un frame pixel art.
        """
        if isinstance(image_or_path, (str, Path)):
            with Image.open(Path(image_or_path)) as img:
                return self.estimate_pose_anchors(img, alpha_threshold=alpha_threshold)

        img = image_or_path.convert("RGBA")
        bbox = BoundingBoxDetector.detect_sprite_bbox(img, alpha_threshold=alpha_threshold)

        if not bbox:
            return PoseSkeleton()

        min_x, min_y, max_x, max_y = bbox
        w = max_x - min_x + 1
        h = max_y - min_y + 1
        cx = int((min_x + max_x) // 2)

        # Heurísticas de proporción vertical para personajes pixel art de videojuego:
        # head: 15% de la altura
        # neck: 32%
        # shoulders: 36%, desplazados +/- 30% del ancho
        # elbows: 48%
        # hands: 60%
        # hip: 58%
        # knees: 75%
        # feet: 98%
        head_y = int(min_y + h * 0.15)
        neck_y = int(min_y + h * 0.32)

        shoulder_y = int(min_y + h * 0.38)
        shoulder_offset = max(2, int(w * 0.30))

        elbow_y = int(min_y + h * 0.50)
        elbow_offset = max(3, int(w * 0.38))

        hand_y = int(min_y + h * 0.62)
        hand_offset = max(3, int(w * 0.40))

        hip_y = int(min_y + h * 0.58)

        knee_y = int(min_y + h * 0.76)
        knee_offset = max(2, int(w * 0.18))

        foot_y = int(max_y)
        foot_offset = max(2, int(w * 0.22))

        anchors = {
            "head": AnchorPoint("head", cx, head_y, 0.9),
            "neck": AnchorPoint("neck", cx, neck_y, 0.85),
            "left_shoulder": AnchorPoint("left_shoulder", cx - shoulder_offset, shoulder_y, 0.8),
            "right_shoulder": AnchorPoint("right_shoulder", cx + shoulder_offset, shoulder_y, 0.8),
            "left_elbow": AnchorPoint("left_elbow", cx - elbow_offset, elbow_y, 0.75),
            "right_elbow": AnchorPoint("right_elbow", cx + elbow_offset, elbow_y, 0.75),
            "left_hand": AnchorPoint("left_hand", cx - hand_offset, hand_y, 0.7),
            "right_hand": AnchorPoint("right_hand", cx + hand_offset, hand_y, 0.7),
            "hip": AnchorPoint("hip", cx, hip_y, 0.85),
            "left_knee": AnchorPoint("left_knee", cx - knee_offset, knee_y, 0.8),
            "right_knee": AnchorPoint("right_knee", cx + knee_offset, knee_y, 0.8),
            "left_foot": AnchorPoint("left_foot", cx - foot_offset, foot_y, 0.9),
            "right_foot": AnchorPoint("right_foot", cx + foot_offset, foot_y, 0.9),
        }

        return PoseSkeleton(anchors=anchors)
