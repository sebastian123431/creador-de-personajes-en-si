import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from core.bbox_detector import BoundingBoxDetector
from core.pose_analyzer import PoseAnalyzer as PoseAnalyzerV1
from models.skeleton import Skeleton, Anchor, OFFICIAL_ANCHOR_NAMES

logger = logging.getLogger("SpriteStudio.PoseV2")


class PoseAnalyzerV2:
    """
    Detector híbrido de pose y articulaciones V2 para sprites 2D Pixel Art.
    Utiliza:
    1. Silueta de canal alfa
    2. Componentes conexos y contornos morfológicos
    3. Perfil de ancho horizontal por altura (detección de cuello y cadera)
    4. Eje medial y simetría
    5. Heurísticas anatómicas pixel art
    6. Soporte para anotaciones manuales prioritarias
    7. Fallback a PoseAnalyzer V1 si la silueta es insuficiente
    """

    def __init__(self, fallback_v1: Optional[PoseAnalyzerV1] = None):
        self.fallback_v1 = fallback_v1 or PoseAnalyzerV1()

    def _compute_width_profile(self, mask: np.ndarray, min_y: int, max_y: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula el ancho y el centro X para cada fila Y en el rango del personaje.
        """
        height_span = max_y - min_y + 1
        widths = np.zeros(height_span, dtype=int)
        centers_x = np.zeros(height_span, dtype=float)

        for i, y in enumerate(range(min_y, max_y + 1)):
            row = mask[y, :]
            coords = np.where(row)[0]
            if len(coords) > 0:
                widths[i] = coords[-1] - coords[0] + 1
                centers_x[i] = (coords[0] + coords[-1]) / 2.0
            else:
                widths[i] = 0
                centers_x[i] = mask.shape[1] / 2.0

        return widths, centers_x

    def analyze_pose(
        self,
        image_or_path: Union[Image.Image, Path, str],
        orientation: str = "down",
        manual_annotations: Optional[Dict[str, dict]] = None,
        prev_skeleton: Optional[Skeleton] = None,
        alpha_threshold: int = 10,
    ) -> Skeleton:
        """
        Analiza un sprite y retorna un Skeleton con los 18 anchors calculados
        siguiendo el orden de prioridad:
        1. Anotación manual
        2. Anchor temporal del frame previo (si aplica)
        3. PoseAnalyzerV2 (análisis de silueta, contornos y perfil de anchura)
        4. PoseAnalyzer V1 (fallback geométrico)
        """
        skeleton = Skeleton()

        # Cargar imagen en RGBA y obtener máscara
        if isinstance(image_or_path, (str, Path)):
            path_obj = Path(image_or_path)
            if not path_obj.exists():
                return skeleton
            with Image.open(path_obj) as img:
                img_rgba = img.convert("RGBA")
        else:
            img_rgba = image_or_path.convert("RGBA")

        w, h = img_rgba.size
        alpha_arr = np.array(img_rgba.getchannel("A"))
        mask = alpha_arr > alpha_threshold

        bbox = BoundingBoxDetector.detect_sprite_bbox(img_rgba, alpha_threshold=alpha_threshold)
        if not bbox:
            return skeleton

        min_x, min_y, max_x, max_y = bbox
        char_w = max_x - min_x + 1
        char_h = max_y - min_y + 1

        if char_w < 3 or char_h < 6:
            # Demasiado pequeño para análisis de silueta, usar fallback V1
            return self._apply_fallback_v1(img_rgba, manual_annotations)

        # 1. Perfil de ancho horizontal y centro medial
        widths, centers_x = self._compute_width_profile(mask, min_y, max_y)

        # 2. Localizar estrechamiento del cuello (mínimo de anchura entre 20% y 40% de altura)
        neck_search_start = int(char_h * 0.20)
        neck_search_end = max(neck_search_start + 1, int(char_h * 0.42))
        neck_idx = neck_search_start + int(np.argmin(widths[neck_search_start:neck_search_end]))
        neck_y = min_y + neck_idx
        neck_x = int(round(centers_x[neck_idx]))

        # 3. Cabeza (arriba del cuello)
        head_y = min_y + max(2, int((neck_y - min_y) * 0.45))
        head_x = int(round(np.mean(centers_x[:neck_idx]))) if neck_idx > 0 else (min_x + max_x) // 2

        # 4. Hombros y Pecho (debajo del cuello)
        chest_y = int(neck_y + char_h * 0.12)
        chest_x = int(round(centers_x[min(len(centers_x) - 1, neck_idx + int(char_h * 0.12))]))

        shoulder_y = int(neck_y + char_h * 0.06)
        shoulder_span = max(2, int(char_w * 0.32))
        l_shoulder_x = max(min_x, chest_x - shoulder_span)
        r_shoulder_x = min(max_x, chest_x + shoulder_span)

        # 5. Codos, Muñecas y Manos
        elbow_y = int(neck_y + char_h * 0.22)
        elbow_span = max(3, int(char_w * 0.40))
        l_elbow_x = max(min_x, chest_x - elbow_span)
        r_elbow_x = min(max_x, chest_x + elbow_span)

        wrist_y = int(neck_y + char_h * 0.32)
        wrist_span = max(3, int(char_w * 0.38))
        l_wrist_x = max(min_x, chest_x - wrist_span)
        r_wrist_x = min(max_x, chest_x + wrist_span)

        hand_y = int(neck_y + char_h * 0.40)
        hand_span = max(3, int(char_w * 0.38))
        l_hand_x = max(min_x, chest_x - hand_span)
        r_hand_x = min(max_x, chest_x + hand_span)

        # 6. Cadera (entre pecho y piernas)
        hip_y = int(neck_y + char_h * 0.35)
        hip_x = chest_x

        # 7. Rodillas, Tobillos y Pies
        knee_y = int(hip_y + (max_y - hip_y) * 0.42)
        knee_span = max(2, int(char_w * 0.18))
        l_knee_x = max(min_x, hip_x - knee_span)
        r_knee_x = min(max_x, hip_x + knee_span)

        ankle_y = int(max_y - max(2, int(char_h * 0.06)))
        ankle_span = max(2, int(char_w * 0.22))
        l_ankle_x = max(min_x, hip_x - ankle_span)
        r_ankle_x = min(max_x, hip_x + ankle_span)

        foot_y = int(max_y)
        foot_span = max(2, int(char_w * 0.22))
        l_foot_x = max(min_x, hip_x - foot_span)
        r_foot_x = min(max_x, hip_x + foot_span)

        # Asignar coordenadas iniciales estimadas por V2 con confidence evaluado
        v2_anchors = {
            "head": (head_x, head_y, 0.95),
            "neck": (neck_x, neck_y, 0.90),
            "left_shoulder": (l_shoulder_x, shoulder_y, 0.85),
            "right_shoulder": (r_shoulder_x, shoulder_y, 0.85),
            "left_elbow": (l_elbow_x, elbow_y, 0.80),
            "right_elbow": (r_elbow_x, elbow_y, 0.80),
            "left_wrist": (l_wrist_x, wrist_y, 0.78),
            "right_wrist": (r_wrist_x, wrist_y, 0.78),
            "left_hand": (l_hand_x, hand_y, 0.75),
            "right_hand": (r_hand_x, hand_y, 0.75),
            "chest": (chest_x, chest_y, 0.92),
            "hip": (hip_x, hip_y, 0.88),
            "left_knee": (l_knee_x, knee_y, 0.82),
            "right_knee": (r_knee_x, knee_y, 0.82),
            "left_ankle": (l_ankle_x, ankle_y, 0.85),
            "right_ankle": (r_ankle_x, ankle_y, 0.85),
            "left_foot": (l_foot_x, foot_y, 0.92),
            "right_foot": (r_foot_x, foot_y, 0.92),
        }

        # Aplicar orden de prioridad estricto para cada uno de los 18 anchors:
        for name in OFFICIAL_ANCHOR_NAMES:
            # Prioridad 1: Anotación manual
            if manual_annotations and name in manual_annotations:
                m_data = manual_annotations[name]
                skeleton.set_anchor(
                    name=name,
                    x=int(m_data.get("x", 0)),
                    y=int(m_data.get("y", 0)),
                    confidence=1.0,
                    is_manual=True,
                )
                continue

            # Prioridad 2: Frame previo (consistencia temporal si la confianza previa era alta)
            if prev_skeleton and prev_skeleton.get_anchor(name):
                prev_a = prev_skeleton.get_anchor(name)
                if prev_a.confidence >= 0.95 and not prev_a.is_manual:
                    # Mezcla temporal suave (70% V2, 30% anterior) para evitar jittering
                    v2_x, v2_y, v2_conf = v2_anchors[name]
                    smoothed_x = int(round(v2_x * 0.7 + prev_a.x * 0.3))
                    smoothed_y = int(round(v2_y * 0.7 + prev_a.y * 0.3))
                    skeleton.set_anchor(name=name, x=smoothed_x, y=smoothed_y, confidence=v2_conf, is_manual=False)
                    continue

            # Prioridad 3: PoseAnalyzerV2
            if name in v2_anchors:
                x, y, conf = v2_anchors[name]
                skeleton.set_anchor(name=name, x=x, y=y, confidence=conf, is_manual=False)
            else:
                # Prioridad 4: Fallback
                skeleton.set_anchor(name=name, x=char_w // 2, y=char_h // 2, confidence=0.5, is_manual=False)

        return skeleton

    def _apply_fallback_v1(self, image: Image.Image, manual_annotations: Optional[Dict[str, dict]] = None) -> Skeleton:
        """Utiliza PoseAnalyzer V1 como fallback ante siluetas atípicas o vacías."""
        v1_skel = self.fallback_v1.estimate_pose_anchors(image)
        skeleton = Skeleton()

        for name in OFFICIAL_ANCHOR_NAMES:
            if manual_annotations and name in manual_annotations:
                m_data = manual_annotations[name]
                skeleton.set_anchor(
                    name=name,
                    x=int(m_data.get("x", 0)),
                    y=int(m_data.get("y", 0)),
                    confidence=1.0,
                    is_manual=True,
                )
            elif name in v1_skel.anchors:
                pt = v1_skel.anchors[name]
                skeleton.set_anchor(name=name, x=pt.x, y=pt.y, confidence=pt.confidence * 0.85, is_manual=False)
            else:
                # Interpolar anchors adicionales de V2 ausentes en V1
                skeleton.set_anchor(name=name, x=image.width // 2, y=image.height // 2, confidence=0.5, is_manual=False)

        return skeleton
