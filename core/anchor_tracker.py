import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from core.bbox_detector import BoundingBoxDetector
from core.pose_analyzer_v2 import PoseAnalyzerV2
from models.frame import Frame
from models.skeleton import Skeleton, Anchor, OFFICIAL_ANCHOR_NAMES

logger = logging.getLogger("SpriteStudio.AnchorTracker")


class AnchorTracker:
    """
    Rastreador cinemático temporal coordinado entre los 4 frames de una animación.
    Evita la detección aislada e inconexa entre frames contiguos, utilizando:
    - Búsqueda en vecindad local basada en la posición del frame anterior
    - Restricciones de distancia máxima anatómica por paso de tiempo (dt)
    - Ponderación de consistencia temporal
    - Cierre de ciclo suave (frame 4 -> frame 1)
    """

    def __init__(self, pose_analyzer: Optional[PoseAnalyzerV2] = None, max_drift_px: int = 14):
        self.pose_analyzer = pose_analyzer or PoseAnalyzerV2()
        self.max_drift_px = max_drift_px

    def track_animation_anchors(
        self,
        frames: List[Union[Frame, Path, str]],
        orientation: str = "down",
        manual_annotations_by_frame: Optional[Dict[int, Dict[str, dict]]] = None
    ) -> List[Skeleton]:
        """
        Analiza conjuntamente la secuencia de frames (típicamente 4) y retorna
        la lista de Skeletons con continuidad temporal asegurada.
        """
        skeletons: List[Skeleton] = []
        if not frames:
            return skeletons

        manual_dict = manual_annotations_by_frame or {}

        # 1. Analizar Frame 1 como ancla temporal base
        f0_path = frames[0].image_path if isinstance(frames[0], Frame) else Path(frames[0])
        f0_manual = manual_dict.get(1)
        base_skeleton = self.pose_analyzer.analyze_pose(
            f0_path,
            orientation=orientation,
            manual_annotations=f0_manual
        )
        skeletons.append(base_skeleton)

        # 2. Rastreo progresivo para Frames 2, 3, 4 guiado por el frame precedente
        for idx in range(1, len(frames)):
            frame_num = idx + 1
            f_path = frames[idx].image_path if isinstance(frames[idx], Frame) else Path(frames[idx])
            cur_manual = manual_dict.get(frame_num)

            prev_skel = skeletons[idx - 1]

            # Detección V2 asistida por el esqueleto previo
            raw_skel = self.pose_analyzer.analyze_pose(
                f_path,
                orientation=orientation,
                manual_annotations=cur_manual,
                prev_skeleton=prev_skel
            )

            # Refinamiento y control de deriva anatómica (distance constraints)
            refined_skel = Skeleton()
            for name in OFFICIAL_ANCHOR_NAMES:
                cur_a = raw_skel.get_anchor(name)
                prev_a = prev_skel.get_anchor(name)

                if not cur_a:
                    continue

                if cur_a.is_manual:
                    # Anotaciones manuales siempre prioritarias
                    refined_skel.set_anchor(name, cur_a.x, cur_a.y, 1.0, is_manual=True)
                    continue

                if prev_a:
                    dist = np.hypot(cur_a.x - prev_a.x, cur_a.y - prev_a.y)
                    if dist > self.max_drift_px:
                        # Si el salto excede el umbral anatómico entre frames adyacentes,
                        # amortiguar hacia la vecindad del frame anterior
                        scale = self.max_drift_px / dist
                        clamped_x = int(round(prev_a.x + (cur_a.x - prev_a.x) * scale))
                        clamped_y = int(round(prev_a.y + (cur_a.y - prev_a.y) * scale))
                        # Penalizar confianza ante salto brusco
                        penalized_conf = max(0.4, cur_a.confidence * 0.75)
                        refined_skel.set_anchor(name, clamped_x, clamped_y, penalized_conf, is_manual=False)
                        continue

                refined_skel.set_anchor(name, cur_a.x, cur_a.y, cur_a.confidence, is_manual=False)

            skeletons.append(refined_skel)

        return skeletons
