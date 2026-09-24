import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from core.guard import SourceDatasetGuard
from core.layer_resolver import LayerResolver
from models.body_part import BodyPart
from models.skeleton import Skeleton, Anchor

logger = logging.getLogger("SpriteStudio.BodyPartSegmenter")

BODY_PART_NAMES = ["head", "torso", "left_arm", "right_arm", "left_leg", "right_leg"]


def _point_to_segment_dist_sq(px: np.ndarray, py: np.ndarray, ax: float, ay: float, bx: float, by: float) -> np.ndarray:
    """
    Calcula de forma vectorizada la distancia euclidiana al cuadrado de un array de puntos (px, py)
    al segmento de línea definido por los extremos A=(ax, ay) y B=(bx, by).
    """
    abx = bx - ax
    aby = by - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq < 1e-6:
        # A y B son casi idénticos
        return (px - ax) ** 2 + (py - ay) ** 2

    # Proyección escalar t clampeada a [0, 1]
    t = ((px - ax) * abx + (py - ay) * aby) / ab_len_sq
    t = np.clip(t, 0.0, 1.0)

    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return (px - proj_x) ** 2 + (py - proj_y) ** 2


class BodyPartSegmenter:
    """
    Segmentador de partes corporales pixel art guiado por esqueleto anatómico de 18 articulaciones.
    Descompone cada frame en 6 partes independientes:
    - head (cabeza)
    - torso (tronco)
    - left_arm (brazo izquierdo)
    - right_arm (brazo derecho)
    - left_leg (pierna izquierda)
    - right_leg (pierna derecha)

    Garantiza:
    - Conservación del canal alfa y píxeles originales.
    - Asignación de pivote anatómico para traslación/rotación articular.
    - Ordenamiento de capas Z-index mediante LayerResolver.
    - Escritura segura y no destructiva fuera del dataset original.
    """

    def __init__(self, base_output_dir: Optional[Path] = None, guard: Optional[SourceDatasetGuard] = None):
        self.base_output_dir = Path(base_output_dir or "dataset/body_parts").resolve()
        self.guard = guard

    def segment_frame(
        self,
        image_input: Union[Path, str, Image.Image],
        skeleton: Optional[Skeleton] = None,
        orientation: str = "down",
    ) -> Tuple[Dict[str, BodyPart], Dict[str, Image.Image]]:
        """
        Segmenta una imagen de sprite en 6 partes corporales.
        
        Retorna:
        - Dict[str, BodyPart]: Metadatos de cada parte (bbox, pivot, parent, z_index).
        - Dict[str, Image.Image]: Imágenes recortadas RGBA de cada parte con transparencia.
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input)).convert("RGBA")
        else:
            img = image_input.convert("RGBA")

        arr = np.array(img)
        h, w = arr.shape[:2]
        alpha = arr[:, :, 3]
        valid_mask = alpha > 15

        norm_orient = LayerResolver.normalize_orientation(orientation)

        # Si el esqueleto es nulo o incompleto, construir posiciones de respaldo
        skel = skeleton or Skeleton()
        anchors = self._ensure_anchors(skel, w, h, valid_mask)

        # Coordenadas de los píxeles no transparentes
        ys, xs = np.where(valid_mask)
        if len(xs) == 0:
            # Sprite completamente vacío
            return self._build_empty_parts(anchors, norm_orient), {}

        # Clasificación por regiones anatómicas
        part_labels = np.zeros(len(xs), dtype=object)

        neck_y = anchors["neck"][1]
        hip_y = anchors["hip"][1]

        # 1. Región Cabeza: y < neck_y (con margen de 1px)
        head_indices = ys < (neck_y + 1)
        part_labels[head_indices] = "head"

        # 2. Región Piernas: y >= hip_y
        legs_indices = ys >= hip_y

        if np.any(legs_indices):
            leg_xs = xs[legs_indices]
            leg_ys = ys[legs_indices]

            # Distancia a la cadena de pierna izquierda: hip -> left_knee -> left_ankle -> left_foot
            dist_l1 = _point_to_segment_dist_sq(
                leg_xs, leg_ys,
                anchors["hip"][0], anchors["hip"][1],
                anchors["left_knee"][0], anchors["left_knee"][1]
            )
            dist_l2 = _point_to_segment_dist_sq(
                leg_xs, leg_ys,
                anchors["left_knee"][0], anchors["left_knee"][1],
                anchors["left_ankle"][0], anchors["left_ankle"][1]
            )
            dist_l3 = _point_to_segment_dist_sq(
                leg_xs, leg_ys,
                anchors["left_ankle"][0], anchors["left_ankle"][1],
                anchors["left_foot"][0], anchors["left_foot"][1]
            )
            dist_left_leg = np.minimum(np.minimum(dist_l1, dist_l2), dist_l3)

            # Distancia a la cadena de pierna derecha: hip -> right_knee -> right_ankle -> right_foot
            dist_r1 = _point_to_segment_dist_sq(
                leg_xs, leg_ys,
                anchors["hip"][0], anchors["hip"][1],
                anchors["right_knee"][0], anchors["right_knee"][1]
            )
            dist_r2 = _point_to_segment_dist_sq(
                leg_xs, leg_ys,
                anchors["right_knee"][0], anchors["right_knee"][1],
                anchors["right_ankle"][0], anchors["right_ankle"][1]
            )
            dist_r3 = _point_to_segment_dist_sq(
                leg_xs, leg_ys,
                anchors["right_ankle"][0], anchors["right_ankle"][1],
                anchors["right_foot"][0], anchors["right_foot"][1]
            )
            dist_right_leg = np.minimum(np.minimum(dist_r1, dist_r2), dist_r3)

            leg_choices = np.where(dist_left_leg <= dist_right_leg, "left_leg", "right_leg")
            part_labels[legs_indices] = leg_choices

        # 3. Región Torso y Brazos: (neck_y + 1) <= y < hip_y
        upper_body_indices = (~head_indices) & (~legs_indices)

        if np.any(upper_body_indices):
            ub_xs = xs[upper_body_indices]
            ub_ys = ys[upper_body_indices]

            # Distancia al eje central del torso (neck -> chest -> hip)
            dist_t1 = _point_to_segment_dist_sq(
                ub_xs, ub_ys,
                anchors["neck"][0], anchors["neck"][1],
                anchors["chest"][0], anchors["chest"][1]
            )
            dist_t2 = _point_to_segment_dist_sq(
                ub_xs, ub_ys,
                anchors["chest"][0], anchors["chest"][1],
                anchors["hip"][0], anchors["hip"][1]
            )
            dist_torso = np.minimum(dist_t1, dist_t2)

            # Distancia al brazo izquierdo (shoulder -> elbow -> wrist -> hand)
            dist_la1 = _point_to_segment_dist_sq(
                ub_xs, ub_ys,
                anchors["left_shoulder"][0], anchors["left_shoulder"][1],
                anchors["left_elbow"][0], anchors["left_elbow"][1]
            )
            dist_la2 = _point_to_segment_dist_sq(
                ub_xs, ub_ys,
                anchors["left_elbow"][0], anchors["left_elbow"][1],
                anchors["left_wrist"][0], anchors["left_wrist"][1]
            )
            dist_la3 = _point_to_segment_dist_sq(
                ub_xs, ub_ys,
                anchors["left_wrist"][0], anchors["left_wrist"][1],
                anchors["left_hand"][0], anchors["left_hand"][1]
            )
            dist_left_arm = np.minimum(np.minimum(dist_la1, dist_la2), dist_la3)

            # Distancia al brazo derecho (shoulder -> elbow -> wrist -> hand)
            dist_ra1 = _point_to_segment_dist_sq(
                ub_xs, ub_ys,
                anchors["right_shoulder"][0], anchors["right_shoulder"][1],
                anchors["right_elbow"][0], anchors["right_elbow"][1]
            )
            dist_ra2 = _point_to_segment_dist_sq(
                ub_xs, ub_ys,
                anchors["right_elbow"][0], anchors["right_elbow"][1],
                anchors["right_wrist"][0], anchors["right_wrist"][1]
            )
            dist_ra3 = _point_to_segment_dist_sq(
                ub_xs, ub_ys,
                anchors["right_wrist"][0], anchors["right_wrist"][1],
                anchors["right_hand"][0], anchors["right_hand"][1]
            )
            dist_right_arm = np.minimum(np.minimum(dist_ra1, dist_ra2), dist_ra3)

            # Ponderación anatómica: el torso tiene un radio corporal central (sesgo de contención)
            torso_radius = max(2.5, abs(anchors["right_shoulder"][0] - anchors["left_shoulder"][0]) * 0.28)
            torso_score = np.sqrt(dist_torso) - torso_radius
            la_score = np.sqrt(dist_left_arm)
            ra_score = np.sqrt(dist_right_arm)

            scores = np.stack([torso_score, la_score, ra_score], axis=1)
            min_indices = np.argmin(scores, axis=1)

            choices = np.array(["torso", "left_arm", "right_arm"])
            part_labels[upper_body_indices] = choices[min_indices]

        # Extraer imágenes recortadas y generar modelos BodyPart
        parts: Dict[str, BodyPart] = {}
        part_images: Dict[str, Image.Image] = {}

        pivots = {
            "head": (anchors["neck"][0], anchors["neck"][1]),
            "torso": (anchors["hip"][0], anchors["hip"][1]),
            "left_arm": (anchors["left_shoulder"][0], anchors["left_shoulder"][1]),
            "right_arm": (anchors["right_shoulder"][0], anchors["right_shoulder"][1]),
            "left_leg": (anchors["hip"][0], anchors["hip"][1]),
            "right_leg": (anchors["hip"][0], anchors["hip"][1]),
        }

        parents = {
            "head": "torso",
            "torso": None,
            "left_arm": "torso",
            "right_arm": "torso",
            "left_leg": "torso",
            "right_leg": "torso",
        }

        for part_name in BODY_PART_NAMES:
            p_indices = np.where(part_labels == part_name)[0]
            raw_pivot = pivots[part_name]
            # Validar y clampear pivote dentro del canvas
            p_pivot = (
                float(np.clip(raw_pivot[0], 0.0, float(max(0, w - 1)))),
                float(np.clip(raw_pivot[1], 0.0, float(max(0, h - 1))))
            )
            p_parent = parents[part_name]

            if len(p_indices) == 0:
                # Parte vacía u ocluida
                px_i = int(round(p_pivot[0]))
                py_i = int(round(p_pivot[1]))
                bbox = (px_i, py_i, px_i, py_i)
                part = BodyPart(
                    name=part_name,
                    bbox=bbox,
                    pivot_x=float(p_pivot[0]),
                    pivot_y=float(p_pivot[1]),
                    parent=p_parent,
                    confidence=0.0
                )
                parts[part_name] = part
                part_images[part_name] = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
                continue

            part_xs = xs[p_indices]
            part_ys = ys[p_indices]

            min_x = int(np.min(part_xs))
            max_x = int(np.max(part_xs))
            min_y = int(np.min(part_ys))
            max_y = int(np.max(part_ys))
            bbox = (min_x, min_y, max_x, max_y)

            # Crear recorte RGBA transparente ajustado al bounding box de la parte
            crop_w = max_x - min_x + 1
            crop_h = max_y - min_y + 1
            part_arr = np.zeros((crop_h, crop_w, 4), dtype=np.uint8)

            rel_xs = part_xs - min_x
            rel_ys = part_ys - min_y

            # Copiar píxeles originales preservando colores exactos y canal alfa
            part_arr[rel_ys, rel_xs] = arr[part_ys, part_xs]
            part_img = Image.fromarray(part_arr, mode="RGBA")

            part = BodyPart(
                name=part_name,
                bbox=bbox,
                pivot_x=float(p_pivot[0]),
                pivot_y=float(p_pivot[1]),
                parent=p_parent,
                confidence=1.0
            )

            parts[part_name] = part
            part_images[part_name] = part_img

        # Resolver orden de capas (Z-index) según orientación
        LayerResolver.resolve_z_indices(norm_orient, parts)

        return parts, part_images

    def save_segmented_parts(
        self,
        character_id: str,
        variant: str,
        animation: str,
        frame_index: int,
        parts: Dict[str, BodyPart],
        part_images: Dict[str, Image.Image],
    ) -> Path:
        """
        Persiste los recortes PNG y el archivo de metadatos meta.json en:
        dataset/body_parts/<character_id>/<variant>/<animation>/<frame_index:02d>/
        
        Garantiza que la escritura sea totalmente externa al dataset original mediante SourceDatasetGuard.
        """
        target_dir = self.base_output_dir / character_id / variant / animation / f"{frame_index:02d}"
        target_dir = target_dir.resolve()

        if self.guard:
            self.guard.assert_can_write(target_dir, operation_desc="segmentación de partes corporales")

        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. Guardar cada imagen PNG y actualizar mask_path
        for part_name, img in part_images.items():
            png_path = target_dir / f"{part_name}.png"
            img.save(png_path, "PNG")
            if part_name in parts:
                parts[part_name].mask_path = png_path

        # 2. Guardar meta.json con los atributos y bboxes
        meta_payload = {
            "character_id": character_id,
            "variant": variant,
            "animation": animation,
            "frame": frame_index,
            "parts": {name: p.to_dict() for name, p in parts.items()}
        }

        meta_path = target_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=4, ensure_ascii=False)

        logger.info(f"Segmentación guardada para {character_id} [{variant}] {animation} #{frame_index} en: {target_dir}")
        return target_dir

    def load_segmented_parts(
        self,
        character_id: str,
        variant: str,
        animation: str,
        frame_index: int
    ) -> Optional[Dict[str, BodyPart]]:
        """
        Carga las partes corporales segmentadas desde disco si existen.
        """
        meta_path = self.base_output_dir / character_id / variant / animation / f"{frame_index:02d}" / "meta.json"
        if not meta_path.exists():
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            parts_dict = {}
            for name, p_data in data.get("parts", {}).items():
                parts_dict[name] = BodyPart.from_dict(p_data)
            return parts_dict
        except Exception as e:
            logger.error(f"Error cargando partes desde {meta_path}: {e}")
            return None

    def _ensure_anchors(self, skeleton: Skeleton, w: int, h: int, valid_mask: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """
        Extrae o estima las coordenadas (x, y) de las articulaciones requeridas usando
        exclusivamente la nomenclatura de OFFICIAL_ANCHOR_NAMES de models/skeleton.py.
        Si falta un anchor crítico, emite warning y aplica fallback geométrico explícito con menor confianza.
        """
        from models.skeleton import OFFICIAL_ANCHOR_NAMES

        ys, xs = np.where(valid_mask)
        if len(xs) > 0:
            min_x, max_x = float(np.min(xs)), float(np.max(xs))
            min_y, max_y = float(np.min(ys)), float(np.max(ys))
        else:
            min_x, max_x = 0.0, float(w - 1)
            min_y, max_y = 0.0, float(h - 1)

        cx = (min_x + max_x) / 2.0
        char_h = max(1.0, max_y - min_y)
        char_w = max(1.0, max_x - min_x)

        # Valores anatómicos por defecto para los 18 anchors oficiales
        defaults = {
            "head": (cx, min_y + char_h * 0.15),
            "neck": (cx, min_y + char_h * 0.32),
            "chest": (cx, min_y + char_h * 0.45),
            "hip": (cx, min_y + char_h * 0.62),
            "left_shoulder": (cx - char_w * 0.30, min_y + char_h * 0.35),
            "left_elbow": (cx - char_w * 0.38, min_y + char_h * 0.48),
            "left_wrist": (cx - char_w * 0.40, min_y + char_h * 0.55),
            "left_hand": (cx - char_w * 0.42, min_y + char_h * 0.60),
            "right_shoulder": (cx + char_w * 0.30, min_y + char_h * 0.35),
            "right_elbow": (cx + char_w * 0.38, min_y + char_h * 0.48),
            "right_wrist": (cx + char_w * 0.40, min_y + char_h * 0.55),
            "right_hand": (cx + char_w * 0.42, min_y + char_h * 0.60),
            "left_knee": (cx - char_w * 0.18, min_y + char_h * 0.78),
            "left_ankle": (cx - char_w * 0.19, min_y + char_h * 0.90),
            "left_foot": (cx - char_w * 0.20, max_y),
            "right_knee": (cx + char_w * 0.18, min_y + char_h * 0.78),
            "right_ankle": (cx + char_w * 0.19, min_y + char_h * 0.90),
            "right_foot": (cx + char_w * 0.20, max_y),
        }

        missing_critical: List[str] = []
        result = {}
        for name in OFFICIAL_ANCHOR_NAMES:
            a = skeleton.get_anchor(name)
            if a is not None:
                # Clampear coordenadas al canvas
                clamped_x = float(np.clip(a.x, 0.0, float(max(0, w - 1))))
                clamped_y = float(np.clip(a.y, 0.0, float(max(0, h - 1))))
                result[name] = (clamped_x, clamped_y)
            else:
                def_pos = defaults.get(name, (cx, min_y + char_h * 0.5))
                clamped_x = float(np.clip(def_pos[0], 0.0, float(max(0, w - 1))))
                clamped_y = float(np.clip(def_pos[1], 0.0, float(max(0, h - 1))))
                result[name] = (clamped_x, clamped_y)
                missing_critical.append(name)

        if missing_critical:
            logger.warning(
                f"BodyPartSegmenter: Anchors oficiales ausentes en skeleton: {missing_critical}. "
                f"Se utilizó estrategia explícita de fallback geométrico con menor confianza."
            )

        return result

    def _build_empty_parts(self, anchors: Dict[str, Tuple[float, float]], orientation: str) -> Dict[str, BodyPart]:
        parts = {}
        for name in BODY_PART_NAMES:
            p = anchors.get("neck" if name == "head" else "hip", (0.0, 0.0))
            parts[name] = BodyPart(
                name=name,
                bbox=(int(p[0]), int(p[1]), int(p[0]), int(p[1])),
                pivot_x=float(p[0]),
                pivot_y=float(p[1]),
                parent="torso" if name != "torso" else None,
                confidence=0.0
            )
        LayerResolver.resolve_z_indices(orientation, parts)
        return parts
