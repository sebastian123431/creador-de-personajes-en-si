import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from core.guard import SourceDatasetGuard
from core.layer_resolver import LayerResolver
from models.articulated_motion_template import (
    ArticulatedMotionTemplate,
    ArticulatedFrameTemplate,
    PartMotion,
)
from models.skeleton import Skeleton, Anchor, OFFICIAL_ANCHOR_NAMES

logger = logging.getLogger("SpriteStudio.TemplateExtractorV2")


def _normalize_angle_deg(deg: float) -> float:
    """Normaliza un ángulo en grados al rango [-180, 180)."""
    return ((deg + 180.0) % 360.0) - 180.0


def _calculate_segment_angle_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Calcula el ángulo en grados de la dirección desde el punto A hacia el punto B."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.degrees(math.atan2(dy, dx))


def _filter_outliers_mad(values: List[float], threshold: float = 3.5, min_deviation: float = 3.0) -> Tuple[List[float], int]:
    """
    Filtra anomalías estadísticas (outliers) utilizando Desviación Absoluta de la Mediana (MAD).
    Si mad es cercano a 0 (la mayoría de personajes aprobados tienen el mismo valor exacto),
    filtra muestras con desviación mayor a min_deviation.
    Retorna la lista de valores válidos (inliers) y el conteo de outliers descartados.
    """
    if len(values) <= 2:
        return values, 0

    arr = np.array(values, dtype=float)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))

    if mad < 1e-5:
        deviations = np.abs(arr - med)
        inlier_mask = deviations <= min_deviation
        inliers = arr[inlier_mask].tolist()
        outlier_count = int(np.sum(~inlier_mask))
        if not inliers:
            return values, 0
        return inliers, outlier_count

    # Modified Z-score de Boris Iglewicz y David Hoaglin
    mod_z_scores = 0.6745 * np.abs(arr - med) / mad
    inlier_mask = mod_z_scores <= threshold

    inliers = arr[inlier_mask].tolist()
    outlier_count = int(np.sum(~inlier_mask))

    if not inliers:
        return values, 0

    return inliers, outlier_count


class TemplateExtractorV2:
    """
    Extractor cinemático de plantillas de movimiento articulado V2 a partir de personajes aprobados.
    Aprende las trayectorias angulares (Delta theta) y desplazamientos (Delta x, Delta y)
    de cada articulación a lo largo de los 4 frames del ciclo de animación.
    
    Aplica filtrado de anomalías mediante MAD (Median Absolute Deviation) para garantizar
    que personajes con proporciones atípicas no contaminen la plantilla dorada.
    """

    def __init__(self, base_templates_dir: Optional[Path] = None, guard: Optional[SourceDatasetGuard] = None):
        self.base_dir = Path(base_templates_dir or "dataset/templates_v2").resolve()
        self.guard = guard

    def extract_motion_from_sequence(self, skeletons: List[Skeleton]) -> List[Dict[str, Any]]:
        """
        Extrae los desplazamientos y rotaciones relativas para una secuencia de frames (típicamente 4).
        El Frame 1 se toma como referencia de pose base (Delta = 0).
        """
        if not skeletons:
            return []

        base_skel = skeletons[0]
        base_pelvis = self._get_anchor_pos(base_skel, "pelvis", (32.0, 48.0))

        sequence_data = []

        for frame_idx, cur_skel in enumerate(skeletons, start=1):
            cur_pelvis = self._get_anchor_pos(cur_skel, "pelvis", (32.0, 48.0))
            root_dx = cur_pelvis[0] - base_pelvis[0]
            root_dy = cur_pelvis[1] - base_pelvis[1]

            parts_kinematics: Dict[str, Dict[str, float]] = {}

            # 1. Head (rotación y offset relativo al cuello)
            b_neck = self._get_anchor_pos(base_skel, "neck", (32.0, 24.0))
            b_head = self._get_anchor_pos(base_skel, "head", (32.0, 16.0))
            c_neck = self._get_anchor_pos(cur_skel, "neck", (32.0, 24.0))
            c_head = self._get_anchor_pos(cur_skel, "head", (32.0, 16.0))

            base_head_ang = _calculate_segment_angle_deg(b_neck, b_head)
            cur_head_ang = _calculate_segment_angle_deg(c_neck, c_head)
            d_theta_head = _normalize_angle_deg(cur_head_ang - base_head_ang)

            parts_kinematics["head"] = {
                "dx": (c_head[0] - c_neck[0]) - (b_head[0] - b_neck[0]),
                "dy": (c_head[1] - c_neck[1]) - (b_head[1] - b_neck[1]),
                "angle_deg": d_theta_head,
            }

            # 2. Torso (eje pelvis -> neck)
            base_torso_ang = _calculate_segment_angle_deg(base_pelvis, b_neck)
            cur_torso_ang = _calculate_segment_angle_deg(cur_pelvis, c_neck)
            d_theta_torso = _normalize_angle_deg(cur_torso_ang - base_torso_ang)

            parts_kinematics["torso"] = {
                "dx": root_dx,
                "dy": root_dy,
                "angle_deg": d_theta_torso,
            }

            # 3. Brazo izquierdo (left_shoulder -> left_hand)
            b_ls = self._get_anchor_pos(base_skel, "left_shoulder", (20.0, 28.0))
            b_lh = self._get_anchor_pos(base_skel, "left_hand", (16.0, 48.0))
            c_ls = self._get_anchor_pos(cur_skel, "left_shoulder", (20.0, 28.0))
            c_lh = self._get_anchor_pos(cur_skel, "left_hand", (16.0, 48.0))

            base_la_ang = _calculate_segment_angle_deg(b_ls, b_lh)
            cur_la_ang = _calculate_segment_angle_deg(c_ls, c_lh)
            d_theta_la = _normalize_angle_deg(cur_la_ang - base_la_ang)

            parts_kinematics["left_arm"] = {
                "dx": (c_lh[0] - c_ls[0]) - (b_lh[0] - b_ls[0]),
                "dy": (c_lh[1] - c_ls[1]) - (b_lh[1] - b_ls[1]),
                "angle_deg": d_theta_la,
            }

            # 4. Brazo derecho (right_shoulder -> right_hand)
            b_rs = self._get_anchor_pos(base_skel, "right_shoulder", (44.0, 28.0))
            b_rh = self._get_anchor_pos(base_skel, "right_hand", (48.0, 48.0))
            c_rs = self._get_anchor_pos(cur_skel, "right_shoulder", (44.0, 28.0))
            c_rh = self._get_anchor_pos(cur_skel, "right_hand", (48.0, 48.0))

            base_ra_ang = _calculate_segment_angle_deg(b_rs, b_rh)
            cur_ra_ang = _calculate_segment_angle_deg(c_rs, c_rh)
            d_theta_ra = _normalize_angle_deg(cur_ra_ang - base_ra_ang)

            parts_kinematics["right_arm"] = {
                "dx": (c_rh[0] - c_rs[0]) - (b_rh[0] - b_rs[0]),
                "dy": (c_rh[1] - c_rs[1]) - (b_rh[1] - b_rs[1]),
                "angle_deg": d_theta_ra,
            }

            # 5. Pierna izquierda (left_hip -> left_foot)
            b_lhip = self._get_anchor_pos(base_skel, "left_hip", (24.0, 48.0))
            b_lfoot = self._get_anchor_pos(base_skel, "left_foot", (24.0, 72.0))
            c_lhip = self._get_anchor_pos(cur_skel, "left_hip", (24.0, 48.0))
            c_lfoot = self._get_anchor_pos(cur_skel, "left_foot", (24.0, 72.0))

            base_lleg_ang = _calculate_segment_angle_deg(b_lhip, b_lfoot)
            cur_lleg_ang = _calculate_segment_angle_deg(c_lhip, c_lfoot)
            d_theta_lleg = _normalize_angle_deg(cur_lleg_ang - base_lleg_ang)

            parts_kinematics["left_leg"] = {
                "dx": (c_lfoot[0] - c_lhip[0]) - (b_lfoot[0] - b_lhip[0]),
                "dy": (c_lfoot[1] - c_lhip[1]) - (b_lfoot[1] - b_lhip[1]),
                "angle_deg": d_theta_lleg,
            }

            # 6. Pierna derecha (right_hip -> right_foot)
            b_rhip = self._get_anchor_pos(base_skel, "right_hip", (40.0, 48.0))
            b_rfoot = self._get_anchor_pos(base_skel, "right_foot", (40.0, 72.0))
            c_rhip = self._get_anchor_pos(cur_skel, "right_hip", (40.0, 48.0))
            c_rfoot = self._get_anchor_pos(cur_skel, "right_foot", (40.0, 72.0))

            base_rleg_ang = _calculate_segment_angle_deg(b_rhip, b_rfoot)
            cur_rleg_ang = _calculate_segment_angle_deg(c_rhip, c_rfoot)
            d_theta_rleg = _normalize_angle_deg(cur_rleg_ang - base_rleg_ang)

            parts_kinematics["right_leg"] = {
                "dx": (c_rfoot[0] - c_rhip[0]) - (b_rfoot[0] - b_rhip[0]),
                "dy": (c_rfoot[1] - c_rhip[1]) - (b_rfoot[1] - b_rhip[1]),
                "angle_deg": d_theta_rleg,
            }

            # Anchors relativos al pelvis del frame 1
            anchors_rel: Dict[str, Dict[str, float]] = {}
            for name in OFFICIAL_ANCHOR_NAMES:
                a_pos = self._get_anchor_pos(cur_skel, name, base_pelvis)
                b_pos = self._get_anchor_pos(base_skel, name, base_pelvis)
                anchors_rel[name] = {
                    "dx": a_pos[0] - b_pos[0],
                    "dy": a_pos[1] - b_pos[1],
                }

            sequence_data.append({
                "frame_index": frame_idx,
                "root_dx": root_dx,
                "root_dy": root_dy,
                "parts": parts_kinematics,
                "anchors_rel": anchors_rel,
            })

        return sequence_data

    def build_articulated_template(
        self,
        animation_name: str,
        character_sequences: List[List[Skeleton]]
    ) -> ArticulatedMotionTemplate:
        """
        Agrega cinemáticamente las secuencias de múltiples personajes para una animación,
        aplicando filtrado de anomalías (MAD) en cada variable articular.
        """
        if not character_sequences:
            return ArticulatedMotionTemplate(animation_name=animation_name)

        frame_count = max(len(seq) for seq in character_sequences)
        total_outliers = 0
        samples_used = len(character_sequences)

        # Extraer cinemática de cada personaje
        extracted_samples = [self.extract_motion_from_sequence(seq) for seq in character_sequences if len(seq) == frame_count]
        if not extracted_samples:
            return ArticulatedMotionTemplate(animation_name=animation_name)

        aggregated_frames: List[ArticulatedFrameTemplate] = []

        part_names = ["head", "torso", "left_arm", "right_arm", "left_leg", "right_leg"]

        for f_idx in range(frame_count):
            frame_num = f_idx + 1

            # Recolectar root_dx y root_dy de todas las muestras
            root_dx_vals = [sample[f_idx]["root_dx"] for sample in extracted_samples]
            root_dy_vals = [sample[f_idx]["root_dy"] for sample in extracted_samples]

            clean_rdx, o1 = _filter_outliers_mad(root_dx_vals)
            clean_rdy, o2 = _filter_outliers_mad(root_dy_vals)
            total_outliers += (o1 + o2)

            agg_root_dx = float(np.median(clean_rdx))
            agg_root_dy = float(np.median(clean_rdy))

            # Recolectar datos por cada parte corporal
            agg_parts: Dict[str, PartMotion] = {}
            for pname in part_names:
                dx_vals = [sample[f_idx]["parts"][pname]["dx"] for sample in extracted_samples]
                dy_vals = [sample[f_idx]["parts"][pname]["dy"] for sample in extracted_samples]
                ang_vals = [sample[f_idx]["parts"][pname]["angle_deg"] for sample in extracted_samples]

                c_dx, o_dx = _filter_outliers_mad(dx_vals)
                c_dy, o_dy = _filter_outliers_mad(dy_vals)
                c_ang, o_ang = _filter_outliers_mad(ang_vals)
                total_outliers += (o_dx + o_dy + o_ang)

                agg_parts[pname] = PartMotion(
                    dx=float(np.median(c_dx)),
                    dy=float(np.median(c_dy)),
                    angle_deg=float(np.median(c_ang)),
                    scale=1.0,
                    confidence=1.0
                )

            # Anchors relativos agregados
            agg_anchors: Dict[str, Dict[str, float]] = {}
            for aname in OFFICIAL_ANCHOR_NAMES:
                adx_vals = [sample[f_idx]["anchors_rel"].get(aname, {}).get("dx", 0.0) for sample in extracted_samples]
                ady_vals = [sample[f_idx]["anchors_rel"].get(aname, {}).get("dy", 0.0) for sample in extracted_samples]

                c_adx, o_adx = _filter_outliers_mad(adx_vals)
                c_ady, o_ady = _filter_outliers_mad(ady_vals)
                total_outliers += (o_adx + o_ady)

                agg_anchors[aname] = {
                    "dx": round(float(np.median(c_adx)), 2),
                    "dy": round(float(np.median(c_ady)), 2),
                }

            agg_frame = ArticulatedFrameTemplate(
                frame_index=frame_num,
                root_dx=agg_root_dx,
                root_dy=agg_root_dy,
                parts=agg_parts,
                anchors_rel=agg_anchors,
            )
            aggregated_frames.append(agg_frame)

        template = ArticulatedMotionTemplate(
            animation_name=animation_name,
            frame_count=frame_count,
            samples_used=samples_used,
            outliers_detected=total_outliers,
            frames=aggregated_frames,
        )
        return template

    def save_template(self, template: ArticulatedMotionTemplate) -> Path:
        """
        Persiste la plantilla en: dataset/templates_v2/<animation_name>.json
        Garantiza que la escritura sea totalmente externa al dataset de origen mediante SourceDatasetGuard.
        """
        target_path = self.base_dir / f"{template.animation_name}.json"
        target_path = target_path.resolve()

        if self.guard:
            self.guard.assert_can_write(target_path, operation_desc="guardado de plantilla de movimiento articulado")

        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = template.to_dict()
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)

        logger.info(f"Plantilla de movimiento articulado guardada en: {target_path}")
        return target_path

    def load_template(self, animation_name: str) -> Optional[ArticulatedMotionTemplate]:
        """Carga una plantilla existente desde disco."""
        target_path = self.base_dir / f"{animation_name}.json"
        if not target_path.exists():
            return None

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ArticulatedMotionTemplate.from_dict(data)
        except Exception as e:
            logger.error(f"Error cargando plantilla desde {target_path}: {e}")
            return None

    def _get_anchor_pos(self, skel: Skeleton, name: str, default: Tuple[float, float]) -> Tuple[float, float]:
        a = skel.get_anchor(name)
        if a is not None:
            return float(a.x), float(a.y)
        return default
