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
        base_hip = self._get_anchor_pos(base_skel, "hip", (32.0, 48.0))

        # Estimar dimensiones de silueta a partir de base_skel para normalización relativa
        xs = [a.x for a in base_skel.anchors.values() if a.confidence > 0]
        ys = [a.y for a in base_skel.anchors.values() if a.confidence > 0]
        char_w = float(max(xs) - min(xs)) if xs and max(xs) > min(xs) else 32.0
        char_h = float(max(ys) - min(ys)) if ys and max(ys) > min(ys) else 64.0
        char_w = max(char_w, 1.0)
        char_h = max(char_h, 1.0)

        sequence_data = []

        def _make_part_kinematics(dx: float, dy: float, angle_deg: float) -> Dict[str, float]:
            return {
                "dx": float(dx),
                "dy": float(dy),
                "dx_ratio": float(dx / char_w),
                "dy_ratio": float(dy / char_h),
                "angle_deg": float(angle_deg),
            }

        for frame_idx, cur_skel in enumerate(skeletons, start=1):
            cur_hip = self._get_anchor_pos(cur_skel, "hip", (32.0, 48.0))
            root_dx = cur_hip[0] - base_hip[0]
            root_dy = cur_hip[1] - base_hip[1]

            parts_kinematics: Dict[str, Dict[str, float]] = {}

            # 1. Head (rotación y offset relativo al cuello)
            b_neck = self._get_anchor_pos(base_skel, "neck", (32.0, 24.0))
            b_head = self._get_anchor_pos(base_skel, "head", (32.0, 16.0))
            c_neck = self._get_anchor_pos(cur_skel, "neck", (32.0, 24.0))
            c_head = self._get_anchor_pos(cur_skel, "head", (32.0, 16.0))

            base_head_ang = _calculate_segment_angle_deg(b_neck, b_head)
            cur_head_ang = _calculate_segment_angle_deg(c_neck, c_head)
            d_theta_head = _normalize_angle_deg(cur_head_ang - base_head_ang)

            parts_kinematics["head"] = _make_part_kinematics(
                dx=(c_head[0] - c_neck[0]) - (b_head[0] - b_neck[0]),
                dy=(c_head[1] - c_neck[1]) - (b_head[1] - b_neck[1]),
                angle_deg=d_theta_head,
            )

            # 2. Torso (eje hip -> neck)
            base_torso_ang = _calculate_segment_angle_deg(base_hip, b_neck)
            cur_torso_ang = _calculate_segment_angle_deg(cur_hip, c_neck)
            d_theta_torso = _normalize_angle_deg(cur_torso_ang - base_torso_ang)

            parts_kinematics["torso"] = _make_part_kinematics(
                dx=root_dx,
                dy=root_dy,
                angle_deg=d_theta_torso,
            )

            # 3. Brazo izquierdo completo y subcadenas (shoulder -> elbow -> wrist -> hand)
            b_ls = self._get_anchor_pos(base_skel, "left_shoulder", (20.0, 28.0))
            b_le = self._get_anchor_pos(base_skel, "left_elbow", (18.0, 38.0))
            b_lw = self._get_anchor_pos(base_skel, "left_wrist", (16.0, 44.0))
            b_lh = self._get_anchor_pos(base_skel, "left_hand", (16.0, 48.0))
            c_ls = self._get_anchor_pos(cur_skel, "left_shoulder", (20.0, 28.0))
            c_le = self._get_anchor_pos(cur_skel, "left_elbow", (18.0, 38.0))
            c_lw = self._get_anchor_pos(cur_skel, "left_wrist", (16.0, 44.0))
            c_lh = self._get_anchor_pos(cur_skel, "left_hand", (16.0, 48.0))

            base_la_ang = _calculate_segment_angle_deg(b_ls, b_lh)
            cur_la_ang = _calculate_segment_angle_deg(c_ls, c_lh)
            d_theta_la = _normalize_angle_deg(cur_la_ang - base_la_ang)

            parts_kinematics["left_arm"] = _make_part_kinematics(
                dx=(c_lh[0] - c_ls[0]) - (b_lh[0] - b_ls[0]),
                dy=(c_lh[1] - c_ls[1]) - (b_lh[1] - b_ls[1]),
                angle_deg=d_theta_la,
            )

            # Subpartes anatómicas del brazo izquierdo
            d_theta_l_upper = _normalize_angle_deg(_calculate_segment_angle_deg(c_ls, c_le) - _calculate_segment_angle_deg(b_ls, b_le))
            d_theta_l_fore = _normalize_angle_deg(_calculate_segment_angle_deg(c_le, c_lw) - _calculate_segment_angle_deg(b_le, b_lw))
            d_theta_l_hand = _normalize_angle_deg(_calculate_segment_angle_deg(c_lw, c_lh) - _calculate_segment_angle_deg(b_lw, b_lh))

            parts_kinematics["left_upper_arm"] = _make_part_kinematics(
                dx=(c_le[0] - c_ls[0]) - (b_le[0] - b_ls[0]),
                dy=(c_le[1] - c_ls[1]) - (b_le[1] - b_ls[1]),
                angle_deg=d_theta_l_upper,
            )
            parts_kinematics["left_forearm"] = _make_part_kinematics(
                dx=(c_lw[0] - c_le[0]) - (b_lw[0] - b_le[0]),
                dy=(c_lw[1] - c_le[1]) - (b_lw[1] - b_le[1]),
                angle_deg=d_theta_l_fore,
            )
            parts_kinematics["left_hand"] = _make_part_kinematics(
                dx=(c_lh[0] - c_lw[0]) - (b_lh[0] - b_lw[0]),
                dy=(c_lh[1] - c_lw[1]) - (b_lh[1] - b_lw[1]),
                angle_deg=d_theta_l_hand,
            )

            # 4. Brazo derecho completo y subcadenas (shoulder -> elbow -> wrist -> hand)
            b_rs = self._get_anchor_pos(base_skel, "right_shoulder", (44.0, 28.0))
            b_re = self._get_anchor_pos(base_skel, "right_elbow", (46.0, 38.0))
            b_rw = self._get_anchor_pos(base_skel, "right_wrist", (48.0, 44.0))
            b_rh = self._get_anchor_pos(base_skel, "right_hand", (48.0, 48.0))
            c_rs = self._get_anchor_pos(cur_skel, "right_shoulder", (44.0, 28.0))
            c_re = self._get_anchor_pos(cur_skel, "right_elbow", (46.0, 38.0))
            c_rw = self._get_anchor_pos(cur_skel, "right_wrist", (48.0, 44.0))
            c_rh = self._get_anchor_pos(cur_skel, "right_hand", (48.0, 48.0))

            base_ra_ang = _calculate_segment_angle_deg(b_rs, b_rh)
            cur_ra_ang = _calculate_segment_angle_deg(c_rs, c_rh)
            d_theta_ra = _normalize_angle_deg(cur_ra_ang - base_ra_ang)

            parts_kinematics["right_arm"] = _make_part_kinematics(
                dx=(c_rh[0] - c_rs[0]) - (b_rh[0] - b_rs[0]),
                dy=(c_rh[1] - c_rs[1]) - (b_rh[1] - b_rs[1]),
                angle_deg=d_theta_ra,
            )

            # Subpartes anatómicas del brazo derecho
            d_theta_r_upper = _normalize_angle_deg(_calculate_segment_angle_deg(c_rs, c_re) - _calculate_segment_angle_deg(b_rs, b_re))
            d_theta_r_fore = _normalize_angle_deg(_calculate_segment_angle_deg(c_re, c_rw) - _calculate_segment_angle_deg(b_re, b_rw))
            d_theta_r_hand = _normalize_angle_deg(_calculate_segment_angle_deg(c_rw, c_rh) - _calculate_segment_angle_deg(b_rw, b_rh))

            parts_kinematics["right_upper_arm"] = _make_part_kinematics(
                dx=(c_re[0] - c_rs[0]) - (b_re[0] - b_rs[0]),
                dy=(c_re[1] - c_rs[1]) - (b_re[1] - b_rs[1]),
                angle_deg=d_theta_r_upper,
            )
            parts_kinematics["right_forearm"] = _make_part_kinematics(
                dx=(c_rw[0] - c_re[0]) - (b_rw[0] - b_re[0]),
                dy=(c_rw[1] - c_re[1]) - (b_rw[1] - b_re[1]),
                angle_deg=d_theta_r_fore,
            )
            parts_kinematics["right_hand"] = _make_part_kinematics(
                dx=(c_rh[0] - c_rw[0]) - (b_rh[0] - b_rw[0]),
                dy=(c_rh[1] - c_rw[1]) - (b_rh[1] - b_rw[1]),
                angle_deg=d_theta_r_hand,
            )

            # 5. Pierna izquierda completa y subcadenas (hip -> knee -> ankle -> foot)
            b_lk = self._get_anchor_pos(base_skel, "left_knee", (24.0, 62.0))
            b_la = self._get_anchor_pos(base_skel, "left_ankle", (24.0, 68.0))
            b_lfoot = self._get_anchor_pos(base_skel, "left_foot", (24.0, 72.0))
            c_lk = self._get_anchor_pos(cur_skel, "left_knee", (24.0, 62.0))
            c_la = self._get_anchor_pos(cur_skel, "left_ankle", (24.0, 68.0))
            c_lfoot = self._get_anchor_pos(cur_skel, "left_foot", (24.0, 72.0))

            base_lleg_ang = _calculate_segment_angle_deg(base_hip, b_lfoot)
            cur_lleg_ang = _calculate_segment_angle_deg(cur_hip, c_lfoot)
            d_theta_lleg = _normalize_angle_deg(cur_lleg_ang - base_lleg_ang)

            parts_kinematics["left_leg"] = _make_part_kinematics(
                dx=(c_lfoot[0] - cur_hip[0]) - (b_lfoot[0] - base_hip[0]),
                dy=(c_lfoot[1] - cur_hip[1]) - (b_lfoot[1] - base_hip[1]),
                angle_deg=d_theta_lleg,
            )

            # Subpartes anatómicas pierna izquierda
            d_theta_l_thigh = _normalize_angle_deg(_calculate_segment_angle_deg(cur_hip, c_lk) - _calculate_segment_angle_deg(base_hip, b_lk))
            d_theta_l_lower = _normalize_angle_deg(_calculate_segment_angle_deg(c_lk, c_la) - _calculate_segment_angle_deg(b_lk, b_la))
            d_theta_l_foot = _normalize_angle_deg(_calculate_segment_angle_deg(c_la, c_lfoot) - _calculate_segment_angle_deg(b_la, b_lfoot))

            parts_kinematics["left_thigh"] = _make_part_kinematics(
                dx=(c_lk[0] - cur_hip[0]) - (b_lk[0] - base_hip[0]),
                dy=(c_lk[1] - cur_hip[1]) - (b_lk[1] - base_hip[1]),
                angle_deg=d_theta_l_thigh,
            )
            parts_kinematics["left_lower_leg"] = _make_part_kinematics(
                dx=(c_la[0] - c_lk[0]) - (b_la[0] - b_lk[0]),
                dy=(c_la[1] - c_lk[1]) - (b_la[1] - b_lk[1]),
                angle_deg=d_theta_l_lower,
            )
            parts_kinematics["left_foot"] = _make_part_kinematics(
                dx=(c_lfoot[0] - c_la[0]) - (b_lfoot[0] - b_la[0]),
                dy=(c_lfoot[1] - c_la[1]) - (b_lfoot[1] - b_la[1]),
                angle_deg=d_theta_l_foot,
            )

            # 6. Pierna derecha completa y subcadenas (hip -> knee -> ankle -> foot)
            b_rk = self._get_anchor_pos(base_skel, "right_knee", (40.0, 62.0))
            b_ra = self._get_anchor_pos(base_skel, "right_ankle", (40.0, 68.0))
            b_rfoot = self._get_anchor_pos(base_skel, "right_foot", (40.0, 72.0))
            c_rk = self._get_anchor_pos(cur_skel, "right_knee", (40.0, 62.0))
            c_ra = self._get_anchor_pos(cur_skel, "right_ankle", (40.0, 68.0))
            c_rfoot = self._get_anchor_pos(cur_skel, "right_foot", (40.0, 72.0))

            base_rleg_ang = _calculate_segment_angle_deg(base_hip, b_rfoot)
            cur_rleg_ang = _calculate_segment_angle_deg(cur_hip, c_rfoot)
            d_theta_rleg = _normalize_angle_deg(cur_rleg_ang - base_rleg_ang)

            parts_kinematics["right_leg"] = _make_part_kinematics(
                dx=(c_rfoot[0] - cur_hip[0]) - (b_rfoot[0] - base_hip[0]),
                dy=(c_rfoot[1] - cur_hip[1]) - (b_rfoot[1] - base_hip[1]),
                angle_deg=d_theta_rleg,
            )

            # Subpartes anatómicas pierna derecha
            d_theta_r_thigh = _normalize_angle_deg(_calculate_segment_angle_deg(cur_hip, c_rk) - _calculate_segment_angle_deg(base_hip, b_rk))
            d_theta_r_lower = _normalize_angle_deg(_calculate_segment_angle_deg(c_rk, c_ra) - _calculate_segment_angle_deg(b_rk, b_ra))
            d_theta_r_foot = _normalize_angle_deg(_calculate_segment_angle_deg(c_ra, c_rfoot) - _calculate_segment_angle_deg(b_ra, b_rfoot))

            parts_kinematics["right_thigh"] = _make_part_kinematics(
                dx=(c_rk[0] - cur_hip[0]) - (b_rk[0] - base_hip[0]),
                dy=(c_rk[1] - cur_hip[1]) - (b_rk[1] - base_hip[1]),
                angle_deg=d_theta_r_thigh,
            )
            parts_kinematics["right_lower_leg"] = _make_part_kinematics(
                dx=(c_ra[0] - c_rk[0]) - (b_ra[0] - b_rk[0]),
                dy=(c_ra[1] - c_rk[1]) - (b_ra[1] - b_rk[1]),
                angle_deg=d_theta_r_lower,
            )
            parts_kinematics["right_foot"] = _make_part_kinematics(
                dx=(c_rfoot[0] - c_ra[0]) - (b_rfoot[0] - b_ra[0]),
                dy=(c_rfoot[1] - c_ra[1]) - (b_rfoot[1] - b_ra[1]),
                angle_deg=d_theta_r_foot,
            )

            # Anchors relativos al hip del frame 1
            anchors_rel: Dict[str, Dict[str, float]] = {}
            for name in OFFICIAL_ANCHOR_NAMES:
                a_pos = self._get_anchor_pos(cur_skel, name, base_hip)
                b_pos = self._get_anchor_pos(base_skel, name, base_hip)
                anchors_rel[name] = {
                    "dx": a_pos[0] - b_pos[0],
                    "dy": a_pos[1] - b_pos[1],
                }

            sequence_data.append({
                "frame_index": frame_idx,
                "root_dx": root_dx,
                "root_dy": root_dy,
                "root_dx_ratio": root_dx / char_w,
                "root_dy_ratio": root_dy / char_h,
                "parts": parts_kinematics,
                "anchors_rel": anchors_rel,
                "confidence": cur_skel.average_confidence,
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
        Garantiza la preservación de las 18 partes corporales completas y subcadenas anatómicas,
        así como el filtrado de secuencias con baja confianza (< 0.60 REVIEW_REQUIRED).
        """
        if not character_sequences:
            return ArticulatedMotionTemplate(animation_name=animation_name)

        frame_count = max(len(seq) for seq in character_sequences)
        total_outliers = 0

        # Filtrado de confianza: secuencias con frames de confianza < 0.60
        # se marcan REVIEW_REQUIRED y no contaminan la agregación principal
        clean_sequences = []
        review_sequences = []
        for seq in character_sequences:
            if len(seq) != frame_count:
                continue
            min_conf = min((skel.average_confidence for skel in seq), default=1.0)
            if min_conf < 0.60:
                logger.warning(
                    f"TemplateExtractorV2: Secuencia para '{animation_name}' con baja confianza "
                    f"({min_conf:.2f} < 0.60). Marcada REVIEW_REQUIRED y excluida de agregación primaria."
                )
                review_sequences.append(seq)
            else:
                if min_conf < 0.75:
                    logger.info(
                        f"TemplateExtractorV2: Secuencia para '{animation_name}' utilizada con advertencia "
                        f"(0.60 <= confianza {min_conf:.2f} < 0.75)."
                    )
                clean_sequences.append(seq)

        usable_sequences = clean_sequences if clean_sequences else review_sequences
        samples_used = len(usable_sequences)

        # Extraer cinemática de cada secuencia utilizable
        extracted_samples = [self.extract_motion_from_sequence(seq) for seq in usable_sequences]
        if not extracted_samples:
            return ArticulatedMotionTemplate(animation_name=animation_name)

        aggregated_frames: List[ArticulatedFrameTemplate] = []

        # Las 18 partes anatómicas canónicas y subcadenas completas
        ALL_PART_NAMES = [
            "head", "torso",
            "left_arm", "left_upper_arm", "left_forearm", "left_hand",
            "right_arm", "right_upper_arm", "right_forearm", "right_hand",
            "left_leg", "left_thigh", "left_lower_leg", "left_foot",
            "right_leg", "right_thigh", "right_lower_leg", "right_foot",
        ]

        for f_idx in range(frame_count):
            frame_num = f_idx + 1

            # Recolectar root_dx y root_dy absolutos y relativos de todas las muestras
            root_dx_vals = [sample[f_idx]["root_dx"] for sample in extracted_samples]
            root_dy_vals = [sample[f_idx]["root_dy"] for sample in extracted_samples]
            root_dxr_vals = [sample[f_idx]["root_dx_ratio"] for sample in extracted_samples]
            root_dyr_vals = [sample[f_idx]["root_dy_ratio"] for sample in extracted_samples]

            clean_rdx, o1 = _filter_outliers_mad(root_dx_vals)
            clean_rdy, o2 = _filter_outliers_mad(root_dy_vals)
            clean_rdxr, _ = _filter_outliers_mad(root_dxr_vals)
            clean_rdyr, _ = _filter_outliers_mad(root_dyr_vals)
            total_outliers += (o1 + o2)

            agg_root_dx = float(np.median(clean_rdx))
            agg_root_dy = float(np.median(clean_rdy))
            agg_root_dx_ratio = float(np.median(clean_rdxr))
            agg_root_dy_ratio = float(np.median(clean_rdyr))

            # Recolectar datos por cada una de las 18 partes corporales
            agg_parts: Dict[str, PartMotion] = {}
            for pname in ALL_PART_NAMES:
                dx_vals = [sample[f_idx]["parts"][pname]["dx"] for sample in extracted_samples]
                dy_vals = [sample[f_idx]["parts"][pname]["dy"] for sample in extracted_samples]
                dxr_vals = [sample[f_idx]["parts"][pname]["dx_ratio"] for sample in extracted_samples]
                dyr_vals = [sample[f_idx]["parts"][pname]["dy_ratio"] for sample in extracted_samples]
                ang_vals = [sample[f_idx]["parts"][pname]["angle_deg"] for sample in extracted_samples]

                c_dx, o_dx = _filter_outliers_mad(dx_vals)
                c_dy, o_dy = _filter_outliers_mad(dy_vals)
                c_dxr, _ = _filter_outliers_mad(dxr_vals)
                c_dyr, _ = _filter_outliers_mad(dyr_vals)
                c_ang, o_ang = _filter_outliers_mad(ang_vals)
                total_outliers += (o_dx + o_dy + o_ang)

                # Confianza promedio de frame para esta parte
                frame_confs = [sample[f_idx].get("confidence", 1.0) for sample in extracted_samples]
                avg_part_conf = float(np.mean(frame_confs)) if frame_confs else 1.0

                agg_parts[pname] = PartMotion(
                    dx=float(np.median(c_dx)),
                    dy=float(np.median(c_dy)),
                    dx_ratio=float(np.median(c_dxr)),
                    dy_ratio=float(np.median(c_dyr)),
                    angle_deg=float(np.median(c_ang)),
                    scale=1.0,
                    confidence=round(avg_part_conf, 3),
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
                root_dx_ratio=agg_root_dx_ratio,
                root_dy_ratio=agg_root_dy_ratio,
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

    def save_template(
        self,
        template: ArticulatedMotionTemplate,
        version_tag: Optional[str] = None,
    ) -> Path:
        """
        Persiste la plantilla en:
        1. dataset/templates_v2/<animation_name>.json (base)
        2. dataset/templates_v2/current/<animation_name>.json (referencia activa)
        3. dataset/templates_v2/versions/<version_tag>/<animation_name>.json (versión inmutable si version_tag se provee)
        Garantiza que la escritura sea totalmente externa al dataset de origen mediante SourceDatasetGuard.
        """
        target_path = (self.base_dir / f"{template.animation_name}.json").resolve()

        if self.guard:
            self.guard.assert_can_write(target_path, operation_desc="guardado de plantilla de movimiento articulado")

        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = template.to_dict()
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)

        # Copiar / guardar en current/
        current_dir = self.base_dir / "current"
        if self.guard:
            self.guard.assert_can_write(current_dir, operation_desc="guardado en templates_v2/current")
        current_dir.mkdir(parents=True, exist_ok=True)
        current_path = current_dir / f"{template.animation_name}.json"
        with open(current_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)

        # Guardar en versión inmutable si corresponde
        if version_tag:
            version_dir = self.base_dir / "versions" / version_tag
            if self.guard:
                self.guard.assert_can_write(version_dir, operation_desc="guardado en templates_v2/versions")
            version_dir.mkdir(parents=True, exist_ok=True)
            version_path = version_dir / f"{template.animation_name}.json"
            with open(version_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4, ensure_ascii=False)

        logger.info(f"Plantilla de movimiento articulado guardada en: {target_path}")
        return target_path

    def load_template(self, animation_name: str) -> Optional[ArticulatedMotionTemplate]:
        """Carga una plantilla existente desde disco (buscando primero en current/ y luego en base)."""
        current_path = self.base_dir / "current" / f"{animation_name}.json"
        base_path = self.base_dir / f"{animation_name}.json"

        target_path = current_path if current_path.exists() else base_path
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
        logger.warning(f"TemplateExtractorV2: Anchor '{name}' ausente en Skeleton. Usando fallback explícito: {default}")
        return default
