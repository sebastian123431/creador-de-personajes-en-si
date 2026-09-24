"""
Módulo de métricas de validación V2 (ValidationMetricsV2).
Evalúa de manera desacoplada y diagnóstica:
1. motion_activity_score
2. head_identity_score
3. palette_integrity_score
4. baseline_stability_score
5. anchor_continuity_score
6. limb_continuity_score
7. silhouette_consistency_score
8. diagnostic_identity_score
"""

import math
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from PIL import Image

from models.skeleton import Skeleton, Anchor, OFFICIAL_ANCHOR_NAMES
from models.validation_metrics import ValidationMetrics, ValidationResult


class ValidationMetricsV2:
    """
    Motor de evaluación diagnóstica de animaciones generadas.
    Calcula métricas desacopladas y emite advertencias técnicas.
    """

    def __init__(
        self,
        max_anchor_jump_px: float = 30.0,
        max_baseline_variance_px: float = 4.0,
        max_unexpected_color_ratio: float = 0.05,
        min_diagnostic_score: float = 0.70,
    ):
        self.max_anchor_jump_px = max_anchor_jump_px
        self.max_baseline_variance_px = max_baseline_variance_px
        self.max_unexpected_color_ratio = max_unexpected_color_ratio
        self.min_diagnostic_score = min_diagnostic_score

    # --------------------------------------------------------------------------
    # 1. motion_activity_score
    # --------------------------------------------------------------------------
    @staticmethod
    def compute_motion_activity(frames: List[Image.Image]) -> float:
        """
        Mide cuánto movimiento existe entre frames consecutivos.
        NO llamarlo quality score: mide actividad cinemática relativa.
        """
        if not frames or len(frames) < 2:
            return 0.0

        diffs = []
        for i in range(len(frames) - 1):
            arr1 = np.array(frames[i].convert("RGBA"), dtype=np.float32)
            arr2 = np.array(frames[i + 1].convert("RGBA"), dtype=np.float32)

            # Diferencia absoluta sobre canales RGB ponderada por alfa
            alpha_union = np.maximum(arr1[:, :, 3], arr2[:, :, 3]) / 255.0
            if np.sum(alpha_union) == 0:
                diffs.append(0.0)
                continue

            pixel_diff = np.abs(arr1[:, :, :3] - arr2[:, :, :3]).mean(axis=-1) / 255.0
            alpha_diff = np.abs(arr1[:, :, 3] - arr2[:, :, 3]) / 255.0
            frame_diff = float(np.mean(pixel_diff * alpha_union + alpha_diff * 0.5))
            diffs.append(frame_diff)

        avg_diff = float(np.mean(diffs)) if diffs else 0.0
        # Normalizar: una actividad típica de pixel art oscila entre 0.01 y 0.20
        activity_score = min(1.0, avg_diff * 5.0)
        return float(activity_score)

    # --------------------------------------------------------------------------
    # 2. head_identity_score
    # --------------------------------------------------------------------------
    @staticmethod
    def compute_head_identity(
        reference: Image.Image,
        frames: List[Image.Image],
        head_anchors: Optional[List[Tuple[float, float]]] = None,
    ) -> float:
        """
        Compara la región de la cabeza (silueta, posición y preservación)
        entre la imagen de referencia y los frames generados.
        """
        if not frames:
            return 0.0

        ref_arr = np.array(reference.convert("RGBA"))
        h, w = ref_arr.shape[:2]

        # Estimar región de cabeza: tercio superior o área alrededor de head anchor
        head_top = 0
        head_bottom = int(h * 0.45)

        ref_head_alpha = ref_arr[head_top:head_bottom, :, 3] > 0
        ref_head_pixels = int(np.sum(ref_head_alpha))
        if ref_head_pixels == 0:
            return 1.0  # Sin cabeza detectable

        scores = []
        for frame in frames:
            f_arr = np.array(frame.convert("RGBA"))
            f_head_alpha = f_arr[head_top:head_bottom, :, 3] > 0
            f_head_pixels = int(np.sum(f_head_alpha))

            # IoU / Similitud de silueta de cabeza
            intersection = int(np.sum(ref_head_alpha & f_head_alpha))
            union = int(np.sum(ref_head_alpha | f_head_alpha))

            iou = (intersection / union) if union > 0 else 0.0
            scores.append(iou)

        return float(np.clip(np.mean(scores), 0.0, 1.0))

    # --------------------------------------------------------------------------
    # 3. palette_integrity_score
    # --------------------------------------------------------------------------
    def compute_palette_integrity(
        self,
        reference_palette: Set[Tuple[int, int, int, int]],
        frames: List[Image.Image],
        allowed_props_palette: Optional[Set[Tuple[int, int, int, int]]] = None,
    ) -> Tuple[float, List[str]]:
        """
        Detecta colores nuevos inesperados (palette drift).
        """
        warnings = []
        if not frames or not reference_palette:
            return 1.0, warnings

        # Filtrar colores transparentes de la paleta permitida
        valid_palette = {c for c in reference_palette if c[3] > 0}
        if allowed_props_palette:
            valid_palette.update({c for c in allowed_props_palette if c[3] > 0})

        unexpected_pixel_counts = []
        total_opaque_counts = []

        for frame in frames:
            arr = np.array(frame.convert("RGBA"))
            opaque_mask = arr[:, :, 3] > 0
            n_opaque = int(np.sum(opaque_mask))
            if n_opaque == 0:
                continue

            opaque_pixels = arr[opaque_mask]
            # Convertir a tuplas
            tuples = [tuple(p) for p in opaque_pixels]
            unexpected = sum(1 for p in tuples if p not in valid_palette)

            unexpected_pixel_counts.append(unexpected)
            total_opaque_counts.append(n_opaque)

        total_unexpected = sum(unexpected_pixel_counts)
        total_opaque = sum(total_opaque_counts)

        if total_opaque == 0:
            return 1.0, warnings

        drift_ratio = total_unexpected / total_opaque
        integrity_score = max(0.0, 1.0 - (drift_ratio * 3.0))

        # Alerta si cualquier frame presenta colores anómalos significativos
        frame_drifts = [
            (unexp / opaq)
            for unexp, opaq in zip(unexpected_pixel_counts, total_opaque_counts)
            if opaq > 0
        ]
        if any(d > self.max_unexpected_color_ratio for d in frame_drifts):
            warnings.append("PALETTE_DRIFT")

        return float(integrity_score), warnings

    # --------------------------------------------------------------------------
    # 4. baseline_stability_score
    # --------------------------------------------------------------------------
    def compute_baseline_stability(
        self,
        frames: List[Image.Image],
        skeletons: Optional[List[Skeleton]] = None,
    ) -> Tuple[float, List[str]]:
        """
        Mide la estabilidad de los pies / baseline a lo largo de la secuencia.
        """
        warnings = []
        if not frames:
            return 1.0, warnings

        baselines = []
        for frame in frames:
            arr = np.array(frame.convert("RGBA"))
            alpha = arr[:, :, 3]
            rows_with_pixels = np.where(alpha > 0)[0]
            if len(rows_with_pixels) > 0:
                baselines.append(float(np.max(rows_with_pixels)))
            else:
                baselines.append(float(frame.height))

        if len(baselines) < 2:
            return 1.0, warnings

        std_dev = float(np.std(baselines))
        max_jump = float(np.max(np.abs(np.diff(baselines))))

        if std_dev > self.max_baseline_variance_px or max_jump > (self.max_baseline_variance_px * 2.0):
            warnings.append("BASELINE_INSTABILITY")

        # Penalización gradual
        stability_score = max(0.0, 1.0 - (std_dev / (self.max_baseline_variance_px * 3.0)))
        return float(stability_score), warnings

    # --------------------------------------------------------------------------
    # 5. anchor_continuity_score
    # --------------------------------------------------------------------------
    def compute_anchor_continuity(
        self,
        skeletons: List[Skeleton],
    ) -> Tuple[float, List[str]]:
        """
        Detecta saltos abruptos en anchors entre frames adyacentes
        (ej. 50 -> 52 -> 110 -> 53).
        """
        warnings = []
        if not skeletons or len(skeletons) < 2:
            return 1.0, warnings

        has_jump = False
        penalties = []

        first_anchors = skeletons[0].anchors
        if isinstance(first_anchors, dict):
            anchor_names = list(first_anchors.keys())
        else:
            anchor_names = [a.name for a in first_anchors]

        for name in anchor_names:
            coords = []
            for sk in skeletons:
                anc = sk.get_anchor(name)
                if anc:
                    coords.append((anc.x, anc.y))
                else:
                    coords.append(None)

            for i in range(len(coords) - 1):
                p1, p2 = coords[i], coords[i + 1]
                if p1 is not None and p2 is not None:
                    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                    if dist > self.max_anchor_jump_px:
                        has_jump = True
                        penalties.append((dist - self.max_anchor_jump_px) / self.max_anchor_jump_px)

        if has_jump:
            warnings.append("ANCHOR_JUMP")

        avg_penalty = float(np.mean(penalties)) if penalties else 0.0
        score = max(0.0, 1.0 - min(1.0, avg_penalty * 0.5))
        return float(score), warnings

    # --------------------------------------------------------------------------
    # 6. limb_continuity_score
    # --------------------------------------------------------------------------
    @staticmethod
    def compute_limb_continuity(skeletons: List[Skeleton]) -> float:
        """
        Mide la continuidad y conservación de longitudes de extremidades:
        upper_arm, forearm, thigh, lower_leg.
        """
        if not skeletons or len(skeletons) < 2:
            return 1.0

        limb_pairs = [
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_hand"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_hand"),
            ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
        ]

        variances = []
        for p1_name, p2_name in limb_pairs:
            lengths = []
            for sk in skeletons:
                a1 = sk.get_anchor(p1_name)
                a2 = sk.get_anchor(p2_name)
                if a1 and a2:
                    lengths.append(math.hypot(a2.x - a1.x, a2.y - a1.y))
            if len(lengths) >= 2 and np.mean(lengths) > 0:
                cv = float(np.std(lengths) / (np.mean(lengths) + 1e-5))
                variances.append(cv)

        if not variances:
            return 1.0

        mean_cv = float(np.mean(variances))
        score = max(0.0, 1.0 - min(1.0, mean_cv * 2.0))
        return float(score)

    # --------------------------------------------------------------------------
    # 7. silhouette_consistency_score
    # --------------------------------------------------------------------------
    @staticmethod
    def compute_silhouette_consistency(
        reference: Image.Image,
        frames: List[Image.Image],
    ) -> Tuple[float, List[str]]:
        """
        Detecta escalado extraño, deformación corporal o variaciones extremas
        de área alfa.
        """
        warnings = []
        if not frames:
            return 1.0, warnings

        ref_arr = np.array(reference.convert("RGBA"))
        ref_area = int(np.sum(ref_arr[:, :, 3] > 0))
        if ref_area == 0:
            return 1.0, warnings

        area_ratios = []
        for frame in frames:
            f_arr = np.array(frame.convert("RGBA"))
            f_area = int(np.sum(f_arr[:, :, 3] > 0))
            ratio = f_area / ref_area
            area_ratios.append(ratio)

            if ratio < 0.60 or ratio > 1.50:
                if "SILHOUETTE_ANOMALY" not in warnings:
                    warnings.append("SILHOUETTE_ANOMALY")

        # Evaluar desviación respecto a 1.0
        avg_deviation = float(np.mean([abs(r - 1.0) for r in area_ratios]))
        score = max(0.0, 1.0 - min(1.0, avg_deviation * 2.0))
        return float(score), warnings

    # --------------------------------------------------------------------------
    # 8. diagnostic_identity_score
    # --------------------------------------------------------------------------
    @staticmethod
    def compute_diagnostic_identity(
        head_score: float,
        palette_score: float,
        silhouette_score: float,
        limb_score: float,
    ) -> float:
        """
        Combina de forma diagnóstica y ponderada la fidelidad de identidad:
        35% cabeza, 30% paleta, 20% silueta, 15% proporciones de extremidades.
        """
        score = (
            0.35 * head_score
            + 0.30 * palette_score
            + 0.20 * silhouette_score
            + 0.15 * limb_score
        )
        return float(np.clip(score, 0.0, 1.0))

    # --------------------------------------------------------------------------
    # EVALUACIÓN INTEGRAL
    # --------------------------------------------------------------------------
    def evaluate_animation(
        self,
        character_id: str,
        variant: str,
        animation: str,
        template_version: str,
        reference_sprite: Image.Image,
        generated_frames: List[Image.Image],
        skeletons: Optional[List[Skeleton]] = None,
        reference_palette: Optional[Set[Tuple[int, int, int, int]]] = None,
        allowed_props_palette: Optional[Set[Tuple[int, int, int, int]]] = None,
    ) -> ValidationResult:
        """
        Ejecuta la batería diagnóstica completa sobre una animación generada.
        """
        warnings: List[str] = []

        # 1. Motion Activity
        motion_act = self.compute_motion_activity(generated_frames)

        # 2. Head Identity
        head_ident = self.compute_head_identity(reference_sprite, generated_frames)

        # 3. Palette Integrity
        if reference_palette is None:
            # Extraer paleta del sprite de referencia
            ref_rgba = reference_sprite.convert("RGBA")
            reference_palette = {p for _, p in ref_rgba.getcolors(maxcolors=65536) or []}

        palette_int, pal_warns = self.compute_palette_integrity(
            reference_palette, generated_frames, allowed_props_palette
        )
        warnings.extend(pal_warns)

        # 4. Baseline Stability
        base_stab, base_warns = self.compute_baseline_stability(generated_frames, skeletons)
        warnings.extend(base_warns)

        # 5. Anchor Continuity
        if skeletons:
            anc_cont, anc_warns = self.compute_anchor_continuity(skeletons)
            warnings.extend(anc_warns)
        else:
            anc_cont = 1.0

        # 6. Limb Continuity
        if skeletons:
            limb_cont = self.compute_limb_continuity(skeletons)
        else:
            limb_cont = 1.0

        # 7. Silhouette Consistency
        sil_cons, sil_warns = self.compute_silhouette_consistency(reference_sprite, generated_frames)
        warnings.extend(sil_warns)

        # 8. Diagnostic Identity Score
        diag_id = self.compute_diagnostic_identity(
            head_score=head_ident,
            palette_score=palette_int,
            silhouette_score=sil_cons,
            limb_score=limb_cont,
        )

        if diag_id < self.min_diagnostic_score:
            warnings.append("IDENTITY_WARNING")

        # Determinar si requiere revisión humana
        review_required = bool(warnings) or (diag_id < self.min_diagnostic_score)

        metrics = ValidationMetrics(
            motion_activity_score=motion_act,
            head_identity_score=head_ident,
            palette_integrity_score=palette_int,
            baseline_stability_score=base_stab,
            anchor_continuity_score=anc_cont,
            limb_continuity_score=limb_cont,
            silhouette_consistency_score=sil_cons,
            diagnostic_identity_score=diag_id,
        )

        return ValidationResult(
            character_id=character_id,
            variant=variant,
            animation=animation,
            template_version=template_version,
            metrics=metrics,
            warnings=list(dict.fromkeys(warnings)),  # Deduplicar
            review_required=review_required,
        )
