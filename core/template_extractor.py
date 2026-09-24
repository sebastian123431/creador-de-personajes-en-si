import logging
from typing import List, Tuple
import numpy as np

from core.motion_analyzer import MovementDescriptor
from models.motion_template import MotionTemplate, TemplateFrame

logger = logging.getLogger("SpriteStudio.TemplateExtractor")


class TemplateExtractor:
    """
    Extrae plantillas de movimiento consolidadas a partir de múltiples personajes aprobados.
    NUNCA promedia píxeles. Aprende exclusivamente la GEOMETRÍA DEL MOVIMIENTO mediante MEDIANAS.
    """

    def __init__(self, outlier_std_threshold: float = 2.5):
        self.outlier_std_threshold = outlier_std_threshold

    def filter_outliers(self, sample_descriptors: List[MovementDescriptor]) -> Tuple[List[MovementDescriptor], int]:
        """
        Detecta descriptores anómalos cuyo desplazamiento medio dista excesivamente del grupo.
        """
        if len(sample_descriptors) < 4:
            return sample_descriptors, 0

        # Evaluar la magnitud de desplazamiento acumulado de cada personaje
        magnitudes = []
        for desc in sample_descriptors:
            total_disp = sum(abs(f.body_dx) + abs(f.body_dy) + abs(f.head_dx) + abs(f.head_dy) for f in desc.frames)
            magnitudes.append(total_disp)

        med = np.median(magnitudes)
        std = np.std(magnitudes)

        valid_samples = []
        outliers_count = 0

        for i, desc in enumerate(sample_descriptors):
            if std > 0 and abs(magnitudes[i] - med) > (self.outlier_std_threshold * std):
                logger.warning(
                    f"Outlier descartado en {desc.animation}: {desc.character_id}:{desc.variant} "
                    f"(Magnitud: {magnitudes[i]:.1f}, Mediana: {med:.1f})"
                )
                outliers_count += 1
            else:
                valid_samples.append(desc)

        return valid_samples, outliers_count

    def extract_template(self, animation_name: str, sample_descriptors: List[MovementDescriptor]) -> MotionTemplate:
        """
        Calcula el movimiento promedio robusto usando la mediana de desplazamientos para cada frame.
        """
        if not sample_descriptors:
            # Plantilla estática neutral si no hay muestras
            empty_frames = [TemplateFrame(index=i+1) for i in range(4)]
            return MotionTemplate(name=animation_name, frame_count=4, samples_used=0, frames=empty_frames)

        clean_samples, outliers_count = self.filter_outliers(sample_descriptors)
        samples_to_use = clean_samples if clean_samples else sample_descriptors

        template_frames: List[TemplateFrame] = []
        num_frames = 4

        for f_idx in range(num_frames):
            body_dxs, body_dys = [], []
            head_dxs, head_dys = [], []
            lf_dxs, lf_dys = [], []
            rf_dxs, rf_dys = [], []

            for desc in samples_to_use:
                if len(desc.frames) > f_idx:
                    f = desc.frames[f_idx]
                    body_dxs.append(f.body_dx)
                    body_dys.append(f.body_dy)
                    head_dxs.append(f.head_dx)
                    head_dys.append(f.head_dy)
                    lf_dxs.append(f.left_foot_dx)
                    lf_dys.append(f.left_foot_dy)
                    rf_dxs.append(f.right_foot_dx)
                    rf_dys.append(f.right_foot_dy)

            tf = TemplateFrame(
                index=f_idx + 1,
                body={
                    "x": int(round(float(np.median(body_dxs)))) if body_dxs else 0,
                    "y": int(round(float(np.median(body_dys)))) if body_dys else 0,
                },
                head={
                    "x": int(round(float(np.median(head_dxs)))) if head_dxs else 0,
                    "y": int(round(float(np.median(head_dys)))) if head_dys else 0,
                },
                left_foot={
                    "x": int(round(float(np.median(lf_dxs)))) if lf_dxs else 0,
                    "y": int(round(float(np.median(lf_dys)))) if lf_dys else 0,
                },
                right_foot={
                    "x": int(round(float(np.median(rf_dxs)))) if rf_dxs else 0,
                    "y": int(round(float(np.median(rf_dys)))) if rf_dys else 0,
                },
            )
            template_frames.append(tf)

        template = MotionTemplate(
            name=animation_name,
            frame_count=num_frames,
            loop=True,
            samples_used=len(samples_to_use),
            outliers_detected=outliers_count,
            frames=template_frames
        )
        logger.info(f"Plantilla generada para '{animation_name}' con {len(samples_to_use)} muestras.")
        return template
