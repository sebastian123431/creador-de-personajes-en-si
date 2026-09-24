"""
Módulo de Validación Cruzada entre Personajes (CrossCharacterValidator - Fase 3D).
Demuestra la transferencia de movimiento articulado desde personajes de entrenamiento
a personajes de validación completamente desacoplados.
Genera muestras de auditoría visual (audit_samples/validation/) con comparativas y métricas.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.guard import SourceDatasetGuard
from core.motion_transfer_v2 import MotionTransferV2
from core.pose_analyzer_v2 import PoseAnalyzerV2
from core.validation_metrics_v2 import ValidationMetricsV2
from models.articulated_motion_template import ArticulatedMotionTemplate
from models.dataset_split import DatasetSplit
from models.skeleton import Skeleton, Anchor, SKELETON_BONES
from models.validation_metrics import ValidationResult
from services.dataset_service import DatasetService

logger = logging.getLogger("SpriteStudio.CrossCharacterValidator")

DEFAULT_VALIDATION_ANIMATIONS = [
    "walk_down",
    "walk_left",
    "cook_down",
    "pickup",
    "celebrate",
]


class CrossCharacterValidator:
    """
    Coordina la validación cruzada:
    1. Carga partición de personajes (Train vs Validation).
    2. Utiliza plantillas generadas exclusivamente con personajes de Train.
    3. Transfiere movimiento articulado a los personajes de Validation.
    4. Evalúa con ValidationMetricsV2.
    5. Exporta audit_samples/ para inspección visual y diagnóstico técnico.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        guard: Optional[SourceDatasetGuard] = None,
    ):
        self.base_dir = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()
        self.guard = guard or SourceDatasetGuard(
            self.base_dir / "dataset" / "finished_characters" / "approved" / "personajes al 100%"
        )
        self.templates_dir = self.base_dir / "dataset" / "templates_v2"
        self.audit_root = self.base_dir / "audit_samples" / "validation"
        self.motion_transfer = MotionTransferV2(guard=self.guard)
        self.pose_analyzer = PoseAnalyzerV2()
        self.metrics_evaluator = ValidationMetricsV2()

    def _render_debug_comparison(
        self,
        source_frames: List[Image.Image],
        generated_frames: List[Image.Image],
        output_path: Path,
    ) -> Image.Image:
        """
        Crea una tira visual comparativa:
        Fila 1: SOURCE MOTION (frames 1..4)
        Fila 2: GENERATED MOTION (frames 1..4)
        """
        f_w, f_h = generated_frames[0].size
        cols = max(len(source_frames), len(generated_frames))
        header_h = 24
        row_gap = 10
        total_w = cols * f_w
        total_h = header_h * 2 + f_h * 2 + row_gap

        comp_img = Image.new("RGBA", (total_w, total_h), (30, 32, 36, 255))
        draw = ImageDraw.Draw(comp_img)

        # Header fila 1: SOURCE MOTION
        draw.rectangle([(0, 0), (total_w, header_h)], fill=(45, 50, 58, 255))
        draw.text((10, 5), "SOURCE MOVEMENT (TRAIN)", fill=(200, 220, 255, 255))

        # Pegar frames de origen
        y_src = header_h
        for idx, sf in enumerate(source_frames):
            comp_img.paste(sf, (idx * f_w, y_src), mask=sf.convert("RGBA"))

        # Header fila 2: GENERATED MOTION
        y_hdr2 = y_src + f_h + row_gap
        draw.rectangle([(0, y_hdr2), (total_w, y_hdr2 + header_h)], fill=(40, 55, 45, 255))
        draw.text((10, y_hdr2 + 5), "GENERATED MOVEMENT (VALIDATION TRANSFER)", fill=(180, 255, 180, 255))

        # Pegar frames generados
        y_gen = y_hdr2 + header_h
        for idx, gf in enumerate(generated_frames):
            comp_img.paste(gf, (idx * f_w, y_gen), mask=gf.convert("RGBA"))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        comp_img.save(output_path)
        return comp_img

    def _render_skeleton_overlay(
        self,
        frames: List[Image.Image],
        skeletons: List[Skeleton],
        output_path: Path,
    ) -> Image.Image:
        """
        Dibuja los esqueletos articulares (huesos y anchors) sobre los frames generados.
        """
        f_w, f_h = frames[0].size
        total_w = f_w * len(frames)
        total_h = f_h

        strip = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))

        for idx, (frame, skel) in enumerate(zip(frames, skeletons)):
            frame_overlay = frame.copy().convert("RGBA")
            draw = ImageDraw.Draw(frame_overlay)

            # Dibujar huesos (conexiones)
            for b1, b2 in SKELETON_BONES:
                a1 = skel.get_anchor(b1)
                a2 = skel.get_anchor(b2)
                if a1 and a2 and a1.confidence > 0.3 and a2.confidence > 0.3:
                    draw.line([(a1.x, a1.y), (a2.x, a2.y)], fill=(0, 255, 128, 200), width=1)

            # Dibujar anchors (articulaciones)
            anchor_dict = skel.anchors if isinstance(skel.anchors, dict) else {a.name: a for a in skel.anchors}
            for a in anchor_dict.values():
                r = 2
                color = (255, 80, 80, 255) if "hand" in a.name or "foot" in a.name else (80, 180, 255, 255)
                draw.ellipse([(a.x - r, a.y - r), (a.x + r, a.y + r)], fill=color)

            strip.paste(frame_overlay, (idx * f_w, 0), mask=frame_overlay)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        strip.save(output_path)
        return strip

    def find_reference_image(
        self,
        character_id: str,
        variant: str,
        dataset_service: Optional[DatasetService] = None,
    ) -> Optional[Image.Image]:
        """
        Localiza la imagen de referencia para un personaje y variante específicos.
        """
        if dataset_service is None:
            dataset_service = DatasetService(base_dir=self.base_dir)

        chars = dataset_service.scan_approved()
        char = chars.get(character_id)
        if not char:
            return None

        var_obj = char.get_variant(variant)
        if var_obj and var_obj.reference_image and var_obj.reference_image.is_file():
            return Image.open(var_obj.reference_image).convert("RGBA")

        # Fallback a extracted_frames si ya fueron extraídos
        extracted_ref = (
            self.base_dir
            / "dataset"
            / "extracted_frames"
            / character_id
            / variant
            / "idle_down"
            / "01.png"
        )
        if extracted_ref.is_file():
            return Image.open(extracted_ref).convert("RGBA")

        # Fallback a primer frame de la spritesheet si existe
        if var_obj and var_obj.spritesheet and var_obj.spritesheet.is_file():
            sheet = Image.open(var_obj.spritesheet).convert("RGBA")
            cell_w, cell_h = sheet.width // 4, sheet.height // 16
            return sheet.crop((0, 0, cell_w, cell_h))

        return None

    def find_source_motion_frames(
        self,
        animation: str,
        variant: str,
        train_characters: List[str],
    ) -> List[Image.Image]:
        """
        Busca frames de origen de algún personaje de entrenamiento para la tira comparativa.
        """
        for t_char in train_characters:
            anim_dir = (
                self.base_dir
                / "dataset"
                / "extracted_frames"
                / t_char
                / variant
                / animation
            )
            if anim_dir.is_dir():
                frames = []
                for idx in range(1, 5):
                    f_path = anim_dir / f"{idx:02d}.png"
                    if f_path.is_file():
                        frames.append(Image.open(f_path).convert("RGBA"))
                if len(frames) == 4:
                    return frames

        # Fallback: crear 4 frames neutrales vacíos
        return [Image.new("RGBA", (64, 64), (0, 0, 0, 0)) for _ in range(4)]

    def validate_single(
        self,
        character_id: str,
        variant: str,
        animation: str,
        template: ArticulatedMotionTemplate,
        reference_image: Image.Image,
        train_characters: Optional[List[str]] = None,
    ) -> ValidationResult:
        """
        Ejecuta la validación cruzada para una combinación character/variant/animación
        y construye la carpeta en audit_samples/validation/<char>/<variant>/<anim>/.
        """
        # 1. Transferir movimiento articulado
        generated_frames = self.motion_transfer.generate_animation_frames(
            reference_image=reference_image,
            template=template,
        )

        # 2. Analizar esqueletos de los frames generados
        generated_skeletons = [
            self.pose_analyzer.analyze_pose(gf, orientation=template.orientation)
            for gf in generated_frames
        ]

        # 3. Evaluar métricas diagnósticas V2
        val_result = self.metrics_evaluator.evaluate_animation(
            character_id=character_id,
            variant=variant,
            animation=animation,
            template_version=template.animation_name or "v2",
            reference_sprite=reference_image,
            generated_frames=generated_frames,
            skeletons=generated_skeletons,
        )

        # 4. Construir audit sample
        sample_dir = self.audit_root / character_id / variant / animation
        self.guard.assert_can_write(sample_dir, operation_desc="generación de audit sample")
        sample_dir.mkdir(parents=True, exist_ok=True)

        # 4.1 reference.png
        reference_image.save(sample_dir / "reference.png")

        # 4.2 generated_01.png .. generated_04.png
        for idx, gf in enumerate(generated_frames, 1):
            gf.save(sample_dir / f"generated_{idx:02d}.png")

        # 4.3 source_motion_01.png .. source_motion_04.png
        source_frames = self.find_source_motion_frames(
            animation, variant, train_characters or []
        )
        for idx, sf in enumerate(source_frames, 1):
            sf.save(sample_dir / f"source_motion_{idx:02d}.png")

        # 4.4 debug_comparison.png
        self._render_debug_comparison(
            source_frames=source_frames,
            generated_frames=generated_frames,
            output_path=sample_dir / "debug_comparison.png",
        )

        # 4.5 skeleton_overlay.png
        self._render_skeleton_overlay(
            frames=generated_frames,
            skeletons=generated_skeletons,
            output_path=sample_dir / "skeleton_overlay.png",
        )

        # 4.6 anchors.json
        anchors_data = {}
        for idx, sk in enumerate(generated_skeletons, 1):
            anchors_data[f"frame_{idx:02d}"] = sk.to_dict() if hasattr(sk, "to_dict") else {
                a.name: a.to_dict() for a in (sk.anchors.values() if isinstance(sk.anchors, dict) else sk.anchors)
            }
        with open(sample_dir / "anchors.json", "w", encoding="utf-8") as f:
            json.dump(anchors_data, f, indent=2, ensure_ascii=False)

        # 4.7 metrics.json
        val_result.save(sample_dir / "metrics.json")

        # 4.8 template_used.json
        with open(sample_dir / "template_used.json", "w", encoding="utf-8") as f:
            json.dump(template.to_dict(), f, indent=2, ensure_ascii=False)

        return val_result

    def run_cross_validation(
        self,
        split: Optional[DatasetSplit] = None,
        animations: Optional[List[str]] = None,
        max_validation_characters: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta el ciclo de validación cruzada completo.

        Args:
            split: Partición a usar (si es None, carga split_seed_42.json).
            animations: Lista de animaciones a evaluar (por defecto DEFAULT_VALIDATION_ANIMATIONS).
            max_validation_characters: Límite opcional de personajes de validación a procesar.

        Returns:
            Diccionario de resumen con métricas, conteo de reviews y muestras generadas.
        """
        if split is None:
            split_file = self.base_dir / "dataset" / "training_v2" / "splits" / "split_seed_42.json"
            if split_file.is_file():
                split = DatasetSplit.load(split_file)
            else:
                raise FileNotFoundError(f"Archivo de partición no encontrado: {split_file}")

        target_anims = animations or DEFAULT_VALIDATION_ANIMATIONS
        val_chars = split.validation_characters
        if max_validation_characters:
            val_chars = val_chars[:max_validation_characters]

        dataset_service = DatasetService(base_dir=self.base_dir)

        results: List[Dict[str, Any]] = []
        audit_samples_count = 0
        review_count = 0
        errors_count = 0

        logger.info(
            "Iniciando Cross-Character Validation sobre %d personajes de validación: %s",
            len(val_chars),
            val_chars,
        )

        for char_id in val_chars:
            for variant in ("rnormal", "rbchef", "rnchef"):
                ref_img = self.find_reference_image(char_id, variant, dataset_service)
                if not ref_img:
                    logger.warning("No se encontró imagen de referencia para '%s:%s'. Omitiendo.", char_id, variant)
                    continue

                for anim in target_anims:
                    # Cargar plantilla
                    template_path = self.templates_dir / f"{anim}.json"
                    if not template_path.is_file():
                        logger.warning("Plantilla '%s' no disponible en %s. Omitiendo.", anim, self.templates_dir)
                        continue

                    try:
                        template = ArticulatedMotionTemplate.load(template_path)
                        val_res = self.validate_single(
                            character_id=char_id,
                            variant=variant,
                            animation=anim,
                            template=template,
                            reference_image=ref_img,
                            train_characters=split.train_characters,
                        )
                        results.append(val_res.to_dict())
                        audit_samples_count += 1
                        if val_res.review_required:
                            review_count += 1
                    except Exception as e:
                        logger.error("Error validando '%s:%s:%s': %s", char_id, variant, anim, e)
                        errors_count += 1

        # Resumen
        summary = {
            "validation_characters_count": len(val_chars),
            "validation_characters": val_chars,
            "animations_evaluated": target_anims,
            "audit_samples_generated": audit_samples_count,
            "reviews_required": review_count,
            "errors": errors_count,
            "results": results,
        }

        # Guardar reporte resumen
        summary_path = self.base_dir / "audit_samples" / "validation_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary
