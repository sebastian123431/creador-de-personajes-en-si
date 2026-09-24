import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from PySide6.QtCore import QObject, Signal, QThread

from core.frame_extractor import FrameExtractor, OFFICIAL_ANIMATION_ROWS
from core.frame_normalizer import FrameNormalizer
from core.motion_analyzer import MotionAnalyzer, MovementDescriptor
from core.template_extractor import TemplateExtractor
from core.template_library import TemplateLibrary
from models.character import Character
from models.motion_template import MotionTemplate
from services.dataset_service import DatasetService

logger = logging.getLogger("SpriteStudio.TrainingService")


@dataclass
class TrainingReport:
    characters_count: int = 0
    variants_count: int = 0
    spritesheets_count: int = 0
    frames_extracted: int = 0
    animations_count: int = 16
    templates_generated: int = 0
    outliers_detected: int = 0
    errors: int = 0
    warnings: List[str] = field(default_factory=list)

    def summary_text(self) -> str:
        return (
            f"=== REPORTE DE ENTRENAMIENTO DE MOVIMIENTO ===\n"
            f"Personajes analizados: {self.characters_count}\n"
            f"Variantes procesadas:  {self.variants_count}\n"
            f"Spritesheets leídos:   {self.spritesheets_count}\n"
            f"Frames extraídos:      {self.frames_extracted}\n"
            f"Animaciones oficiales: {self.animations_count}\n"
            f"Templates generados:   {self.templates_generated}\n"
            f"Outliers detectados:   {self.outliers_detected}\n"
            f"Errores:               {self.errors}\n"
            f"Advertencias:          {len(self.warnings)}\n"
            f"==============================================="
        )


class TrainingWorker(QObject):
    """
    Worker para ejecutar el pipeline de entrenamiento en segundo plano (QThread)
    sin bloquear la interfaz gráfica.
    """
    progress = Signal(str, int, int)  # mensaje, paso_actual, total_pasos
    finished = Signal(object)        # TrainingReport
    error = Signal(str)

    def __init__(self, dataset_service: DatasetService, base_dir: Optional[Path] = None):
        super().__init__()
        self.dataset_service = dataset_service
        self.base_dir = Path(base_dir or Path.cwd()).resolve()

        self.extractor = FrameExtractor(output_base_dir=self.base_dir / "dataset" / "extracted_frames")
        self.normalizer = FrameNormalizer(output_base_dir=self.base_dir / "dataset" / "normalized")
        self.motion_analyzer = MotionAnalyzer()
        self.template_extractor = TemplateExtractor()
        self.template_library = TemplateLibrary(templates_dir=self.base_dir / "animations" / "learned")

    def run_training(self):
        """Ejecuta los 10 pasos del pipeline de entrenamiento."""
        report = TrainingReport()
        try:
            self.progress.emit("Escaneando dataset approved...", 1, 10)
            chars = self.dataset_service.scan_approved()
            report.characters_count = len(chars)

            if report.characters_count == 0:
                report.warnings.append("No hay personajes en la carpeta 'approved'.")
                self.finished.emit(report)
                return

            self.progress.emit("Indexando dataset...", 2, 10)
            self.dataset_service.update_index()

            # Recolectar spritesheets
            spritesheet_jobs = []
            for char_id, char in chars.items():
                for v_name, v_obj in char.variants.items():
                    if v_obj.has_spritesheet:
                        spritesheet_jobs.append((char_id, v_name, v_obj.spritesheet))

            report.variants_count = sum(len(c.variants) for c in chars.values())
            report.spritesheets_count = len(spritesheet_jobs)

            # 3. Extraer frames
            self.progress.emit(f"Extrayendo frames de {len(spritesheet_jobs)} spritesheets...", 3, 10)
            extracted_anim_by_char = {}  # { (char_id, v_name): {anim_name: Animation} }

            for char_id, v_name, sheet_path in spritesheet_jobs:
                anims = self.extractor.extract_from_sheet(sheet_path, char_id, v_name)
                extracted_anim_by_char[(char_id, v_name)] = anims
                report.frames_extracted += sum(len(a.frames) for a in anims.values())

            # 4. Normalizar frames
            self.progress.emit("Normalizando frames a canvas 256x192...", 4, 10)
            normalized_anim_by_char = {}
            for (char_id, v_name), anims in extracted_anim_by_char.items():
                norm_dict = {}
                for anim_name, anim in anims.items():
                    norm_frames = []
                    for f in anim.frames:
                        norm_f = self.normalizer.normalize_frame(
                            f,
                            character_id=char_id,
                            variant=v_name,
                            animation_name=anim_name
                        )
                        norm_frames.append(norm_f)
                    norm_dict[anim_name] = norm_frames
                normalized_anim_by_char[(char_id, v_name)] = norm_dict

            # 5 & 6. Analizar pose y cinemática de movimiento
            self.progress.emit("Analizando deltas cinemáticos de movimiento...", 5, 10)
            motion_samples_by_anim: Dict[str, List[MovementDescriptor]] = {
                a: [] for a in OFFICIAL_ANIMATION_ROWS
            }

            for (char_id, v_name), norm_dict in normalized_anim_by_char.items():
                for anim_name, frames in norm_dict.items():
                    if anim_name in motion_samples_by_anim:
                        desc = self.motion_analyzer.analyze_animation_motion(
                            frames,
                            animation_name=anim_name,
                            character_id=char_id,
                            variant=v_name
                        )
                        motion_samples_by_anim[anim_name].append(desc)

            # 7 & 8. Crear templates consolidados para las 16 animaciones
            self.progress.emit("Generando plantillas consolidadas (medianas)...", 7, 10)
            for anim_name in OFFICIAL_ANIMATION_ROWS:
                samples = motion_samples_by_anim.get(anim_name, [])
                template = self.template_extractor.extract_template(anim_name, samples)
                self.template_library.save_template(template)
                report.templates_generated += 1
                report.outliers_detected += template.outliers_detected

            # 9. Validación
            self.progress.emit("Validando biblioteca de plantillas...", 9, 10)
            if not self.template_library.has_all_templates():
                report.warnings.append("Algunas plantillas no se pudieron generar por falta de muestras.")

            self.progress.emit("Entrenamiento completado exitosamente.", 10, 10)
            self.finished.emit(report)

        except Exception as e:
            logger.exception("Error durante el entrenamiento:")
            self.error.emit(str(e))


@dataclass
class ArticulatedTrainingReport:
    characters_count: int = 0
    variants_count: int = 0
    templates_v2_generated: int = 0
    outliers_detected: int = 0
    frames_processed: int = 0
    frames_cached: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    session_id: str = ""
    status: str = "completed"
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def summary_text(self) -> str:
        return (
            f"=== REPORTE ENTRENAMIENTO ARTICULADO V2 ===\n"
            f"Estado:                        {self.status}\n"
            f"Sesión ID:                     {self.session_id}\n"
            f"Personajes maestros analizados: {self.characters_count}\n"
            f"Variantes procesadas:          {self.variants_count}\n"
            f"Plantillas V2 generadas:       {self.templates_v2_generated}\n"
            f"Frames procesados:             {self.frames_processed}\n"
            f"Frames desde caché:            {self.frames_cached}\n"
            f"Aciertos de caché (Hits):      {self.cache_hits}\n"
            f"Fallos de caché (Misses):      {self.cache_misses}\n"
            f"Errores:                       {len(self.errors)}\n"
            f"Advertencias:                  {len(self.warnings)}\n"
            f"============================================"
        )


class ArticulatedTrainingWorker(QObject):
    """
    Worker para entrenar las plantillas cinemáticas V2 con motor incremental,
    filtrado MAD, caché SHA256 y soporte de cancelación limpia en segundo plano.
    """
    progress = Signal(str, int, int)
    finished = Signal(object)  # ArticulatedTrainingReport
    error = Signal(str)

    def __init__(
        self,
        dataset_service: DatasetService,
        base_dir: Optional[Path] = None,
        is_smoke: bool = False,
        force_rebuild: bool = False,
    ):
        super().__init__()
        self.dataset_service = dataset_service
        self.base_dir = Path(base_dir or Path.cwd()).resolve()
        self.is_smoke = is_smoke
        self.force_rebuild = force_rebuild

        from core.incremental_training_engine import IncrementalTrainingEngineV2

        def on_engine_progress(msg: str, cur: int, tot: int, details: dict):
            self.progress.emit(msg, cur, tot)

        self.engine = IncrementalTrainingEngineV2(
            base_dir=self.base_dir,
            dataset_service=self.dataset_service,
            guard=self.dataset_service.guard,
            progress_callback=on_engine_progress,
        )

        # Exponer submódulos para compatibilidad directa con tests e inspección
        self.sheet_detector = self.engine.sheet_detector
        self.frame_extractor = self.engine.frame_extractor
        self.pose_analyzer = self.engine.pose_analyzer
        self.anchor_tracker = self.engine.anchor_tracker
        self.template_extractor_v2 = self.engine.template_extractor

    def cancel(self):
        """Solicita la cancelación limpia del entrenamiento."""
        self.engine.request_cancel()

    def run_training(self):
        try:
            session = self.engine.run_training(
                is_smoke=self.is_smoke,
                force_rebuild=self.force_rebuild,
            )

            report = ArticulatedTrainingReport(
                characters_count=session.characters_processed,
                variants_count=session.variants_processed,
                templates_v2_generated=session.templates_generated,
                frames_processed=session.frames_processed,
                frames_cached=session.frames_cached,
                cache_hits=session.cache_hits,
                cache_misses=session.cache_misses,
                session_id=session.session_id,
                status=session.status,
                warnings=session.warnings,
                errors=session.errors,
            )

            if session.status == "failed":
                self.error.emit("; ".join(session.errors) if session.errors else "Error desconocido")
            else:
                self.finished.emit(report)

        except Exception as e:
            logger.exception("Error durante el entrenamiento articulado V2:")
            self.error.emit(str(e))


class TrainingService:
    """
    Servicio de alto nivel que permite lanzar el entrenamiento en un QThread
    o ejecutarlo sincrónicamente según sea necesario.
    """

    def __init__(self, dataset_service: DatasetService, base_dir: Optional[Path] = None):
        self.dataset_service = dataset_service
        self.base_dir = Path(base_dir or Path.cwd()).resolve()

    def create_worker(self) -> TrainingWorker:
        return TrainingWorker(dataset_service=self.dataset_service, base_dir=self.base_dir)

    def create_articulated_worker(
        self,
        is_smoke: bool = False,
        force_rebuild: bool = False,
    ) -> ArticulatedTrainingWorker:
        return ArticulatedTrainingWorker(
            dataset_service=self.dataset_service,
            base_dir=self.base_dir,
            is_smoke=is_smoke,
            force_rebuild=force_rebuild,
        )

    def run_synchronous_training(self) -> TrainingReport:
        worker = self.create_worker()
        result_report: Optional[TrainingReport] = None

        def on_finished(rep):
            nonlocal result_report
            result_report = rep

        worker.finished.connect(on_finished)
        worker.run_training()
        return result_report or TrainingReport()

    def run_synchronous_articulated_training(
        self,
        is_smoke: bool = False,
        force_rebuild: bool = False,
    ) -> ArticulatedTrainingReport:
        worker = self.create_articulated_worker(is_smoke=is_smoke, force_rebuild=force_rebuild)
        result_report: Optional[ArticulatedTrainingReport] = None

        def on_finished(rep):
            nonlocal result_report
            result_report = rep

        worker.finished.connect(on_finished)
        worker.run_training()
        return result_report or ArticulatedTrainingReport()



