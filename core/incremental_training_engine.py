import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from core.anchor_tracker import AnchorTracker
from core.frame_extractor import FrameExtractor, OFFICIAL_ANIMATION_ROWS
from core.guard import SourceDatasetGuard
from core.pose_analyzer_v2 import PoseAnalyzerV2
from core.sheet_detector_v2 import SheetDetectorV2
from core.template_extractor_v2 import TemplateExtractorV2
from core.training_logger import setup_training_logger
from core.training_manifest import TrainingManifest, TrainingManifestEntry
from models.motion_descriptor import MotionDescriptor
from models.skeleton import Skeleton, Anchor, OFFICIAL_ANCHOR_NAMES
from models.training_session import TrainingSession
from services.dataset_service import DatasetService

logger = logging.getLogger("SpriteStudio.IncrementalTrainingEngine")


class IncrementalTrainingEngineV2:
    """
    Motor de entrenamiento cinemático articulado incremental V2.
    - 100% offline, local en CPU (Windows compatible).
    - Basado en caché SHA256 y manifiesto para evitar reprocesamiento.
    - Soporta cancelación limpia y reanudación automática.
    - Modo Smoke (3 personajes, 2 animaciones) y Full (todos los personajes, 16 animaciones).
    - Preservación estricta de inmutabilidad del dataset maestro.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        config_path: Optional[Path] = None,
        dataset_service: Optional[DatasetService] = None,
        guard: Optional[SourceDatasetGuard] = None,
        progress_callback: Optional[Callable[[str, int, int, Dict[str, Any]], None]] = None,
    ):
        self.base_dir = Path(base_dir or Path.cwd()).resolve()
        self.config_path = Path(config_path or (self.base_dir / "config" / "training_v2.json")).resolve()
        self.guard = guard or SourceDatasetGuard()
        self.dataset_service = dataset_service or DatasetService(base_dir=self.base_dir)
        self.progress_callback = progress_callback

        # Configurar logger rotativo en logs/training_v2.log
        self.train_logger = setup_training_logger(self.base_dir / "logs" / "training_v2.log")

        # Cargar configuración
        self.config = self._load_config()

        # Componentes del pipeline
        self.manifest_path = self.base_dir / "dataset" / "training_v2" / "training_manifest.json"
        self.manifest = TrainingManifest(manifest_path=self.manifest_path, guard=self.guard)

        self.descriptors_dir = self.base_dir / "dataset" / "training_v2" / "descriptors"
        self.templates_dir = self.base_dir / "dataset" / "templates_v2"

        self.sheet_detector = SheetDetectorV2(guard=self.guard)
        self.frame_extractor = FrameExtractor(
            output_base_dir=self.base_dir / "dataset" / "extracted_frames",
            sheet_detector=self.sheet_detector,
            guard=self.guard,
        )
        self.pose_analyzer = PoseAnalyzerV2()
        self.anchor_tracker = AnchorTracker()
        self.template_extractor = TemplateExtractorV2(
            base_templates_dir=self.templates_dir,
            guard=self.guard,
        )

        self.current_session: Optional[TrainingSession] = None
        self._cancel_requested: bool = False

    def _load_config(self) -> Dict[str, Any]:
        defaults = {
            "offline_mode": true if hasattr(__builtins__, "true") else True,
            "min_anchor_confidence": 0.60,
            "preferred_anchor_confidence": 0.75,
            "use_manual_annotations": True,
            "use_temporal_tracking": True,
            "use_mad_outlier_filter": True,
            "incremental_training": True,
            "template_version": "2.2",
            "pose_analyzer_version": "2.2",
            "force_rebuild": False,
        }
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                defaults.update(data)
            except Exception as e:
                logger.warning(f"Error cargando {self.config_path}: {e}. Usando defaults.")
        return defaults

    def request_cancel(self) -> None:
        """Solicita la cancelación limpia del entrenamiento."""
        self._cancel_requested = True
        if self.current_session:
            self.current_session.cancel_requested = True
        logger.info("Cancelación solicitada por el usuario.")
        self.train_logger.info("CANCEL_REQUESTED: El usuario solicitó detener el entrenamiento.")

    def _compute_file_sha256(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _snapshot_protected_dataset(self) -> Dict[str, str]:
        """Calcula los hashes SHA256 de todos los archivos del dataset maestro protegido."""
        approved_dir = self.base_dir / "dataset" / "finished_characters" / "approved" / "personajes al 100%"
        hashes = {}
        if approved_dir.exists():
            for f in sorted(approved_dir.rglob("*")):
                if f.is_file():
                    rel = f.relative_to(approved_dir).as_posix()
                    hashes[rel] = self._compute_file_sha256(f)
        return hashes

    def _emit_progress(self, message: str, current: int, total: int, details: Optional[Dict[str, Any]] = None):
        if self.progress_callback:
            try:
                self.progress_callback(message, current, total, details or {})
            except Exception as e:
                logger.warning(f"Error en progress_callback: {e}")

    def run_training(
        self,
        is_smoke: bool = False,
        force_rebuild: Optional[bool] = None,
        max_characters: Optional[int] = None,
    ) -> TrainingSession:
        """
        Ejecuta el entrenamiento incremental V2.
        - is_smoke: si True, procesa únicamente 3 personajes y las animaciones walk_down y cook_down.
        - force_rebuild: si True, ignora la caché existente y reprocesa todo.
        """
        start_time = time.time()
        self._cancel_requested = False

        if force_rebuild is None:
            force_rebuild = bool(self.config.get("force_rebuild", False))

        pose_version = str(self.config.get("pose_analyzer_version", "2.2"))
        tpl_version = str(self.config.get("template_version", "2.2"))

        # Snapshot hash del dataset maestro ANTES de entrenar
        hash_before = self._snapshot_protected_dataset()

        # Crear sesión
        session = TrainingSession(is_smoke=is_smoke)
        self.current_session = session

        # Directorio de este run específico
        run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = self.base_dir / "dataset" / "training_v2" / "runs" / f"run_{run_timestamp}"
        self.guard.assert_can_write(run_dir, operation_desc="creación de directorio de run")
        run_dir.mkdir(parents=True, exist_ok=True)

        # Guardar configuración y hash inicial del run
        with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump({
                "session_id": session.session_id,
                "is_smoke": is_smoke,
                "force_rebuild": force_rebuild,
                "pose_analyzer_version": pose_version,
                "template_version": tpl_version,
                "config": self.config,
            }, f, indent=2)

        with open(run_dir / "dataset_hash_before.json", "w", encoding="utf-8") as f:
            json.dump(hash_before, f, indent=2)

        self.train_logger.info(
            f"=== INICIO SESION ENTRENAMIENTO V2 (id: {session.session_id}, smoke: {is_smoke}) ==="
        )

        try:
            # 1. Escaneo dinámico de personajes aprobados
            self._emit_progress("Escaneando dataset maestro...", 1, 10)
            chars = self.dataset_service.scan_approved()
            if not chars:
                msg = "No se encontraron personajes aprobados en el dataset maestro."
                session.warnings.append(msg)
                self.train_logger.warning(msg)
                session.status = "completed"
                session.completed_at = datetime.now(timezone.utc).isoformat()
                return session

            # Filtrar personajes si es smoke training o max_characters
            char_keys = sorted(list(chars.keys()))
            if is_smoke:
                # Tomar los 3 primeros personajes aprobados
                char_keys = char_keys[:3]
                target_animations = ["walk_down", "cook_down"]
            else:
                if max_characters:
                    char_keys = char_keys[:max_characters]
                target_animations = list(OFFICIAL_ANIMATION_ROWS)

            session.characters_total = len(char_keys)
            session.animations_total = len(target_animations)

            # Contar variantes utilizables
            variants_to_process = []
            for cid in char_keys:
                c_obj = chars[cid]
                for vname in ("rnormal", "rbchef", "rnchef"):
                    if vname in c_obj.variants and c_obj.variants[vname].has_spritesheet:
                        variants_to_process.append((cid, vname, c_obj.variants[vname].spritesheet))

            session.variants_total = len(variants_to_process)

            # Estructura en memoria para agregar templates por animación
            # anim_name -> List[List[Skeleton]]
            aggregated_animation_skeletons: Dict[str, List[List[Skeleton]]] = {
                a: [] for a in target_animations
            }

            total_jobs = len(variants_to_process)
            current_job_idx = 0

            # 2. Procesamiento incremental por personaje / variante
            for char_id in char_keys:
                if self._cancel_requested:
                    session.status = "cancelled"
                    break

                c_obj = chars[char_id]
                for vname in ("rnormal", "rbchef", "rnchef"):
                    if self._cancel_requested:
                        session.status = "cancelled"
                        break

                    if vname not in c_obj.variants or not c_obj.variants[vname].has_spritesheet:
                        continue

                    current_job_idx += 1
                    sheet_path = c_obj.variants[vname].spritesheet
                    sheet_sha256 = self._compute_file_sha256(sheet_path)

                    details = {
                        "character_id": char_id,
                        "variant": vname,
                        "char_index": char_keys.index(char_id) + 1,
                        "char_total": len(char_keys),
                        "job_index": current_job_idx,
                        "job_total": total_jobs,
                        "cache_hits": session.cache_hits,
                        "cache_misses": session.cache_misses,
                    }

                    # Comprobar validez de caché
                    is_cached = (
                        not force_rebuild
                        and self.manifest.is_cache_valid(
                            character_id=char_id,
                            variant=vname,
                            current_sha256=sheet_sha256,
                            pose_version=pose_version,
                            template_version=tpl_version,
                            required_animations=target_animations,
                        )
                    )

                    manifest_entry = self.manifest.get_entry(char_id, vname) or TrainingManifestEntry(
                        character_id=char_id,
                        variant=vname,
                        spritesheet_path=str(sheet_path),
                        spritesheet_sha256=sheet_sha256,
                        pose_analyzer_version=pose_version,
                        template_version=tpl_version,
                    )

                    if is_cached:
                        # --- CACHE HIT ---
                        session.cache_hits += 1
                        session.frames_cached += len(target_animations) * 4
                        self.train_logger.info(
                            f"[CACHE HIT] {char_id} [{vname}] (SHA256: {sheet_sha256[:8]}...)"
                        )
                        self._emit_progress(
                            f"[CACHE HIT] {char_id} ({vname})",
                            current_job_idx,
                            total_jobs,
                            details,
                        )

                        # Cargar descriptores existentes desde disco y reconstruir esqueletos
                        for anim_name in target_animations:
                            desc_file = self.descriptors_dir / anim_name / char_id / vname / "descriptor.json"
                            if desc_file.exists():
                                descriptor = MotionDescriptor.load(desc_file)
                                skel_seq = self._reconstruct_skeletons_from_descriptor(descriptor)
                                if skel_seq and len(skel_seq) == 4:
                                    aggregated_animation_skeletons[anim_name].append(skel_seq)
                                session.animations_processed += 1
                                if descriptor.review_required:
                                    session.review_required_count += 1
                    else:
                        # --- CACHE MISS ---
                        session.cache_misses += 1
                        manifest_entry.status = "processing"
                        self.manifest.update_entry(manifest_entry)

                        self.train_logger.info(
                            f"[CACHE MISS] {char_id} [{vname}] Procesando extracción y cinemática..."
                        )
                        self._emit_progress(
                            f"[PROCESANDO] {char_id} ({vname})",
                            current_job_idx,
                            total_jobs,
                            details,
                        )

                        # Extraer frames mediante SheetDetectorV2 + FrameExtractor
                        extracted_anims = self.frame_extractor.extract_from_sheet(
                            sheet_path, char_id, vname
                        )

                        variant_has_low_confidence = False

                        for anim_name in target_animations:
                            if self._cancel_requested:
                                session.status = "cancelled"
                                break

                            anim_obj = extracted_anims.get(anim_name)
                            if not anim_obj or len(anim_obj.frames) < 4:
                                session.warnings.append(
                                    f"Animación '{anim_name}' incompleta para {char_id}/{vname}"
                                )
                                continue

                            frame_paths = [f.image_path for f in anim_obj.frames[:4]]

                            # Calcular dimensiones reales de la silueta a partir de la máscara alfa
                            char_w, char_h = self._compute_sprite_bbox(frame_paths[0])

                            # Memory safety: analizar frames y liberar imágenes
                            skeletons = self.anchor_tracker.track_animation_anchors(
                                frame_paths, orientation=anim_name
                            )

                            if len(skeletons) == 4:
                                min_anchor_conf = float(self.config.get("min_anchor_confidence", 0.60))

                                # Contar frames individuales de baja confianza
                                for skel in skeletons:
                                    if skel.average_confidence < min_anchor_conf:
                                        session.low_confidence_frames += 1

                                # Comprobar confianza del ciclo
                                avg_conf = sum(s.average_confidence for s in skeletons) / 4.0
                                review_req = avg_conf < min_anchor_conf

                                if review_req:
                                    session.review_required_count += 1
                                    variant_has_low_confidence = True

                                # Extraer cinemática para MotionDescriptor usando dimensiones reales
                                motion_seq_data = self.template_extractor.extract_motion_from_sequence(
                                    skeletons, character_size=(char_w, char_h)
                                )

                                # Crear MotionDescriptor con dimensiones de silueta real
                                desc = self._build_motion_descriptor(
                                    character_id=char_id,
                                    variant=vname,
                                    animation=anim_name,
                                    source_path=str(sheet_path),
                                    source_sha256=sheet_sha256,
                                    skeletons=skeletons,
                                    motion_seq_data=motion_seq_data,
                                    confidence=avg_conf,
                                    review_required=review_req,
                                    pose_version=pose_version,
                                    template_version=tpl_version,
                                    character_size=(char_w, char_h),
                                )

                                # Guardar descriptor en disco
                                desc_target_path = (
                                    self.descriptors_dir / anim_name / char_id / vname / "descriptor.json"
                                )
                                desc.save(desc_target_path, guard=self.guard)

                                # Registrar en manifest
                                manifest_entry.descriptor_paths[anim_name] = str(desc_target_path)
                                if anim_name not in manifest_entry.animations_processed:
                                    manifest_entry.animations_processed.append(anim_name)

                                aggregated_animation_skeletons[anim_name].append(skeletons)
                                session.frames_processed += 4
                                session.animations_processed += 1

                        # Actualizar estado de la entrada en el manifiesto
                        manifest_entry.spritesheet_sha256 = sheet_sha256
                        manifest_entry.pose_analyzer_version = pose_version
                        manifest_entry.template_version = tpl_version
                        manifest_entry.processed_at = datetime.now(timezone.utc).isoformat()
                        manifest_entry.status = "review_required" if variant_has_low_confidence else "completed"

                        self.manifest.update_entry(manifest_entry)
                        self.manifest.save()

                    session.variants_processed += 1

                session.characters_processed += 1

            # 3. Construir plantillas cinemáticas V2 con filtrado MAD
            if not self._cancel_requested:
                self._emit_progress("Construyendo plantillas agregadas con filtrado MAD...", 9, 10)
                version_tag = f"v{tpl_version}_{run_timestamp}"

                for anim_name in target_animations:
                    seqs = aggregated_animation_skeletons.get(anim_name, [])
                    if seqs:
                        tpl = self.template_extractor.build_articulated_template(anim_name, seqs)
                        # Guardar plantilla en versión inmutable y en current/
                        self.template_extractor.save_template(tpl, version_tag=version_tag)
                        session.templates_generated += 1
                        self.train_logger.info(
                            f"[TEMPLATE] Plantilla generada para '{anim_name}' con {len(seqs)} secuencias."
                        )

            if self._cancel_requested:
                session.status = "cancelled"
                self.train_logger.info("Entrenamiento cancelado limpiamente por solicitud de usuario.")
            else:
                session.status = "completed"
                self.train_logger.info("Entrenamiento completado exitosamente.")

        except Exception as e:
            session.status = "failed"
            err_msg = f"Error crítico durante el entrenamiento: {e}"
            session.errors.append(err_msg)
            self.train_logger.exception(err_msg)
            logger.exception(err_msg)

        # 4. Finalización y verificación de dataset protegido
        elapsed = time.time() - start_time
        session.completed_at = datetime.now(timezone.utc).isoformat()

        hash_after = self._snapshot_protected_dataset()

        # Guardar hashes finales y reportes
        with open(run_dir / "dataset_hash_after.json", "w", encoding="utf-8") as f:
            json.dump(hash_after, f, indent=2)

        session.save(run_dir / "report.json", guard=self.guard)

        # Si hubo errores, persistir errors.json
        if session.errors:
            with open(run_dir / "errors.json", "w", encoding="utf-8") as f:
                json.dump({"errors": session.errors}, f, indent=2)

        # Guardar latest_report.json
        session.save(self.base_dir / "dataset" / "training_v2" / "latest_report.json", guard=self.guard)

        if is_smoke:
            smoke_dir = self.base_dir / "dataset" / "training_v2" / "smoke"
            self.guard.assert_can_write(smoke_dir, operation_desc="creación de directorio smoke")
            smoke_dir.mkdir(parents=True, exist_ok=True)
            session.save(smoke_dir / "latest_report.json", guard=self.guard)

        # Verificar inmutabilidad del dataset maestro
        assert hash_before == hash_after, "FATAL: ¡El dataset maestro fue alterado durante el entrenamiento!"

        self.train_logger.info(
            f"=== FIN SESION (duración: {elapsed:.2f}s, status: {session.status}, templates: {session.templates_generated}) ==="
        )

        return session

    def _compute_sprite_bbox(self, img_path: Path) -> Tuple[float, float]:
        """Calcula el ancho y alto real a partir de la máscara alfa (> 0)."""
        try:
            with Image.open(str(img_path)) as img:
                arr = np.array(img.convert("RGBA"))
            alpha = arr[:, :, 3]
            ys, xs = np.where(alpha > 0)
            if len(xs) > 0 and len(ys) > 0:
                w = float(np.max(xs) - np.min(xs) + 1)
                h = float(np.max(ys) - np.min(ys) + 1)
                return max(w, 1.0), max(h, 1.0)
        except Exception as e:
            logger.warning(f"Error calculando sprite bbox para {img_path}: {e}")
        return 32.0, 64.0

    def _build_motion_descriptor(
        self,
        character_id: str,
        variant: str,
        animation: str,
        source_path: str,
        source_sha256: str,
        skeletons: List[Skeleton],
        motion_seq_data: List[Dict[str, Any]],
        confidence: float,
        review_required: bool,
        pose_version: str,
        template_version: str,
        character_size: Optional[Tuple[float, float]] = None,
    ) -> MotionDescriptor:
        """Construye un MotionDescriptor libre de imágenes a partir del análisis cinemático."""
        root_motion = []
        anchors_list = []
        parts_list = []
        translations_px = []
        translations_ratio = []
        angles_list = []

        for f_idx, skel in enumerate(skeletons):
            frame_data = motion_seq_data[f_idx] if f_idx < len(motion_seq_data) else {}
            parts_dict = frame_data.get("parts", {})

            root_motion.append({
                "frame_index": f_idx + 1,
                "root_dx": frame_data.get("root_dx", 0.0),
                "root_dy": frame_data.get("root_dy", 0.0),
                "root_dx_ratio": frame_data.get("root_dx_ratio", 0.0),
                "root_dy_ratio": frame_data.get("root_dy_ratio", 0.0),
            })

            anchors_list.append(skel.to_dict())
            parts_list.append(parts_dict)

            px_dict = {pname: {"dx": pinfo.get("dx", 0.0), "dy": pinfo.get("dy", 0.0)} for pname, pinfo in parts_dict.items()}
            ratio_dict = {pname: {"dx_ratio": pinfo.get("dx_ratio", 0.0), "dy_ratio": pinfo.get("dy_ratio", 0.0)} for pname, pinfo in parts_dict.items()}
            ang_dict = {pname: pinfo.get("angle_deg", 0.0) for pname, pinfo in parts_dict.items()}

            translations_px.append(px_dict)
            translations_ratio.append(ratio_dict)
            angles_list.append(ang_dict)

        # Dimensiones de silueta: prioridad máscara alfa real, fallback anchors
        if character_size and character_size[0] > 0 and character_size[1] > 0:
            char_w = float(character_size[0])
            char_h = float(character_size[1])
        else:
            base_skel = skeletons[0]
            xs = [a.x for a in base_skel.anchors.values() if a.confidence > 0]
            ys = [a.y for a in base_skel.anchors.values() if a.confidence > 0]
            char_w = float(max(xs) - min(xs)) if xs and max(xs) > min(xs) else 32.0
            char_h = float(max(ys) - min(ys)) if ys and max(ys) > min(ys) else 64.0

        char_w = max(char_w, 1.0)
        char_h = max(char_h, 1.0)

        return MotionDescriptor(
            character_id=character_id,
            variant=variant,
            animation=animation,
            source_path=source_path,
            source_sha256=source_sha256,
            frame_count=len(skeletons),
            orientation=animation.split("_")[-1] if "_" in animation else "down",
            character_width=char_w,
            character_height=char_h,
            root_motion=root_motion,
            anchors=anchors_list,
            parts=parts_list,
            translations_px=translations_px,
            translations_ratio=translations_ratio,
            angles=angles_list,
            confidence=confidence,
            review_required=review_required,
            pose_analyzer_version=pose_version,
            template_version=template_version,
        )

    def _reconstruct_skeletons_from_descriptor(self, descriptor: MotionDescriptor) -> List[Skeleton]:
        """Reconstruye la secuencia de Skeleton a partir de los datos guardados en descriptor.json."""
        skeletons = []
        for a_dict in descriptor.anchors:
            skel = Skeleton.from_dict(a_dict)
            skeletons.append(skel)
        return skeletons
