import logging
from pathlib import Path
from typing import Dict, Optional
from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QStatusBar, QMessageBox, QPushButton,
    QLabel, QFrame, QProgressDialog, QFileDialog
)

from core.frame_extractor import OFFICIAL_ANIMATION_ROWS
from core.similarity import FrameComparator
from models.animation import Animation
from models.character import Character
from services.animation_service import AnimationService
from services.dataset_service import DatasetService
from services.export_service import ExportService
from services.training_service import TrainingService, TrainingWorker, TrainingReport, ArticulatedTrainingReport
from ui.character_panel import CharacterPanel
from ui.comparison_panel import ComparisonDialog
from ui.compare_v1_v2_dialog import CompareV1V2Dialog
from ui.dataset_panel import DatasetPanel
from ui.preview_panel import PreviewPanel
from ui.timeline_panel import TimelinePanel
from ui.theme import DARK_THEME_QSS

logger = logging.getLogger("SpriteStudio.MainWindow")


class MainWindow(QMainWindow):
    """
    Ventana principal de Villa del Chef - Sprite Studio.
    Integra Dataset, Visor Pixel-Perfect, Inspector de Personajes,
    Línea de Tiempo con Onion Skinning, Entrenamiento Cinemático,
    Transferencia de Movimiento y Exportación para Unity.
    """
    def __init__(
        self,
        dataset_service: Optional[DatasetService] = None,
        animation_service: Optional[AnimationService] = None,
        export_service: Optional[ExportService] = None,
        training_service: Optional[TrainingService] = None,
    ):
        super().__init__()
        self.setWindowTitle("Villa del Chef - Sprite Studio")
        self.resize(1420, 860)
        self.setMinimumSize(1080, 720)

        # Servicios
        self.dataset_service = dataset_service or DatasetService()
        self.animation_service = animation_service or AnimationService()
        self.export_service = export_service or ExportService()
        self.training_service = training_service or TrainingService(self.dataset_service)

        # Estado activo
        self.current_character: Optional[Character] = None
        self.current_variant_name: str = "rnormal"
        self.current_animations: Dict[str, Animation] = {}
        self.v2_animations: Dict[str, Animation] = {}
        self.training_thread: Optional[QThread] = None
        self.training_worker: Optional[TrainingWorker] = None
        self.training_v2_thread: Optional[QThread] = None
        self.training_v2_worker: Optional[Any] = None

        self._setup_ui()
        self._setup_menu()
        self._setup_statusbar()
        self._connect_signals()

        # Cargar dataset inicial
        self.dataset_panel.reload_dataset()

    def _setup_ui(self):
        self.setStyleSheet(DARK_THEME_QSS)

        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # Splitter de tres columnas principales (Dataset | Preview | Character)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.dataset_panel = DatasetPanel(self.dataset_service)
        self.preview_panel = PreviewPanel()
        self.character_panel = CharacterPanel()

        splitter.addWidget(self.dataset_panel)
        splitter.addWidget(self.preview_panel)
        splitter.addWidget(self.character_panel)

        splitter.setStretchFactor(0, 2)  # Dataset
        splitter.setStretchFactor(1, 5)  # Preview
        splitter.setStretchFactor(2, 3)  # Character

        main_layout.addWidget(splitter, 1)

        # Barra inferior interactiva: Timeline y Acciones de Pipeline
        bottom_box = QFrame()
        bottom_box.setStyleSheet("background: #181b22; border: 1px solid #28303e; border-radius: 6px; padding: 6px;")
        bottom_layout = QVBoxLayout(bottom_box)
        bottom_layout.setSpacing(8)

        # Timeline Panel
        self.timeline_panel = TimelinePanel()
        bottom_layout.addWidget(self.timeline_panel)

        # Fila de Acciones de Pipeline
        pipeline_bar = QHBoxLayout()

        self.train_btn = QPushButton("⚡ TRAIN DATASET")
        self.train_btn.setObjectName("primaryButton")
        self.train_btn.setToolTip("Aprender templates de movimiento (medianas) a partir del dataset Approved")
        self.train_btn.clicked.connect(self._on_train_clicked)

        self.gen_btn = QPushButton("✨ GENERATE MOVEMENT")
        self.gen_btn.setToolTip("Transferir movimiento aprendido preservando identidad visual del personaje")
        self.gen_btn.clicked.connect(self._on_generate_movement_clicked)

        self.export_btn = QPushButton("📦 EXPORT SPRITESHEET (4x16)")
        self.export_btn.setToolTip("Exportar spritesheet 4x16 (64 frames), frames individuales y metadata Unity")
        self.export_btn.clicked.connect(self._on_export_clicked)

        self.edit_anchors_btn = QPushButton("🦴 EDIT ANCHORS")
        self.edit_anchors_btn.setToolTip("Abrir editor interactivo de articulaciones manuales")
        self.edit_anchors_btn.clicked.connect(self._on_edit_anchors_clicked)

        self.diff_btn = QPushButton("🔍 COMPARAR FRAMES (DIFF)")
        self.diff_btn.clicked.connect(self._on_compare_frames_clicked)

        pipeline_bar.addWidget(self.train_btn)
        pipeline_bar.addWidget(self.gen_btn)
        pipeline_bar.addWidget(self.export_btn)
        pipeline_bar.addWidget(self.edit_anchors_btn)
        pipeline_bar.addWidget(self.diff_btn)

        # Fila V2 de Acciones Articuladas
        v2_bar = QHBoxLayout()

        self.train_v2_btn = QPushButton("🦴 TRAIN ARTICULATED (V2)")
        self.train_v2_btn.setStyleSheet("background: #0284c7; color: white; font-weight: bold;")
        self.train_v2_btn.setToolTip("Entrenar plantillas cinemáticas V2 con filtrado MAD desde personajes aprobados")
        self.train_v2_btn.clicked.connect(self._on_train_v2_clicked)

        self.gen_v2_btn = QPushButton("🚀 GENERATE V2")
        self.gen_v2_btn.setStyleSheet("background: #059669; color: white; font-weight: bold;")
        self.gen_v2_btn.setToolTip("Generar movimiento V2 con 18 articulaciones, HeadIdentityLock y PaletteGuard")
        self.gen_v2_btn.clicked.connect(self._on_generate_v2_clicked)

        self.compare_v1_v2_btn = QPushButton("⚖️ COMPARAR V1 vs V2")
        self.compare_v1_v2_btn.setToolTip("Inspección animada lado a lado de movimiento V1 vs V2")
        self.compare_v1_v2_btn.clicked.connect(self._on_compare_v1_v2_clicked)

        self.export_v2_btn = QPushButton("📦 EXPORT UNITY V2")
        self.export_v2_btn.setToolTip("Exportar spritesheet empaquetado y metadata JSON enriquecida para Unity")
        self.export_v2_btn.clicked.connect(self._on_export_v2_clicked)

        v2_bar.addWidget(self.train_v2_btn)
        v2_bar.addWidget(self.gen_v2_btn)
        v2_bar.addWidget(self.compare_v1_v2_btn)
        v2_bar.addWidget(self.export_v2_btn)

        bottom_layout.addLayout(pipeline_bar)
        bottom_layout.addLayout(v2_bar)
        main_layout.addWidget(bottom_box)

        self.setCentralWidget(central_widget)

    def _setup_menu(self):
        menubar = self.menuBar()

        # Menú Dataset
        dataset_menu = menubar.addMenu("Dataset")

        import_act = dataset_menu.addAction("Importar personajes terminados...")
        import_act.triggered.connect(self.dataset_panel._on_import_clicked)

        rescan_act = dataset_menu.addAction("Re-escanear Dataset")
        rescan_act.triggered.connect(self.dataset_panel.reload_dataset)

        rebuild_act = dataset_menu.addAction("Reconstruir dataset_index.json")
        rebuild_act.triggered.connect(self.dataset_panel._on_rebuild_index)

        validate_act = dataset_menu.addAction("Validar Integridad del Dataset")
        validate_act.triggered.connect(self._on_validate_dataset)

        dataset_menu.addSeparator()
        exit_act = dataset_menu.addAction("Salir")
        exit_act.triggered.connect(self.close)

        # Menú Herramientas
        tools_menu = menubar.addMenu("Herramientas")
        train_act = tools_menu.addAction("Entrenar Plantillas de Movimiento")
        train_act.triggered.connect(self._on_train_clicked)

        diff_act = tools_menu.addAction("Comparar Dos Frames con cv2.absdiff")
        diff_act.triggered.connect(self._on_compare_frames_clicked)

        # Menú Ayuda
        help_menu = menubar.addMenu("Ayuda")
        about_act = help_menu.addAction("Acerca de Sprite Studio")
        about_act.triggered.connect(self._on_about)

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.status_label = QLabel("Listo.")
        self.statusbar.addWidget(self.status_label)

    def _connect_signals(self):
        self.dataset_panel.character_selected.connect(self._on_character_selected)
        self.character_panel.variant_changed.connect(self._on_variant_changed)
        self.dataset_panel.dataset_modified.connect(self._update_status_stats)

        # Timeline signals
        self.timeline_panel.frame_changed.connect(self._on_timeline_frame_changed)
        self.timeline_panel.onion_skin_toggled.connect(self._on_onion_skin_toggled)

    def _on_character_selected(self, character: Optional[Character]):
        self.current_character = character
        self.character_panel.set_character(character)

    def _on_variant_changed(self, variant_name: str, ref_path: Optional[Path], sheet_path: Optional[Path]):
        self.current_variant_name = variant_name
        self.preview_panel.load_variant_assets(ref_path, sheet_path)

        # Cargar animaciones si existen
        if self.current_character:
            anims = self.animation_service.get_or_extract_animations(self.current_character, variant_name)
            self.current_animations = anims
            self.timeline_panel.set_animations(anims)
        else:
            self.current_animations = {}
            self.timeline_panel.set_animations({})

    def _on_timeline_frame_changed(self, frame_obj, frame_idx: int):
        if frame_obj and frame_obj.image_path.exists():
            # Si el usuario activa Onion Skin, renderizarlo
            if self.timeline_panel.onion_check.isChecked():
                prev_f = self.timeline_panel.get_frame_at((frame_idx - 1) % 4)
                next_f = self.timeline_panel.get_frame_at((frame_idx + 1) % 4)
                onion_img = FrameComparator.create_onion_skin_frame(
                    frame_obj.image_path,
                    prev_f.image_path if prev_f else None,
                    next_f.image_path if next_f else None
                )
                if onion_img:
                    # Guardar temporal para visualización
                    temp_onion_path = Path(".cache") / "temp_onion.png"
                    temp_onion_path.parent.mkdir(parents=True, exist_ok=True)
                    onion_img.save(temp_onion_path, "PNG")
                    self.preview_panel.canvas.set_image(temp_onion_path)
                    return

            # Actualizar Skeleton para overlay si está activado
            if not hasattr(self, "pose_analyzer_v2"):
                from core.pose_analyzer_v2 import PoseAnalyzerV2
                self.pose_analyzer_v2 = PoseAnalyzerV2()

            skel = self.pose_analyzer_v2.analyze_pose(frame_obj.image_path)
            self.preview_panel.canvas.set_skeleton(skel)

            self.preview_panel.canvas.set_image(frame_obj.image_path)

    def _on_edit_anchors_clicked(self):
        """Abre el editor manual interactivo de articulaciones."""
        if not self.current_character:
            QMessageBox.warning(self, "Selección requerida", "Seleccione un personaje para editar articulaciones.")
            return

        cur_frame = self.timeline_panel.get_current_frame()
        img_path = cur_frame.image_path if cur_frame else None

        if not img_path or not img_path.exists():
            # Usar imagen de referencia si no hay frame de animación
            variant = self.current_character.variants.get(self.current_variant_name)
            if variant and variant.reference_image and variant.reference_image.exists():
                img_path = variant.reference_image
            else:
                QMessageBox.warning(self, "Sin imagen", "No hay imagen disponible para editar articulaciones.")
                return

        from core.annotation_manager import AnnotationManager
        from core.pose_analyzer_v2 import PoseAnalyzerV2
        from ui.anchor_editor import AnchorEditorDialog

        ann_mgr = AnnotationManager()
        anim_name = self.timeline_panel.current_anim_name
        frame_idx = self.timeline_panel.current_frame_idx + 1

        # Cargar anotación existente o generar con V2
        existing_skel = ann_mgr.load_annotation(
            self.current_character.character_id,
            self.current_variant_name,
            anim_name,
            frame_idx
        )
        if not existing_skel:
            analyzer = PoseAnalyzerV2()
            existing_skel = analyzer.analyze_pose(img_path)

        dlg = AnchorEditorDialog(
            image_path=img_path,
            skeleton=existing_skel,
            character_id=self.current_character.character_id,
            variant=self.current_variant_name,
            animation=anim_name,
            frame_index=frame_idx,
            annotation_manager=ann_mgr,
            parent=self
        )
        if dlg.exec():
            # Actualizar skeleton en el canvas
            self.preview_panel.canvas.set_skeleton(dlg.canvas.skeleton)

    def _on_onion_skin_toggled(self, checked: bool):
        cur_frame = self.timeline_panel.get_current_frame()
        if cur_frame:
            self._on_timeline_frame_changed(cur_frame, self.timeline_panel.current_frame_idx)

    def _update_status_stats(self):
        appr_count = len(self.dataset_service.approved_characters)
        inc_count = len(self.dataset_service.incoming_characters)
        rej_count = len(self.dataset_service.rejected_characters)

        total_variants = sum(c.variant_count for c in self.dataset_service.approved_characters.values())
        total_sheets = sum(c.spritesheet_count for c in self.dataset_service.approved_characters.values())
        total_refs = sum(c.reference_count for c in self.dataset_service.approved_characters.values())

        msg = (
            f"Approved: {appr_count} personajes | "
            f"Variantes: {total_variants} ({total_refs} refs, {total_sheets} sheets) | "
            f"Incoming: {inc_count} | Rejected: {rej_count}"
        )
        self.status_label.setText(msg)

    def _on_validate_dataset(self):
        report = self.dataset_service.validate_all_approved()
        if report.is_valid:
            QMessageBox.information(
                self,
                "Validación de Dataset",
                f"Validación exitosa:\n"
                f"- Personajes verificados: {report.total_characters}\n"
                f"- Válidos sin errores: {report.valid_characters}\n"
                f"- Errores: {report.error_count}\n"
                f"- Advertencias: {report.warning_count}"
            )
        else:
            first_errs = "\n".join(f"[{i.severity}] {i.character_id} ({i.variant}): {i.message}" for i in report.issues[:10])
            QMessageBox.warning(
                self,
                "Validación con Observaciones",
                f"Se detectaron {report.error_count} errores y {report.warning_count} advertencias:\n\n"
                f"{first_errs}\n..."
            )

    def _on_train_clicked(self):
        """Ejecuta el entrenamiento cinemático en QThread sin bloquear la GUI."""
        if len(self.dataset_service.approved_characters) == 0:
            QMessageBox.warning(self, "Dataset Vacío", "No hay personajes en 'approved' para aprender movimiento.")
            return

        progress_dlg = QProgressDialog("Iniciando entrenamiento...", "Cancelar", 0, 10, self)
        progress_dlg.setWindowTitle("Entrenando Plantillas de Movimiento")
        progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dlg.setMinimumDuration(0)
        progress_dlg.setValue(0)

        self.training_thread = QThread()
        self.training_worker = self.training_service.create_worker()
        self.training_worker.moveToThread(self.training_thread)

        self.training_thread.started.connect(self.training_worker.run_training)

        def on_progress(msg, step, total):
            progress_dlg.setLabelText(msg)
            progress_dlg.setValue(step)

        def on_finished(report: TrainingReport):
            progress_dlg.close()
            self.training_thread.quit()
            self.training_thread.wait()

            QMessageBox.information(
                self,
                "Entrenamiento Completado",
                report.summary_text()
            )
            self._update_status_stats()

        def on_error(err_msg: str):
            progress_dlg.close()
            self.training_thread.quit()
            self.training_thread.wait()
            QMessageBox.critical(self, "Error de Entrenamiento", f"Ocurrió un error:\n{err_msg}")

        self.training_worker.progress.connect(on_progress)
        self.training_worker.finished.connect(on_finished)
        self.training_worker.error.connect(on_error)

        self.training_thread.start()

    def _on_generate_movement_clicked(self):
        """Aplica la transferencia de movimiento aprendida a un personaje nuevo."""
        if not self.current_character:
            QMessageBox.warning(self, "Selección requerida", "Seleccione un personaje para generar movimiento.")
            return

        variant = self.current_character.variants.get(self.current_variant_name)
        if not variant or not variant.reference_image or not variant.reference_image.exists():
            QMessageBox.warning(self, "Falta Referencia", f"La variante '{self.current_variant_name}' no posee imagen de referencia.")
            return

        try:
            anims = self.animation_service.generate_animations_for_new_character(
                variant.reference_image,
                self.current_character.character_id,
                self.current_variant_name
            )
            self.current_animations = anims
            self.timeline_panel.set_animations(anims)
            QMessageBox.information(
                self,
                "Movimiento Generado",
                f"Se generaron los 64 frames (16 animaciones x 4 frames) exitosamente para "
                f"'{self.current_character.display_name}' ({self.current_variant_name}) respetando al 100% su identidad visual."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error de Generación", f"Error al generar movimiento: {e}")

    def _on_export_clicked(self):
        """Exporta spritesheet 4x16 y metadata Unity."""
        if not self.current_character or not self.current_animations:
            QMessageBox.warning(self, "Sin Animaciones", "Seleccione un personaje con animaciones extraídas o generadas para exportar.")
            return

        try:
            sheet_p, meta_p, metrics = self.export_service.export_character_variant(
                self.current_character.character_id,
                self.current_variant_name,
                self.current_animations
            )
            QMessageBox.information(
                self,
                "Exportación Exitosa para Unity",
                f"Spritesheet maestro exportado exitosamente:\n\n"
                f"• Spritesheet: {sheet_p.name}\n"
                f"• Metadata Unity: {meta_p.name}\n"
                f"• Carpeta de frames individuales: {sheet_p.parent / 'frames'}\n\n"
                f"Consistency Score: {metrics.overall_score}%\n"
                f"  - Baseline stability: {metrics.baseline_stability}%\n"
                f"  - BBox consistency: {metrics.bbox_consistency}%\n"
                f"  - Alpha area stability: {metrics.alpha_area_stability}%"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error de Exportación", f"Ocurrió un error al exportar: {e}")

    def _on_compare_frames_clicked(self):
        """Abre diálogo para comparar dos frames usando cv2.absdiff."""
        file_a, _ = QFileDialog.getOpenFileName(self, "Seleccionar Frame A", filter="Imágenes (*.png)")
        if not file_a:
            return
        file_b, _ = QFileDialog.getOpenFileName(self, "Seleccionar Frame B", filter="Imágenes (*.png)")
        if not file_b:
            return

        dlg = ComparisonDialog(Path(file_a), Path(file_b), self)
        dlg.exec()

    def _on_about(self):
        QMessageBox.about(
            self,
            "Villa del Chef - Sprite Studio",
            "<b>Villa del Chef - Sprite Studio v2.0.0 (Articulated Edition)</b><br><br>"
            "Herramienta integral de cinemática y generación de animaciones pixel art 2D.<br>"
            "• Extracción de 64 frames (4 columnas x 16 filas)<br>"
            "• Esqueleto anatómico de 18 anclajes con suavizado temporal<br>"
            "• Segmentación por partes corporales y resolución de capas Z<br>"
            "• Aprendizaje cinemático con filtrado de anomalías MAD<br>"
            "• HeadIdentityLock y PaletteGuard: Máxima preservación de identidad visual<br>"
            "• Exportación de Spritesheet y metadatos para Unity / Godot.<br><br>"
            "Desarrollado para el videojuego <i>Villa del Chef</i>."
        )

    def _on_train_v2_clicked(self):
        """Entrena las 16 plantillas cinemáticas V2 con filtrado MAD de forma asíncrona en QThread."""
        progress_dlg = QProgressDialog("Iniciando entrenamiento cinemático V2...", "Cancelar", 0, 6, self)
        progress_dlg.setWindowTitle("Entrenamiento Articulado V2")
        progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dlg.setMinimumDuration(0)
        progress_dlg.setValue(0)
        progress_dlg.show()

        self.training_v2_thread = QThread()
        self.training_v2_worker = self.training_service.create_articulated_worker()
        self.training_v2_worker.moveToThread(self.training_v2_thread)

        self.training_v2_thread.started.connect(self.training_v2_worker.run_training)

        def on_prog(msg, cur, tot):
            progress_dlg.setLabelText(msg)
            progress_dlg.setValue(cur)

        def on_finished(rep: ArticulatedTrainingReport):
            progress_dlg.close()
            if self.training_v2_thread:
                self.training_v2_thread.quit()
                self.training_v2_thread.wait()

            QMessageBox.information(
                self,
                "Entrenamiento V2 Completado",
                rep.summary_text()
            )
            self._update_status_stats()

        def on_error(err_msg: str):
            progress_dlg.close()
            if self.training_v2_thread:
                self.training_v2_thread.quit()
                self.training_v2_thread.wait()
            QMessageBox.critical(self, "Error de Entrenamiento V2", f"Ocurrió un error:\n{err_msg}")

        def on_cancel():
            if self.training_v2_thread and self.training_v2_thread.isRunning():
                self.training_v2_thread.requestInterruption()
                self.training_v2_thread.quit()

        progress_dlg.canceled.connect(on_cancel)
        self.training_v2_worker.progress.connect(on_prog)
        self.training_v2_worker.finished.connect(on_finished)
        self.training_v2_worker.error.connect(on_error)

        self.training_v2_thread.start()

    def _on_generate_v2_clicked(self):
        """Genera movimiento articulado V2 respetando HeadIdentityLock y PaletteGuard."""
        if not self.current_character:
            QMessageBox.warning(self, "Sin Personaje", "Seleccione primero un personaje del árbol de dataset.")
            return

        variant = self.current_character.variants.get(self.current_variant_name)
        if not variant or not variant.reference_image or not variant.reference_image.exists():
            QMessageBox.warning(self, "Falta Referencia", f"La variante '{self.current_variant_name}' no posee imagen de referencia.")
            return

        try:
            anims_v2 = self.animation_service.generate_articulated_animations_v2(
                variant.reference_image,
                self.current_character.character_id,
                self.current_variant_name
            )
            self.v2_animations = anims_v2
            self.timeline_panel.set_animations(anims_v2)
            QMessageBox.information(
                self,
                "Movimiento Articulado V2 Generado",
                f"Se generaron exitosamente las 16 animaciones V2 para "
                f"'{self.current_character.display_name}' ({self.current_variant_name}).\n\n"
                f"• HeadIdentityLock: ACTIVO (Máxima preservación de identidad visual)\n"
                f"• PaletteGuard: ACTIVO (Sin colores espurios)\n"
                f"• Cinemática: 18 articulaciones anatómicas"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error de Generación V2", f"Ocurrió un error al generar V2: {e}")

    def _on_compare_v1_v2_clicked(self):
        """Abre el diálogo interactivo para comparar visualmente V1 vs V2."""
        if not self.current_character:
            QMessageBox.warning(self, "Sin Personaje", "Seleccione un personaje para comparar animaciones.")
            return

        dlg = CompareV1V2Dialog(
            character_id=self.current_character.character_id,
            variant=self.current_variant_name,
            v1_animations=self.current_animations,
            v2_animations=self.v2_animations,
            parent=self
        )
        dlg.exec()

    def _on_export_v2_clicked(self):
        """Exporta paquete Unity V2 con spritesheet y metadata de rig articulado."""
        if not self.current_character:
            QMessageBox.warning(self, "Sin Personaje", "Seleccione un personaje para exportar.")
            return

        target_anims = self.v2_animations or self.current_animations
        if not target_anims:
            QMessageBox.warning(self, "Sin Animaciones", "Primero extraiga o genere animaciones (V1 o V2).")
            return

        try:
            sheet_p, meta_p, metrics = self.export_service.export_unity_package_v2(
                self.current_character.character_id,
                self.current_variant_name,
                target_anims
            )
            QMessageBox.information(
                self,
                "Exportación Unity V2 Exitosa",
                f"Paquete Unity V2 generado con éxito:\n\n"
                f"• Spritesheet V2: {sheet_p.name}\n"
                f"• Metadata Rig V2: {meta_p.name}\n"
                f"• Carpeta destino: {sheet_p.parent}\n\n"
                f"Consistency Score: {metrics.overall_score}%"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error de Exportación V2", f"Ocurrió un error al exportar: {e}")

