from pathlib import Path
from typing import Dict, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTabWidget, QListWidget, QListWidgetItem,
    QLineEdit, QPushButton, QMessageBox, QFileDialog
)

from models.character import Character
from services.dataset_service import DatasetService


class DatasetPanel(QWidget):
    """
    Panel de gestión del dataset maestro:
    - Approved (Read-Only)
    - Incoming (Entrantes para revisión)
    - Rejected (Rechazados)
    """
    character_selected = Signal(object)  # Emite Character o None
    dataset_modified = Signal()

    def __init__(self, dataset_service: DatasetService, parent=None):
        super().__init__(parent)
        self.dataset_service = dataset_service
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Barra de búsqueda rápida
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Filtrar personaje...")
        self.search_input.textChanged.connect(self._filter_lists)
        layout.addWidget(self.search_input)

        # Pestañas de categorías
        self.tabs = QTabWidget()

        # Tab Approved
        self.approved_list = QListWidget()
        self.approved_list.itemClicked.connect(self._on_approved_clicked)
        self.tabs.addTab(self.approved_list, "Aprobados (0)")

        # Tab Incoming
        incoming_container = QWidget()
        inc_layout = QVBoxLayout(incoming_container)
        inc_layout.setContentsMargins(0, 4, 0, 0)
        self.incoming_list = QListWidget()
        self.incoming_list.itemClicked.connect(self._on_incoming_clicked)
        inc_layout.addWidget(self.incoming_list)

        # Botones de acción para incoming
        inc_actions = QHBoxLayout()
        self.approve_btn = QPushButton("Aprobar")
        self.approve_btn.setObjectName("successButton")
        self.approve_btn.clicked.connect(self._on_approve_incoming)

        self.reject_btn = QPushButton("Rechazar")
        self.reject_btn.setObjectName("dangerButton")
        self.reject_btn.clicked.connect(self._on_reject_incoming)

        inc_actions.addWidget(self.approve_btn)
        inc_actions.addWidget(self.reject_btn)
        inc_layout.addLayout(inc_actions)
        self.tabs.addTab(incoming_container, "Entrantes (0)")

        # Tab Rejected
        self.rejected_list = QListWidget()
        self.rejected_list.itemClicked.connect(self._on_rejected_clicked)
        self.tabs.addTab(self.rejected_list, "Rechazados (0)")

        layout.addWidget(self.tabs, 1)

        # Botones de Operaciones Globales
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(6)

        self.import_btn = QPushButton("📥 Importar personajes terminados...")
        self.import_btn.setObjectName("primaryButton")
        self.import_btn.clicked.connect(self._on_import_clicked)
        btn_layout.addWidget(self.import_btn)

        self.rescan_btn = QPushButton("🔄 Re-escanear Dataset")
        self.rescan_btn.clicked.connect(self.reload_dataset)
        btn_layout.addWidget(self.rescan_btn)

        self.rebuild_index_btn = QPushButton("📑 Reconstruir dataset_index.json")
        self.rebuild_index_btn.clicked.connect(self._on_rebuild_index)
        btn_layout.addWidget(self.rebuild_index_btn)

        layout.addLayout(btn_layout)

    def reload_dataset(self):
        """Escanea todos los directorios y refresca las listas."""
        appr, inc, rej = self.dataset_service.scan_all()
        self._populate_list(self.approved_list, appr)
        self._populate_list(self.incoming_list, inc)
        self._populate_list(self.rejected_list, rej)

        self.tabs.setTabText(0, f"Aprobados ({len(appr)})")
        self.tabs.setTabText(1, f"Entrantes ({len(inc)})")
        self.tabs.setTabText(2, f"Rechazados ({len(rej)})")

        self.dataset_modified.emit()

    def _populate_list(self, list_widget: QListWidget, characters: Dict[str, Character]):
        list_widget.clear()
        query = self.search_input.text().strip().lower()

        for char_id, char in characters.items():
            if query and query not in char.display_name.lower() and query not in char_id:
                continue

            ref_c = char.reference_count
            sheet_c = char.spritesheet_count
            status_summary = f"{char.display_name}  [{ref_c}/3 refs, {sheet_c}/3 sheets]"

            item = QListWidgetItem(status_summary)
            item.setData(Qt.ItemDataRole.UserRole, char)

            # Color visual según integridad
            if ref_c == 3 and sheet_c == 3:
                item.setForeground(Qt.GlobalColor.white)
            elif ref_c > 0 or sheet_c > 0:
                item.setForeground(Qt.GlobalColor.yellow)
            else:
                item.setForeground(Qt.GlobalColor.gray)

            list_widget.addItem(item)

    def _filter_lists(self):
        self._populate_list(self.approved_list, self.dataset_service.approved_characters)
        self._populate_list(self.incoming_list, self.dataset_service.incoming_characters)
        self._populate_list(self.rejected_list, self.dataset_service.rejected_characters)

    def _on_approved_clicked(self, item: QListWidgetItem):
        char = item.data(Qt.ItemDataRole.UserRole)
        self.character_selected.emit(char)

    def _on_incoming_clicked(self, item: QListWidgetItem):
        char = item.data(Qt.ItemDataRole.UserRole)
        self.character_selected.emit(char)

    def _on_rejected_clicked(self, item: QListWidgetItem):
        char = item.data(Qt.ItemDataRole.UserRole)
        self.character_selected.emit(char)

    def _on_import_clicked(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Seleccionar carpeta con personajes (ej. 'personajes al 100%')"
        )
        if not folder:
            return

        source_path = Path(folder)
        # Importar a approved
        try:
            imported = self.dataset_service.import_external_directory_to_approved(source_path)
            self.reload_dataset()
            QMessageBox.information(
                self,
                "Importación Exitosa",
                f"Se importaron {len(imported)} personajes correctamente a 'approved'.\n"
                f"El archivo 'dataset_index.json' ha sido actualizado automáticamente."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error de Importación", f"Ocurrió un error al importar: {e}")

    def _on_approve_incoming(self):
        item = self.incoming_list.currentItem()
        if not item:
            QMessageBox.warning(self, "Atención", "Seleccione un personaje de la lista 'Entrantes'.")
            return

        char: Character = item.data(Qt.ItemDataRole.UserRole)
        self.dataset_service.approve_incoming_character(char.character_id)
        self.reload_dataset()
        QMessageBox.information(self, "Aprobado", f"Personaje '{char.display_name}' aprobado y copiado al dataset.")

    def _on_reject_incoming(self):
        item = self.incoming_list.currentItem()
        if not item:
            QMessageBox.warning(self, "Atención", "Seleccione un personaje de la lista 'Entrantes'.")
            return

        char: Character = item.data(Qt.ItemDataRole.UserRole)
        self.dataset_service.reject_incoming_character(char.character_id)
        self.reload_dataset()
        QMessageBox.information(self, "Rechazado", f"Personaje '{char.display_name}' movido a 'rejected'.")

    def _on_rebuild_index(self):
        idx_path = self.dataset_service.update_index()
        QMessageBox.information(self, "Índice Actualizado", f"Índice maestro generado exitosamente en:\n{idx_path}")
