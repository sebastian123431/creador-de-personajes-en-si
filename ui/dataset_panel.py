from pathlib import Path
from typing import Dict, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTreeWidget, QTreeWidgetItem, QLineEdit,
    QPushButton, QMessageBox, QFileDialog, QFrame
)

from core.naming import OFFICIAL_VARIANTS, VARIANT_LABELS
from models.character import Character
from services.dataset_service import DatasetService


class DatasetPanel(QWidget):
    """
    Panel de gestión y navegación del Dataset Maestro.
    Permite configurar 'Dataset Root', seleccionar carpetas con [ Browse ],
    lanzar [ Scan Dataset ] y visualizar el árbol jerárquico de personajes y variantes.
    """
    character_selected = Signal(object)  # Emite Character o None
    dataset_modified = Signal()

    def __init__(self, dataset_service: DatasetService, parent=None):
        super().__init__(parent)
        self.dataset_service = dataset_service
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # 1. Selector de Dataset Root y Browse
        root_box = QFrame()
        root_box.setStyleSheet("background: #181b22; border: 1px solid #28303e; border-radius: 6px; padding: 6px;")
        root_layout = QVBoxLayout(root_box)
        root_layout.setSpacing(6)

        root_layout.addWidget(QLabel("<b>Dataset Root:</b>"))

        path_row = QHBoxLayout()
        self.root_path_input = QLineEdit()
        self.root_path_input.setText(str(self.dataset_service.approved_root).replace("\\", "/"))
        self.root_path_input.setReadOnly(True)
        self.root_path_input.setStyleSheet("font-family: monospace; font-size: 11px;")

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setFixedWidth(70)
        self.browse_btn.clicked.connect(self._on_browse_clicked)

        path_row.addWidget(self.root_path_input)
        path_row.addWidget(self.browse_btn)
        root_layout.addLayout(path_row)

        self.scan_btn = QPushButton("🔍 Scan Dataset")
        self.scan_btn.setObjectName("primaryButton")
        self.scan_btn.clicked.connect(self.reload_dataset)
        root_layout.addWidget(self.scan_btn)

        layout.addWidget(root_box)

        # 2. Barra de búsqueda rápida
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Filtrar personaje en árbol...")
        self.search_input.textChanged.connect(self._filter_tree)
        layout.addWidget(self.search_input)

        # 3. Árbol de visualización jerárquico
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemClicked.connect(self._on_tree_item_clicked)
        layout.addWidget(self.tree, 1)

        # 4. Botones de acción adicionales
        bottom_actions = QHBoxLayout()
        self.rebuild_idx_btn = QPushButton("📑 Reconstruir Índice")
        self.rebuild_idx_btn.clicked.connect(self._on_rebuild_index)
        bottom_actions.addWidget(self.rebuild_idx_btn)

        layout.addLayout(bottom_actions)

    def _on_browse_clicked(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Seleccionar Carpeta Root de Personajes (ej. 'personajes al 100%')",
            str(self.dataset_service.approved_root)
        )
        if folder:
            self.dataset_service.set_approved_root(folder)
            self.root_path_input.setText(str(self.dataset_service.approved_root).replace("\\", "/"))
            self.reload_dataset()

    def _on_import_clicked(self):
        self._on_browse_clicked()

    def reload_dataset(self):
        """Escanea la carpeta raíz y actualiza el árbol."""
        self.dataset_service.scan_approved()
        self._populate_tree()
        self.dataset_service.update_index()
        self.dataset_modified.emit()

    def _populate_tree(self):
        self.tree.clear()
        characters = self.dataset_service.approved_characters
        query = self.search_input.text().strip().lower()

        # Nodo raíz del árbol (nombre de la carpeta contenedora, ej. 'personajes al 100%')
        container_name = self.dataset_service.approved_root.name
        root_item = QTreeWidgetItem([f"📁 {container_name} ({len(characters)} personajes)"])
        root_item.setExpanded(True)
        root_item.setForeground(0, QColor("#38bdf8"))
        self.tree.addTopLevelItem(root_item)

        for char_id, char in characters.items():
            if query and query not in char.display_name.lower() and query not in char_id:
                continue

            ref_c = char.reference_count
            sheet_c = char.spritesheet_count
            char_text = f"👤 {char.character_id}"
            char_item = QTreeWidgetItem([char_text])
            char_item.setData(0, Qt.ItemDataRole.UserRole, char)
            char_item.setExpanded(True)

            # Sub-items de variantes (✓ rnormal, ✓ rbchef, ✓ rnchef)
            for v_name in OFFICIAL_VARIANTS:
                variant_obj = char.variants.get(v_name)
                has_ref = variant_obj and variant_obj.has_reference
                has_sheet = variant_obj and variant_obj.has_spritesheet

                check_symbol = "✓" if (has_ref and has_sheet) else ("~" if (has_ref or has_sheet) else "✗")
                label = VARIANT_LABELS.get(v_name, v_name)
                var_text = f"    {check_symbol} {v_name} ({label})"

                var_item = QTreeWidgetItem([var_text])
                var_item.setData(0, Qt.ItemDataRole.UserRole, char)
                var_item.setData(0, Qt.ItemDataRole.UserRole + 1, v_name)

                if has_ref and has_sheet:
                    var_item.setForeground(0, QColor("#22c55e"))
                elif has_ref or has_sheet:
                    var_item.setForeground(0, QColor("#f59e0b"))
                else:
                    var_item.setForeground(0, QColor("#ef4444"))

                char_item.addChild(var_item)

            root_item.addChild(char_item)

    def _filter_tree(self):
        self._populate_tree()

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, column: int):
        char = item.data(0, Qt.ItemDataRole.UserRole)
        if char:
            self.character_selected.emit(char)

    def _on_rebuild_index(self):
        idx_p = self.dataset_service.update_index()
        QMessageBox.information(
            self,
            "Índice Generado",
            f"Índice maestro dataset_index.json generado exitosamente en:\n{idx_p}\n\n"
            f"Ubicado FUERA de la carpeta de personajes."
        )
