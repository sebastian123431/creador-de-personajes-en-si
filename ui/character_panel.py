from pathlib import Path
from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QRadioButton, QButtonGroup, QFrame,
    QPushButton, QTextEdit
)

from core.naming import OFFICIAL_VARIANTS, VARIANT_LABELS, make_unique_key
from models.character import Character
from models.character_variant import CharacterVariant


class CharacterPanel(QWidget):
    """
    Panel de inspección de personaje y variantes oficiales (rnormal, rbchef, rnchef).
    """
    variant_changed = Signal(str, object, object)  # variant_name, ref_path, sheet_path

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_character: Optional[Character] = None
        self.current_variant_name: str = "rnormal"
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # Header Info
        header_group = QGroupBox("IDENTIDAD DEL PERSONAJE")
        header_layout = QVBoxLayout(header_group)

        self.name_label = QLabel("Seleccione un personaje...")
        self.name_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #38bdf8;")
        header_layout.addWidget(self.name_label)

        self.id_label = QLabel("ID único: -")
        self.id_label.setStyleSheet("color: #94a3b8; font-family: monospace;")
        header_layout.addWidget(self.id_label)

        self.folder_label = QLabel("Directorio: -")
        self.folder_label.setStyleSheet("color: #64748b; font-size: 11px;")
        self.folder_label.setWordWrap(True)
        header_layout.addWidget(self.folder_label)

        self.status_badge = QLabel("ESTADO: -")
        self.status_badge.setStyleSheet("color: #e2e8f0; font-weight: bold; padding: 4px 8px; background: #1e293b; border-radius: 4px;")
        header_layout.addWidget(self.status_badge)

        layout.addWidget(header_group)

        # Selector de Variante Oficial
        variant_group = QGroupBox("VARIANTES OFICIALES")
        var_layout = QVBoxLayout(variant_group)

        self.btn_group = QButtonGroup(self)
        self.radio_buttons = {}

        for v in OFFICIAL_VARIANTS:
            label_text = f"{VARIANT_LABELS[v]} ({v})"
            rb = QRadioButton(label_text)
            if v == "rnormal":
                rb.setChecked(True)
            self.btn_group.addButton(rb)
            self.radio_buttons[v] = rb
            var_layout.addWidget(rb)

        self.btn_group.buttonClicked.connect(self._on_variant_radio_clicked)
        layout.addWidget(variant_group)

        # Detalle de la variante
        detail_group = QGroupBox("DETALLES DE LA VARIANTE")
        detail_layout = QVBoxLayout(detail_group)

        self.key_label = QLabel("Clave canónica: -")
        self.key_label.setStyleSheet("color: #38bdf8; font-family: monospace; font-weight: bold;")
        detail_layout.addWidget(self.key_label)

        # Referencia
        ref_box = QFrame()
        ref_box.setStyleSheet("background: #181b22; border: 1px solid #28303e; border-radius: 4px; padding: 6px;")
        ref_l = QVBoxLayout(ref_box)
        ref_l.setSpacing(3)
        self.ref_status_label = QLabel("🖼️ Referencia: No encontrada")
        self.ref_hash_label = QLabel("SHA256: -")
        self.ref_hash_label.setStyleSheet("color: #64748b; font-size: 10px; font-family: monospace;")
        ref_l.addWidget(self.ref_status_label)
        ref_l.addWidget(self.ref_hash_label)
        detail_layout.addWidget(ref_box)

        # Spritesheet
        sheet_box = QFrame()
        sheet_box.setStyleSheet("background: #181b22; border: 1px solid #28303e; border-radius: 4px; padding: 6px;")
        sheet_l = QVBoxLayout(sheet_box)
        sheet_l.setSpacing(3)
        self.sheet_status_label = QLabel("🎬 Spritesheet: No encontrado")
        self.sheet_hash_label = QLabel("SHA256: -")
        self.sheet_hash_label.setStyleSheet("color: #64748b; font-size: 10px; font-family: monospace;")
        sheet_l.addWidget(self.sheet_status_label)
        sheet_l.addWidget(self.sheet_hash_label)
        detail_layout.addWidget(sheet_box)

        layout.addWidget(detail_group)
        layout.addStretch()

    def set_character(self, character: Optional[Character]):
        self.current_character = character
        if not character:
            self.name_label.setText("Sin selección")
            self.id_label.setText("ID único: -")
            self.folder_label.setText("Directorio: -")
            self.status_badge.setText("ESTADO: -")
            self._update_variant_ui()
            return

        self.name_label.setText(character.display_name)
        self.id_label.setText(f"ID único: {character.character_id}")
        self.folder_label.setText(f"Dir: {character.source_dir}")

        if character.is_approved:
            self.status_badge.setText("APROBADO (READ-ONLY)")
            self.status_badge.setStyleSheet("color: #22c55e; font-weight: bold; background: #052e16; border: 1px solid #16a34a; border-radius: 4px; padding: 4px;")
        else:
            self.status_badge.setText("EVALUACIÓN PENDIENTE")
            self.status_badge.setStyleSheet("color: #f59e0b; font-weight: bold; background: #451a03; border: 1px solid #d97706; border-radius: 4px; padding: 4px;")

        self._update_variant_ui()

    def _on_variant_radio_clicked(self, button):
        for v_name, rb in self.radio_buttons.items():
            if rb == button:
                self.current_variant_name = v_name
                break
        self._update_variant_ui()

    def _update_variant_ui(self):
        if not self.current_character:
            self.key_label.setText("Clave canónica: -")
            self.ref_status_label.setText("🖼️ Referencia: -")
            self.ref_hash_label.setText("SHA256: -")
            self.sheet_status_label.setText("🎬 Spritesheet: -")
            self.sheet_hash_label.setText("SHA256: -")
            self.variant_changed.emit(self.current_variant_name, None, None)
            return

        variant = self.current_character.variants.get(self.current_variant_name)
        canonical_key = make_unique_key(self.current_character.character_id, self.current_variant_name)
        self.key_label.setText(f"Clave: {canonical_key}")

        ref_path = None
        sheet_path = None

        if variant and variant.has_reference:
            ref_path = variant.reference_image
            hash_val = variant.file_hashes.get(variant.reference_image.name, "-")
            self.ref_status_label.setText(f"🖼️ Referencia: {variant.reference_image.name}")
            self.ref_status_label.setStyleSheet("color: #22c55e; font-weight: 500;")
            self.ref_hash_label.setText(f"SHA: {hash_val[:16]}...")
        else:
            self.ref_status_label.setText("🖼️ Referencia: FALTANTE")
            self.ref_status_label.setStyleSheet("color: #ef4444; font-weight: 500;")
            self.ref_hash_label.setText("SHA: -")

        if variant and variant.has_spritesheet:
            sheet_path = variant.spritesheet
            hash_val = variant.file_hashes.get(variant.spritesheet.name, "-")
            self.sheet_status_label.setText(f"🎬 Spritesheet: {variant.spritesheet.name}")
            self.sheet_status_label.setStyleSheet("color: #22c55e; font-weight: 500;")
            self.sheet_hash_label.setText(f"SHA: {hash_val[:16]}...")
        else:
            self.sheet_status_label.setText("🎬 Spritesheet: FALTANTE")
            self.sheet_status_label.setStyleSheet("color: #f59e0b; font-weight: 500;")
            self.sheet_hash_label.setText("SHA: -")

        self.variant_changed.emit(self.current_variant_name, ref_path, sheet_path)
