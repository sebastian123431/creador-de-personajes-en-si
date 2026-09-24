from pathlib import Path
from typing import Dict, Optional, Tuple
from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import QPainter, QPixmap, QColor, QPen, QBrush, QFont, QMouseEvent
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QScrollArea, QFrame, QMessageBox
)

from core.annotation_manager import AnnotationManager
from models.skeleton import Skeleton, Anchor, OFFICIAL_ANCHOR_NAMES, SKELETON_BONES


class InteractiveAnchorCanvas(QWidget):
    """
    Lienzo interactivo ampliado para edición de anchors con click y arrastre.
    """
    anchor_moved = Signal(str, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap: Optional[QPixmap] = None
        self.skeleton: Skeleton = Skeleton()
        self.scale_factor: float = 4.0
        self.selected_anchor_name: Optional[str] = None
        self.dragging: bool = False
        self.point_radius: int = 6
        self.setMouseTracking(True)

    def set_data(self, image_path: Path, skeleton: Skeleton, scale: float = 4.0):
        if image_path.exists():
            self.pixmap = QPixmap(str(image_path))
        else:
            self.pixmap = None
        self.skeleton = skeleton
        self.scale_factor = scale
        self.update_canvas_size()
        self.update()

    def update_canvas_size(self):
        if self.pixmap and not self.pixmap.isNull():
            w = int(self.pixmap.width() * self.scale_factor)
            h = int(self.pixmap.height() * self.scale_factor)
            self.setFixedSize(w, h)
        else:
            self.setFixedSize(400, 400)

    def _screen_to_sprite_coords(self, px: float, py: float) -> Tuple[int, int]:
        sx = int(px / self.scale_factor)
        sy = int(py / self.scale_factor)
        return sx, sy

    def _sprite_to_screen_coords(self, sx: int, sy: int) -> Tuple[float, float]:
        px = sx * self.scale_factor + (self.scale_factor / 2.0)
        py = sy * self.scale_factor + (self.scale_factor / 2.0)
        return px, py

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            m_pos = event.position()
            # Buscar el anchor más cercano al cursor
            closest_name = None
            min_dist = float("inf")

            for name, anchor in self.skeleton.anchors.items():
                ax, ay = self._sprite_to_screen_coords(anchor.x, anchor.y)
                dist = ((m_pos.x() - ax) ** 2 + (m_pos.y() - ay) ** 2) ** 0.5
                if dist < (self.point_radius * 2.5) and dist < min_dist:
                    min_dist = dist
                    closest_name = name

            if closest_name:
                self.selected_anchor_name = closest_name
                self.dragging = True
                self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.dragging and self.selected_anchor_name:
            m_pos = event.position()
            sx, sy = self._screen_to_sprite_coords(m_pos.x(), m_pos.y())
            if self.pixmap:
                sx = max(0, min(sx, self.pixmap.width() - 1))
                sy = max(0, min(sy, self.pixmap.height() - 1))

            self.skeleton.set_anchor(self.selected_anchor_name, sx, sy, confidence=1.0, is_manual=True)
            self.anchor_moved.emit(self.selected_anchor_name, sx, sy)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

        # 1. Dibujar fondo tablero de ajedrez
        check_size = 12
        c1, c2 = QColor(30, 34, 42), QColor(22, 26, 32)
        for y in range(0, self.height(), check_size):
            for x in range(0, self.width(), check_size):
                is_even = ((x // check_size) + (y // check_size)) % 2 == 0
                painter.fillRect(x, y, check_size, check_size, c1 if is_even else c2)

        if not self.pixmap or self.pixmap.isNull():
            return

        # 2. Dibujar imagen de sprite pixel-perfect
        painter.drawPixmap(0, 0, self.width(), self.height(), self.pixmap)

        # 3. Dibujar huesos (líneas entre anchors)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        bone_pen = QPen(QColor(56, 189, 248, 180), 2, Qt.PenStyle.SolidLine)
        painter.setPen(bone_pen)

        for b_from, b_to in SKELETON_BONES:
            a_from = self.skeleton.get_anchor(b_from)
            a_to = self.skeleton.get_anchor(b_to)
            if a_from and a_to:
                x1, y1 = self._sprite_to_screen_coords(a_from.x, a_from.y)
                x2, y2 = self._sprite_to_screen_coords(a_to.x, a_to.y)
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # 4. Dibujar puntos de anclaje (anchors)
        font = QFont("Segoe UI", 8)
        painter.setFont(font)

        for name, a in self.skeleton.anchors.items():
            px, py = self._sprite_to_screen_coords(a.x, a.y)
            is_selected = (name == self.selected_anchor_name)

            # Color según estado: verde para manual, celeste para auto, rojo para seleccionado
            if is_selected:
                brush = QBrush(QColor(239, 68, 68))
                pen = QPen(QColor(255, 255, 255), 2)
                rad = self.point_radius + 2
            elif a.is_manual:
                brush = QBrush(QColor(34, 197, 94))
                pen = QPen(QColor(240, 253, 244), 1)
                rad = self.point_radius
            else:
                brush = QBrush(QColor(56, 189, 248))
                pen = QPen(QColor(224, 242, 254), 1)
                rad = self.point_radius

            painter.setBrush(brush)
            painter.setPen(pen)
            painter.drawEllipse(QPointF(px, py), rad, rad)

            # Etiqueta de texto sutil
            if is_selected:
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(int(px + 8), int(py + 4), f"{name} ({a.x}, {a.y})")


class AnchorEditorDialog(QDialog):
    """
    Diálogo para inspección y edición manual de anclajes esqueléticos (EDIT ANCHORS).
    """
    def __init__(
        self,
        image_path: Path,
        skeleton: Skeleton,
        character_id: str,
        variant: str,
        animation: str,
        frame_index: int,
        annotation_manager: Optional[AnnotationManager] = None,
        parent=None
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Editor de Articulaciones - {character_id} ({variant}) - {animation} #{frame_index}")
        self.resize(950, 680)

        self.image_path = image_path
        self.skeleton = skeleton
        self.character_id = character_id
        self.variant = variant
        self.animation = animation
        self.frame_index = frame_index
        self.annotation_manager = annotation_manager or AnnotationManager()

        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # 1. Área central de dibujo interactivo con scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setStyleSheet("background: #0f172a; border: 1px solid #334155; border-radius: 6px;")

        self.canvas = InteractiveAnchorCanvas()
        self.canvas.set_data(self.image_path, self.skeleton, scale=4.0)
        self.canvas.anchor_moved.connect(self._on_anchor_moved)
        scroll.setWidget(self.canvas)
        layout.addWidget(scroll, 1)

        # 2. Panel lateral de controles y lista de articulaciones
        side_panel = QFrame()
        side_panel.setFixedWidth(280)
        side_panel.setStyleSheet("background: #181b22; border: 1px solid #28303e; border-radius: 6px; padding: 6px;")
        side_l = QVBoxLayout(side_panel)

        side_l.addWidget(QLabel("<b>ARTICULACIONES (18 ANCHORS)</b>"))
        self.selected_info = QLabel("Haz clic y arrastra cualquier articulación.")
        self.selected_info.setStyleSheet("color: #38bdf8; font-size: 11px;")
        self.selected_info.setWordWrap(True)
        side_l.addWidget(self.selected_info)

        side_l.addSpacing(10)
        self.anchor_selector = QComboBox()
        self.anchor_selector.addItems(OFFICIAL_ANCHOR_NAMES)
        self.anchor_selector.currentTextChanged.connect(self._on_combo_selected)
        side_l.addWidget(self.anchor_selector)

        side_l.addStretch()

        # Botones de acción
        self.save_btn = QPushButton("💾 Guardar Anotación Manual")
        self.save_btn.setObjectName("successButton")
        self.save_btn.clicked.connect(self._on_save_clicked)
        side_l.addWidget(self.save_btn)

        self.close_btn = QPushButton("Cerrar")
        self.close_btn.clicked.connect(self.accept)
        side_l.addWidget(self.close_btn)

        layout.addWidget(side_panel)

    def _on_anchor_moved(self, name: str, x: int, y: int):
        self.selected_info.setText(f"Articulación: <b>{name}</b><br>Posición: X={x}, Y={y}<br>Estado: <i>Manual (Confianza: 100%)</i>")

    def _on_combo_selected(self, name: str):
        self.canvas.selected_anchor_name = name
        a = self.canvas.skeleton.get_anchor(name)
        if a:
            self._on_anchor_moved(name, a.x, a.y)
        self.canvas.update()

    def _on_save_clicked(self):
        saved_p = self.annotation_manager.save_annotation(
            self.character_id,
            self.variant,
            self.animation,
            self.frame_index,
            self.canvas.skeleton
        )
        QMessageBox.information(
            self,
            "Anotación Guardada",
            f"Anclajes manuales guardados exitosamente en:\n{saved_p}"
        )
