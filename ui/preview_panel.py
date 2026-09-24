from pathlib import Path
from typing import Optional
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QPixmap, QColor, QPen, QWheelEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QSlider, QCheckBox, QScrollArea, QFrame
)


class PixelArtCanvas(QWidget):
    """
    Lienzo de dibujo Pixel-Perfect con escalado Nearest-Neighbor estricto,
    patrón de transparencia y cuadrícula de depuración.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap: Optional[QPixmap] = None
        self.scale_factor: float = 2.0
        self.show_grid: bool = False
        self.grid_cols: int = 4
        self.grid_rows: int = 16
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)

    def set_image(self, image_path: Optional[Path]):
        if image_path and image_path.exists():
            self.pixmap = QPixmap(str(image_path))
        else:
            self.pixmap = None
        self.update_canvas_size()
        self.update()

    def set_scale(self, scale: float):
        self.scale_factor = max(0.5, min(scale, 16.0))
        self.update_canvas_size()
        self.update()

    def set_grid(self, enabled: bool, cols: int = 4, rows: int = 16):
        self.show_grid = enabled
        self.grid_cols = cols
        self.grid_rows = rows
        self.update()

    def update_canvas_size(self):
        if self.pixmap and not self.pixmap.isNull():
            w = int(self.pixmap.width() * self.scale_factor)
            h = int(self.pixmap.height() * self.scale_factor)
            self.setFixedSize(w, h)
        else:
            self.setFixedSize(400, 300)

    def paintEvent(self, event):
        painter = QPainter(self)
        # Desactivar suavizado para Pixel Art absoluto
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

        # 1. Dibujar fondo tablero de ajedrez (transparencia)
        check_size = 12
        light_color = QColor(36, 40, 48)
        dark_color = QColor(26, 30, 36)

        for y in range(0, self.height(), check_size):
            for x in range(0, self.width(), check_size):
                is_even = ((x // check_size) + (y // check_size)) % 2 == 0
                painter.fillRect(x, y, check_size, check_size, light_color if is_even else dark_color)

        if not self.pixmap or self.pixmap.isNull():
            painter.setPen(QColor(140, 150, 170))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Sin imagen cargada")
            return

        # 2. Dibujar imagen pixel-perfect
        target_rect = QRectF(0, 0, self.pixmap.width() * self.scale_factor, self.pixmap.height() * self.scale_factor)
        painter.drawPixmap(target_rect.toRect(), self.pixmap)

        # 3. Dibujar cuadrícula si está activa
        if self.show_grid and self.grid_cols > 0 and self.grid_rows > 0:
            grid_pen = QPen(QColor(56, 189, 248, 120), 1, Qt.PenStyle.DashLine)
            painter.setPen(grid_pen)

            col_w = target_rect.width() / self.grid_cols
            row_h = target_rect.height() / self.grid_rows

            for c in range(1, self.grid_cols):
                x = int(c * col_w)
                painter.drawLine(x, 0, x, int(target_rect.height()))

            for r in range(1, self.grid_rows):
                y = int(r * row_h)
                painter.drawLine(0, y, int(target_rect.width()), y)


class PreviewPanel(QWidget):
    """
    Panel central para visualización y análisis visual de referencias y spritesheets.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ref_path: Optional[Path] = None
        self.sheet_path: Optional[Path] = None

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Barra superior de controles de visualización
        top_bar = QHBoxLayout()

        self.view_selector = QComboBox()
        self.view_selector.addItems(["Referencia (Master)", "Spritesheet (Movimientos)"])
        self.view_selector.currentIndexChanged.connect(self._on_view_changed)
        top_bar.addWidget(QLabel("Vista:"))
        top_bar.addWidget(self.view_selector)

        top_bar.addSpacing(15)

        self.zoom_label = QLabel("Zoom: 200%")
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(1, 8)  # 1x a 8x
        self.zoom_slider.setValue(2)
        self.zoom_slider.setFixedWidth(120)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)

        top_bar.addWidget(self.zoom_label)
        top_bar.addWidget(self.zoom_slider)

        top_bar.addSpacing(15)

        self.grid_toggle = QCheckBox("Mostrar Grid (4x16)")
        self.grid_toggle.toggled.connect(self._on_grid_toggled)
        top_bar.addWidget(self.grid_toggle)

        top_bar.addStretch()

        layout.addLayout(top_bar)

        # Área de scroll con canvas
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setStyleSheet("background-color: #101216; border: 1px solid #232832; border-radius: 6px;")

        self.canvas = PixelArtCanvas()
        self.scroll_area.setWidget(self.canvas)
        layout.addWidget(self.scroll_area, 1)

        # Barra inferior de metadata
        self.info_label = QLabel("Dimensiones: - x - | Formato: -")
        self.info_label.setStyleSheet("color: #64748b; font-size: 11px;")
        layout.addWidget(self.info_label)

    def load_variant_assets(self, ref_path: Optional[Path], sheet_path: Optional[Path]):
        self.ref_path = ref_path
        self.sheet_path = sheet_path
        self._on_view_changed()

    def _on_view_changed(self):
        is_sheet = self.view_selector.currentIndex() == 1
        current_path = self.sheet_path if is_sheet else self.ref_path

        self.canvas.set_image(current_path)
        self.grid_toggle.setEnabled(is_sheet)
        if not is_sheet:
            self.canvas.set_grid(False)
        else:
            self.canvas.set_grid(self.grid_toggle.isChecked())

        if current_path and current_path.exists() and self.canvas.pixmap:
            p = self.canvas.pixmap
            self.info_label.setText(
                f"Archivo: {current_path.name} | Dimensiones: {p.width()} x {p.height()} px | "
                f"Alpha: {'Sí' if p.hasAlpha() else 'No'} | Ruta: {current_path}"
            )
        else:
            self.info_label.setText("Sin archivo disponible para esta vista.")

    def _on_zoom_changed(self, value: int):
        self.zoom_label.setText(f"Zoom: {value * 100}%")
        self.canvas.set_scale(float(value))

    def _on_grid_toggled(self, checked: bool):
        is_sheet = self.view_selector.currentIndex() == 1
        self.canvas.set_grid(checked and is_sheet)
