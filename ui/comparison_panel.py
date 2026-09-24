from pathlib import Path
from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QFrame, QSplitter
)
from core.similarity import FrameComparator


class ComparisonDialog(QDialog):
    """
    Diálogo para comparar visual y matemáticamente dos frames con cv2.absdiff.
    """
    def __init__(self, frame_a_path: Optional[Path] = None, frame_b_path: Optional[Path] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Comparador de Frames (cv2.absdiff)")
        self.resize(900, 480)
        self.frame_a = frame_a_path
        self.frame_b = frame_b_path
        self._init_ui()
        self.update_comparison()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        cols = QHBoxLayout()

        # Frame A
        box_a = QVBoxLayout()
        box_a.addWidget(QLabel("<b>Frame A</b>"))
        self.img_a_lbl = QLabel("Sin imagen")
        self.img_a_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_a_lbl.setStyleSheet("background: #181b22; border: 1px solid #334155; min-height: 250px;")
        box_a.addWidget(self.img_a_lbl)
        cols.addLayout(box_a)

        # Frame B
        box_b = QVBoxLayout()
        box_b.addWidget(QLabel("<b>Frame B</b>"))
        self.img_b_lbl = QLabel("Sin imagen")
        self.img_b_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_b_lbl.setStyleSheet("background: #181b22; border: 1px solid #334155; min-height: 250px;")
        box_b.addWidget(self.img_b_lbl)
        cols.addLayout(box_b)

        # Diff (cv2.absdiff)
        box_diff = QVBoxLayout()
        box_diff.addWidget(QLabel("<b>Diferencia Absoluta (cv2.absdiff)</b>"))
        self.img_diff_lbl = QLabel("Calculando...")
        self.img_diff_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_diff_lbl.setStyleSheet("background: #181b22; border: 1px solid #38bdf8; min-height: 250px;")
        box_diff.addWidget(self.img_diff_lbl)
        cols.addLayout(box_diff)

        layout.addLayout(cols)

        # Botón de cerrar
        btn_box = QHBoxLayout()
        btn_box.addStretch()
        close_btn = QPushButton("Cerrar")
        close_btn.clicked.connect(self.accept)
        btn_box.addWidget(close_btn)
        layout.addLayout(btn_box)

    def update_comparison(self):
        if self.frame_a and self.frame_a.exists():
            pm_a = QPixmap(str(self.frame_a)).scaled(240, 240, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
            self.img_a_lbl.setPixmap(pm_a)

        if self.frame_b and self.frame_b.exists():
            pm_b = QPixmap(str(self.frame_b)).scaled(240, 240, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
            self.img_b_lbl.setPixmap(pm_b)

        if self.frame_a and self.frame_b and self.frame_a.exists() and self.frame_b.exists():
            diff_arr = FrameComparator.compute_absdiff(self.frame_a, self.frame_b)
            if diff_arr is not None:
                # Convertir arreglo OpenCV a QPixmap
                h, w, c = diff_arr.shape
                bytes_per_line = c * w
                # cv2 es BGR(A), Qt espera RGB(A)
                if c == 4:
                    fmt = QImage.Format.Format_RGBA8888
                    rgb_diff = cv2.cvtColor(diff_arr, cv2.COLOR_BGRA2RGBA)
                else:
                    fmt = QImage.Format.Format_RGB888
                    rgb_diff = cv2.cvtColor(diff_arr, cv2.COLOR_BGR2RGB)

                qimg = QImage(rgb_diff.data, w, h, bytes_per_line, fmt)
                pm_diff = QPixmap.fromImage(qimg).scaled(240, 240, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
                self.img_diff_lbl.setPixmap(pm_diff)
