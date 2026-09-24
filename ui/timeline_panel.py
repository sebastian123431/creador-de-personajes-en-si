from pathlib import Path
from typing import Dict, List, Optional
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QSlider, QCheckBox, QFrame
)

from core.frame_extractor import OFFICIAL_ANIMATION_ROWS
from models.animation import Animation
from models.frame import Frame


class TimelinePanel(QWidget):
    """
    Panel de línea de tiempo y control de reproducción de animación (4 frames).
    """
    frame_changed = Signal(object, int)  # (Frame | Path | None, current_index_0_based)
    onion_skin_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.animations: Dict[str, Animation] = {}
        self.current_anim_name: str = "walk_down"
        self.current_frame_idx: int = 0
        self.is_playing: bool = False
        self.fps: int = 6

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer_tick)

        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)

        # 1. Selector de Animación (16 animaciones oficiales)
        layout.addWidget(QLabel("<b>Animación:</b>"))
        self.anim_combo = QComboBox()
        self.anim_combo.addItems(OFFICIAL_ANIMATION_ROWS)
        self.anim_combo.setCurrentText("walk_down")
        self.anim_combo.currentTextChanged.connect(self._on_anim_selected)
        layout.addWidget(self.anim_combo)

        layout.addSpacing(10)

        # 2. Botones de Frame Directo [1] [2] [3] [4]
        self.frame_buttons: List[QPushButton] = []
        for i in range(1, 5):
            btn = QPushButton(f"[{i}]")
            btn.setFixedWidth(34)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, idx=i-1: self.set_frame_index(idx))
            self.frame_buttons.append(btn)
            layout.addWidget(btn)

        layout.addSpacing(10)

        # 3. Controles de Reproducción
        self.prev_btn = QPushButton("◀")
        self.prev_btn.setFixedWidth(30)
        self.prev_btn.clicked.connect(self.step_backward)
        layout.addWidget(self.prev_btn)

        self.play_btn = QPushButton("▶ PLAY")
        self.play_btn.setFixedWidth(75)
        self.play_btn.clicked.connect(self.toggle_play)
        layout.addWidget(self.play_btn)

        self.next_btn = QPushButton("▶")
        self.next_btn.setFixedWidth(30)
        self.next_btn.clicked.connect(self.step_forward)
        layout.addWidget(self.next_btn)

        layout.addSpacing(15)

        # 4. Velocidad FPS
        self.fps_label = QLabel(f"{self.fps} FPS")
        self.fps_slider = QSlider(Qt.Orientation.Horizontal)
        self.fps_slider.setRange(1, 24)
        self.fps_slider.setValue(self.fps)
        self.fps_slider.setFixedWidth(90)
        self.fps_slider.valueChanged.connect(self._on_fps_changed)
        layout.addWidget(self.fps_label)
        layout.addWidget(self.fps_slider)

        layout.addSpacing(15)

        # 5. Opciones avanzadas (Loop, Onion Skin)
        self.loop_check = QCheckBox("Loop")
        self.loop_check.setChecked(True)
        layout.addWidget(self.loop_check)

        self.onion_check = QCheckBox("Onion Skin")
        self.onion_check.toggled.connect(self.onion_skin_toggled.emit)
        layout.addWidget(self.onion_check)

        layout.addStretch()

    def set_animations(self, animations: Dict[str, Animation]):
        """Carga las animaciones extraídas y actualiza el estado de la línea de tiempo."""
        self.animations = animations
        if self.current_anim_name not in self.animations and self.animations:
            self.current_anim_name = list(self.animations.keys())[0]
            self.anim_combo.setCurrentText(self.current_anim_name)

        has_data = len(self.animations) > 0
        for btn in self.frame_buttons:
            btn.setEnabled(has_data)
        self.play_btn.setEnabled(has_data)
        self.prev_btn.setEnabled(has_data)
        self.next_btn.setEnabled(has_data)

        self.set_frame_index(0)

    def _on_anim_selected(self, anim_name: str):
        self.current_anim_name = anim_name
        self.set_frame_index(0)

    def set_frame_index(self, idx: int):
        self.current_frame_idx = idx % 4

        # Actualizar visualización de botones
        for i, btn in enumerate(self.frame_buttons):
            btn.setChecked(i == self.current_frame_idx)

        frame_obj = self.get_current_frame()
        self.frame_changed.emit(frame_obj, self.current_frame_idx)

    def get_current_frame(self) -> Optional[Frame]:
        anim = self.animations.get(self.current_anim_name)
        if anim and anim.frames and len(anim.frames) > self.current_frame_idx:
            return anim.frames[self.current_frame_idx]
        return None

    def get_frame_at(self, idx: int) -> Optional[Frame]:
        anim = self.animations.get(self.current_anim_name)
        if anim and anim.frames and 0 <= idx < len(anim.frames):
            return anim.frames[idx]
        return None

    def toggle_play(self):
        if self.is_playing:
            self.stop()
        else:
            self.play()

    def play(self):
        self.is_playing = True
        self.play_btn.setText("⏸ PAUSE")
        interval_ms = int(1000 / self.fps)
        self.timer.start(interval_ms)

    def stop(self):
        self.is_playing = False
        self.play_btn.setText("▶ PLAY")
        self.timer.stop()

    def step_forward(self):
        self.set_frame_index((self.current_frame_idx + 1) % 4)

    def step_backward(self):
        self.set_frame_index((self.current_frame_idx - 1) % 4)

    def _on_timer_tick(self):
        next_idx = self.current_frame_idx + 1
        if next_idx >= 4:
            if not self.loop_check.isChecked():
                self.stop()
                return
            next_idx = 0
        self.set_frame_index(next_idx)

    def _on_fps_changed(self, val: int):
        self.fps = val
        self.fps_label.setText(f"{val} FPS")
        if self.is_playing:
            self.timer.setInterval(int(1000 / self.fps))
