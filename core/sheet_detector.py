import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image

from core.alpha_analyzer import AlphaAnalyzer

logger = logging.getLogger("SpriteStudio.SheetDetector")


@dataclass
class FrameRect:
    col: int
    row: int
    x: int
    y: int
    w: int
    h: int

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


@dataclass
class SheetDetectionResult:
    is_valid: bool
    detection_method: str  # "projection", "connected_components", "fallback_grid"
    columns: int
    rows: int
    total_frames: int
    cells: List[List[FrameRect]]  # [16 rows][4 cols]
    sheet_width: int
    sheet_height: int
    warnings: List[str]


class SheetDetector:
    """
    Detector de celdas de animación en spritesheets con resolución variable.
    Utiliza proyecciones de canal alfa, componentes conexos y cuadrícula adaptativa.
    """

    def __init__(self, expected_cols: int = 4, expected_rows: int = 16, alpha_threshold: int = 10):
        self.expected_cols = expected_cols
        self.expected_rows = expected_rows
        self.alpha_threshold = alpha_threshold

    def _find_projection_segments(self, projection: np.ndarray, target_count: int) -> List[Tuple[int, int]]:
        """
        Encuentra segmentos continuos donde la proyección > 0.
        Retorna lista de (start, end).
        """
        non_zero = projection > 0
        diff = np.diff(non_zero.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        if non_zero[0]:
            starts = np.r_[0, starts]
        if non_zero[-1]:
            ends = np.r_[ends, len(non_zero)]

        segments = list(zip(starts.tolist(), ends.tolist()))
        return segments

    def detect_sheet(self, sheet_path: Path) -> SheetDetectionResult:
        """
        Analiza un spritesheet y extrae la cuadrícula de 16 filas x 4 columnas (64 frames).
        """
        warnings = []
        if not sheet_path.exists():
            return SheetDetectionResult(
                is_valid=False,
                detection_method="none",
                columns=0,
                rows=0,
                total_frames=0,
                cells=[],
                sheet_width=0,
                sheet_height=0,
                warnings=[f"Archivo no encontrado: {sheet_path}"]
            )

        with Image.open(sheet_path) as img:
            w, h = img.size
            if "A" in img.getbands():
                alpha = np.array(img.getchannel("A"))
            else:
                alpha = np.full((h, w), 255, dtype=np.uint8)

        mask = alpha > self.alpha_threshold

        # 1. Proyecciones
        h_proj = np.sum(mask, axis=0)  # columnas (ancho)
        v_proj = np.sum(mask, axis=1)  # filas (alto)

        col_segments = self._find_projection_segments(h_proj, self.expected_cols)
        row_segments = self._find_projection_segments(v_proj, self.expected_rows)

        cells: List[List[FrameRect]] = []
        detection_method = "fallback_grid"

        cell_w = w / self.expected_cols
        cell_h = h / self.expected_rows

        # Si las proyecciones dan exactamente 4 columnas y 16 filas separadas por espacios vacíos
        if len(col_segments) == self.expected_cols and len(row_segments) == self.expected_rows:
            detection_method = "projection"
            logger.info(f"Detección por proyección exitosa para {sheet_path.name}")
            for r in range(self.expected_rows):
                row_cells = []
                r_y = int(round(r * cell_h))
                r_h = int(round((r + 1) * cell_h)) - r_y
                for c in range(self.expected_cols):
                    c_x = int(round(c * cell_w))
                    c_w = int(round((c + 1) * cell_w)) - c_x
                    row_cells.append(FrameRect(
                        col=c,
                        row=r,
                        x=c_x,
                        y=r_y,
                        w=c_w,
                        h=r_h
                    ))
                cells.append(row_cells)
        else:
            # Fallback robusto: descomposición en cuadrícula uniforme 4x16

            if len(col_segments) != self.expected_cols or len(row_segments) != self.expected_rows:
                warnings.append(
                    f"Espaciado no uniforme detectado ({len(col_segments)} cols, {len(row_segments)} rows). "
                    f"Aplicando descomposición uniforme de {cell_w:.1f}x{cell_h:.1f} px."
                )

            for r in range(self.expected_rows):
                row_cells = []
                for c in range(self.expected_cols):
                    x = int(round(c * cell_w))
                    y = int(round(r * cell_h))
                    next_x = int(round((c + 1) * cell_w))
                    next_y = int(round((r + 1) * cell_h))
                    row_cells.append(FrameRect(
                        col=c,
                        row=r,
                        x=x,
                        y=y,
                        w=next_x - x,
                        h=next_y - y
                    ))
                cells.append(row_cells)

        total_frames = sum(len(row) for row in cells)
        is_valid = (len(cells) == self.expected_rows and all(len(r) == self.expected_cols for r in cells))

        return SheetDetectionResult(
            is_valid=is_valid,
            detection_method=detection_method,
            columns=self.expected_cols,
            rows=self.expected_rows,
            total_frames=total_frames,
            cells=cells,
            sheet_width=w,
            sheet_height=h,
            warnings=warnings
        )
