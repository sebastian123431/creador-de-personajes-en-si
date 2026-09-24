import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.frame_extractor import OFFICIAL_ANIMATION_ROWS
from core.guard import SourceDatasetGuard

logger = logging.getLogger("SpriteStudio.SheetDetectorV2")


@dataclass
class DetectedCellV2:
    """
    Representa una celda detectada en un spritesheet de movimientos V2.
    """
    row: int
    col: int
    animation_name: str
    frame_index: int
    cell_rect: Tuple[int, int, int, int]  # (x, y, w, h)
    tight_bbox: Optional[Tuple[int, int, int, int]] = None  # (min_x, min_y, max_x, max_y)
    is_empty: bool = False

    @property
    def box(self) -> Tuple[int, int, int, int]:
        x, y, w, h = self.cell_rect
        return (x, y, x + w, y + h)


@dataclass
class SheetDetectionResultV2:
    """
    Resultado estructurado del análisis de cuadrícula de spritesheet V2.
    """
    is_valid: bool
    detection_method: str  # "adaptive_valleys", "projection", "uniform_grid"
    background_type: str  # "alpha", "solid_chroma"
    background_color: Optional[Tuple[int, int, int]]
    columns: int
    rows: int
    total_frames: int
    cells: List[List[DetectedCellV2]]  # [16 rows][4 cols]
    sheet_width: int
    sheet_height: int
    warnings: List[str] = field(default_factory=list)

    def get_cell(self, row: int, col: int) -> Optional[DetectedCellV2]:
        if 0 <= row < len(self.cells) and 0 <= col < len(self.cells[row]):
            return self.cells[row][col]
        return None

    def get_animation_cells(self, animation_name: str) -> List[DetectedCellV2]:
        for r_cells in self.cells:
            if r_cells and r_cells[0].animation_name == animation_name:
                return r_cells
        return []


class SheetDetectorV2:
    """
    Detector avanzado de cuadrícula para spritesheets de movimientos (16 filas x 4 columnas).
    Capacidades V2:
    - Detección adaptativa de valles de proyección ante espaciado y márgenes irregulares.
    - Detección automática de fondo transparente (alfa) o color chroma-key/sólido.
    - Extracción de tight_bbox por personaje dentro de cada celda.
    - Generador de overlay visual de depuración con anotaciones (debug_sheet_grid.png).
    """

    def __init__(
        self,
        expected_cols: int = 4,
        expected_rows: int = 16,
        alpha_threshold: int = 15,
        guard: Optional[SourceDatasetGuard] = None,
    ):
        self.expected_cols = expected_cols
        self.expected_rows = expected_rows
        self.alpha_threshold = alpha_threshold
        self.guard = guard

    def detect_sheet(self, sheet_input: Union[Path, str, Image.Image]) -> SheetDetectionResultV2:
        """
        Analiza el spritesheet y detecta la cuadrícula de 16 filas x 4 columnas con precisión.
        """
        warnings: List[str] = []

        if isinstance(sheet_input, (str, Path)):
            path = Path(sheet_input)
            if not path.exists():
                return SheetDetectionResultV2(
                    is_valid=False,
                    detection_method="none",
                    background_type="none",
                    background_color=None,
                    columns=0,
                    rows=0,
                    total_frames=0,
                    cells=[],
                    sheet_width=0,
                    sheet_height=0,
                    warnings=[f"Archivo no encontrado: {path}"]
                )
            img = Image.open(path)
        else:
            img = sheet_input

        w, h = img.size

        # 1. Determinar tipo de fondo (Alfa vs Color sólido/Chroma-Key)
        foreground_mask, bg_type, bg_color = self._extract_foreground_mask(img)

        # 2. Partición horizontal (4 columnas) y vertical (16 filas)
        col_splits, method_cols = self._find_split_boundaries(
            projection=np.sum(foreground_mask, axis=0),
            dimension_len=w,
            target_count=self.expected_cols
        )

        row_splits, method_rows = self._find_split_boundaries(
            projection=np.sum(foreground_mask, axis=1),
            dimension_len=h,
            target_count=self.expected_rows
        )

        detection_method = "adaptive_valleys" if (method_cols == "valley" or method_rows == "valley") else "uniform_grid"

        # 3. Construir celdas y detectar bounding box ajustado (tight_bbox) de cada personaje
        cells: List[List[DetectedCellV2]] = []

        for r_idx in range(self.expected_rows):
            anim_name = OFFICIAL_ANIMATION_ROWS[r_idx] if r_idx < len(OFFICIAL_ANIMATION_ROWS) else f"anim_{r_idx}"
            row_y1 = row_splits[r_idx]
            row_y2 = row_splits[r_idx + 1]
            row_cells: List[DetectedCellV2] = []

            for c_idx in range(self.expected_cols):
                col_x1 = col_splits[c_idx]
                col_x2 = col_splits[c_idx + 1]
                cw = col_x2 - col_x1
                ch = row_y2 - row_y1

                # Extraer máscara local para encontrar píxeles del personaje
                sub_mask = foreground_mask[row_y1:row_y2, col_x1:col_x2]
                sub_ys, sub_xs = np.where(sub_mask)

                if len(sub_xs) == 0:
                    tight_box = None
                    is_empty = True
                else:
                    tight_box = (
                        col_x1 + int(np.min(sub_xs)),
                        row_y1 + int(np.min(sub_ys)),
                        col_x1 + int(np.max(sub_xs)),
                        row_y1 + int(np.max(sub_ys)),
                    )
                    is_empty = False

                cell = DetectedCellV2(
                    row=r_idx,
                    col=c_idx,
                    animation_name=anim_name,
                    frame_index=c_idx + 1,
                    cell_rect=(col_x1, row_y1, cw, ch),
                    tight_bbox=tight_box,
                    is_empty=is_empty
                )
                row_cells.append(cell)
            cells.append(row_cells)

        total_frames = sum(len(r) for r in cells)
        is_valid = len(cells) == self.expected_rows and all(len(r) == self.expected_cols for r in cells)

        return SheetDetectionResultV2(
            is_valid=is_valid,
            detection_method=detection_method,
            background_type=bg_type,
            background_color=bg_color,
            columns=self.expected_cols,
            rows=self.expected_rows,
            total_frames=total_frames,
            cells=cells,
            sheet_width=w,
            sheet_height=h,
            warnings=warnings
        )

    def generate_debug_overlay(
        self,
        sheet_input: Union[Path, str, Image.Image],
        detection_result: Optional[SheetDetectionResultV2] = None,
        output_path: Optional[Path] = None,
    ) -> Image.Image:
        """
        Genera una imagen con anotaciones visuales y cuadrículas de depuración:
        - Líneas cian: celdas de cuadrícula detectadas.
        - Rectángulos amarillos: tight_bbox del personaje dentro de la celda.
        - Etiquetas de texto con el nombre de animación y número de frame.
        """
        if isinstance(sheet_input, (str, Path)):
            base_img = Image.open(str(sheet_input)).convert("RGBA")
        else:
            base_img = sheet_input.convert("RGBA")

        result = detection_result or self.detect_sheet(base_img)

        # Crear capa de dibujo
        overlay = base_img.copy()
        draw = ImageDraw.Draw(overlay)

        # Colores de anotación
        grid_color = (0, 220, 255, 180)      # Cian semitransparente para celda
        tight_color = (255, 230, 0, 230)     # Amarillo brillante para personaje
        text_bg = (15, 23, 42, 200)          # Fondo oscuro para legibilidad

        for r_cells in result.cells:
            for cell in r_cells:
                x, y, w, h = cell.cell_rect
                # 1. Dibujar celda de cuadrícula
                draw.rectangle([x, y, x + w - 1, y + h - 1], outline=grid_color, width=1)

                # 2. Dibujar tight_bbox si existe
                if cell.tight_bbox:
                    bx1, by1, bx2, by2 = cell.tight_bbox
                    draw.rectangle([bx1, by1, bx2, by2], outline=tight_color, width=1)

                # 3. Dibujar indicador sutil de frame
                label = f"{cell.animation_name} #{cell.frame_index}"
                draw.rectangle([x + 2, y + 2, x + min(w - 2, 70), y + 12], fill=text_bg)
                draw.text((x + 4, y + 2), label[:11], fill=(255, 255, 255))

        if output_path:
            out_p = Path(output_path).resolve()
            if self.guard:
                self.guard.assert_can_write(out_p, operation_desc="guardado de overlay de depuración")
            out_p.parent.mkdir(parents=True, exist_ok=True)
            overlay.save(out_p, "PNG")
            logger.info(f"Overlay de depuración guardado en: {out_p}")

        return overlay

    def _extract_foreground_mask(self, img: Image.Image) -> Tuple[np.ndarray, str, Optional[Tuple[int, int, int]]]:
        """
        Determina si el spritesheet usa transparencia de canal alfa o color sólido/chroma-key.
        Retorna (foreground_mask_bool, "alpha"|"solid_chroma", bg_color).
        """
        w, h = img.size
        has_alpha_band = "A" in img.getbands()

        if has_alpha_band:
            alpha = np.array(img.getchannel("A"))
            # Si hay píxeles transparentes reales, usar canal alfa
            if np.any(alpha <= self.alpha_threshold):
                return alpha > self.alpha_threshold, "alpha", None

        # Si no hay canal alfa o todos los píxeles tienen alfa=255, analizar color de fondo por esquinas
        rgb_img = img.convert("RGB")
        rgb_arr = np.array(rgb_img)

        # Muestrear las 4 esquinas
        c1 = rgb_arr[0, 0]
        c2 = rgb_arr[0, w - 1]
        c3 = rgb_arr[h - 1, 0]
        c4 = rgb_arr[h - 1, w - 1]

        corners = [tuple(c1), tuple(c2), tuple(c3), tuple(c4)]
        # Determinar color de fondo dominante en las esquinas
        most_common_bg = max(set(corners), key=corners.count)
        bg_col = np.array(most_common_bg, dtype=np.int32)

        # Distancia euclidiana de color respecto al color de fondo
        diff = np.abs(rgb_arr.astype(np.int32) - bg_col)
        dist = np.sum(diff, axis=-1)

        # Los píxeles con diferencia significativa son primer plano (personajes)
        foreground_mask = dist > 18

        # Si el fondo abarca al menos el 15% del total de píxeles, confirmar solid_chroma
        if np.sum(~foreground_mask) > (w * h * 0.15):
            return foreground_mask, "solid_chroma", most_common_bg

        # Si todo parece ocupado o fondo ambiguo, tratar todo como foreground
        return np.ones((h, w), dtype=bool), "alpha", None

    def _find_split_boundaries(
        self,
        projection: np.ndarray,
        dimension_len: int,
        target_count: int
    ) -> Tuple[List[int], str]:
        """
        Encuentra los límites de división de columnas o filas.
        Busca valles locales en la proyección dentro de ventanas alrededor de la cuadrícula uniforme.
        """
        step = dimension_len / target_count
        splits = [0]
        used_valleys = False

        window_radius = max(3, int(round(step * 0.18)))

        for i in range(1, target_count):
            nominal_pos = int(round(i * step))
            w_start = max(splits[-1] + 1, nominal_pos - window_radius)
            w_end = min(dimension_len - 1, nominal_pos + window_radius)

            if w_end > w_start:
                sub_proj = projection[w_start:w_end]
                min_idx_rel = int(np.argmin(sub_proj))
                best_split = w_start + min_idx_rel

                # Si el valle es significativamente más bajo que la media local
                if sub_proj[min_idx_rel] < (np.mean(sub_proj) + 1):
                    splits.append(best_split)
                    used_valleys = True
                    continue

            splits.append(nominal_pos)

        splits.append(dimension_len)
        method = "valley" if used_valleys else "uniform"
        return splits, method
