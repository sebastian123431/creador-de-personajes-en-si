import pytest
from pathlib import Path
from PIL import Image
import numpy as np

from core.guard import SourceDatasetGuard, SourceDatasetWriteError
from core.sheet_detector_v2 import SheetDetectorV2, SheetDetectionResultV2
from core.frame_extractor import OFFICIAL_ANIMATION_ROWS


def create_realistic_spritesheet_v2(
    filepath: Path,
    cols: int = 4,
    rows: int = 16,
    cell_w: int = 32,
    cell_h: int = 48,
    bg_mode: str = "alpha",  # "alpha" o "chroma_magenta"
) -> Path:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    total_w = cols * cell_w
    total_h = rows * cell_h

    if bg_mode == "alpha":
        img = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    else:
        # Chroma key magenta
        img = Image.new("RGB", (total_w, total_h), (255, 0, 255))

    for r in range(rows):
        for c in range(cols):
            cx = c * cell_w + cell_w // 4
            cy = r * cell_h + cell_h // 4
            # Dibujar figura pixel art dentro de la celda
            color = (50 + r * 10, 100 + c * 30, 180, 255) if bg_mode == "alpha" else (50 + r * 10, 100 + c * 30, 180)
            for y in range(cy, cy + cell_h // 2):
                for x in range(cx, cx + cell_w // 2):
                    img.putpixel((x, y), color)

    img.save(filepath, "PNG")
    return filepath


def test_sheet_detector_v2_alpha_transparent(tmp_path: Path):
    """
    Verifica la detección completa de 64 frames (16 filas x 4 columnas)
    en un spritesheet con canal alfa transparente.
    """
    sheet_path = tmp_path / "sheet_alpha.png"
    create_realistic_spritesheet_v2(sheet_path, bg_mode="alpha")

    detector = SheetDetectorV2()
    result = detector.detect_sheet(sheet_path)

    assert result.is_valid
    assert result.columns == 4
    assert result.rows == 16
    assert result.total_frames == 64
    assert result.background_type == "alpha"
    assert len(result.cells) == 16

    # Comprobar que los nombres de animación coincidan con los 16 oficiales
    for r_idx, anim_name in enumerate(OFFICIAL_ANIMATION_ROWS):
        cells = result.get_animation_cells(anim_name)
        assert len(cells) == 4
        for c_idx, cell in enumerate(cells):
            assert cell.frame_index == c_idx + 1
            assert cell.animation_name == anim_name
            assert not cell.is_empty
            assert cell.tight_bbox is not None


def test_sheet_detector_v2_chroma_solid(tmp_path: Path):
    """
    Verifica la detección automática de fondo chroma-key (magenta sólido)
    y aislamiento de los 64 frames en un spritesheet RGB sin canal alfa.
    """
    sheet_path = tmp_path / "sheet_chroma.png"
    create_realistic_spritesheet_v2(sheet_path, bg_mode="chroma_magenta")

    detector = SheetDetectorV2()
    result = detector.detect_sheet(sheet_path)

    assert result.is_valid
    assert result.columns == 4
    assert result.rows == 16
    assert result.total_frames == 64
    assert result.background_type == "solid_chroma"
    assert result.background_color == (255, 0, 255)

    # Verificar que los tight_bbox se hayan aislado correctamente del fondo magenta
    c0 = result.get_cell(0, 0)
    assert c0 is not None
    assert not c0.is_empty
    assert c0.tight_bbox is not None


def test_sheet_detector_v2_debug_overlay_and_guard(tmp_path: Path):
    """
    Verifica la generación de la imagen de overlay visual con cuadrículas y etiquetas,
    y comprueba que SourceDatasetGuard impida escribir el overlay en el dataset original.
    """
    sheet_path = tmp_path / "sheet_overlay.png"
    create_realistic_spritesheet_v2(sheet_path, bg_mode="alpha")

    approved_dir = tmp_path / "finished_characters" / "approved" / "personajes al 100"
    approved_dir.mkdir(parents=True)
    guard = SourceDatasetGuard(protected_root=approved_dir)

    detector = SheetDetectorV2(guard=guard)
    out_overlay_path = tmp_path / "debug_sheet_grid.png"

    overlay_img = detector.generate_debug_overlay(sheet_path, output_path=out_overlay_path)

    assert out_overlay_path.exists()
    assert overlay_img.size == (32 * 4, 48 * 16)

    # Intento de guardar overlay dentro de carpeta protegida
    illegal_out = approved_dir / "alex" / "debug.png"
    with pytest.raises(SourceDatasetWriteError):
        detector.generate_debug_overlay(sheet_path, output_path=illegal_out)
