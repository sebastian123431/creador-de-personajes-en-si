import pytest
from pathlib import Path
from PIL import Image


def create_dummy_png(filepath: Path, width: int = 64, height: int = 64, color=(255, 100, 100, 255)):
    """Crea una imagen PNG válida con canal alfa RGBA."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (width, height), color)
    img.save(filepath, "PNG")
    return filepath


@pytest.fixture
def sample_dataset_dir(tmp_path: Path) -> Path:
    """
    Crea una estructura de prueba realista con personajes de nombres simples
    y compuestos (apellido, ciudad, apodo).
    """
    characters = [
        "alex",
        "andrea",
        "diego_vallenar",
        "diego_serena",
        "andres_arica",
        "benja_bacaba",
    ]
    variants = ["rnormal", "rbchef", "rnchef"]

    root = tmp_path / "personajes_al_100"
    root.mkdir()

    for char in characters:
        char_dir = root / char
        char_dir.mkdir()

        for var in variants:
            # Imagen de referencia: alex_rnormal.png
            ref_path = char_dir / f"{char}_{var}.png"
            create_dummy_png(ref_path, width=64, height=96, color=(100, 150, 200, 255))

            # Spritesheet de movimiento: movimientos_rnormal.png (ej. 4x16 grid: 256x192)
            sheet_path = char_dir / f"movimientos_{var}.png"
            create_dummy_png(sheet_path, width=256, height=192, color=(50, 80, 120, 255))

    return root
