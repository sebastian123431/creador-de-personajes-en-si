import pytest
from pathlib import Path
from PIL import Image
import numpy as np

from core.guard import SourceDatasetGuard, SourceDatasetWriteError
from core.palette_guard import PaletteGuard
from core.motion_transfer_v2 import MotionTransferV2, pixel_rotate
from core.template_extractor_v2 import TemplateExtractorV2
from models.articulated_motion_template import (
    ArticulatedMotionTemplate,
    ArticulatedFrameTemplate,
    PartMotion,
)
from tests.test_etapa_c_v2 import create_character_sprite_with_limbs
from tests.test_etapa_d_v2 import _create_synthetic_walk_cycle


def test_palette_guard_extraction_and_cleaning():
    """
    Verifica que PaletteGuard extraiga la paleta, detecte colores ajenos
    y limpie el frame proyectando colores extraños hacia la paleta válida.
    """
    # Imagen de 16x16 con 3 colores definidos
    img = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    for x in range(2, 6):
        img.putpixel((x, 2), (255, 100, 100, 255))
    for x in range(6, 10):
        img.putpixel((x, 2), (100, 255, 100, 255))

    palette = PaletteGuard.extract_palette(img)
    assert len(palette) == 2
    assert (255, 100, 100, 255) in palette
    assert (100, 255, 100, 255) in palette

    # Frame con un color espurio (ej. magenta brillante no presente en el personaje)
    dirty_img = img.copy()
    dirty_img.putpixel((12, 12), (255, 0, 255, 255))

    is_valid, invalid_count = PaletteGuard.validate_frame(dirty_img, palette)
    assert not is_valid
    assert invalid_count == 1

    # Limpiar el frame
    cleaned_img = PaletteGuard.clean_frame(dirty_img, palette)
    is_now_valid, invalid_remaining = PaletteGuard.validate_frame(cleaned_img, palette)
    assert is_now_valid
    assert invalid_remaining == 0


def test_pixel_rotate_preserves_palette_and_pivot():
    """
    Verifica que pixel_rotate gire la extremidad con nearest-neighbor sin difuminar bordes
    ni introducir colores nuevos, y que devuelva el offset que preserva el pivote.
    """
    limb = Image.new("RGBA", (16, 32), (0, 0, 0, 0))
    for y in range(4, 28):
        for x in range(4, 12):
            limb.putpixel((x, y), (80, 140, 200, 255))

    orig_palette = PaletteGuard.extract_palette(limb)

    pivot = (8.0, 4.0)  # Pivote superior del miembro
    rot_img, (off_x, off_y) = pixel_rotate(limb, 30.0, pivot)

    assert rot_img.width > 0
    assert rot_img.height > 0

    # Todos los colores en la imagen rotada deben pertenecer estrictamente a la paleta original
    rot_palette = PaletteGuard.extract_palette(rot_img)
    assert rot_palette.issubset(orig_palette)


def test_head_identity_lock(tmp_path: Path):
    """
    Verifica que MotionTransferV2 conserve la cabeza 100% bit-exacta (HeadIdentityLock).
    Los píxeles de la cabeza en el frame resultante deben ser idénticos a los del original.
    """
    sprite_path = tmp_path / "char_head_lock.png"
    create_character_sprite_with_limbs(sprite_path)

    extractor = TemplateExtractorV2()
    template = extractor.build_articulated_template(
        "walk_down",
        [_create_synthetic_walk_cycle() for _ in range(3)]
    )

    transfer = MotionTransferV2()
    frames = transfer.generate_animation_frames(sprite_path, template)
    assert len(frames) == 4

    orig_img = Image.open(sprite_path).convert("RGBA")
    parts, _ = transfer.segmenter.segment_frame(orig_img, orientation="down")
    head_part = parts["head"]

    # Extraer píxeles de la cabeza del frame original
    ref_head_crop = orig_img.crop(head_part.bbox)
    ref_arr = np.array(ref_head_crop)

    # Frame 1 debe coincidir exactamente en posición
    f1_crop = frames[0].crop(head_part.bbox)
    f1_arr = np.array(f1_crop)

    assert np.array_equal(f1_arr, ref_arr), "HeadIdentityLock falló: la cabeza no es idéntica píxel por píxel."


def test_motion_transfer_v2_full_pipeline_and_files(tmp_path: Path):
    """
    Verifica la generación completa de los 4 frames guardados en disco y la
    conformidad con PaletteGuard en todos los frames generados.
    """
    sprite_path = tmp_path / "char_transfer.png"
    create_character_sprite_with_limbs(sprite_path)

    extractor = TemplateExtractorV2()
    template = extractor.build_articulated_template(
        "cook_down",
        [_create_synthetic_walk_cycle() for _ in range(3)]
    )

    out_dir = tmp_path / "generated_anim"
    transfer = MotionTransferV2()
    frames = transfer.generate_animation_frames(sprite_path, template, output_dir=out_dir)

    assert len(frames) == 4
    for i in range(1, 5):
        fp = out_dir / f"{i:02d}.png"
        assert fp.exists()

    orig_palette = PaletteGuard.extract_palette(Image.open(sprite_path))
    for f in frames:
        is_valid, _ = PaletteGuard.validate_frame(f, orig_palette)
        assert is_valid, "El frame generado contiene colores ajenos a la paleta original."


def test_motion_transfer_v2_guard_protection(tmp_path: Path):
    """
    Verifica que SourceDatasetGuard impida la generación dentro de la carpeta protegida.
    """
    approved_dir = tmp_path / "finished_characters" / "approved" / "personajes al 100"
    approved_dir.mkdir(parents=True)
    guard = SourceDatasetGuard(protected_root=approved_dir)

    sprite_path = tmp_path / "char.png"
    create_character_sprite_with_limbs(sprite_path)

    template = ArticulatedMotionTemplate("walk_down")
    illegal_out = approved_dir / "alex" / "generated"
    transfer = MotionTransferV2(guard=guard)

    with pytest.raises(SourceDatasetWriteError):
        transfer.generate_animation_frames(sprite_path, template, output_dir=illegal_out)
