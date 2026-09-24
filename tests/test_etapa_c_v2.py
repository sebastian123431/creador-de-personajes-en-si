import json
import pytest
from pathlib import Path
from PIL import Image
import numpy as np

from core.body_part_segmenter import BodyPartSegmenter, BODY_PART_NAMES
from core.layer_resolver import LayerResolver
from core.guard import SourceDatasetGuard, SourceDatasetWriteError
from core.pose_analyzer_v2 import PoseAnalyzerV2
from models.body_part import BodyPart
from models.skeleton import Skeleton
from tests.conftest import create_dummy_png


def create_character_sprite_with_limbs(filepath: Path) -> Path:
    """
    Crea un sprite sintético pixel art de 64x96 con regiones anatómicas diferenciadas:
    - Cabeza: y=16 a y=36, x=24 a x=40
    - Cuello/Torso: y=37 a y=60, x=22 a x=42
    - Brazo izquierdo: y=38 a y=58, x=12 a x=20
    - Brazo derecho: y=38 a y=58, x=44 a x=52
    - Pierna izquierda: y=61 a y=88, x=20 a x=30
    - Pierna derecha: y=61 a y=88, x=34 a x=44
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (64, 96), (0, 0, 0, 0))

    # Cabeza (piel / cabello)
    for y in range(16, 37):
        for x in range(24, 41):
            img.putpixel((x, y), (255, 200, 160, 255))

    # Torso (camisa)
    for y in range(37, 61):
        for x in range(22, 43):
            img.putpixel((x, y), (50, 120, 220, 255))

    # Brazo izquierdo
    for y in range(38, 59):
        for x in range(12, 21):
            img.putpixel((x, y), (40, 100, 190, 255))

    # Brazo derecho
    for y in range(38, 59):
        for x in range(44, 53):
            img.putpixel((x, y), (40, 100, 190, 255))

    # Pierna izquierda (pantalón)
    for y in range(61, 89):
        for x in range(20, 31):
            img.putpixel((x, y), (40, 40, 60, 255))

    # Pierna derecha (pantalón)
    for y in range(61, 89):
        for x in range(34, 45):
            img.putpixel((x, y), (40, 40, 60, 255))

    img.save(filepath, "PNG")
    return filepath


def test_layer_resolver_orientations():
    """
    Verifica que LayerResolver normalice animaciones y resuelva el orden de capas (Z-index)
    adecuadamente para cada punto de vista.
    """
    assert LayerResolver.normalize_orientation("walk_down") == "down"
    assert LayerResolver.normalize_orientation("cook_up") == "up"
    assert LayerResolver.normalize_orientation("idle_left") == "left"
    assert LayerResolver.normalize_orientation("serve_right") == "right"
    assert LayerResolver.normalize_orientation("unknown") == "down"

    # En vista lateral izquierda (left), el brazo derecho es el más lejano (z=0)
    z_left = LayerResolver.get_z_indices_for_orientation("left")
    assert z_left["right_arm"] == 0
    assert z_left["left_arm"] == 4
    assert z_left["head"] == 5

    # En vista lateral derecha (right), el brazo izquierdo es el más lejano (z=0)
    z_right = LayerResolver.get_z_indices_for_orientation("right")
    assert z_right["left_arm"] == 0
    assert z_right["right_arm"] == 4
    assert z_right["head"] == 5

    # En vista frontal (down), la cabeza está arriba (z=4)
    z_down = LayerResolver.get_z_indices_for_orientation("down")
    assert z_down["head"] == 4
    assert z_down["torso"] == 2


def test_layer_resolver_sort_parts():
    """Verifica que sort_parts_for_rendering ordene las partes según el Painter's Algorithm."""
    parts = {
        "head": BodyPart("head", (0, 0, 10, 10), 5.0, 5.0, z_index=5),
        "torso": BodyPart("torso", (0, 11, 10, 20), 5.0, 15.0, z_index=2),
        "right_arm": BodyPart("right_arm", (11, 11, 15, 20), 12.0, 12.0, z_index=0),
    }
    sorted_parts = LayerResolver.sort_parts_for_rendering(parts)
    assert [p.name for p in sorted_parts] == ["right_arm", "torso", "head"]


def test_body_part_segmentation_decomposition(tmp_path: Path):
    """
    Verifica que BodyPartSegmenter descomponga el sprite en las 6 partes canónicas,
    con bboxes correctos, pivotes anatómicos y preservación sin pérdida de píxeles.
    """
    sprite_path = tmp_path / "char_full.png"
    create_character_sprite_with_limbs(sprite_path)

    analyzer = PoseAnalyzerV2()
    skeleton = analyzer.analyze_pose(sprite_path, orientation="down")

    segmenter = BodyPartSegmenter()
    parts, part_images = segmenter.segment_frame(sprite_path, skeleton=skeleton, orientation="down")

    # 1. Comprobar que existen las 6 partes
    assert set(parts.keys()) == set(BODY_PART_NAMES)
    assert set(part_images.keys()) == set(BODY_PART_NAMES)

    # 2. Comprobar que cada parte tiene dimensiones consistentes
    for name in BODY_PART_NAMES:
        p = parts[name]
        img = part_images[name]
        assert p.width > 0
        assert p.height > 0
        assert img.size == (p.width, p.height)
        assert p.pivot_x >= 0
        assert p.pivot_y >= 0

    # 3. Conservación de píxeles: la suma de píxeles con alfa > 0 de todas las partes
    # debe ser exactamente igual al total de píxeles opacos del sprite original.
    orig_img = Image.open(sprite_path).convert("RGBA")
    orig_opaque = np.count_nonzero(np.array(orig_img)[:, :, 3] > 15)

    parts_opaque = sum(
        np.count_nonzero(np.array(img)[:, :, 3] > 15)
        for img in part_images.values()
    )
    assert parts_opaque == orig_opaque, f"Pérdida de píxeles: original {orig_opaque} vs partes {parts_opaque}"


def test_body_part_segmentation_persistence(tmp_path: Path):
    """
    Verifica que save_segmented_parts persista adecuadamente las 6 imágenes PNG y meta.json
    bajo dataset/body_parts/<char>/<variant>/<anim>/<frame>/ y que load_segmented_parts los recupere.
    """
    sprite_path = tmp_path / "char_save.png"
    create_character_sprite_with_limbs(sprite_path)

    body_parts_dir = tmp_path / "body_parts"
    segmenter = BodyPartSegmenter(base_output_dir=body_parts_dir)

    parts, part_images = segmenter.segment_frame(sprite_path, orientation="walk_left")
    saved_dir = segmenter.save_segmented_parts(
        character_id="diego_vallenar",
        variant="rbchef",
        animation="walk_left",
        frame_index=2,
        parts=parts,
        part_images=part_images
    )

    expected_dir = body_parts_dir / "diego_vallenar" / "rbchef" / "walk_left" / "02"
    assert saved_dir == expected_dir
    assert expected_dir.exists()

    # Verificar existencia de los 6 PNGs y meta.json
    for name in BODY_PART_NAMES:
        assert (expected_dir / f"{name}.png").exists()
    assert (expected_dir / "meta.json").exists()

    # Recargar con load_segmented_parts
    loaded_parts = segmenter.load_segmented_parts("diego_vallenar", "rbchef", "walk_left", 2)
    assert loaded_parts is not None
    assert len(loaded_parts) == 6
    assert loaded_parts["head"].name == "head"
    assert loaded_parts["head"].mask_path.exists()
    assert loaded_parts["head"].z_index == 5  # z_index para head en 'left'


def test_body_part_segmentation_guard_protection(tmp_path: Path):
    """
    Verifica que SourceDatasetGuard prevenga cualquier intento de guardar
    partes segmentadas dentro de la carpeta protegida del dataset original.
    """
    approved_dir = tmp_path / "finished_characters" / "approved" / "personajes al 100"
    approved_dir.mkdir(parents=True)
    guard = SourceDatasetGuard(protected_root=approved_dir)

    # Intentar configurar salida dentro de carpeta protegida
    illegal_dir = approved_dir / "alex" / "body_parts"
    segmenter = BodyPartSegmenter(base_output_dir=illegal_dir, guard=guard)

    dummy_parts = {
        "head": BodyPart("head", (0, 0, 5, 5), 2.0, 2.0)
    }
    dummy_imgs = {
        "head": Image.new("RGBA", (5, 5), (255, 0, 0, 255))
    }

    with pytest.raises(SourceDatasetWriteError):
        segmenter.save_segmented_parts(
            "alex", "rnormal", "walk_down", 1, dummy_parts, dummy_imgs
        )


def test_body_part_segmenter_empty_sprite(tmp_path: Path):
    """Verifica que un sprite vacío o transparente no cause excepciones y devuelva partes válidas."""
    empty_path = tmp_path / "empty.png"
    img = Image.new("RGBA", (48, 64), (0, 0, 0, 0))
    img.save(empty_path)

    segmenter = BodyPartSegmenter()
    parts, part_images = segmenter.segment_frame(empty_path, orientation="down")
    assert len(parts) == 6
    for name in BODY_PART_NAMES:
        assert parts[name].confidence == 0.0
