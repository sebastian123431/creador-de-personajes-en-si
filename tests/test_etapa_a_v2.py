import pytest
from pathlib import Path
from PIL import Image

from core.pose_analyzer_v2 import PoseAnalyzerV2
from models.body_part import BodyPart
from models.skeleton import Skeleton, Anchor, OFFICIAL_ANCHOR_NAMES, SKELETON_BONES
from models.pose_frame import PoseFrame
from tests.conftest import create_dummy_png


def test_models_v2_instantiation():
    """Verifica la correcta instanciación y serialización de los modelos V2."""
    bp = BodyPart(
        name="left_forearm",
        bbox=(20, 40, 28, 55),
        pivot_x=24.0,
        pivot_y=40.0,
        parent="left_upper_arm",
        z_index=3,
        mask_path=Path("dataset/body_parts/alex/rnormal/walk_down/01/left_forearm.png"),
        confidence=0.92
    )
    assert bp.name == "left_forearm"
    assert bp.width == 9
    assert bp.height == 16
    d = bp.to_dict()
    assert d["z_index"] == 3
    assert d["confidence"] == 0.92

    skel = Skeleton()
    assert skel.anchor_count == 0
    skel.set_anchor("head", 64, 20, confidence=0.95)
    assert skel.anchor_count == 1
    assert skel.get_anchor("head").x == 64

    pf = PoseFrame(
        character_id="diego_vallenar",
        variant="rbchef",
        animation="walk_down",
        frame_index=1,
        skeleton=skel,
        baseline=176,
        center=(128, 100),
        orientation="down",
        estimated_direction=False
    )
    assert pf.character_id == "diego_vallenar"
    assert not pf.estimated_direction
    assert pf.to_dict()["baseline"] == 176


def test_pose_v2_anchor_count(tmp_path: Path):
    """
    Verifica que PoseAnalyzerV2 detecte e inicialice los 18 anchors oficiales
    sobre un sprite de personaje pixel art.
    """
    sprite_path = tmp_path / "char_sprite.png"
    # Crear un sprite sintético con proporciones de personaje (48x80)
    img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    for y in range(24, 104):
        for x in range(40, 88):
            img.putpixel((x, y), (150, 180, 210, 255))
    img.save(sprite_path, "PNG")

    analyzer = PoseAnalyzerV2()
    skeleton = analyzer.analyze_pose(sprite_path, orientation="down")

    assert skeleton.anchor_count == 18, f"Se esperaban 18 anchors, se obtuvieron {skeleton.anchor_count}"
    for name in OFFICIAL_ANCHOR_NAMES:
        anchor = skeleton.get_anchor(name)
        assert anchor is not None, f"Falta el anchor oficial '{name}'"
        assert anchor.confidence > 0.0
        assert not anchor.is_manual


def test_anchor_manual_override(tmp_path: Path):
    """
    Verifica que una anotación manual del usuario tenga PRIORIDAD ABSOLUTA (Prioridad 1)
    sobre las estimaciones automáticas, con confidence=1.0 e is_manual=True.
    """
    sprite_path = tmp_path / "char_manual.png"
    create_dummy_png(sprite_path, width=64, height=96)

    analyzer = PoseAnalyzerV2()

    # Usuario ajusta manualmente la posición de left_hand y right_foot
    manual = {
        "left_hand": {"x": 75, "y": 110},
        "right_foot": {"x": 82, "y": 125},
    }

    skeleton = analyzer.analyze_pose(sprite_path, manual_annotations=manual)

    l_hand = skeleton.get_anchor("left_hand")
    assert l_hand is not None
    assert l_hand.x == 75
    assert l_hand.y == 110
    assert l_hand.is_manual
    assert l_hand.confidence == 1.0

    r_foot = skeleton.get_anchor("right_foot")
    assert r_foot is not None
    assert r_foot.x == 82
    assert r_foot.y == 125
    assert r_foot.is_manual
    assert r_foot.confidence == 1.0

    # Los anchors no modificados manualmente deben generarse automáticamente
    head = skeleton.get_anchor("head")
    assert head is not None
    assert not head.is_manual


def test_pose_v2_fallback_on_empty(tmp_path: Path):
    """Verifica que un frame vacío use el fallback sin lanzar excepciones."""
    empty_path = tmp_path / "empty.png"
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    img.save(empty_path, "PNG")

    analyzer = PoseAnalyzerV2()
    skeleton = analyzer.analyze_pose(empty_path)
    # Debe retornar un skeleton vacío o con anchors neutrales sin crashear
    assert skeleton is not None


def test_pose_v2_temporal_smoothing(tmp_path: Path):
    """
    Verifica que al proporcionar prev_skeleton de alta confianza, se aplique
    suavizado temporal para evitar saltos bruscos de articulaciones.
    """
    sprite_path = tmp_path / "char_temporal.png"
    create_dummy_png(sprite_path, width=64, height=96)

    analyzer = PoseAnalyzerV2()

    # Frame previo
    prev_skel = Skeleton()
    prev_skel.set_anchor("neck", 32, 40, confidence=0.98)

    smoothed_skel = analyzer.analyze_pose(sprite_path, prev_skeleton=prev_skel)
    neck = smoothed_skel.get_anchor("neck")
    assert neck is not None
    # Debe haber considerado el frame previo en el cálculo
    assert neck.confidence >= 0.80
