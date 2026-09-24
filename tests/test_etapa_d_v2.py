import json
from typing import Dict, List, Optional
import pytest
from pathlib import Path
import numpy as np

from core.guard import SourceDatasetGuard, SourceDatasetWriteError
from core.template_extractor_v2 import (
    TemplateExtractorV2,
    _normalize_angle_deg,
    _calculate_segment_angle_deg,
    _filter_outliers_mad,
)
from models.articulated_motion_template import (
    ArticulatedMotionTemplate,
    ArticulatedFrameTemplate,
    PartMotion,
)
from models.skeleton import Skeleton, OFFICIAL_ANCHOR_NAMES


def test_models_instantiation_and_serialization():
    """Verifica la serialización y deserialización de modelos de plantillas V2."""
    pm = PartMotion(dx=2.5, dy=-1.2, angle_deg=14.5, scale=1.0, confidence=0.95)
    pm_dict = pm.to_dict()
    assert pm_dict["dx"] == 2.5
    assert pm_dict["angle_deg"] == 14.5

    re_pm = PartMotion.from_dict(pm_dict)
    assert re_pm.dx == 2.5
    assert re_pm.angle_deg == 14.5

    ft = ArticulatedFrameTemplate(
        frame_index=2,
        root_dx=0.0,
        root_dy=-1.0,
        parts={"head": pm, "torso": PartMotion(dx=0.0, dy=0.0)},
        anchors_rel={"head": {"dx": 0.0, "dy": -1.0}}
    )
    ft_dict = ft.to_dict()
    assert ft_dict["frame_index"] == 2
    assert "head" in ft_dict["parts"]

    re_ft = ArticulatedFrameTemplate.from_dict(ft_dict)
    assert re_ft.frame_index == 2
    assert re_ft.parts["head"].dx == 2.5

    template = ArticulatedMotionTemplate(
        animation_name="walk_down",
        frame_count=4,
        samples_used=14,
        outliers_detected=3,
        frames=[ft]
    )
    assert template.orientation == "down"
    t_dict = template.to_dict()
    assert t_dict["animation_name"] == "walk_down"
    assert t_dict["samples_used"] == 14

    re_t = ArticulatedMotionTemplate.from_dict(t_dict)
    assert re_t.animation_name == "walk_down"
    assert re_t.samples_used == 14
    assert len(re_t.frames) == 1


def test_mad_outlier_filtering():
    """
    Verifica que el algoritmo MAD descarte muestras anómalas (outliers)
    sin distorsionar la mediana representativa.
    """
    # Muestras normales de desplazamiento de cadera: ~4.0px
    normal_values = [3.8, 4.0, 4.1, 3.9, 4.2, 4.0, 3.9, 4.1]
    # Muestra corrupta/anómala: 48.0px
    values_with_outlier = normal_values + [48.0]

    inliers, outliers_count = _filter_outliers_mad(values_with_outlier, threshold=3.5)

    assert outliers_count == 1
    assert 48.0 not in inliers
    assert len(inliers) == len(normal_values)
    assert abs(np.median(inliers) - 4.0) < 0.15


def test_angle_helpers():
    """Verifica las funciones matemáticas de ángulos y normalización."""
    assert _normalize_angle_deg(0.0) == 0.0
    assert _normalize_angle_deg(190.0) == -170.0
    assert _normalize_angle_deg(-200.0) == 160.0

    # Ángulo horizontal a la derecha: 0 grados
    assert abs(_calculate_segment_angle_deg((10, 10), (20, 10)) - 0.0) < 1e-4
    # Ángulo vertical hacia abajo (coordenadas de pantalla Y invertido): +90 grados
    assert abs(_calculate_segment_angle_deg((10, 10), (10, 20)) - 90.0) < 1e-4


def _create_synthetic_walk_cycle() -> List[Skeleton]:
    """Genera 4 esqueletos sintéticos que simulan un ciclo de caminata estándar."""
    cycle = []
    for f in range(1, 5):
        sk = Skeleton()
        # Centro base
        sk.set_anchor("head", 32, 16)
        sk.set_anchor("neck", 32, 24)
        sk.set_anchor("spine", 32, 36)
        sk.set_anchor("pelvis", 32, 48)

        # Brazos oscilantes
        swing = (1 if f % 2 == 0 else -1) * 4
        sk.set_anchor("left_shoulder", 22, 26)
        sk.set_anchor("left_elbow", 18 + swing, 36)
        sk.set_anchor("left_hand", 16 + swing * 2, 48)

        sk.set_anchor("right_shoulder", 42, 26)
        sk.set_anchor("right_elbow", 46 - swing, 36)
        sk.set_anchor("right_hand", 48 - swing * 2, 48)

        # Piernas oscilantes
        sk.set_anchor("left_hip", 26, 48)
        sk.set_anchor("left_knee", 26 - swing, 64)
        sk.set_anchor("left_foot", 24 - swing * 2, 80)
        sk.set_anchor("left_toe", 24 - swing * 2, 84)

        sk.set_anchor("right_hip", 38, 48)
        sk.set_anchor("right_knee", 38 + swing, 64)
        sk.set_anchor("right_foot", 40 + swing * 2, 80)
        sk.set_anchor("right_toe", 40 + swing * 2, 84)

        cycle.append(sk)
    return cycle


def test_extract_motion_from_sequence():
    """
    Verifica que la extracción de una secuencia de 4 frames calcule
    desplazamientos Delta = 0 para el frame 1 y magnitudes cinemáticas para los siguientes.
    """
    extractor = TemplateExtractorV2()
    cycle = _create_synthetic_walk_cycle()

    seq_data = extractor.extract_motion_from_sequence(cycle)
    assert len(seq_data) == 4

    # Frame 1: Pose base de referencia
    f1 = seq_data[0]
    assert f1["frame_index"] == 1
    assert f1["root_dx"] == 0.0
    assert f1["root_dy"] == 0.0
    assert f1["parts"]["left_arm"]["dx"] == 0.0
    assert f1["parts"]["left_arm"]["angle_deg"] == 0.0

    # Frame 2: Hay oscilación en brazos y piernas
    f2 = seq_data[1]
    assert f2["frame_index"] == 2
    assert f2["parts"]["left_arm"]["angle_deg"] != 0.0
    assert f2["parts"]["right_arm"]["angle_deg"] != 0.0


def test_build_articulated_template_robustness():
    """
    Verifica que al agregar múltiples personajes para una animación,
    un personaje anómalo no corrompa la plantilla final.
    """
    extractor = TemplateExtractorV2()

    # 4 personajes normales
    sequences = [_create_synthetic_walk_cycle() for _ in range(4)]

    # 1 personaje anómalo con salto extremo de 200px en brazo izquierdo durante la animación
    corrupt_cycle = _create_synthetic_walk_cycle()
    corrupt_cycle[1].set_anchor("left_hand", 250, 250)
    corrupt_cycle[2].set_anchor("left_hand", 250, 250)
    sequences.append(corrupt_cycle)

    template = extractor.build_articulated_template("walk_down", sequences)

    assert template.animation_name == "walk_down"
    assert template.orientation == "down"
    assert template.samples_used == 5
    assert template.outliers_detected > 0
    assert len(template.frames) == 4

    # Verificar que el brazo izquierdo en frame 2 no tenga la distorsión de 250px
    f2 = template.get_frame(2)
    assert f2 is not None
    assert abs(f2.parts["left_arm"].dx) < 30.0  # Protegido por filtro MAD


def test_template_io_and_guard_protection(tmp_path: Path):
    """
    Verifica guardado y carga de la plantilla en disco, y protección
    del dataset original mediante SourceDatasetGuard.
    """
    templates_dir = tmp_path / "templates_v2"
    extractor = TemplateExtractorV2(base_templates_dir=templates_dir)

    template = extractor.build_articulated_template(
        "cook_up",
        [_create_synthetic_walk_cycle() for _ in range(3)]
    )

    # 1. Guardar plantilla
    saved_path = extractor.save_template(template)
    expected_path = templates_dir / "cook_up.json"
    assert saved_path == expected_path
    assert expected_path.exists()

    # 2. Cargar plantilla
    loaded_t = extractor.load_template("cook_up")
    assert loaded_t is not None
    assert loaded_t.animation_name == "cook_up"
    assert loaded_t.orientation == "up"
    assert len(loaded_t.frames) == 4

    # 3. Protección de SourceDatasetGuard
    approved_dir = tmp_path / "finished_characters" / "approved" / "personajes al 100"
    approved_dir.mkdir(parents=True)
    guard = SourceDatasetGuard(protected_root=approved_dir)

    illegal_dir = approved_dir / "alex" / "templates_v2"
    guarded_extractor = TemplateExtractorV2(base_templates_dir=illegal_dir, guard=guard)

    with pytest.raises(SourceDatasetWriteError):
        guarded_extractor.save_template(template)
