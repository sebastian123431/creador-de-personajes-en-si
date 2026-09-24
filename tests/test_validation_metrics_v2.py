"""
Pruebas para ValidationMetricsV2 y modelos de validación (Fase 3C).
Verifica métricas desacopladas:
- motion_activity_score
- head_identity_score
- palette_integrity_score
- baseline_stability_score
- anchor_continuity_score
- limb_continuity_score
- silhouette_consistency_score
- diagnostic_identity_score
- warnings diagnósticos y review_required
"""

import json
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from core.validation_metrics_v2 import ValidationMetricsV2
from models.skeleton import Skeleton, Anchor
from models.validation_metrics import ValidationMetrics, ValidationResult


def _create_sample_sprite(color=(120, 180, 220, 255), size=(64, 64)) -> Image.Image:
    """Crea un sprite con cabeza, cuerpo y pies para pruebas."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    arr = np.array(img)
    # Cabeza (y: 10..24, x: 22..42)
    arr[10:25, 22:43] = [230, 190, 160, 255]
    # Torso (y: 25..45, x: 20..44)
    arr[25:46, 20:45] = color
    # Piernas / Pies (y: 46..58, x: 22..42)
    arr[46:59, 22:43] = [50, 50, 70, 255]
    return Image.fromarray(arr, mode="RGBA")


def _create_sample_skeleton(frame_idx: int = 0, hand_offset: float = 0.0) -> Skeleton:
    """Crea un skeleton con anchors canónicos oficiales."""
    anchors_list = [
        Anchor("head", 32, 18, 0.95),
        Anchor("neck", 32, 25, 0.95),
        Anchor("chest", 32, 32, 0.95),
        Anchor("hip", 32, 44, 0.95),
        Anchor("left_shoulder", 24, 28, 0.95),
        Anchor("left_elbow", 20, 35, 0.95),
        Anchor("left_hand", int(18 + hand_offset), 42, 0.95),
        Anchor("right_shoulder", 40, 28, 0.95),
        Anchor("right_elbow", 44, 35, 0.95),
        Anchor("right_hand", 46, 42, 0.95),
        Anchor("left_hip", 26, 44, 0.95),
        Anchor("left_knee", 26, 50, 0.95),
        Anchor("left_ankle", 26, 56, 0.95),
        Anchor("left_foot", 26, 58, 0.95),
        Anchor("right_hip", 38, 44, 0.95),
        Anchor("right_knee", 38, 50, 0.95),
        Anchor("right_ankle", 38, 56, 0.95),
        Anchor("right_foot", 38, 58, 0.95),
        Anchor("baseline", 32, 58, 0.95),
    ]
    return Skeleton(anchors={a.name: a for a in anchors_list})


def test_validation_metrics_instantiation_and_serialization(tmp_path):
    """
    Verifica serialización, roundtrip y persistencia atómica de ValidationResult.
    """
    metrics = ValidationMetrics(
        motion_activity_score=0.45,
        head_identity_score=0.98,
        palette_integrity_score=1.0,
        baseline_stability_score=0.95,
        anchor_continuity_score=0.92,
        limb_continuity_score=0.94,
        silhouette_consistency_score=0.96,
        diagnostic_identity_score=0.97,
    )

    result = ValidationResult(
        character_id="alex",
        variant="rnormal",
        animation="walk_down",
        template_version="v2_global",
        metrics=metrics,
        warnings=[],
        review_required=False,
    )

    out_file = tmp_path / "alex_rnormal_walk_down.json"
    result.save(out_file)
    assert out_file.is_file()

    loaded = ValidationResult.load(out_file)
    assert loaded.character_id == "alex"
    assert loaded.variant == "rnormal"
    assert loaded.animation == "walk_down"
    assert loaded.metrics.motion_activity_score == 0.45
    assert loaded.metrics.diagnostic_identity_score == 0.97
    assert not loaded.review_required


def test_motion_activity_score_static_vs_moving():
    """
    Verifica que frames idénticos den actividad 0.0,
    mientras que frames animados den un score positivo.
    """
    sprite = _create_sample_sprite()
    static_frames = [sprite.copy() for _ in range(4)]

    # Movimiento: desplazar brazo o torso en cada frame
    moving_frames = []
    for i in range(4):
        f = sprite.copy()
        arr = np.array(f)
        arr[25:35, 10 + i * 4 : 20 + i * 4] = [200, 100, 100, 255]
        moving_frames.append(Image.fromarray(arr))

    evaluator = ValidationMetricsV2()
    static_score = evaluator.compute_motion_activity(static_frames)
    moving_score = evaluator.compute_motion_activity(moving_frames)

    assert static_score == 0.0
    assert moving_score > 0.05


def test_anchor_jump_detection():
    """
    Detecta saltos abruptos (ej. 50 -> 52 -> 110 -> 53) y emite advertencia ANCHOR_JUMP.
    """
    evaluator = ValidationMetricsV2(max_anchor_jump_px=25.0)

    # Secuencia normal con movimiento suave
    sk0 = _create_sample_skeleton(0, hand_offset=0.0)
    sk1 = _create_sample_skeleton(1, hand_offset=2.0)
    sk2 = _create_sample_skeleton(2, hand_offset=4.0)
    sk3 = _create_sample_skeleton(3, hand_offset=2.0)
    score_normal, warns_normal = evaluator.compute_anchor_continuity([sk0, sk1, sk2, sk3])
    assert score_normal > 0.90
    assert "ANCHOR_JUMP" not in warns_normal

    # Secuencia con salto abrupto anormal (salto de 50px en left_hand)
    sk2_jump = _create_sample_skeleton(2, hand_offset=50.0)
    score_jump, warns_jump = evaluator.compute_anchor_continuity([sk0, sk1, sk2_jump, sk3])
    assert score_jump < score_normal
    assert "ANCHOR_JUMP" in warns_jump


def test_palette_drift_detection():
    """
    Detecta colores nuevos inesperados y emite PALETTE_DRIFT.
    """
    ref = _create_sample_sprite()
    ref_palette = {p for _, p in ref.getcolors() or []}

    evaluator = ValidationMetricsV2(max_unexpected_color_ratio=0.05)

    # Frames con la misma paleta
    clean_frames = [ref.copy() for _ in range(4)]
    score_clean, warns_clean = evaluator.compute_palette_integrity(ref_palette, clean_frames)
    assert score_clean == 1.0
    assert "PALETTE_DRIFT" not in warns_clean

    # Frame con colores extraños (píxeles verde fosforescente inesperados)
    drift_frame = ref.copy()
    drift_arr = np.array(drift_frame)
    drift_arr[15:25, 25:35] = [0, 255, 0, 255]  # Color ajeno a la paleta
    drift_frames = [ref.copy(), Image.fromarray(drift_arr), ref.copy(), ref.copy()]

    score_drift, warns_drift = evaluator.compute_palette_integrity(ref_palette, drift_frames)
    assert score_drift < 1.0
    assert "PALETTE_DRIFT" in warns_drift


def test_baseline_stability():
    """
    Mide la estabilidad de los pies en el suelo y detecta saltos bruscos verticales.
    """
    evaluator = ValidationMetricsV2(max_baseline_variance_px=3.0)

    sprite = _create_sample_sprite()
    stable_frames = [sprite.copy() for _ in range(4)]
    score_stable, warns_stable = evaluator.compute_baseline_stability(stable_frames)
    assert score_stable > 0.90
    assert "BASELINE_INSTABILITY" not in warns_stable

    # Frame con salto vertical anormal (personaje flotando 10px arriba)
    jump_img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    jump_img.paste(sprite, (0, -10))
    unstable_frames = [sprite.copy(), jump_img, sprite.copy(), sprite.copy()]

    score_unstable, warns_unstable = evaluator.compute_baseline_stability(unstable_frames)
    assert score_unstable < score_stable
    assert "BASELINE_INSTABILITY" in warns_unstable


def test_silhouette_consistency_anomaly():
    """
    Detecta escalado extraño o deformación corporal excesiva (SILHOUETTE_ANOMALY).
    """
    ref = _create_sample_sprite()
    evaluator = ValidationMetricsV2()

    # Deformación: agrandar el sprite un 80% o reducirlo a la mitad
    shrunk_img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    small = ref.crop((10, 10, 54, 54)).resize((20, 20), Image.Resampling.NEAREST)
    shrunk_img.paste(small, (22, 22))

    frames_anomaly = [ref.copy(), shrunk_img, ref.copy(), ref.copy()]
    score, warns = evaluator.compute_silhouette_consistency(ref, frames_anomaly)
    assert score < 0.85
    assert "SILHOUETTE_ANOMALY" in warns


def test_diagnostic_identity_score_calculation():
    """
    Verifica que el cálculo diagnóstico ponderado se aplique correctamente
    sin prometer '100% fidelity' o nombres engañosos.
    """
    evaluator = ValidationMetricsV2()
    # 0.35 * 1.0 + 0.30 * 1.0 + 0.20 * 1.0 + 0.15 * 1.0 = 1.0
    perfect = evaluator.compute_diagnostic_identity(1.0, 1.0, 1.0, 1.0)
    assert round(perfect, 4) == 1.0

    # Pérdida de cabeza y paleta
    degraded = evaluator.compute_diagnostic_identity(0.50, 0.60, 0.90, 0.85)
    expected = 0.35 * 0.50 + 0.30 * 0.60 + 0.20 * 0.90 + 0.15 * 0.85  # 0.175 + 0.18 + 0.18 + 0.1275 = 0.6625
    assert abs(degraded - expected) < 1e-4


def test_validation_result_review_required_trigger():
    """
    Verifica que evaluate_animation active review_required cuando hay anomalías.
    """
    evaluator = ValidationMetricsV2(min_diagnostic_score=0.80)
    ref = _create_sample_sprite()
    skels = [_create_sample_skeleton(i) for i in range(4)]
    frames = [ref.copy() for _ in range(4)]

    # Caso limpio
    res_clean = evaluator.evaluate_animation(
        character_id="alex",
        variant="rnormal",
        animation="walk_down",
        template_version="v2_global",
        reference_sprite=ref,
        generated_frames=frames,
        skeletons=skels,
    )
    assert not res_clean.review_required
    assert len(res_clean.warnings) == 0

    # Caso con salto de anchor
    skels_bad = [_create_sample_skeleton(0), _create_sample_skeleton(1, hand_offset=60.0), _create_sample_skeleton(2), _create_sample_skeleton(3)]
    res_bad = evaluator.evaluate_animation(
        character_id="alex",
        variant="rnormal",
        animation="walk_down",
        template_version="v2_global",
        reference_sprite=ref,
        generated_frames=frames,
        skeletons=skels_bad,
    )
    assert res_bad.review_required
    assert "ANCHOR_JUMP" in res_bad.warnings
