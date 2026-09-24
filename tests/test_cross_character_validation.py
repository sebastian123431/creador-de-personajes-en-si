"""
Pruebas para CrossCharacterValidator y Auditoría Visual (Fase 3D).
Valida transferencia entre personajes desacoplados, estructura de audit_samples,
comparativas visuales y protección estricta del dataset maestro.
"""

import json
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from core.cross_character_validator import CrossCharacterValidator
from core.guard import SourceDatasetGuard
from core.incremental_training_engine import IncrementalTrainingEngineV2
from core.template_extractor_v2 import TemplateExtractorV2
from models.articulated_motion_template import ArticulatedMotionTemplate
from models.dataset_split import DatasetSplit
from models.skeleton import Skeleton, Anchor
from services.dataset_service import DatasetService


def _create_synthetic_character(char_dir: Path):
    char_dir.mkdir(parents=True, exist_ok=True)
    for v in ("rnormal", "rbchef", "rnchef"):
        # Imagen de referencia
        ref_img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        arr = np.array(ref_img)
        arr[10:25, 22:43] = [230, 190, 160, 255]  # Cabeza
        arr[25:46, 20:45] = [80, 140, 200, 255]   # Torso
        arr[46:59, 22:43] = [40, 40, 60, 255]     # Piernas/Pies
        ref_img = Image.fromarray(arr)
        ref_img.save(char_dir / f"{char_dir.name}_{v}.png")

        # Spritesheet sintético 4 cols x 16 rows (256 x 1024)
        sheet = Image.new("RGBA", (256, 1024), (0, 0, 0, 0))
        for row in range(16):
            for col in range(4):
                sheet.paste(ref_img, (col * 64, row * 64))
        sheet.save(char_dir / f"movimientos_{v}.png")


def test_cross_validation_pipeline_runs_on_synthetic_data(tmp_path):
    """
    Verifica que el pipeline de validación cruzada cargue el split,
    transfiera movimiento al personaje de validación y retorne métricas diagnósticas.
    """
    approved = tmp_path / "dataset" / "finished_characters" / "approved" / "personajes al 100%"
    # 4 personajes: 3 para train, 1 para validation
    for i in range(1, 5):
        _create_synthetic_character(approved / f"char_{i:02d}")

    # 1. Crear split
    split = DatasetSplit(
        train_characters=["char_01", "char_02", "char_03"],
        validation_characters=["char_04"],
        seed=42,
        ratio=0.75,
    )
    splits_dir = tmp_path / "dataset" / "training_v2" / "splits"
    split.save(splits_dir / "split_seed_42.json")

    # 2. Entrenar plantilla para train
    engine = IncrementalTrainingEngineV2(base_dir=tmp_path)
    engine.run_training(is_smoke=True)

    # 3. Ejecutar Cross-Character Validator
    validator = CrossCharacterValidator(base_dir=tmp_path)
    summary = validator.run_cross_validation(
        split=split,
        animations=["walk_down"],
        max_validation_characters=1,
    )

    assert summary["validation_characters_count"] == 1
    assert summary["audit_samples_generated"] >= 1
    assert summary["errors"] == 0
    assert len(summary["results"]) >= 1


def test_audit_samples_structure_and_files(tmp_path):
    """
    Verifica que cada muestra de auditoría contenga los 13 archivos requeridos:
    - reference.png
    - source_motion_01.png .. 04.png
    - generated_01.png .. 04.png
    - debug_comparison.png
    - skeleton_overlay.png
    - anchors.json
    - metrics.json
    - template_used.json
    """
    approved = tmp_path / "dataset" / "finished_characters" / "approved" / "personajes al 100%"
    for i in range(1, 5):
        _create_synthetic_character(approved / f"char_{i:02d}")

    split = DatasetSplit(
        train_characters=["char_01", "char_02", "char_03"],
        validation_characters=["char_04"],
        seed=42,
    )

    engine = IncrementalTrainingEngineV2(base_dir=tmp_path)
    engine.run_training(is_smoke=True)

    validator = CrossCharacterValidator(base_dir=tmp_path)
    validator.run_cross_validation(split=split, animations=["walk_down"])

    sample_dir = tmp_path / "audit_samples" / "validation" / "char_04" / "rnormal" / "walk_down"
    assert sample_dir.is_dir()

    expected_files = [
        "reference.png",
        "source_motion_01.png",
        "source_motion_02.png",
        "source_motion_03.png",
        "source_motion_04.png",
        "generated_01.png",
        "generated_02.png",
        "generated_03.png",
        "generated_04.png",
        "debug_comparison.png",
        "skeleton_overlay.png",
        "anchors.json",
        "metrics.json",
        "template_used.json",
    ]

    for fname in expected_files:
        fpath = sample_dir / fname
        assert fpath.is_file(), f"Falta el archivo esperado en audit_samples: {fname}"
        assert fpath.stat().st_size > 0, f"El archivo {fname} está vacío"

    # Verificar contenido de metrics.json
    with open(sample_dir / "metrics.json", "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    assert metrics_data["character_id"] == "char_04"
    assert metrics_data["animation"] == "walk_down"
    assert "metrics" in metrics_data
    assert "diagnostic_identity_score" in metrics_data["metrics"]


def test_debug_comparison_image_generated(tmp_path):
    """
    Verifica que debug_comparison.png tenga las dimensiones correctas
    (tira comparativa con encabezados y 4 columnas).
    """
    approved = tmp_path / "dataset" / "finished_characters" / "approved" / "personajes al 100%"
    for i in range(1, 5):
        _create_synthetic_character(approved / f"char_{i:02d}")

    split = DatasetSplit(
        train_characters=["char_01", "char_02", "char_03"],
        validation_characters=["char_04"],
    )

    engine = IncrementalTrainingEngineV2(base_dir=tmp_path)
    engine.run_training(is_smoke=True)

    validator = CrossCharacterValidator(base_dir=tmp_path)
    validator.run_cross_validation(split=split, animations=["walk_down"])

    comp_path = tmp_path / "audit_samples" / "validation" / "char_04" / "rnormal" / "walk_down" / "debug_comparison.png"
    assert comp_path.is_file()

    img = Image.open(comp_path)
    # Ancho debe ser 4 frames * 64px = 256px
    assert img.width == 256
    # Alto incluye 2 headers + 2 filas de 64px + gap
    assert img.height > 128


def test_source_dataset_guard_during_validation(tmp_path):
    """
    Verifica que la ejecución de CrossCharacterValidator no modifique en absoluto
    el dataset maestro aprobado.
    """
    approved = tmp_path / "dataset" / "finished_characters" / "approved" / "personajes al 100%"
    for i in range(1, 5):
        _create_synthetic_character(approved / f"char_{i:02d}")

    files_before = {p: p.stat().st_mtime for p in approved.rglob("*") if p.is_file()}

    split = DatasetSplit(
        train_characters=["char_01", "char_02", "char_03"],
        validation_characters=["char_04"],
    )

    engine = IncrementalTrainingEngineV2(base_dir=tmp_path)
    engine.run_training(is_smoke=True)

    validator = CrossCharacterValidator(base_dir=tmp_path)
    validator.run_cross_validation(split=split, animations=["walk_down"])

    files_after = {p: p.stat().st_mtime for p in approved.rglob("*") if p.is_file()}
    assert files_before == files_after
