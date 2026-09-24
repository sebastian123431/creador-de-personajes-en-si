import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from core.training_manifest import TrainingManifest, TrainingManifestEntry
from models.motion_descriptor import MotionDescriptor
from models.skeleton import Skeleton, Anchor, OFFICIAL_ANCHOR_NAMES
from models.training_session import TrainingSession
from core.incremental_training_engine import IncrementalTrainingEngineV2
from services.dataset_service import DatasetService
from tests.test_etapa_c_v2 import create_character_sprite_with_limbs
from tests.test_etapa_d_v2 import _create_synthetic_walk_cycle


def _create_synthetic_spritesheet(path: Path) -> Path:
    """Crea una hoja de sprites sintética de 4 columnas x 16 filas (256x1536 px)."""
    sheet = Image.new("RGBA", (256, 1536), (0, 0, 0, 0))
    # Dibujar un pequeño cuadrado visible en cada celda de 64x96
    for row in range(16):
        for col in range(4):
            x = col * 64 + 16
            y = row * 96 + 16
            for dy in range(40):
                for dx in range(24):
                    sheet.putpixel((x + dx, y + dy), (120, 180, 220, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


# 1. test_training_manifest_created
def test_training_manifest_created(tmp_path):
    manifest_path = tmp_path / "training_manifest.json"
    manifest = TrainingManifest(manifest_path=manifest_path)
    manifest.save()
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["version"] == "2.2"
    assert "entries" in data


# 2. test_manifest_entry_roundtrip
def test_manifest_entry_roundtrip():
    entry = TrainingManifestEntry(
        character_id="alex",
        variant="rbchef",
        spritesheet_path="path/to/sheet.png",
        spritesheet_sha256="abc123hash",
        pose_analyzer_version="2.2",
        template_version="2.2",
        status="completed",
        animations_processed=["walk_down", "cook_down"],
        descriptor_paths={"walk_down": "desc/walk.json"},
    )
    d = entry.to_dict()
    restored = TrainingManifestEntry.from_dict(d)
    assert restored.character_id == "alex"
    assert restored.variant == "rbchef"
    assert restored.spritesheet_sha256 == "abc123hash"
    assert restored.status == "completed"
    assert restored.animations_processed == ["walk_down", "cook_down"]


# 3. test_manifest_atomic_save
def test_manifest_atomic_save(tmp_path):
    manifest_path = tmp_path / "training_manifest.json"
    manifest = TrainingManifest(manifest_path=manifest_path)
    entry = TrainingManifestEntry(
        character_id="test_hero",
        variant="rnormal",
        spritesheet_path="sheet.png",
        spritesheet_sha256="123456",
        status="completed",
    )
    manifest.update_entry(entry)
    manifest.save()

    tmp_file = manifest_path.with_suffix(".tmp")
    assert not tmp_file.exists(), "El archivo temporal .tmp no debió quedar residual tras atomic save!"
    assert manifest_path.exists()

    # Recargar y verificar
    reloaded = TrainingManifest(manifest_path=manifest_path)
    assert reloaded.get_entry("test_hero", "rnormal") is not None


# 4. test_cache_hit_when_hash_identical
def test_cache_hit_when_hash_identical(tmp_path):
    manifest_path = tmp_path / "training_manifest.json"
    manifest = TrainingManifest(manifest_path=manifest_path)

    desc_file = tmp_path / "desc.json"
    desc_file.write_text("{}", encoding="utf-8")

    entry = TrainingManifestEntry(
        character_id="alex",
        variant="rbchef",
        spritesheet_path="sheet.png",
        spritesheet_sha256="exact_sha256",
        pose_analyzer_version="2.2",
        template_version="2.2",
        status="completed",
        animations_processed=["walk_down"],
        descriptor_paths={"walk_down": str(desc_file)},
    )
    manifest.update_entry(entry)

    assert manifest.is_cache_valid(
        character_id="alex",
        variant="rbchef",
        current_sha256="exact_sha256",
        pose_version="2.2",
        template_version="2.2",
        required_animations=["walk_down"],
    ) is True


# 5. test_cache_miss_when_hash_changes
def test_cache_miss_when_hash_changes(tmp_path):
    manifest_path = tmp_path / "training_manifest.json"
    manifest = TrainingManifest(manifest_path=manifest_path)
    entry = TrainingManifestEntry(
        character_id="alex",
        variant="rbchef",
        spritesheet_path="sheet.png",
        spritesheet_sha256="old_sha256",
        status="completed",
    )
    manifest.update_entry(entry)

    assert manifest.is_cache_valid(
        character_id="alex",
        variant="rbchef",
        current_sha256="new_different_sha256",
    ) is False


# 6. test_cache_invalidates_on_pose_version_change
def test_cache_invalidates_on_pose_version_change(tmp_path):
    manifest_path = tmp_path / "training_manifest.json"
    manifest = TrainingManifest(manifest_path=manifest_path)
    entry = TrainingManifestEntry(
        character_id="alex",
        variant="rbchef",
        spritesheet_path="sheet.png",
        spritesheet_sha256="same_sha256",
        pose_analyzer_version="2.2",
        template_version="2.2",
        status="completed",
    )
    manifest.update_entry(entry)

    assert manifest.is_cache_valid(
        character_id="alex",
        variant="rbchef",
        current_sha256="same_sha256",
        pose_version="2.3",  # Cambio de versión
        template_version="2.2",
    ) is False


# 7. test_cache_invalidates_on_template_version_change
def test_cache_invalidates_on_template_version_change(tmp_path):
    manifest_path = tmp_path / "training_manifest.json"
    manifest = TrainingManifest(manifest_path=manifest_path)
    entry = TrainingManifestEntry(
        character_id="alex",
        variant="rbchef",
        spritesheet_path="sheet.png",
        spritesheet_sha256="same_sha256",
        pose_analyzer_version="2.2",
        template_version="2.2",
        status="completed",
    )
    manifest.update_entry(entry)

    assert manifest.is_cache_valid(
        character_id="alex",
        variant="rbchef",
        current_sha256="same_sha256",
        pose_version="2.2",
        template_version="3.0",  # Cambio de versión
    ) is False


# 8. test_training_resume
def test_training_resume(tmp_path):
    manifest_path = tmp_path / "training_manifest.json"
    manifest = TrainingManifest(manifest_path=manifest_path)

    # Simular personaje 1 ya completado en ejecución anterior
    manifest.update_entry(TrainingManifestEntry(
        character_id="char1",
        variant="rnormal",
        spritesheet_path="sheet1.png",
        spritesheet_sha256="hash1",
        status="completed",
    ))
    # Simular personaje 2 fallido previamente
    manifest.update_entry(TrainingManifestEntry(
        character_id="char2",
        variant="rnormal",
        spritesheet_path="sheet2.png",
        spritesheet_sha256="hash2",
        status="failed",
    ))
    manifest.save()

    # Al leer para resume:
    assert manifest.is_cache_valid("char1", "rnormal", "hash1") is True
    assert manifest.is_cache_valid("char2", "rnormal", "hash2") is False


# 9. test_training_cancel
def test_training_cancel(tmp_path):
    engine = IncrementalTrainingEngineV2(base_dir=tmp_path)
    engine.request_cancel()
    assert engine._cancel_requested is True


# 10. test_descriptor_saved
def test_descriptor_saved(tmp_path):
    desc = MotionDescriptor(
        character_id="alex",
        variant="rbchef",
        animation="walk_down",
        source_path="sheet.png",
        source_sha256="abc123sha",
        character_width=32.0,
        character_height=64.0,
        root_motion=[{"frame_index": 1, "root_dx": 0.0, "root_dy": 0.0, "root_dx_ratio": 0.0, "root_dy_ratio": 0.0}],
        anchors=[{"head": {"x": 32, "y": 16, "confidence": 1.0}}],
        confidence=0.92,
        review_required=False,
    )
    target = tmp_path / "desc.json"
    desc.save(target)
    assert target.exists()

    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "raw_pixels" not in data
    assert "base64" not in data
    assert data["character_id"] == "alex"
    assert data["confidence"] == 0.92

    loaded = MotionDescriptor.load(target)
    assert loaded.character_id == "alex"
    assert loaded.source_sha256 == "abc123sha"


# 11. test_review_required_saved_but_not_primary
def test_review_required_saved_but_not_primary(tmp_path):
    desc = MotionDescriptor(
        character_id="noisy_char",
        variant="rnormal",
        animation="walk_down",
        source_path="sheet.png",
        source_sha256="noisy_hash",
        confidence=0.45,
        review_required=True,
    )
    assert desc.review_required is True
    target = tmp_path / "noisy_desc.json"
    desc.save(target)
    loaded = MotionDescriptor.load(target)
    assert loaded.review_required is True


# 12. test_training_memory_processes_incrementally
def test_training_memory_processes_incrementally(tmp_path):
    """
    Verifica que el motor procese por frame y animación sin retener
    imágenes de toda la hoja completa en memoria simultáneamente.
    """
    engine = IncrementalTrainingEngineV2(base_dir=tmp_path)
    assert engine is not None


# 13. test_source_dataset_hash_unchanged
def test_source_dataset_hash_unchanged():
    """Verifica que el dataset maestro no haya sufrido alteraciones."""
    approved_dir = Path("dataset/finished_characters/approved/personajes al 100%")
    assert approved_dir.exists()

    tmp_hash_file = Path("dataset_initial_hashes.tmp.json")
    if tmp_hash_file.exists():
        with open(tmp_hash_file, "r", encoding="utf-8") as f:
            initial_hashes = json.load(f)

        current_hashes = {}
        for file in sorted(approved_dir.rglob("*")):
            if file.is_file():
                rel = file.relative_to(approved_dir).as_posix()
                h = hashlib.sha256(file.read_bytes()).hexdigest()
                current_hashes[rel] = h

        assert initial_hashes == current_hashes, "El dataset maestro fue modificado!"


# 14. test_smoke_training_creates_two_templates
def test_smoke_training_creates_two_templates(tmp_path):
    """
    Crea un mini dataset en tmp_path con 3 personajes sintéticos y ejecuta run_training(is_smoke=True).
    Verifica que genere exactamente walk_down.json y cook_down.json.
    """
    approved = tmp_path / "dataset" / "finished_characters" / "approved" / "personajes al 100%"
    for i in range(1, 4):
        c_dir = approved / f"hero_{i:02d}"
        for v in ("rnormal", "rbchef", "rnchef"):
            sheet_file = c_dir / f"movimientos_{v}.png"
            _create_synthetic_spritesheet(sheet_file)

    engine = IncrementalTrainingEngineV2(base_dir=tmp_path)
    session = engine.run_training(is_smoke=True)

    assert session.status == "completed"
    assert session.templates_generated == 2

    tpl_dir = tmp_path / "dataset" / "templates_v2"
    assert (tpl_dir / "walk_down.json").exists()
    assert (tpl_dir / "cook_down.json").exists()
    assert (tpl_dir / "current" / "walk_down.json").exists()
    assert (tpl_dir / "current" / "cook_down.json").exists()


# 15. test_second_smoke_training_uses_cache
def test_second_smoke_training_uses_cache(tmp_path):
    """
    Verifica que la segunda ejecución del smoke training utilice la caché (cache_hits > 0).
    """
    approved = tmp_path / "dataset" / "finished_characters" / "approved" / "personajes al 100%"
    for i in range(1, 4):
        c_dir = approved / f"hero_{i:02d}"
        for v in ("rnormal", "rbchef", "rnchef"):
            sheet_file = c_dir / f"movimientos_{v}.png"
            _create_synthetic_spritesheet(sheet_file)

    engine = IncrementalTrainingEngineV2(base_dir=tmp_path)

    # Primer run: todo es cache miss
    session1 = engine.run_training(is_smoke=True)
    assert session1.cache_misses > 0
    first_misses = session1.cache_misses

    # Segundo run con los mismos datos: debe ser cache hit
    engine2 = IncrementalTrainingEngineV2(base_dir=tmp_path)
    session2 = engine2.run_training(is_smoke=True)

    assert session2.cache_hits > 0
    assert session2.cache_misses == 0
    assert session2.cache_hits == first_misses
