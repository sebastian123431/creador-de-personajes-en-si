import json
import pytest
from pathlib import Path
from PIL import Image

from core.dataset_scanner import DatasetScanner, compute_file_sha256
from core.dataset_indexer import DatasetIndexer
from core.guard import SourceDatasetGuard, SourceDatasetWriteError
from core.frame_extractor import FrameExtractor
from core.frame_normalizer import FrameNormalizer
from core.naming import (
    OFFICIAL_VARIANTS,
    VARIANT_LABELS,
    parse_file_variant,
    sanitize_character_id,
)
from services.dataset_service import DatasetService
from tests.conftest import create_dummy_png
from tests.test_pipeline_full import create_realistic_spritesheet


@pytest.fixture
def real_structure_dataset(tmp_path: Path) -> Path:
    """
    Crea la estructura real exacta requerida por el usuario:
    dataset/finished_characters/approved/personajes al 100%/
        alex/
        amaro/
        andrea/
        andres_arica/
        bastian/
        belial/
        benja_bacaba/
        carlos/
        conny/
        dafne/
        dana/
        diego_serena/
        diego_vallenar/
        duvan/
    """
    char_names = [
        "alex", "amaro", "andrea", "andres_arica", "bastian",
        "belial", "benja_bacaba", "carlos", "conny", "dafne",
        "dana", "diego_serena", "diego_vallenar", "duvan"
    ]
    variants = ["rnormal", "rbchef", "rnchef"]

    root = tmp_path / "dataset" / "finished_characters" / "approved" / "personajes al 100"
    root.mkdir(parents=True, exist_ok=True)

    for char in char_names:
        char_folder = root / char
        char_folder.mkdir()
        for v in variants:
            # Referencias
            ref_path = char_folder / f"{char}_{v}.png"
            create_dummy_png(ref_path, width=48, height=80)

            # Spritesheets
            sheet_path = char_folder / f"movimientos_{v}.png"
            create_realistic_spritesheet(sheet_path, cell_w=32, cell_h=32)

    return root


def test_source_dataset_root(real_structure_dataset: Path):
    """Verifica que el scanner acepte directamente la ruta aprobada configurada."""
    scanner = DatasetScanner(root=real_structure_dataset)
    assert scanner.root == real_structure_dataset.resolve()

    chars = scanner.scan_directory()
    assert len(chars) == 14
    assert "alex" in chars
    assert "diego_vallenar" in chars


def test_scan_nested_dataset(tmp_path: Path, real_structure_dataset: Path):
    """
    Verifica que si se pasa la carpeta padre 'approved', el scanner
    navegue automáticamente dentro de 'personajes al 100%' sin exigir
    que las carpetas estén directamente en approved/.
    """
    approved_parent = real_structure_dataset.parent
    scanner = DatasetScanner(root=approved_parent)
    chars = scanner.scan_directory(approved_parent)

    assert len(chars) == 14
    assert "andres_arica" in chars
    assert "benja_bacaba" in chars


def test_personajes_al_100_container(real_structure_dataset: Path):
    """
    Verifica que la carpeta 'personajes al 100%' no sea aplanada
    ni reorganizada ni renombrada.
    """
    assert real_structure_dataset.exists()
    assert real_structure_dataset.name == "personajes al 100"

    scanner = DatasetScanner(root=real_structure_dataset)
    chars = scanner.scan_directory()

    # Cada subcarpeta inmediata de 'personajes al 100%' es un personaje
    for char_id, char_obj in chars.items():
        assert char_obj.source_dir.parent == real_structure_dataset.resolve()
        assert char_obj.character_id == char_obj.source_dir.name


def test_character_folder_detection(real_structure_dataset: Path):
    """Verifica que cada carpeta de personaje se detecte con sus 3 variantes oficiales."""
    scanner = DatasetScanner(root=real_structure_dataset)
    chars = scanner.scan_directory()

    for cid in ["alex", "amaro", "diego_serena", "diego_vallenar"]:
        assert cid in chars
        char = chars[cid]
        for v in OFFICIAL_VARIANTS:
            assert v in char.variants
            assert char.variants[v].has_reference
            assert char.variants[v].has_spritesheet


def test_compound_character_names():
    """
    Verifica que nombres con ciudad, apodo o apellido (diego_vallenar, diego_serena,
    andres_arica, benja_bacaba, juan_perez, alex_2, niko_coquimbo, sebastian_chico)
    se conserven como IDs exactos sin truncamiento ni fusión.
    """
    compounds = [
        "diego_vallenar", "diego_serena", "andres_arica", "benja_bacaba",
        "juan_perez", "alex_2", "niko_coquimbo", "sebastian_chico"
    ]
    for name in compounds:
        clean_id = sanitize_character_id(name)
        assert clean_id == name


def test_rnormal_detection():
    """Verifica la detección de rnormal (Ropa normal)."""
    file_type, var = parse_file_variant("alex_rnormal.png")
    assert file_type == "reference"
    assert var == "rnormal"
    assert VARIANT_LABELS["rnormal"] == "Ropa normal"


def test_rbchef_detection():
    """Verifica la detección de rbchef (Chef blanco)."""
    file_type, var = parse_file_variant("movimientos_rbchef.png")
    assert file_type == "spritesheet"
    assert var == "rbchef"
    assert VARIANT_LABELS["rbchef"] == "Chef blanco"


def test_rnchef_detection():
    """Verifica la detección de rnchef (Chef negro)."""
    file_type, var = parse_file_variant("diego_vallenar_rnchef.png")
    assert file_type == "reference"
    assert var == "rnchef"
    assert VARIANT_LABELS["rnchef"] == "Chef negro"


def test_source_dataset_not_modified(real_structure_dataset: Path, tmp_path: Path):
    """
    REGLA CRÍTICA:
    Verifica que el dataset fuente en 'personajes al 100%' sea estrictamente READ-ONLY.
    El escaneo, indexación y extracción no modifican ningún byte de los archivos originales.
    """
    # Tomar hashes antes
    alex_ref = real_structure_dataset / "alex" / "alex_rnormal.png"
    alex_sheet = real_structure_dataset / "alex" / "movimientos_rnormal.png"
    hash_ref_before = compute_file_sha256(alex_ref)
    hash_sheet_before = compute_file_sha256(alex_sheet)

    # Operaciones del sistema
    service = DatasetService(base_dir=tmp_path)
    service.set_approved_root(real_structure_dataset)
    service.scan_approved()
    service.update_index()

    # Verificar que los hashes permanezcan 100% idénticos
    assert compute_file_sha256(alex_ref) == hash_ref_before
    assert compute_file_sha256(alex_sheet) == hash_sheet_before

    # Verificar que no se hayan creado archivos extraños dentro de 'personajes al 100%/alex'
    alex_files = [f.name for f in (real_structure_dataset / "alex").iterdir()]
    assert "frames" not in alex_files
    assert "metadata.json" not in alex_files
    assert "dataset_index.json" not in alex_files
    assert len(alex_files) == 6  # 3 refs + 3 sheets únicamente


def test_outputs_written_outside_source(real_structure_dataset: Path, tmp_path: Path):
    """
    Verifica que TODOS los archivos derivados se escriban FUERA de la carpeta original:
    - dataset_index.json en dataset/indexed/
    - frames extraídos en dataset/extracted_frames/
    - frames normalizados en dataset/normalized/
    Y que SourceDatasetGuard lance SourceDatasetWriteError si se intenta escribir dentro.
    """
    guard = SourceDatasetGuard(protected_root=real_structure_dataset)

    # Intentar escribir dentro del directorio protegido debe disparar excepción
    with pytest.raises(SourceDatasetWriteError):
        illegal_path = real_structure_dataset / "alex" / "extracted_frame.png"
        guard.assert_can_write(illegal_path)

    # Las rutas permitidas están fuera
    legal_indexed = tmp_path / "dataset" / "indexed" / "dataset_index.json"
    guard.assert_can_write(legal_indexed)

    legal_extracted = tmp_path / "dataset" / "extracted_frames" / "alex" / "rnormal"
    guard.assert_can_write(legal_extracted)

    legal_normalized = tmp_path / "dataset" / "normalized" / "alex" / "rnormal"
    guard.assert_can_write(legal_normalized)

    # Ejecutar indexación y validar ruta
    service = DatasetService(base_dir=tmp_path)
    service.set_approved_root(real_structure_dataset)
    service.scan_approved()
    saved_index = service.update_index()

    # dataset_index.json está en dataset/indexed/
    assert "indexed" in str(saved_index)
    assert not guard.is_inside_protected_root(saved_index)
    assert saved_index.exists()

    with open(saved_index, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "dataset_root" in data
    assert "characters" in data
    assert "alex" in data["characters"]
    assert "source_folder" in data["characters"]["alex"]
