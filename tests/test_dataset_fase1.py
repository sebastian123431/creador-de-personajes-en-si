import json
import shutil
from pathlib import Path
from core.dataset_scanner import DatasetScanner, compute_file_sha256
from core.dataset_indexer import DatasetIndexer
from core.naming import (
    OFFICIAL_VARIANTS,
    format_display_name,
    make_unique_key,
    parse_file_variant,
    sanitize_character_id,
)
from core.validation import CharacterValidator
from services.dataset_service import DatasetService


def test_variant_detection():
    """Valida la detección correcta de tipo de archivo y variante."""
    file_type, var = parse_file_variant("alex_rnormal.png")
    assert file_type == "reference"
    assert var == "rnormal"

    file_type, var = parse_file_variant("movimientos_rnormal.png")
    assert file_type == "spritesheet"
    assert var == "rnormal"

    file_type, var = parse_file_variant("diego_vallenar_rbchef.png")
    assert file_type == "reference"
    assert var == "rbchef"

    file_type, var = parse_file_variant("movimientos_rnchef.png")
    assert file_type == "spritesheet"
    assert var == "rnchef"


def test_rnormal_detection():
    file_type, var = parse_file_variant("test_rnormal.png")
    assert var == "rnormal"


def test_rbchef_detection():
    file_type, var = parse_file_variant("movimientos_rbchef.png")
    assert var == "rbchef"


def test_rnchef_detection():
    file_type, var = parse_file_variant("andres_arica_rnchef.png")
    assert var == "rnchef"


def test_character_names_unique(sample_dataset_dir: Path):
    """
    Verifica que personajes con nombres compuestos (ciudad, apodo, apellido)
    como 'diego_vallenar' y 'diego_serena' se traten como identidades ÚNICAS
    y no se fusionen ni se trunquen.
    """
    scanner = DatasetScanner()
    characters = scanner.scan_directory(sample_dataset_dir)

    assert "diego_vallenar" in characters
    assert "diego_serena" in characters
    assert characters["diego_vallenar"].display_name == "Diego Vallenar"
    assert characters["diego_serena"].display_name == "Diego Serena"

    # Verificar que son 6 identidades distintas
    assert len(characters) == 6
    assert "andres_arica" in characters
    assert "benja_bacaba" in characters


def test_scan_dataset(sample_dataset_dir: Path):
    """Verifica el escaneo completo de 6 personajes x 3 variantes = 18 variantes."""
    scanner = DatasetScanner()
    characters = scanner.scan_directory(sample_dataset_dir)

    for char_id, char in characters.items():
        assert len(char.variants) == 3
        for v_name in OFFICIAL_VARIANTS:
            variant = char.variants[v_name]
            assert variant.has_reference, f"Falta referencia para {char_id}:{v_name}"
            assert variant.has_spritesheet, f"Falta spritesheet para {char_id}:{v_name}"
            assert variant.status == "ready"


def test_dataset_index_generation(sample_dataset_dir: Path, tmp_path: Path):
    """Verifica que dataset_index.json se genere con la estructura oficial."""
    scanner = DatasetScanner()
    characters = scanner.scan_directory(sample_dataset_dir)

    index_path = tmp_path / "dataset_index.json"
    indexer = DatasetIndexer(index_path)

    saved_path = indexer.save_index(characters)
    assert saved_path.exists()

    with open(saved_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["version"] == "1.0.0"
    assert data["total_characters"] == 6
    assert "alex" in data["characters"]
    alex_variants = data["characters"]["alex"]["variants"]
    assert "rnormal" in alex_variants
    assert "rbchef" in alex_variants
    assert "rnchef" in alex_variants
    assert alex_variants["rnormal"]["reference"] is not None
    assert alex_variants["rnormal"]["spritesheet"] is not None


def test_incremental_dataset(sample_dataset_dir: Path, tmp_path: Path):
    """Verifica la detección incremental de cambios usando hashes SHA-256."""
    scanner = DatasetScanner()
    characters = scanner.scan_directory(sample_dataset_dir)

    index_path = tmp_path / "dataset_index.json"
    indexer = DatasetIndexer(index_path)
    indexer.save_index(characters)

    # 1. Sin cambios
    changes = indexer.detect_changes(characters)
    assert len(changes["added"]) == 0
    assert len(changes["modified"]) == 0
    assert len(changes["unchanged"]) == 6

    # 2. Modificar un archivo
    alex_ref = characters["alex"].variants["rnormal"].reference_image
    with open(alex_ref, "ab") as f:
        f.write(b"\x00")  # Cambiar un byte para alterar SHA-256

    re_scanned = scanner.scan_directory(sample_dataset_dir)
    changes_after = indexer.detect_changes(re_scanned)
    assert "alex" in changes_after["modified"]
    assert len(changes_after["unchanged"]) == 5


def test_approved_immutability(sample_dataset_dir: Path, tmp_path: Path):
    """
    Verifica que 'approved' sea READ-ONLY: el escaneo, indexación y validación
    no modifican ni un solo byte de los archivos aprobados.
    """
    alex_ref = sample_dataset_dir / "alex" / "alex_rnormal.png"
    original_hash = compute_file_sha256(alex_ref)

    # Ejecutar escaneo, indexación y validación
    scanner = DatasetScanner()
    chars = scanner.scan_directory(sample_dataset_dir, is_approved=True)

    indexer = DatasetIndexer(tmp_path / "dataset_index.json")
    indexer.save_index(chars)

    validator = CharacterValidator()
    validator.validate_dataset(chars)

    final_hash = compute_file_sha256(alex_ref)
    assert original_hash == final_hash, "Los archivos aprobados fueron alterados indebidamente!"


def test_import_external_to_approved(sample_dataset_dir: Path, tmp_path: Path):
    """
    Verifica la importación de una carpeta externa a dataset/finished_characters/approved/
    preservando los archivos originales sin modificarlos.
    """
    service = DatasetService(base_dir=tmp_path)
    imported = service.import_external_directory_to_approved(sample_dataset_dir)

    assert len(imported) == 6
    assert (service.approved_dir / "alex").exists()
    assert (service.approved_dir / "diego_vallenar").exists()
    assert service.index_file.exists()


def test_validation_system(sample_dataset_dir: Path):
    """Verifica el validador de consistencia de imágenes y variantes."""
    scanner = DatasetScanner()
    characters = scanner.scan_directory(sample_dataset_dir)

    validator = CharacterValidator()
    report = validator.validate_dataset(characters)

    assert report.total_characters == 6
    assert report.valid_characters == 6
    assert report.is_valid
    assert report.error_count == 0
