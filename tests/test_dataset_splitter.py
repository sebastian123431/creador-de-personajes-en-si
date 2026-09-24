"""
Pruebas para DatasetSplitter y DatasetSplit (Fase 3B).
Valida determinismo, aislamiento por character_id, persistencia y seguridad de fuentes.
"""

import json
from pathlib import Path
import pytest
from PIL import Image

from core.dataset_splitter import DatasetSplitter
from models.dataset_split import DatasetSplit
from services.dataset_service import DatasetService


def _create_synthetic_character(char_dir: Path):
    char_dir.mkdir(parents=True, exist_ok=True)
    for v in ("rnormal", "rbchef", "rnchef"):
        # Imagen mínima de referencia y spritesheet
        ref_img = Image.new("RGBA", (64, 64), (100, 150, 200, 255))
        ref_img.save(char_dir / f"{char_dir.name}_{v}.png")
        sheet_img = Image.new("RGBA", (256, 1024), (100, 150, 200, 255))
        sheet_img.save(char_dir / f"movimientos_{v}.png")


def test_split_reproducible(tmp_path):
    """
    Mismo dataset + misma seed = mismo split exactamente,
    independientemente del orden inicial de la lista.
    """
    characters = [f"char_{i:02d}" for i in range(15)]
    splitter = DatasetSplitter(base_dir=tmp_path, default_seed=42, default_ratio=0.8)

    split1 = splitter.split_characters(characters, seed=42)
    
    # Lista invertida o desordenada
    reversed_chars = list(reversed(characters))
    split2 = splitter.split_characters(reversed_chars, seed=42)

    assert split1.train_characters == split2.train_characters
    assert split1.validation_characters == split2.validation_characters
    assert split1.dataset_hash == split2.dataset_hash
    assert split1.seed == 42


def test_split_by_character(tmp_path):
    """
    Verifica que la partición opere exclusivamente a nivel de character_id
    y que los porcentajes correspondan aproximadamente al ratio configurado (80/20).
    """
    characters = [f"chef_{i:02d}" for i in range(10)]
    splitter = DatasetSplitter(base_dir=tmp_path, default_seed=42, default_ratio=0.8)
    split = splitter.split_characters(characters)

    assert len(split.train_characters) == 8
    assert len(split.validation_characters) == 2
    assert len(split.train_characters) + len(split.validation_characters) == 10

    # Todos los elementos deben ser strings identificadores de personaje
    for cid in split.train_characters:
        assert isinstance(cid, str)
        assert cid in characters
    for cid in split.validation_characters:
        assert isinstance(cid, str)
        assert cid in characters


def test_character_never_in_both_sets(tmp_path):
    """
    Verifica que la intersección entre train_characters y validation_characters
    sea estrictamente vacía.
    """
    characters = ["alex", "amaro", "andrea", "carlos", "dani", "elena"]
    splitter = DatasetSplitter(base_dir=tmp_path, default_seed=42)
    split = splitter.split_characters(characters)

    train_set = set(split.train_characters)
    val_set = set(split.validation_characters)

    assert train_set.intersection(val_set) == set()
    assert train_set.union(val_set) == set(characters)

    # Validar que si forzamos solapamiento en DatasetSplit, se lanza ValueError
    with pytest.raises(ValueError, match="Contaminación cruzada"):
        DatasetSplit(
            train_characters=["alex", "amaro"],
            validation_characters=["amaro", "andrea"],
            seed=42,
        )


def test_variants_stay_with_character(tmp_path):
    """
    Verifica que todas las variantes de un personaje (rnormal, rbchef, rnchef)
    queden asignadas al mismo conjunto y que un personaje en VALIDATION
    no aparezca bajo ninguna variante en TRAIN.
    """
    approved = tmp_path / "dataset" / "finished_characters" / "approved" / "personajes al 100%"
    for i in range(1, 6):
        _create_synthetic_character(approved / f"personaje_{i:02d}")

    service = DatasetService(base_dir=tmp_path)
    scanned_characters = service.scan_approved()
    assert len(scanned_characters) == 5

    splitter = DatasetSplitter(base_dir=tmp_path, default_seed=42, default_ratio=0.8)
    split = splitter.split_from_dataset(dataset_service=service, save_split=True)

    assert len(split.validation_characters) >= 1
    assert len(split.train_characters) >= 1

    # Comprobar variantes
    for char in scanned_characters.values():
        if split.is_validation(char.character_id):
            # El personaje está en validación: ninguna de sus variantes puede estar en train
            assert not split.is_train(char.character_id)
            # Aseguramos que todas las variantes pertenecen a este único character
            assert set(char.variants.keys()).issubset({"rnormal", "rbchef", "rnchef"})
        else:
            assert split.is_train(char.character_id)
            assert not split.is_validation(char.character_id)


def test_split_changes_with_seed(tmp_path):
    """
    Verifica que cambiando la semilla se obtengan particiones diferentes.
    """
    characters = [f"hero_{i:02d}" for i in range(12)]
    splitter = DatasetSplitter(base_dir=tmp_path)

    split_seed_42 = splitter.split_characters(characters, seed=42)
    split_seed_123 = splitter.split_characters(characters, seed=123)

    # El conjunto de validación debe variar entre diferentes seeds
    assert (
        split_seed_42.validation_characters != split_seed_123.validation_characters
        or split_seed_42.train_characters != split_seed_123.train_characters
    )


def test_split_json_persistence_and_fields(tmp_path):
    """
    Verifica la estructura y campos del archivo JSON generado:
    - train_characters
    - validation_characters
    - seed
    - ratio
    - created_at
    - dataset_hash
    """
    approved = tmp_path / "dataset" / "finished_characters" / "approved" / "personajes al 100%"
    for i in range(1, 7):
        _create_synthetic_character(approved / f"chef_{i:02d}")

    service = DatasetService(base_dir=tmp_path)
    splitter = DatasetSplitter(base_dir=tmp_path, default_seed=42)
    split = splitter.split_from_dataset(dataset_service=service, save_split=True)

    expected_file = tmp_path / "dataset" / "training_v2" / "splits" / "split_seed_42.json"
    assert expected_file.is_file()

    with open(expected_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "train_characters" in data
    assert "validation_characters" in data
    assert "seed" in data
    assert "ratio" in data
    assert "created_at" in data
    assert "dataset_hash" in data

    assert data["seed"] == 42
    assert data["ratio"] == 0.8
    assert len(data["dataset_hash"]) == 64  # SHA256 hex length

    # Cargar usando splitter.load_split
    loaded_split = splitter.load_split(seed=42)
    assert loaded_split.train_characters == split.train_characters
    assert loaded_split.validation_characters == split.validation_characters
    assert loaded_split.dataset_hash == split.dataset_hash


def test_source_dataset_not_modified_during_split(tmp_path):
    """
    Verifica que el proceso de división no agregue, elimine ni modifique archivos
    en la carpeta del dataset maestro.
    """
    approved = tmp_path / "dataset" / "finished_characters" / "approved" / "personajes al 100%"
    _create_synthetic_character(approved / "chef_alpha")
    _create_synthetic_character(approved / "chef_beta")

    # Tomar snapshot de archivos antes
    files_before = {p: p.stat().st_mtime for p in approved.rglob("*") if p.is_file()}

    service = DatasetService(base_dir=tmp_path)
    splitter = DatasetSplitter(base_dir=tmp_path, default_seed=42)
    splitter.split_from_dataset(dataset_service=service, save_split=True)

    files_after = {p: p.stat().st_mtime for p in approved.rglob("*") if p.is_file()}

    assert files_before == files_after
