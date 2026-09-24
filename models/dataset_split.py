"""
Modelo de datos para división de dataset (Train / Validation Split).
Garantiza división a nivel de character_id (nunca por frame o variante).
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Set


@dataclass
class DatasetSplit:
    """
    Representa una partición determinista de personajes entre entrenamiento y validación.
    
    Reglas inviolables:
    - La partición es estrictamente por character_id.
    - Las 3 variantes de un personaje (rnormal, rbchef, rnchef) pertenecen al mismo conjunto.
    - Ningún character_id puede estar presente simultáneamente en train y validation.
    """
    train_characters: List[str] = field(default_factory=list)
    validation_characters: List[str] = field(default_factory=list)
    seed: int = 42
    ratio: float = 0.8  # Porcentaje para train (ej. 0.8 = 80% train, 20% validation)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    dataset_hash: str = ""

    def __post_init__(self):
        self.validate_isolation()

    def validate_isolation(self) -> None:
        """
        Valida que no exista solapamiento entre train y validation.
        Lanza ValueError si se detecta contaminación cruzada.
        """
        train_set = set(self.train_characters)
        val_set = set(self.validation_characters)
        overlap = train_set & val_set
        if overlap:
            raise ValueError(
                f"Contaminación cruzada detectada en DatasetSplit: {overlap} "
                "aparece simultáneamente en train y validation."
            )

    def is_train(self, character_id: str) -> bool:
        """Verifica si un personaje pertenece al conjunto de entrenamiento."""
        return character_id in set(self.train_characters)

    def is_validation(self, character_id: str) -> bool:
        """Verifica si un personaje pertenece al conjunto de validación."""
        return character_id in set(self.validation_characters)

    def to_dict(self) -> Dict[str, Any]:
        """Serializa la división a diccionario primitivo."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetSplit":
        """Instancia DatasetSplit desde un diccionario."""
        return cls(
            train_characters=list(data.get("train_characters", [])),
            validation_characters=list(data.get("validation_characters", [])),
            seed=int(data.get("seed", 42)),
            ratio=float(data.get("ratio", 0.8)),
            created_at=str(data.get("created_at", "")),
            dataset_hash=str(data.get("dataset_hash", "")),
        )

    def save(self, path: Path) -> Path:
        """Guarda la partición en formato JSON de manera atómica."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        tmp_path.replace(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "DatasetSplit":
        """Carga una partición desde un archivo JSON."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Archivo de partición no encontrado: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
