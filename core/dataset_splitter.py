"""
Módulo de partición de dataset para validación cruzada entre personajes.
Garantiza división a nivel de character_id (80% Train / 20% Validation por defecto).
Totalmente determinista mediante seed controlada y orden canónico previo.
"""

import hashlib
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from models.dataset_split import DatasetSplit
from services.dataset_service import DatasetService

logger = logging.getLogger("SpriteStudio.DatasetSplitter")


class DatasetSplitter:
    """
    Divide un conjunto de personajes en particiones de Entrenamiento y Validación.
    - 100% determinista con seed (misma lista + misma seed = mismo split).
    - Agrupación atómica por personaje: todas las variantes (rnormal, rbchef, rnchef) van juntas.
    - Aislamiento estricto: ningún personaje puede estar en ambos grupos.
    - Salida guardada en dataset/training_v2/splits/.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        default_seed: int = 42,
        default_ratio: float = 0.8,
    ):
        self.base_dir = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()
        self.default_seed = default_seed
        self.default_ratio = default_ratio
        self.splits_dir = self.base_dir / "dataset" / "training_v2" / "splits"

    @staticmethod
    def compute_dataset_hash(character_ids: List[str]) -> str:
        """Calcula un hash SHA256 canónico a partir de la lista ordenada de personajes."""
        sorted_ids = sorted(character_ids)
        content = "\n".join(sorted_ids).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def split_characters(
        self,
        character_ids: List[str],
        seed: Optional[int] = None,
        train_ratio: Optional[float] = None,
    ) -> DatasetSplit:
        """
        Divide una lista de character_ids en train y validation.

        Args:
            character_ids: Lista de nombres/identificadores de personajes.
            seed: Semilla aleatoria (por defecto self.default_seed = 42).
            train_ratio: Fracción para entrenamiento (por defecto self.default_ratio = 0.8).

        Returns:
            DatasetSplit con train_characters y validation_characters.
        """
        active_seed = self.default_seed if seed is None else seed
        active_ratio = self.default_ratio if train_ratio is None else train_ratio

        if not 0.0 < active_ratio <= 1.0:
            raise ValueError(f"train_ratio debe estar en el intervalo (0.0, 1.0], recibido: {active_ratio}")

        # Normalizar y ordenar canónicamente para que el orden de entrada no afecte el resultado
        unique_characters = sorted(list(set(character_ids)))
        n_total = len(unique_characters)

        if n_total == 0:
            return DatasetSplit(
                train_characters=[],
                validation_characters=[],
                seed=active_seed,
                ratio=active_ratio,
                dataset_hash="",
            )

        dataset_hash = self.compute_dataset_hash(unique_characters)

        # Barajado determinista usando generador aislado
        rng = random.Random(active_seed)
        shuffled = list(unique_characters)
        rng.shuffle(shuffled)

        if n_total == 1:
            train_chars = shuffled
            val_chars = []
        elif active_ratio >= 1.0:
            train_chars = shuffled
            val_chars = []
        else:
            # Calcular tamaño de train asegurando al menos 1 en validación si n_total >= 2
            n_train = int(round(n_total * active_ratio))
            if n_train >= n_total and n_total >= 2:
                n_train = n_total - 1
            if n_train < 1 and n_total >= 1:
                n_train = 1

            train_chars = sorted(shuffled[:n_train])
            val_chars = sorted(shuffled[n_train:])

        split = DatasetSplit(
            train_characters=train_chars,
            validation_characters=val_chars,
            seed=active_seed,
            ratio=active_ratio,
            created_at=datetime.now(timezone.utc).isoformat(),
            dataset_hash=dataset_hash,
        )
        return split

    def split_from_dataset(
        self,
        dataset_service: Optional[DatasetService] = None,
        seed: Optional[int] = None,
        train_ratio: Optional[float] = None,
        save_split: bool = True,
    ) -> DatasetSplit:
        """
        Escanea el dataset aprobado y genera la partición oficial.

        Args:
            dataset_service: Servicio de dataset existente (o crea uno nuevo si es None).
            seed: Semilla aleatoria (por defecto 42).
            train_ratio: Ratio de entrenamiento (por defecto 0.8).
            save_split: Si True, persiste el archivo en dataset/training_v2/splits/.

        Returns:
            DatasetSplit validado.
        """
        if dataset_service is None:
            dataset_service = DatasetService(base_dir=self.base_dir)

        characters = dataset_service.scan_approved()
        if isinstance(characters, dict):
            character_ids = list(characters.keys())
        elif isinstance(characters, (list, tuple, set)):
            character_ids = [
                getattr(c, "character_id", getattr(c, "name", str(c))) for c in characters
            ]
        else:
            character_ids = []

        split = self.split_characters(character_ids, seed=seed, train_ratio=train_ratio)

        if save_split:
            out_file = self.splits_dir / f"split_seed_{split.seed}.json"
            split.save(out_file)
            logger.info(
                "DatasetSplit guardado en: %s (Train: %d, Val: %d, Seed: %d)",
                out_file,
                len(split.train_characters),
                len(split.validation_characters),
                split.seed,
            )

        return split

    def load_split(self, seed: Optional[int] = None) -> DatasetSplit:
        """
        Carga una división previamente guardada según su seed.
        """
        target_seed = self.default_seed if seed is None else seed
        target_file = self.splits_dir / f"split_seed_{target_seed}.json"
        if not target_file.is_file():
            raise FileNotFoundError(f"No existe partición guardada para seed={target_seed}: {target_file}")
        return DatasetSplit.load(target_file)
