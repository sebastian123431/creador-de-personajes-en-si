import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.dataset_indexer import DatasetIndexer
from core.dataset_scanner import DatasetScanner
from core.validation import CharacterValidator, ValidationReport
from models.character import Character

logger = logging.getLogger("SpriteStudio.DatasetService")


class DatasetService:
    """
    Servicio de alto nivel para la gestión del dataset:
    - Escaneo de approved, incoming y rejected
    - Mantenimiento estricto de la inmutabilidad de 'approved'
    - Importación controlada y validada
    - Indexación en dataset_index.json
    - Detección de cambios incrementales
    """

    def __init__(self, base_dir: Optional[Path] = None, config_path: Optional[Path] = None):
        self.base_dir = Path(base_dir or Path.cwd()).resolve()

        # Rutas por defecto
        self.approved_dir = self.base_dir / "dataset" / "finished_characters" / "approved"
        self.incoming_dir = self.base_dir / "dataset" / "finished_characters" / "incoming"
        self.rejected_dir = self.base_dir / "dataset" / "finished_characters" / "rejected"
        self.index_file = self.base_dir / "dataset" / "dataset_index.json"

        # Cargar config si existe
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    paths = cfg.get("paths", {})
                    if "approved_characters" in paths:
                        self.approved_dir = (self.base_dir / paths["approved_characters"]).resolve()
                    if "incoming_characters" in paths:
                        self.incoming_dir = (self.base_dir / paths["incoming_characters"]).resolve()
                    if "rejected_characters" in paths:
                        self.rejected_dir = (self.base_dir / paths["rejected_characters"]).resolve()
                    if "dataset_index" in paths:
                        self.index_file = (self.base_dir / paths["dataset_index"]).resolve()
            except Exception as e:
                logger.error(f"Error cargando settings.json en DatasetService: {e}")

        # Garantizar que las carpetas existan
        self.ensure_directories()

        self.scanner = DatasetScanner()
        self.indexer = DatasetIndexer(self.index_file)
        self.validator = CharacterValidator()

        # Estado en memoria
        self.approved_characters: Dict[str, Character] = {}
        self.incoming_characters: Dict[str, Character] = {}
        self.rejected_characters: Dict[str, Character] = {}

    def ensure_directories(self):
        """Crea las carpetas del dataset necesarias si no existen."""
        for d in [self.approved_dir, self.incoming_dir, self.rejected_dir, self.index_file.parent]:
            d.mkdir(parents=True, exist_ok=True)

    def scan_approved(self) -> Dict[str, Character]:
        """
        Escanea la carpeta de personajes aprobados (READ-ONLY).
        """
        self.approved_characters = self.scanner.scan_directory(self.approved_dir, is_approved=True)
        return self.approved_characters

    def scan_incoming(self) -> Dict[str, Character]:
        """
        Escanea la carpeta de personajes entrantes para evaluación.
        """
        self.incoming_characters = self.scanner.scan_directory(self.incoming_dir, is_approved=False)
        return self.incoming_characters

    def scan_rejected(self) -> Dict[str, Character]:
        """
        Escanea la carpeta de personajes rechazados.
        """
        self.rejected_characters = self.scanner.scan_directory(self.rejected_dir, is_approved=False)
        return self.rejected_characters

    def scan_all(self) -> Tuple[Dict[str, Character], Dict[str, Character], Dict[str, Character]]:
        """Escanea approved, incoming y rejected."""
        self.scan_approved()
        self.scan_incoming()
        self.scan_rejected()
        return self.approved_characters, self.incoming_characters, self.rejected_characters

    def update_index(self) -> Path:
        """
        Actualiza y serializa dataset_index.json a partir de los personajes aprobados.
        """
        return self.indexer.save_index(self.approved_characters, base_path=self.base_dir)

    def validate_all_approved(self) -> ValidationReport:
        """Valida todos los personajes aprobados."""
        return self.validator.validate_dataset(self.approved_characters)

    def check_incremental_changes(self) -> dict:
        """Determina qué personajes han cambiado desde el último índice guardado."""
        return self.indexer.detect_changes(self.approved_characters)

    def import_external_directory_to_approved(self, source_path: Path) -> Dict[str, Character]:
        """
        Importa personajes desde una carpeta externa (ej: 'personajes al 100%/')
        hacia dataset/finished_characters/approved/ copiando fielmente sin modificar
        los archivos originales de la fuente.
        """
        source_chars = self.scanner.scan_directory(source_path)
        imported: Dict[str, Character] = {}

        for char_id, char in source_chars.items():
            if not char.source_dir or not char.source_dir.is_dir():
                continue

            target_char_dir = self.approved_dir / char.source_dir.name
            target_char_dir.mkdir(parents=True, exist_ok=True)

            for file_path in char.source_dir.iterdir():
                if file_path.is_file():
                    dest_file = target_char_dir / file_path.name
                    # Copia fiel preservando metadata
                    shutil.copy2(file_path, dest_file)

            # Re-escanear el personaje en la carpeta approved
            approved_char = self.scanner.scan_character_folder(target_char_dir, is_approved=True)
            if approved_char:
                self.approved_characters[char_id] = approved_char
                imported[char_id] = approved_char

        self.update_index()
        logger.info(f"Importados {len(imported)} personajes a 'approved'.")
        return imported

    def approve_incoming_character(self, character_id: str) -> Optional[Character]:
        """
        Mueve o copia un personaje de incoming a approved de forma segura.
        """
        char = self.incoming_characters.get(character_id)
        if not char or not char.source_dir:
            logger.error(f"Personaje '{character_id}' no encontrado en incoming.")
            return None

        target_dir = self.approved_dir / char.source_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        for f in char.source_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, target_dir / f.name)

        # Eliminar de incoming una vez copiado a approved
        shutil.rmtree(char.source_dir)
        del self.incoming_characters[character_id]

        approved_char = self.scanner.scan_character_folder(target_dir, is_approved=True)
        if approved_char:
            self.approved_characters[character_id] = approved_char
            self.update_index()

        return approved_char

    def reject_incoming_character(self, character_id: str) -> bool:
        """
        Mueve un personaje de incoming a rejected.
        """
        char = self.incoming_characters.get(character_id)
        if not char or not char.source_dir:
            return False

        target_dir = self.rejected_dir / char.source_dir.name
        if target_dir.exists():
            shutil.rmtree(target_dir)

        shutil.move(str(char.source_dir), str(target_dir))
        del self.incoming_characters[character_id]
        self.scan_rejected()
        return True
