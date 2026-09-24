import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from core.dataset_indexer import DatasetIndexer
from core.dataset_scanner import DatasetScanner
from core.guard import SourceDatasetGuard, SourceDatasetWriteError
from core.validation import CharacterValidator, ValidationReport
from models.character import Character

logger = logging.getLogger("SpriteStudio.DatasetService")


class DatasetService:
    """
    Servicio de alto nivel para la gestión del dataset:
    - Escaneo de approved_dataset_root (ej. 'dataset/finished_characters/approved/personajes al 100%/')
    - Mantenimiento estricto de la inmutabilidad de la carpeta fuente mediante SourceDatasetGuard
    - NO mueve, NO renombra, NO aplana ni reorganiza la carpeta original
    - Indexación en dataset/indexed/dataset_index.json (fuera de la carpeta de personajes)
    - Detección de cambios incrementales por SHA-256
    """

    def __init__(self, base_dir: Optional[Path] = None, config_path: Optional[Path] = None):
        self.base_dir = Path(base_dir or Path.cwd()).resolve()

        # Rutas por defecto según especificación
        self.approved_root = (self.base_dir / "dataset" / "finished_characters" / "approved" / "personajes al 100%").resolve()
        self.rejected_dir = (self.base_dir / "dataset" / "finished_characters" / "rejected").resolve()
        self.incoming_dir = (self.base_dir / "dataset" / "finished_characters" / "incoming").resolve()
        self.index_file = (self.base_dir / "dataset" / "indexed" / "dataset_index.json").resolve()

        # Cargar config si existe
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    if "approved_dataset_root" in cfg:
                        self.approved_root = (self.base_dir / cfg["approved_dataset_root"]).resolve()
                    paths = cfg.get("paths", {})
                    if "rejected_characters" in paths:
                        self.rejected_dir = (self.base_dir / paths["rejected_characters"]).resolve()
                    if "incoming_characters" in paths:
                        self.incoming_dir = (self.base_dir / paths["incoming_characters"]).resolve()
                    if "dataset_index" in paths:
                        self.index_file = (self.base_dir / paths["dataset_index"]).resolve()
            except Exception as e:
                logger.error(f"Error cargando settings.json en DatasetService: {e}")

        # Guardián de inmutabilidad estricto
        self.guard = SourceDatasetGuard(protected_root=self.approved_root)

        self.ensure_derived_directories()

        self.scanner = DatasetScanner(root=self.approved_root)
        self.indexer = DatasetIndexer(self.index_file, guard=self.guard)
        self.validator = CharacterValidator()

        # Estado en memoria
        self.approved_characters: Dict[str, Character] = {}
        self.incoming_characters: Dict[str, Character] = {}
        self.rejected_characters: Dict[str, Character] = {}

    @property
    def approved_dir(self) -> Path:
        return self.approved_root

    def set_approved_root(self, new_root: Union[str, Path]):
        """Permite al usuario seleccionar otra carpeta raíz desde la GUI."""
        self.approved_root = Path(new_root).resolve()
        self.guard.set_protected_root(self.approved_root)
        self.scanner.root = self.approved_root
        logger.info(f"Nueva ruta raíz de dataset approved: {self.approved_root}")

    def ensure_derived_directories(self):
        """Crea las carpetas de archivos derivados (FUERA de la carpeta original de personajes)."""
        derived_dirs = [
            self.base_dir / "dataset" / "indexed",
            self.base_dir / "dataset" / "extracted_frames",
            self.base_dir / "dataset" / "normalized",
            self.base_dir / "dataset" / "training",
            self.rejected_dir,
            self.incoming_dir,
        ]
        for d in derived_dirs:
            # Comprobar con el guard que ninguna carpeta derivada esté dentro de la protegida
            self.guard.assert_can_write(d, operation_desc="creación de carpetas derivadas")
            d.mkdir(parents=True, exist_ok=True)

    def scan_approved(self) -> Dict[str, Character]:
        """
        Escanea la carpeta de personajes aprobados navegando dentro de contenedores
        intermedios como 'personajes al 100%' sin modificar ningún archivo.
        """
        self.approved_characters = self.scanner.scan_directory(self.approved_root, is_approved=True)
        return self.approved_characters

    def scan_incoming(self) -> Dict[str, Character]:
        """Escanea la carpeta de personajes entrantes para evaluación."""
        self.incoming_characters = self.scanner.scan_directory(self.incoming_dir, is_approved=False)
        return self.incoming_characters

    def scan_rejected(self) -> Dict[str, Character]:
        """Escanea la carpeta de personajes rechazados."""
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
        Actualiza y serializa dataset/indexed/dataset_index.json a partir de los personajes aprobados.
        Garantiza que el archivo se escriba fuera de la carpeta original.
        """
        return self.indexer.save_index(
            self.approved_characters,
            dataset_root=self.approved_root,
            base_path=self.base_dir
        )

    def validate_all_approved(self) -> ValidationReport:
        """Valida todos los personajes aprobados."""
        return self.validator.validate_dataset(self.approved_characters)

    def check_incremental_changes(self) -> dict:
        """Determina qué personajes han cambiado desde el último índice guardado."""
        return self.indexer.detect_changes(self.approved_characters)

    def import_external_directory_to_approved(self, source_path: Path) -> Dict[str, Character]:
        """
        Copia personajes desde una carpeta externa hacia approved_root
        preservando íntegramente los archivos fuente sin modificarlos.
        """
        import shutil
        source_path = Path(source_path).resolve()
        source_chars = self.scanner.scan_directory(source_path)
        imported: Dict[str, Character] = {}

        self.approved_root.mkdir(parents=True, exist_ok=True)

        for char_id, char in source_chars.items():
            if not char.source_dir or not char.source_dir.is_dir():
                continue

            target_char_dir = self.approved_root / char.source_dir.name
            target_char_dir.mkdir(parents=True, exist_ok=True)

            for file_path in char.source_dir.iterdir():
                if file_path.is_file():
                    dest_file = target_char_dir / file_path.name
                    shutil.copy2(file_path, dest_file)

            approved_char = self.scanner.scan_character_folder(target_char_dir, is_approved=True)
            if approved_char:
                self.approved_characters[char_id] = approved_char
                imported[char_id] = approved_char

        self.update_index()
        logger.info(f"Importados {len(imported)} personajes a '{self.approved_root}'.")
        return imported
