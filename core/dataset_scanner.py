import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from core.naming import (
    OFFICIAL_VARIANTS,
    format_display_name,
    parse_file_variant,
    sanitize_character_id,
)
from models.character import Character
from models.character_variant import CharacterVariant

logger = logging.getLogger("SpriteStudio.Scanner")


def compute_file_sha256(filepath: Path) -> str:
    """Calcula el hash SHA-256 de un archivo para control de integridad y cache."""
    if not filepath.is_file():
        return ""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class DatasetScanner:
    """
    Escanea la estructura real del dataset (ej: 'approved/personajes al 100%/')
    sin aplanar, mover ni renombrar la carpeta original.
    Navega dentro de carpetas contenedoras intermedias respetando los IDs exactos.
    """

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        supported_extensions: Optional[List[str]] = None,
        max_recursive_depth: int = 1
    ):
        self.root = Path(root).resolve() if root else None
        self.supported_extensions = supported_extensions or [".png", ".PNG"]
        self.max_recursive_depth = max_recursive_depth

    def _resolve_character_container(self, root_dir: Path) -> Path:
        """
        Navega si es necesario dentro de una carpeta contenedora intermedia
        como 'personajes al 100%' o carpetas similares. Si la ruta configurada
        aún no existe pero su padre sí, intenta resolverla.
        """
        target = root_dir.resolve()
        if not target.exists():
            if target.parent.exists() and target.parent.is_dir():
                target = target.parent
            else:
                return target

        direct_subdirs = [d for d in target.iterdir() if d.is_dir() and not d.name.startswith(".")]

        # 1. Buscar si hay alguna subcarpeta con nombre 'personajes al 100%'
        for sd in direct_subdirs:
            if "personajes al 100" in sd.name.lower():
                logger.info(f"Navegando a la carpeta contenedora intermedia: '{sd}'")
                return sd

        # 2. Si solo hay una subcarpeta y dentro contiene carpetas de personajes
        if len(direct_subdirs) == 1:
            child = direct_subdirs[0]
            inner_subdirs = [d for d in child.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if len(inner_subdirs) > 0:
                logger.info(f"Navegando a subcarpeta contenedora única: '{child}'")
                return child

        return target

    def scan_character_folder(self, folder: Path, is_approved: bool = True) -> Optional[Character]:
        """
        Escanea una carpeta individual de personaje y construye el objeto Character.
        El character_id es el nombre exacto de la carpeta.
        """
        if not folder.is_dir():
            return None

        # Identificador oficial inmutable = nombre exacto de la carpeta
        char_id = sanitize_character_id(folder.name)
        display_name = format_display_name(char_id)

        character = Character(
            character_id=char_id,
            display_name=display_name,
            source_dir=folder,
            is_approved=is_approved,
        )

        for variant_key in OFFICIAL_VARIANTS:
            character.variants[variant_key] = CharacterVariant(variant=variant_key)

        files = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in [ext.lower() for ext in self.supported_extensions]
        ]

        for filepath in files:
            file_type, variant_name, warning_msg = parse_file_variant(
                filepath, expected_folder_name=char_id, return_warning=True
            )
            if warning_msg:
                logger.warning(f"[{char_id}] {warning_msg}")

            if not variant_name or variant_name not in character.variants:
                continue

            variant_obj = character.variants[variant_name]
            file_hash = compute_file_sha256(filepath)
            variant_obj.file_hashes[filepath.name] = file_hash

            if file_type == "reference":
                variant_obj.reference_image = filepath
            elif file_type == "spritesheet":
                variant_obj.spritesheet = filepath

        # Determinar status
        for v_name, v_obj in character.variants.items():
            if v_obj.has_reference and v_obj.has_spritesheet:
                v_obj.status = "ready"
            elif v_obj.has_reference:
                v_obj.status = "reference_only"
            elif v_obj.has_spritesheet:
                v_obj.status = "spritesheet_only"
            else:
                v_obj.status = "missing"

        has_any_asset = any(v.has_reference or v.has_spritesheet for v in character.variants.values())
        if not has_any_asset:
            logger.debug(f"Carpeta {folder.name} ignorada: no contiene assets de animación.")
            return None

        return character

    def scan_directory(self, root_dir: Optional[Union[str, Path]] = None, is_approved: bool = True) -> Dict[str, Character]:
        """
        Escanea el directorio raíz navegando dentro de contenedores intermedios
        como 'personajes al 100%' si existen.
        """
        scan_target = Path(root_dir).resolve() if root_dir else (self.root or Path.cwd().resolve())

        # Navegar si existe contenedor intermedio
        effective_root = self._resolve_character_container(scan_target)

        characters: Dict[str, Character] = {}
        if not effective_root.exists() or not effective_root.is_dir():
            logger.warning(f"El directorio no existe: {effective_root}")
            return characters

        entries = sorted([d for d in effective_root.iterdir() if d.is_dir()], key=lambda p: p.name.lower())

        for entry in entries:
            if entry.name.startswith(".") or entry.name.startswith("__"):
                continue

            char = self.scan_character_folder(entry, is_approved=is_approved)
            if char:
                # Cada carpeta representa una identidad única
                characters[char.character_id] = char

        logger.info(
            f"Escaneo finalizado en '{effective_root}': {len(characters)} personajes identificados."
        )
        return characters
