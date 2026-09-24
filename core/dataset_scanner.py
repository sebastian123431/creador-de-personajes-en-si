import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional

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
    Escanea carpetas de personajes respetando las identidades compuestas
    (ej: diego_vallenar vs diego_serena) y asociando referencias y spritesheets
    a las variantes oficiales (rnormal, rbchef, rnchef).
    """

    def __init__(self, supported_extensions: Optional[List[str]] = None):
        self.supported_extensions = supported_extensions or [".png", ".PNG"]

    def scan_character_folder(self, folder: Path, is_approved: bool = False) -> Optional[Character]:
        """
        Escanea una carpeta individual de personaje y construye el objeto Character.
        """
        if not folder.is_dir():
            return None

        raw_id = folder.name
        char_id = sanitize_character_id(raw_id)
        display_name = format_display_name(char_id)

        character = Character(
            character_id=char_id,
            display_name=display_name,
            source_dir=folder,
            is_approved=is_approved,
        )

        # Preparar las 3 variantes oficiales
        for variant_key in OFFICIAL_VARIANTS:
            character.variants[variant_key] = CharacterVariant(variant=variant_key)

        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in [ext.lower() for ext in self.supported_extensions]]

        for filepath in files:
            file_type, variant_name = parse_file_variant(filepath)
            if not variant_name or variant_name not in character.variants:
                continue

            variant_obj = character.variants[variant_name]
            file_hash = compute_file_sha256(filepath)
            variant_obj.file_hashes[filepath.name] = file_hash

            if file_type == "reference":
                variant_obj.reference_image = filepath
            elif file_type == "spritesheet":
                variant_obj.spritesheet = filepath

        # Actualizar status de cada variante
        for v_name, v_obj in character.variants.items():
            if v_obj.has_reference and v_obj.has_spritesheet:
                v_obj.status = "ready"
            elif v_obj.has_reference:
                v_obj.status = "reference_only"
            elif v_obj.has_spritesheet:
                v_obj.status = "spritesheet_only"
            else:
                v_obj.status = "missing"

        # Verificar si al menos una variante tiene algún asset
        has_any_asset = any(v.has_reference or v.has_spritesheet for v in character.variants.values())
        if not has_any_asset:
            logger.debug(f"Carpeta {folder.name} ignorada: no contiene assets reconocibles.")
            return None

        logger.info(
            f"Personaje escaneado: '{char_id}' ({display_name}) - "
            f"Referencias: {character.reference_count}/3, Spritesheets: {character.spritesheet_count}/3"
        )
        return character

    def scan_directory(self, root_dir: Path, is_approved: bool = False) -> Dict[str, Character]:
        """
        Escanea recursivamente o a primer nivel un directorio que contiene subcarpetas de personajes.
        Retorna un diccionario {character_id: Character}.
        """
        characters: Dict[str, Character] = {}
        if not root_dir.exists() or not root_dir.is_dir():
            logger.warning(f"El directorio no existe o no es carpeta: {root_dir}")
            return characters

        # Buscar subcarpetas de personajes
        entries = sorted([d for d in root_dir.iterdir() if d.is_dir()], key=lambda p: p.name.lower())

        for entry in entries:
            # Ignorar carpetas ocultas o de sistema
            if entry.name.startswith(".") or entry.name.startswith("__"):
                continue

            char = self.scan_character_folder(entry, is_approved=is_approved)
            if char:
                # Cada carpeta representa UNA IDENTIDAD ÚNICA
                if char.character_id in characters:
                    logger.warning(f"Colisión de ID de personaje: '{char.character_id}'. Manteniendo la primera encontrada.")
                else:
                    characters[char.character_id] = char

        logger.info(f"Escaneo finalizado en {root_dir}: {len(characters)} personajes identificados.")
        return characters
