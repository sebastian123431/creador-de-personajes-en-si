import re
from pathlib import Path
from typing import Optional, Tuple

OFFICIAL_VARIANTS = ("rnormal", "rbchef", "rnchef")

VARIANT_LABELS = {
    "rnormal": "Ropa Normal",
    "rbchef": "Ropa Blanca Chef",
    "rnchef": "Ropa Negra Chef"
}


def sanitize_character_id(folder_or_name: str) -> str:
    """
    Normaliza el identificador de personaje conservando componentes compuestos
    (ciudad, apellido, apodo, etc., ej: 'diego_vallenar', 'andres_arica').
    No fusiona ni elimina sufijos de nombres compuestos.
    """
    clean = folder_or_name.strip().lower()
    clean = re.sub(r"[^\w\-]", "_", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean


def format_display_name(character_id: str) -> str:
    """
    Convierte 'diego_vallenar' a 'Diego Vallenar', 'alex' a 'Alex'.
    """
    parts = character_id.replace("-", "_").split("_")
    return " ".join(part.capitalize() for part in parts if part)


def make_unique_key(character_id: str, variant: str) -> str:
    """
    Retorna la clave canónica 'character_id:variant', ej: 'diego_vallenar:rnormal'.
    """
    return f"{character_id}:{variant}"


def parse_file_variant(filename_or_path: str | Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Analiza un nombre de archivo y determina:
    - tipo de archivo: 'reference' o 'spritesheet' (o None si no coincide)
    - variante: 'rnormal', 'rbchef', 'rnchef' (o None)

    Ejemplos:
    - 'alex_rnormal.png' -> ('reference', 'rnormal')
    - 'diego_vallenar_rbchef.png' -> ('reference', 'rbchef')
    - 'movimientos_rnormal.png' -> ('spritesheet', 'rnormal')
    - 'movimientos_rnchef.png' -> ('spritesheet', 'rnchef')
    """
    stem = Path(filename_or_path).stem.lower()

    # Detectar cuál de las variantes oficiales está presente
    detected_variant = None
    for v in OFFICIAL_VARIANTS:
        # Match al final del stem o precedido de underscore
        if stem.endswith(f"_{v}") or stem == v or f"_{v}_" in stem:
            detected_variant = v
            break

    if not detected_variant:
        return None, None

    # Detectar si es un spritesheet de movimientos
    if stem.startswith("movimiento") or "spritesheet" in stem or stem.startswith("sheet_"):
        return "spritesheet", detected_variant
    else:
        return "reference", detected_variant
