import re
from pathlib import Path
from typing import Optional, Tuple, Union

OFFICIAL_VARIANTS = ("rnormal", "rbchef", "rnchef")

# Etiquetas visuales para GUI
VARIANT_LABELS = {
    "rnormal": "Ropa normal",
    "rbchef": "Chef blanco",
    "rnchef": "Chef negro"
}


def sanitize_character_id(folder_or_name: str) -> str:
    """
    Retorna el ID oficial del personaje basado estrictamente en el nombre de la carpeta.
    Conserva nombres compuestos (ej: 'diego_vallenar', 'benja_bacaba', 'alex_2', 'niko_coquimbo').
    NO los aplana, NO los fusiona y NO elimina apellidos ni apodos.
    """
    return folder_or_name.strip()


def format_display_name(character_id: str) -> str:
    """
    Genera opcionalmente un display_name estético para la interfaz ('diego_vallenar' -> 'Diego Vallenar').
    El character_id original permanece intacto.
    """
    parts = character_id.replace("-", "_").split("_")
    return " ".join(part.capitalize() for part in parts if part)


def make_unique_key(character_id: str, variant: str) -> str:
    """
    Retorna la clave canónica 'character_id:variant', ej: 'diego_vallenar:rnormal'.
    """
    return f"{character_id}:{variant}"


def parse_file_variant(
    filename_or_path: str | Path,
    expected_folder_name: Optional[str] = None,
    return_warning: bool = False
) -> Union[Tuple[Optional[str], Optional[str]], Tuple[Optional[str], Optional[str], Optional[str]]]:
    """
    Analiza un nombre de archivo y retorna (file_type, variant) por defecto,
    o (file_type, variant, warning_message) si return_warning=True.
    """
    path_obj = Path(filename_or_path)
    stem = path_obj.stem.lower()
    warning_msg = None

    # Detectar si es spritesheet oficial
    for v in OFFICIAL_VARIANTS:
        if stem == f"movimientos_{v}" or stem == f"movimiento_{v}":
            if return_warning:
                return "spritesheet", v, None
            return "spritesheet", v

    # Detectar si es imagen de referencia por sufijo (*_rnormal.png, *_rbchef.png, *_rnchef.png)
    detected_variant = None
    for v in OFFICIAL_VARIANTS:
        if stem.endswith(f"_{v}") or stem == v:
            detected_variant = v
            break

    # Si no detectó por sufijo exacto, buscar si contiene variaciones históricas toleradas
    if not detected_variant:
        if "rbnormal" in stem:
            detected_variant = "rnormal"
            warning_msg = f"Nombre histórico anómalo detectado en '{path_obj.name}'; asociado a 'rnormal'."
        elif "rnormal" in stem:
            detected_variant = "rnormal"
        elif "rbchef" in stem:
            detected_variant = "rbchef"
        elif "rnchef" in stem:
            detected_variant = "rnchef"

    if not detected_variant:
        if return_warning:
            return None, None, None
        return None, None

    # Verificar si el archivo es un spritesheet con otro patrón
    if stem.startswith("movimiento") or "spritesheet" in stem:
        if return_warning:
            return "spritesheet", detected_variant, warning_msg
        return "spritesheet", detected_variant

    # Si es imagen de referencia, chequear si el prefijo difiere del nombre de carpeta
    if expected_folder_name:
        expected_stem_prefix = expected_folder_name.lower()
        if not stem.startswith(expected_stem_prefix) and stem != detected_variant:
            warning_msg = (
                f"Prefijo de referencia '{path_obj.name}' difiere de carpeta '{expected_folder_name}'. "
                f"Asociado exitosamente a '{detected_variant}' sin renombrar."
            )

    if return_warning:
        return "reference", detected_variant, warning_msg
    return "reference", detected_variant
