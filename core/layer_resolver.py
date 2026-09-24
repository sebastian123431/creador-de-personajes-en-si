import logging
from typing import Dict, List, Optional
from models.body_part import BodyPart

logger = logging.getLogger("SpriteStudio.LayerResolver")


class LayerResolver:
    """
    Resuelve el orden de profundidad (Z-Index / Render Order) de las 6 partes corporales
    según la orientación o punto de vista del sprite (down, up, left, right).
    
    Garantiza que:
    - En vista frontal (down), las extremidades y el torso no se solapen incorrectamente.
    - En vista trasera (up), la cabeza y torso cubran adecuadamente los hombros y cuello.
    - En vista lateral (left/right), las extremidades lejanas se rendericen por detrás del torso
      y las extremidades cercanas por delante.
    """

    # Z-Index predeterminado por orientación (menor número = más al fondo, mayor = más al frente)
    DEFAULT_Z_ORDER: Dict[str, Dict[str, int]] = {
        "down": {
            "left_leg": 1,
            "right_leg": 1,
            "torso": 2,
            "left_arm": 3,
            "right_arm": 3,
            "head": 4,
        },
        "up": {
            "left_leg": 1,
            "right_leg": 1,
            "left_arm": 2,
            "right_arm": 2,
            "torso": 3,
            "head": 4,
        },
        "left": {
            "right_arm": 0,  # Brazo lejano al fondo
            "right_leg": 1,  # Pierna lejana
            "torso": 2,      # Torso intermedio
            "left_leg": 3,   # Pierna cercana
            "left_arm": 4,   # Brazo cercano al frente
            "head": 5,       # Cabeza
        },
        "right": {
            "left_arm": 0,   # Brazo lejano al fondo
            "left_leg": 1,   # Pierna lejana
            "torso": 2,      # Torso intermedio
            "right_leg": 3,  # Pierna cercana
            "right_arm": 4,  # Brazo cercano al frente
            "head": 5,       # Cabeza
        },
    }

    @staticmethod
    def normalize_orientation(direction_or_anim: str) -> str:
        """
        Normaliza strings como 'walk_down', 'cook_left', 'up', 'down' a una de las 4 direcciones base:
        'down', 'up', 'left', 'right'.
        """
        s = direction_or_anim.lower()
        if "down" in s:
            return "down"
        elif "up" in s:
            return "up"
        elif "left" in s:
            return "left"
        elif "right" in s:
            return "right"
        return "down"

    @classmethod
    def get_z_indices_for_orientation(cls, orientation: str) -> Dict[str, int]:
        norm = cls.normalize_orientation(orientation)
        return cls.DEFAULT_Z_ORDER.get(norm, cls.DEFAULT_Z_ORDER["down"]).copy()

    @classmethod
    def resolve_z_indices(cls, orientation: str, parts: Dict[str, BodyPart]) -> Dict[str, BodyPart]:
        """
        Aplica los z_indices correspondientes a la orientación a cada BodyPart en el diccionario.
        """
        z_map = cls.get_z_indices_for_orientation(orientation)
        for name, part in parts.items():
            if name in z_map:
                part.z_index = z_map[name]
        return parts

    @classmethod
    def sort_parts_for_rendering(cls, parts: Dict[str, BodyPart]) -> List[BodyPart]:
        """
        Retorna la lista de BodyPart ordenada de menor a mayor z_index (orden de dibujado Painter's Algorithm).
        """
        return sorted(parts.values(), key=lambda p: (p.z_index, p.name))
