from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional
from models.character_variant import CharacterVariant


@dataclass
class Character:
    """
    Representa una identidad única de personaje.
    Nombres como 'diego_vallenar' y 'diego_serena' representan identidades distintas.
    """
    character_id: str
    display_name: str
    variants: Dict[str, CharacterVariant] = field(default_factory=dict)
    source_dir: Optional[Path] = None
    is_approved: bool = False

    def get_variant(self, variant_name: str) -> Optional[CharacterVariant]:
        return self.variants.get(variant_name)

    @property
    def variant_count(self) -> int:
        return len(self.variants)

    @property
    def spritesheet_count(self) -> int:
        return sum(1 for v in self.variants.values() if v.has_spritesheet)

    @property
    def reference_count(self) -> int:
        return sum(1 for v in self.variants.values() if v.has_reference)
