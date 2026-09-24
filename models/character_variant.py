from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional
from models.animation import Animation


@dataclass
class CharacterVariant:
    """
    Representa una variante específica del personaje:
    - rnormal: ropa normal
    - rbchef: ropa blanca chef
    - rnchef: ropa negra chef
    """
    variant: str
    reference_image: Optional[Path] = None
    spritesheet: Optional[Path] = None
    animations: Dict[str, Animation] = field(default_factory=dict)
    file_hashes: Dict[str, str] = field(default_factory=dict)
    status: str = "pending"  # "reference_only", "ready", "extracted", "validated"

    @property
    def has_reference(self) -> bool:
        return self.reference_image is not None and self.reference_image.exists()

    @property
    def has_spritesheet(self) -> bool:
        return self.spritesheet is not None and self.spritesheet.exists()

    @property
    def is_complete(self) -> bool:
        return self.has_reference and self.has_spritesheet
