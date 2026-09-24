import logging
from pathlib import Path
from typing import Dict, Optional

from core.frame_extractor import FrameExtractor
from core.frame_normalizer import FrameNormalizer
from core.motion_transfer import MotionTransferEngine
from core.template_library import TemplateLibrary
from models.animation import Animation
from models.character import Character
from models.character_variant import CharacterVariant

logger = logging.getLogger("SpriteStudio.AnimationService")


class AnimationService:
    """
    Servicio de orquestación de animaciones: extracción, normalización,
    carga de frames y generación por transferencia cinemática.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir or Path.cwd()).resolve()
        self.extractor = FrameExtractor(output_base_dir=self.base_dir / "dataset" / "extracted_frames")
        self.normalizer = FrameNormalizer(output_base_dir=self.base_dir / "dataset" / "normalized")
        self.transfer_engine = MotionTransferEngine(normalizer=self.normalizer)
        self.template_library = TemplateLibrary(templates_dir=self.base_dir / "animations" / "learned")

    def get_or_extract_animations(self, character: Character, variant_name: str) -> Dict[str, Animation]:
        """
        Obtiene las animaciones de un personaje. Si ya fueron extraídas las carga;
        si no, las extrae desde el spritesheet si está disponible.
        """
        variant = character.variants.get(variant_name)
        if not variant:
            return {}

        if variant.animations:
            return variant.animations

        # Extraer si tiene spritesheet
        if variant.has_spritesheet and variant.spritesheet:
            anims = self.extractor.extract_from_sheet(
                variant.spritesheet,
                character.character_id,
                variant_name
            )
            variant.animations = anims
            variant.status = "extracted"
            return anims

        return {}

    def generate_animations_for_new_character(
        self,
        reference_image: Path,
        character_id: str,
        variant_name: str
    ) -> Dict[str, Animation]:
        """
        Genera los 64 frames para un personaje nuevo aplicando la biblioteca de templates aprendidos.
        """
        templates = self.template_library.load_all_templates()
        out_root = self.base_dir / "dataset" / "generated_frames"
        anims = self.transfer_engine.generate_full_64_frames(
            reference_image,
            templates=templates,
            character_id=character_id,
            variant=variant_name,
            output_root=out_root
        )
        return anims
