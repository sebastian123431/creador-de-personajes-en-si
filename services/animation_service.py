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
from models.frame import Frame

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

    def generate_articulated_animations_v2(
        self,
        reference_image: Path,
        character_id: str,
        variant_name: str,
    ) -> Dict[str, Animation]:
        """
        Genera los 64 frames (16 animaciones x 4 frames) utilizando el motor articulado V2:
        - 18 anclajes anatómicos
        - HeadIdentityLock (máxima preservación de identidad visual)
        - pixel_rotate sin difuminado
        - PaletteGuard (garantía de integridad cromática)
        """
        from core.motion_transfer_v2 import MotionTransferV2
        from core.template_extractor_v2 import TemplateExtractorV2
        from core.frame_extractor import OFFICIAL_ANIMATION_ROWS

        transfer_engine_v2 = MotionTransferV2()
        template_extractor_v2 = TemplateExtractorV2(base_templates_dir=self.base_dir / "dataset" / "templates_v2")

        out_root = self.base_dir / "dataset" / "transferred_v2" / character_id / variant_name
        animations_dict: Dict[str, Animation] = {}

        for anim_name in OFFICIAL_ANIMATION_ROWS:
            template = template_extractor_v2.load_template(anim_name)
            if not template:
                # Si no existe la plantilla V2, generar plantilla neutral de respaldo
                from models.articulated_motion_template import ArticulatedMotionTemplate
                template = ArticulatedMotionTemplate(animation_name=anim_name)

            anim_dir = out_root / anim_name
            frames_imgs = transfer_engine_v2.generate_animation_frames(
                reference_image=reference_image,
                template=template,
                output_dir=anim_dir
            )

            frame_objects = []
            for idx, img in enumerate(frames_imgs, start=1):
                f_path = anim_dir / f"{idx:02d}.png"
                fw, fh = img.size
                frame_objects.append(Frame(
                    image_path=f_path,
                    index=idx,
                    bbox=(0, 0, fw, fh),
                    center_x=fw // 2,
                    baseline_y=fh - 1,
                    width=fw,
                    height=fh
                ))

            anim = Animation(
                name=anim_name,
                frames=frame_objects,
                row_index=OFFICIAL_ANIMATION_ROWS.index(anim_name) if anim_name in OFFICIAL_ANIMATION_ROWS else 0
            )
            animations_dict[anim_name] = anim

        logger.info(f"Generadas 16 animaciones articuladas V2 para '{character_id}:{variant_name}' en: {out_root}")
        return animations_dict

