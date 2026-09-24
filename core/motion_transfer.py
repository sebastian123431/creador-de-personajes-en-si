import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from PIL import Image

from core.bbox_detector import BoundingBoxDetector
from core.compositor import PropCompositor, PropTransform
from core.frame_normalizer import FrameNormalizer
from core.frame_extractor import OFFICIAL_ANIMATION_ROWS
from models.animation import Animation
from models.frame import Frame
from models.motion_template import MotionTemplate

logger = logging.getLogger("SpriteStudio.MotionTransfer")


class MotionTransferEngine:
    """
    Motor de transferencia cinemática de movimiento a personajes nuevos.
    PRIORIDAD ABSOLUTA: CONSISTENCIA VISUAL > CREATIVIDAD.
    Preserva la cabeza, rostro y silueta original, aplicando únicamente
    los desplazamientos y props aprendidos en las plantillas.
    """

    def __init__(
        self,
        normalizer: Optional[FrameNormalizer] = None,
        compositor: Optional[PropCompositor] = None
    ):
        self.normalizer = normalizer or FrameNormalizer()
        self.compositor = compositor or PropCompositor()

    def generate_animation_frames(
        self,
        reference_image_path: Path,
        template: MotionTemplate,
        output_dir: Path,
        animation_name: str
    ) -> Animation:
        """
        Genera los 4 frames de una animación aplicando la plantilla de movimiento
        a la imagen de referencia del personaje.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Normalizar la referencia en un canvas base transparente 256x192
        temp_norm_path = output_dir / "_temp_ref_norm.png"
        norm_ref_frame = self.normalizer.normalize_frame(
            reference_image_path,
            custom_output_path=temp_norm_path
        )

        with Image.open(temp_norm_path) as base_canvas:
            base_canvas = base_canvas.convert("RGBA")
            cw, ch = base_canvas.size

            # Segmentar cabeza y cuerpo dentro del canvas normalizado
            head_box, body_box = BoundingBoxDetector.detect_head_body_split(base_canvas)

            # Si no se detectó segmentación, usar el canvas completo
            has_split = head_box[3] > head_box[1] and body_box[3] > body_box[1]

            if has_split:
                head_crop = base_canvas.crop(head_box)
                body_crop = base_canvas.crop(body_box)

            generated_frames: List[Frame] = []

            for f_idx in range(template.frame_count):
                frame_data = template.frames[f_idx] if f_idx < len(template.frames) else None

                frame_canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))

                b_dx = frame_data.body.get("x", 0) if frame_data else 0
                b_dy = frame_data.body.get("y", 0) if frame_data else 0

                h_dx = frame_data.head.get("x", 0) if frame_data else 0
                h_dy = frame_data.head.get("y", 0) if frame_data else 0

                # Para idle y caminatas leves, mantener la cabeza estable con micro-rebote
                if "idle" in animation_name:
                    # Rebote suave de respiración
                    idle_offsets = [0, -1, 0, 1]
                    b_dy = idle_offsets[f_idx % 4]
                    h_dy = idle_offsets[f_idx % 4]

                if has_split:
                    # Pegar cuerpo con desplazamiento
                    frame_canvas.paste(
                        body_crop,
                        (body_box[0] + b_dx, body_box[1] + b_dy),
                        mask=body_crop
                    )
                    # Pegar cabeza (reutilizando cara original para máxima consistencia visual)
                    frame_canvas.paste(
                        head_crop,
                        (head_box[0] + h_dx, head_box[1] + h_dy),
                        mask=head_crop
                    )
                else:
                    # Si no hay split limpio, desplazar sprite entero
                    frame_canvas.paste(
                        base_canvas,
                        (b_dx, b_dy),
                        mask=base_canvas
                    )

                # 2. Superposición de Props según la animación oficial
                hand_anchor = (cw // 2 + 12, ch // 2 + 25)

                if "cook" in animation_name:
                    # Cocinar: bowl fijo + cuchara con movimiento
                    spoon_dy = [-3, 2, -2, 3][f_idx % 4]
                    frame_canvas = self.compositor.composite_prop(
                        frame_canvas, "bowl", hand_anchor, PropTransform(scale=1.0, dy=0)
                    )
                    frame_canvas = self.compositor.composite_prop(
                        frame_canvas, "spoon", (hand_anchor[0], hand_anchor[1] - 8),
                        PropTransform(scale=1.0, dy=spoon_dy, rotation=-15 if f_idx % 2 == 0 else 15)
                    )
                elif animation_name == "pickup":
                    # Caja en las manos
                    frame_canvas = self.compositor.composite_prop(
                        frame_canvas, "box", (cw // 2, hand_anchor[1]), PropTransform(scale=1.0, dy=b_dy)
                    )
                elif animation_name == "serve":
                    # Plato en las manos
                    frame_canvas = self.compositor.composite_prop(
                        frame_canvas, "plate", (cw // 2 + 10, hand_anchor[1]), PropTransform(scale=1.0, dy=b_dy)
                    )

                # Guardar frame final en formato PNG
                frame_filename = f"{f_idx + 1:02d}.png"
                frame_out_path = output_dir / frame_filename
                frame_canvas.save(frame_out_path, "PNG")

                # Analizar bbox
                bbox = BoundingBoxDetector.detect_sprite_bbox(frame_canvas) or (0, 0, cw, ch)
                alpha_arr = np.array(frame_canvas.getchannel("A"))
                alpha_area = int(np.sum(alpha_arr > 10))

                frame_obj = Frame(
                    image_path=frame_out_path,
                    index=f_idx + 1,
                    bbox=bbox,
                    center_x=cw // 2,
                    baseline_y=bbox[3],
                    width=cw,
                    height=ch,
                    alpha_area=alpha_area
                )
                generated_frames.append(frame_obj)

        if temp_norm_path.exists():
            temp_norm_path.unlink()

        return Animation(name=animation_name, frames=generated_frames)

    def generate_full_64_frames(
        self,
        reference_image_path: Path,
        templates: Dict[str, MotionTemplate],
        character_id: str,
        variant: str,
        output_root: Optional[Path] = None
    ) -> Dict[str, Animation]:
        """
        Genera el set completo de 64 frames (16 animaciones x 4 frames)
        para el personaje nuevo utilizando la biblioteca de templates.
        """
        base_out = (output_root or Path("dataset/generated_frames")) / character_id / variant
        base_out.mkdir(parents=True, exist_ok=True)

        full_animations: Dict[str, Animation] = {}

        for r_idx, anim_name in enumerate(OFFICIAL_ANIMATION_ROWS):
            tmpl = templates.get(anim_name)
            if not tmpl:
                # Si no hay template, generar uno estático por defecto
                tmpl = MotionTemplate(name=anim_name, frame_count=4)

            anim_dir = base_out / anim_name
            anim_obj = self.generate_animation_frames(
                reference_image_path,
                template=tmpl,
                output_dir=anim_dir,
                animation_name=anim_name
            )
            anim_obj.row_index = r_idx
            full_animations[anim_name] = anim_obj

        logger.info(f"64 frames generados con éxito para '{character_id}:{variant}' en: {base_out}")
        return full_animations
