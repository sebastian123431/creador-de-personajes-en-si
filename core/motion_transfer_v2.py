import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from core.body_part_segmenter import BodyPartSegmenter
from core.guard import SourceDatasetGuard
from core.layer_resolver import LayerResolver
from core.palette_guard import PaletteGuard
from core.pose_analyzer_v2 import PoseAnalyzerV2
from models.articulated_motion_template import ArticulatedMotionTemplate, PartMotion
from models.body_part import BodyPart
from models.skeleton import Skeleton

logger = logging.getLogger("SpriteStudio.MotionTransferV2")


def pixel_rotate(
    image: Image.Image,
    angle_deg: float,
    pivot: Tuple[float, float]
) -> Tuple[Image.Image, Tuple[int, int]]:
    """
    Rotación pixel art con muestreo de vecino más cercano (Nearest-Neighbor).
    Rota una imagen alrededor de un pivote específico (px, py) sin interpolación bicúbica
    ni anti-aliasing borroso, preservando la paleta exacta de píxeles.
    
    Retorna:
    - Image.Image: Imagen rotada con canal alfa transparente.
    - Tuple[int, int]: Desplazamiento (offset_x, offset_y) respecto al origen original
      para que el punto pivote permanezca exactamente invariante en el espacio del canvas.
    """
    if abs(angle_deg) < 0.2:
        return image.copy(), (0, 0)

    src_arr = np.array(image.convert("RGBA"))
    h, w = src_arr.shape[:2]
    px, py = float(pivot[0]), float(pivot[1])

    # Ángulo en radianes (sentido antihorario en pantalla con Y invertido)
    rad = math.radians(angle_deg)
    cos_t = math.cos(rad)
    sin_t = math.sin(rad)

    # Calcular nuevas coordenadas de las 4 esquinas para determinar el nuevo bbox
    corners = [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]
    rot_corners_x = []
    rot_corners_y = []

    for cx, cy in corners:
        rx = px + (cx - px) * cos_t - (cy - py) * sin_t
        ry = py + (cx - px) * sin_t + (cy - py) * cos_t
        rot_corners_x.append(rx)
        rot_corners_y.append(ry)

    min_rx = min(rot_corners_x)
    max_rx = max(rot_corners_x)
    min_ry = min(rot_corners_y)
    max_ry = max(rot_corners_y)

    offset_x = int(math.floor(min_rx))
    offset_y = int(math.floor(min_ry))

    new_w = int(math.ceil(max_rx)) - offset_x + 1
    new_h = int(math.ceil(max_ry)) - offset_y + 1

    # Mapeo inverso vectorizado con cuadrícula de destino
    dest_y, dest_x = np.indices((new_h, new_w))
    world_x = dest_x + offset_x
    world_y = dest_y + offset_y

    # Rotación inversa hacia coordenadas fuente
    src_x = np.round(px + (world_x - px) * cos_t + (world_y - py) * sin_t).astype(int)
    src_y = np.round(py - (world_x - px) * sin_t + (world_y - py) * cos_t).astype(int)

    valid_mask = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)

    dest_arr = np.zeros((new_h, new_w, 4), dtype=np.uint8)
    dest_arr[dest_y[valid_mask], dest_x[valid_mask]] = src_arr[src_y[valid_mask], src_x[valid_mask]]

    return Image.fromarray(dest_arr, mode="RGBA"), (offset_x, offset_y)


class MotionTransferV2:
    """
    Motor de Transferencia de Movimiento Articulado V2.
    
    PRIORIDAD ABSOLUTA: CONSISTENCIA VISUAL > CREATIVIDAD.
    - HeadIdentityLock: La cabeza y rostro del personaje original se conservan 100% bit-exactos,
      trasladándose según la cinemática de la plantilla sin ninguna deformación ni regeneración.
    - pixel_rotate: Rotación de extremidades sobre pivotes sin aberraciones ni difuminados.
    - LayerResolver: Ordenamiento estricto de capas según la orientación del sprite.
    - PaletteGuard: Garantía de que ningún color extraño entre en el sprite final.
    """

    def __init__(
        self,
        segmenter: Optional[BodyPartSegmenter] = None,
        pose_analyzer: Optional[PoseAnalyzerV2] = None,
        guard: Optional[SourceDatasetGuard] = None,
    ):
        self.segmenter = segmenter or BodyPartSegmenter(guard=guard)
        self.pose_analyzer = pose_analyzer or PoseAnalyzerV2()
        self.guard = guard

    def generate_animation_frames(
        self,
        reference_image: Union[Path, str, Image.Image],
        template: ArticulatedMotionTemplate,
        output_dir: Optional[Path] = None,
        skeleton: Optional[Skeleton] = None,
    ) -> List[Image.Image]:
        """
        Genera los frames de animación transfiriendo la plantilla articulada
        al personaje de referencia.
        """
        if isinstance(reference_image, (str, Path)):
            ref_img = Image.open(str(reference_image)).convert("RGBA")
        else:
            ref_img = reference_image.convert("RGBA")

        canvas_w, canvas_h = ref_img.size

        # 1. Extraer la paleta original del personaje para PaletteGuard
        ref_palette = PaletteGuard.extract_palette(ref_img)

        # 2. Obtener esqueleto de referencia si no se proporcionó
        skel = skeleton
        if skel is None:
            skel = self.pose_analyzer.analyze_pose(ref_img, orientation=template.orientation)

        # 3. Segmentar el personaje de referencia en las 6 partes canónicas
        parts, part_images = self.segmenter.segment_frame(
            ref_img, skeleton=skel, orientation=template.orientation
        )

        generated_frames: List[Image.Image] = []

        if output_dir:
            out_path = Path(output_dir).resolve()
            if self.guard:
                self.guard.assert_can_write(out_path, operation_desc="transferencia de movimiento articulado V2")
            out_path.mkdir(parents=True, exist_ok=True)

        # Ordenar partes según Z-Index para dibujado (Painter's Algorithm)
        ordered_parts = LayerResolver.sort_parts_for_rendering(parts)

        # 4. Generar cada frame del ciclo de animación
        for frame_idx in range(1, template.frame_count + 1):
            frame_template = template.get_frame(frame_idx)
            root_dx = frame_template.root_dx if frame_template else 0.0
            root_dy = frame_template.root_dy if frame_template else 0.0

            frame_canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

            for part in ordered_parts:
                pname = part.name
                if pname not in part_images:
                    continue

                p_img = part_images[pname]
                p_motion = frame_template.parts.get(pname, PartMotion()) if frame_template else PartMotion()

                tot_dx = int(round(p_motion.dx + root_dx))
                tot_dy = int(round(p_motion.dy + root_dy))

                if pname == "head":
                    # --- HEAD IDENTITY LOCK ---
                    # La cabeza se traslada de forma entera para preservar el 100% de píxeles originales
                    paste_x = part.bbox[0] + tot_dx
                    paste_y = part.bbox[1] + tot_dy
                    frame_canvas.paste(p_img, (paste_x, paste_y), mask=p_img)
                else:
                    # Extremidades y torso: rotación cinemática con pixel_rotate sobre el pivote
                    pivot_rel = (part.pivot_x - part.bbox[0], part.pivot_y - part.bbox[1])
                    rot_img, (rot_off_x, rot_off_y) = pixel_rotate(p_img, p_motion.angle_deg, pivot_rel)

                    paste_x = part.bbox[0] + rot_off_x + tot_dx
                    paste_y = part.bbox[1] + rot_off_y + tot_dy
                    frame_canvas.paste(rot_img, (paste_x, paste_y), mask=rot_img)

            # 5. Aplicar saneamiento de paleta cromática (PaletteGuard)
            clean_frame = PaletteGuard.clean_frame(frame_canvas, ref_palette)

            if output_dir:
                file_path = out_path / f"{frame_idx:02d}.png"
                clean_frame.save(file_path, "PNG")

            generated_frames.append(clean_frame)

        logger.info(f"Generados {len(generated_frames)} frames de movimiento articulado para '{template.animation_name}'")
        return generated_frames
