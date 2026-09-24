import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
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


def compute_motion_delta_score(frames: List[Image.Image]) -> float:
    """
    Calcula la métrica de variación de movimiento (motion_delta_score)
    entre los frames generados de una animación. Retorna el porcentaje promedio de píxeles
    que cambian entre frames consecutivos.
    """
    if len(frames) < 2:
        return 0.0

    scores = []
    for i in range(len(frames)):
        f_curr = np.array(frames[i].convert("RGBA"))
        f_next = np.array(frames[(i + 1) % len(frames)].convert("RGBA"))
        # Diferencia de píxeles no idénticos
        diff = np.any(f_curr != f_next, axis=-1)
        scores.append(float(np.mean(diff) * 100.0))

    return float(np.mean(scores))


class MotionTransferV2:
    """
    Motor de Transferencia de Movimiento Articulado V2.
    
    PRIORIDAD ABSOLUTA: CONSISTENCIA VISUAL > CREATIVIDAD.
    - HeadIdentityLock activo: Máxima preservación de identidad visual. La cabeza y rostro
      del personaje de referencia se conservan íntegros, trasladándose con precisión geométrica.
    - Articulación anatómica por cadenas: Brazos (upper arm, forearm, hand) y Piernas (thigh,
      lower leg, foot) rotados en torno a sus pivotes articulares reales (hombro, codo, mano,
      cadera, rodilla, pie).
    - pixel_rotate: Rotación de extremidades con muestreo Nearest-Neighbor estricto sin
      interpolación suave ni antialiasing.
    - LayerResolver: Ordenamiento estricto de capas Z-index según la orientación.
    - PaletteGuard: Garantía de que ningún color espurio penetre en los sprites finales.
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

    def _render_limb_chain(
        self,
        limb_name: str,
        limb_part: BodyPart,
        limb_img: Image.Image,
        skeleton: Skeleton,
        frame_template: Any,
        root_dx: float,
        root_dy: float,
        frame_canvas: Image.Image,
    ):
        """
        Renderiza una extremidad articulada dividiéndola en su cadena anatómica:
        - Brazos: upper_arm (pivote: shoulder), forearm (pivote: elbow), hand (pivote: hand/wrist)
        - Piernas: thigh (pivote: hip), lower_leg (pivote: knee), foot (pivote: foot)
        """
        arr = np.array(limb_img)
        h, w = arr.shape[:2]
        bx0, by0, bx1, by1 = limb_part.bbox

        # Determinar cadena y pivotes
        if "arm" in limb_name:
            prefix = "left_" if "left" in limb_name else "right_"
            p_shoulder = skeleton.get_anchor(f"{prefix}shoulder")
            p_elbow = skeleton.get_anchor(f"{prefix}elbow")
            p_wrist = skeleton.get_anchor(f"{prefix}wrist")
            p_hand = skeleton.get_anchor(f"{prefix}hand")

            elbow_y = p_elbow.y if p_elbow else by0 + h // 3
            wrist_y = p_wrist.y if p_wrist else by0 + (2 * h) // 3

            sub_defs = [
                (f"{prefix}upper_arm", 0, max(1, min(h, elbow_y - by0)),
                 (p_shoulder.x if p_shoulder else bx0, p_shoulder.y if p_shoulder else by0)),
                (f"{prefix}forearm", max(0, min(h, elbow_y - by0)), max(1, min(h, wrist_y - by0)),
                 (p_elbow.x if p_elbow else bx0, p_elbow.y if p_elbow else by0 + h // 3)),
                (f"{prefix}hand", max(0, min(h, wrist_y - by0)), h,
                 (p_hand.x if p_hand else (p_wrist.x if p_wrist else bx0),
                  p_hand.y if p_hand else (p_wrist.y if p_wrist else by0 + (2 * h) // 3))),
            ]
        else:  # leg
            prefix = "left_" if "left" in limb_name else "right_"
            p_hip = skeleton.get_anchor("hip")
            p_knee = skeleton.get_anchor(f"{prefix}knee")
            p_ankle = skeleton.get_anchor(f"{prefix}ankle")
            p_foot = skeleton.get_anchor(f"{prefix}foot")

            knee_y = p_knee.y if p_knee else by0 + h // 2
            ankle_y = p_ankle.y if p_ankle else by0 + (3 * h) // 4

            sub_defs = [
                (f"{prefix}thigh", 0, max(1, min(h, knee_y - by0)),
                 (p_hip.x if p_hip else bx0, p_hip.y if p_hip else by0)),
                (f"{prefix}lower_leg", max(0, min(h, knee_y - by0)), max(1, min(h, ankle_y - by0)),
                 (p_knee.x if p_knee else bx0, p_knee.y if p_knee else by0 + h // 2)),
                (f"{prefix}foot", max(0, min(h, ankle_y - by0)), h,
                 (p_foot.x if p_foot else bx0, p_foot.y if p_foot else by0 + (3 * h) // 4)),
            ]

        # Si el recorte es de altura muy reducida (< 4px), rotar como bloque único
        if h < 4:
            p_motion = frame_template.parts.get(limb_name, PartMotion()) if frame_template else PartMotion()
            tot_dx = int(round(p_motion.dx + root_dx))
            tot_dy = int(round(p_motion.dy + root_dy))
            pivot_rel = (limb_part.pivot_x - bx0, limb_part.pivot_y - by0)
            rot_img, (rot_off_x, rot_off_y) = pixel_rotate(limb_img, p_motion.angle_deg, pivot_rel)
            frame_canvas.paste(rot_img, (bx0 + rot_off_x + tot_dx, by0 + rot_off_y + tot_dy), mask=rot_img)
            return

        rendered_any = False
        for sub_name, y_start, y_end, (piv_x, piv_y) in sub_defs:
            if y_start >= y_end or y_start >= h:
                continue

            sub_arr = np.zeros_like(arr)
            sub_arr[y_start:y_end, :] = arr[y_start:y_end, :]
            # Comprobar si hay píxeles en esta subsección
            if not np.any(sub_arr[:, :, 3] > 10):
                continue

            sub_img = Image.fromarray(sub_arr, mode="RGBA")
            sub_motion = frame_template.parts.get(sub_name) if frame_template else None
            if not sub_motion:
                sub_motion = frame_template.parts.get(limb_name, PartMotion()) if frame_template else PartMotion()

            tot_dx = int(round(sub_motion.dx + root_dx))
            tot_dy = int(round(sub_motion.dy + root_dy))

            piv_rel = (piv_x - bx0, piv_y - by0)
            rot_img, (rot_off_x, rot_off_y) = pixel_rotate(sub_img, sub_motion.angle_deg, piv_rel)

            paste_x = bx0 + rot_off_x + tot_dx
            paste_y = by0 + rot_off_y + tot_dy
            frame_canvas.paste(rot_img, (paste_x, paste_y), mask=rot_img)
            rendered_any = True

        if not rendered_any:
            # Fallback en caso de que las subsecciones no capturaran píxeles
            p_motion = frame_template.parts.get(limb_name, PartMotion()) if frame_template else PartMotion()
            tot_dx = int(round(p_motion.dx + root_dx))
            tot_dy = int(round(p_motion.dy + root_dy))
            pivot_rel = (limb_part.pivot_x - bx0, limb_part.pivot_y - by0)
            rot_img, (rot_off_x, rot_off_y) = pixel_rotate(limb_img, p_motion.angle_deg, pivot_rel)
            frame_canvas.paste(rot_img, (bx0 + rot_off_x + tot_dx, by0 + rot_off_y + tot_dy), mask=rot_img)

    def generate_animation_frames(
        self,
        reference_image: Union[Path, str, Image.Image],
        template: ArticulatedMotionTemplate,
        output_dir: Optional[Path] = None,
        skeleton: Optional[Skeleton] = None,
    ) -> List[Image.Image]:
        """
        Genera los frames de animación transfiriendo la plantilla articulada
        al personaje de referencia con máxima preservación de identidad visual.
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

                if pname == "head":
                    # --- HEAD IDENTITY LOCK ---
                    # Preservación de identidad visual: la cabeza se traslada de forma entera
                    # sin deformación ni rotación distorsionadora, preservando los píxeles originales.
                    p_motion = frame_template.parts.get("head", PartMotion()) if frame_template else PartMotion()
                    tot_dx = int(round(p_motion.dx + root_dx))
                    tot_dy = int(round(p_motion.dy + root_dy))
                    paste_x = part.bbox[0] + tot_dx
                    paste_y = part.bbox[1] + tot_dy
                    frame_canvas.paste(p_img, (paste_x, paste_y), mask=p_img)

                elif "arm" in pname or "leg" in pname:
                    # Articulación anatómica por cadenas (upper/fore/hand, thigh/lower/foot)
                    self._render_limb_chain(
                        pname, part, p_img, skel, frame_template, root_dx, root_dy, frame_canvas
                    )

                else:
                    # Torso: rotación y traslación cinemática sobre pivote
                    p_motion = frame_template.parts.get(pname, PartMotion()) if frame_template else PartMotion()
                    tot_dx = int(round(p_motion.dx + root_dx))
                    tot_dy = int(round(p_motion.dy + root_dy))
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

        # 6. Métrica de validación de movimiento
        delta_score = compute_motion_delta_score(generated_frames)
        if delta_score < 0.05:
            logger.warning(
                f"Motion template '{template.animation_name}' produced insufficient visible motion. "
                f"motion_delta_score={delta_score:.3f}%"
            )
        else:
            logger.info(
                f"Animación '{template.animation_name}': motion_delta_score={delta_score:.2f}%"
            )

        logger.info(f"Generados {len(generated_frames)} frames de movimiento articulado para '{template.animation_name}'")
        return generated_frames
