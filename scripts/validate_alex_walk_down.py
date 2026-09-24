import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.anchor_tracker import AnchorTracker
from core.frame_extractor import OFFICIAL_ANIMATION_ROWS
from core.guard import SourceDatasetGuard
from core.motion_transfer_v2 import MotionTransferV2, compute_motion_delta_score
from core.palette_guard import PaletteGuard
from core.pose_analyzer_v2 import PoseAnalyzerV2
from core.sheet_detector_v2 import SheetDetectorV2
from core.template_extractor_v2 import TemplateExtractorV2
from models.skeleton import SKELETON_BONES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ValidateAlex")


def run_alex_walk_down_validation():
    base_dir = Path.cwd().resolve()
    approved_dir = base_dir / "dataset" / "finished_characters" / "approved" / "personajes al 100%" / "alex"
    alex_ref = approved_dir / "alex_rbchef.png"
    movimientos_sheet = approved_dir / "movimientos_rbchef.png"

    assert alex_ref.exists(), f"Falta referencia: {alex_ref}"
    assert movimientos_sheet.exists(), f"Falta spritesheet: {movimientos_sheet}"

    guard = SourceDatasetGuard(protected_root=approved_dir.parent)

    out_anim_dir = base_dir / "dataset" / "transferred_v2" / "alex" / "rbchef" / "walk_down"
    guard.assert_can_write(out_anim_dir, "creación de frames transferidos V2")
    out_anim_dir.mkdir(parents=True, exist_ok=True)

    # 1. Detectar celdas con SheetDetectorV2
    logger.info("Paso 1: Detección con SheetDetectorV2...")
    sheet_det = SheetDetectorV2(guard=guard)
    detection = sheet_det.detect_sheet(movimientos_sheet)
    assert detection.is_valid, "La detección del spritesheet falló"

    walk_down_row = OFFICIAL_ANIMATION_ROWS.index("walk_down")
    walk_down_cells = detection.cells[walk_down_row]
    assert len(walk_down_cells) == 4, f"Se esperaban 4 celdas, se obtuvieron {len(walk_down_cells)}"

    # 2. Recortar los 4 frames de la animación fuente
    logger.info("Paso 2: Recortando los 4 frames de walk_down de la fuente...")
    with Image.open(movimientos_sheet) as full_img:
        full_rgba = full_img.convert("RGBA")
        source_frames = []
        for c in walk_down_cells:
            box = (c.x, c.y, c.x + c.w, c.y + c.h)
            source_frames.append(full_rgba.crop(box))

    # 3. Rastrear cinemática y articulaciones con AnchorTracker
    logger.info("Paso 3: Extracción cinemática temporal con AnchorTracker...")
    pose_analyzer = PoseAnalyzerV2()
    anchor_tracker = AnchorTracker(pose_analyzer=pose_analyzer)
    skeletons = anchor_tracker.track_animation_anchors(source_frames, orientation="walk_down")
    assert len(skeletons) == 4, "No se obtuvieron los 4 esqueletos"

    # 4. Construir template de movimiento articulado V2
    logger.info("Paso 4: Construyendo plantilla cinemática walk_down V2...")
    tpl_dir = base_dir / "dataset" / "templates_v2"
    tpl_extractor = TemplateExtractorV2(base_templates_dir=tpl_dir, guard=guard)
    template = tpl_extractor.build_articulated_template("walk_down", [skeletons])
    tpl_path = tpl_extractor.save_template(template)
    logger.info(f"Plantilla guardada en: {tpl_path}")

    # 5. Aplicar template a alex_rbchef.png con MotionTransferV2
    logger.info("Paso 5: Transfiriendo movimiento articulado a alex_rbchef.png...")
    ref_image = Image.open(alex_ref).convert("RGBA")
    motion_transfer = MotionTransferV2(guard=guard)
    gen_frames = motion_transfer.generate_animation_frames(
        reference_image=ref_image,
        template=template,
        output_dir=out_anim_dir
    )

    assert len(gen_frames) == 4, f"Se esperaban 4 frames generados, se obtuvieron {len(gen_frames)}"

    # Verificar que los archivos existan en disco
    for idx in range(1, 5):
        fpath = out_anim_dir / f"{idx:02d}.png"
        assert fpath.exists(), f"Falta archivo {fpath}"
        logger.info(f"Frame generado confirmado: {fpath} ({fpath.stat().st_size} bytes)")

    # 6. Calcular métrica motion_delta_score
    delta_score = compute_motion_delta_score(gen_frames)
    logger.info(f"Métrica motion_delta_score: {delta_score:.2f}%")
    if delta_score < 0.05:
        logger.warning("Motion template produced insufficient visible motion.")
    else:
        logger.info(">> Verificación visual: Movimiento visible confirmado!")

    # 7. Generar imagen de depuración visual debug_walk_down.png
    logger.info("Paso 7: Generando debug_walk_down.png con comparación lado a lado...")
    fw, fh = ref_image.size
    total_panels = 5  # Ref + 4 frames
    debug_w = fw * total_panels + 20 * (total_panels - 1) + 40
    debug_h = fh + 60

    debug_img = Image.new("RGBA", (debug_w, debug_h), (20, 24, 34, 255))
    draw = ImageDraw.Draw(debug_img)

    panels = [("Original", ref_image)] + [(f"Frame {i}", gen_frames[i - 1]) for i in range(1, 5)]

    cur_x = 20
    cur_y = 40
    for label, p_img in panels:
        # Fondo de celda
        draw.rectangle([cur_x - 2, cur_y - 2, cur_x + fw + 1, cur_y + fh + 1], outline=(60, 75, 100, 255))
        debug_img.paste(p_img, (cur_x, cur_y), mask=p_img)
        # Etiqueta
        draw.text((cur_x + 4, 12), label, fill=(220, 230, 245, 255))
        cur_x += fw + 20

    debug_path = out_anim_dir / "debug_walk_down.png"
    debug_img.save(debug_path, "PNG")
    logger.info(f"Debug visual guardado exitosamente en: {debug_path}")

    print("\n=======================================================")
    print("PHASE 2.1 — E2E REAL CHARACTER VALIDATION (ALEX RBCHEF)")
    print(f"Frames generados en: {out_anim_dir}")
    print(f"Debug generado en:   {debug_path}")
    print(f"Motion delta score:  {delta_score:.2f}%")
    print("=======================================================\n")
    return delta_score


if __name__ == "__main__":
    run_alex_walk_down_validation()
