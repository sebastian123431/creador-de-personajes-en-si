import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from core.frame_extractor import OFFICIAL_ANIMATION_ROWS
from core.guard import SourceDatasetGuard, SourceDatasetWriteError
from core.motion_transfer_v2 import MotionTransferV2, compute_motion_delta_score
from core.palette_guard import PaletteGuard
from core.sheet_detector_v2 import SheetDetectorV2
from core.template_extractor_v2 import TemplateExtractorV2
from core.body_part_segmenter import BodyPartSegmenter
from models.articulated_motion_template import ArticulatedMotionTemplate
from models.skeleton import Skeleton, OFFICIAL_ANCHOR_NAMES
from services.animation_service import AnimationService
from services.dataset_service import DatasetService
from services.export_service import ExportService
from services.training_service import ArticulatedTrainingWorker
from tests.test_etapa_c_v2 import create_character_sprite_with_limbs
from tests.test_etapa_d_v2 import _create_synthetic_walk_cycle


def _hash_directory(directory: Path) -> dict:
    hashes = {}
    if not directory.exists():
        return hashes
    for file in sorted(directory.rglob("*")):
        if file.is_file():
            rel = file.relative_to(directory).as_posix()
            h = hashlib.sha256(file.read_bytes()).hexdigest()
            hashes[rel] = h
    return hashes


# 1. test_animation_service_v2_no_nameerror
def test_animation_service_v2_no_nameerror(tmp_path):
    """
    Verifica que generate_articulated_animations_v2 no produzca
    NameError: name 'Frame' is not defined y devuelva objetos Frame válidos.
    """
    sprite_path = tmp_path / "test_char.png"
    create_character_sprite_with_limbs(sprite_path)

    svc = AnimationService(base_dir=tmp_path)
    anims = svc.generate_articulated_animations_v2(sprite_path, "test_hero", "default")

    assert isinstance(anims, dict)
    assert len(anims) == 16
    assert "walk_down" in anims
    assert len(anims["walk_down"].frames) == 4
    for f in anims["walk_down"].frames:
        assert f.image_path.exists()
        assert f.width == 64
        assert f.height == 96


# 2. test_v2_uses_official_anchor_names
def test_v2_uses_official_anchor_names(tmp_path):
    """
    Verifica que TemplateExtractorV2, BodyPartSegmenter y MotionTransferV2
    trabajan exclusivamente con los 18 anchors de OFFICIAL_ANCHOR_NAMES.
    """
    forbidden_names = {"pelvis", "spine", "left_hip", "right_hip", "left_toe", "right_toe"}

    # BodyPartSegmenter
    segmenter = BodyPartSegmenter(base_output_dir=tmp_path / "bp")
    dummy_path = tmp_path / "char.png"
    create_character_sprite_with_limbs(dummy_path)
    skel = Skeleton()
    # Inicializar con anchors oficiales
    for name in OFFICIAL_ANCHOR_NAMES:
        skel.set_anchor(name, 24, 40)

    # Asegurar que _ensure_anchors devuelve solo anchors oficiales
    mask = np.ones((96, 64), dtype=bool)
    anchors_dict = segmenter._ensure_anchors(skel, 64, 96, mask)

    for forbidden in forbidden_names:
        assert forbidden not in anchors_dict, f"Nombre no oficial '{forbidden}' encontrado en BodyPartSegmenter!"
    for official in OFFICIAL_ANCHOR_NAMES:
        assert official in anchors_dict, f"Anchor oficial '{official}' ausente en BodyPartSegmenter!"

    # TemplateExtractorV2
    extractor = TemplateExtractorV2(base_templates_dir=tmp_path / "tpl")
    cycle = _create_synthetic_walk_cycle()
    seq_data = extractor.extract_motion_from_sequence(cycle)
    assert len(seq_data) == 4
    rel_anchors = seq_data[0]["anchors_rel"]
    for forbidden in forbidden_names:
        assert forbidden not in rel_anchors, f"Nombre no oficial '{forbidden}' en TemplateExtractorV2!"
    for official in OFFICIAL_ANCHOR_NAMES:
        assert official in rel_anchors, f"Anchor oficial '{official}' ausente en TemplateExtractorV2!"


# 3. test_sheet_detector_v2_integrated_with_training
def test_sheet_detector_v2_integrated_with_training(tmp_path):
    """
    Verifica que ArticulatedTrainingWorker utilice realmente SheetDetectorV2
    para la extracción de spritesheets.
    """
    ds_svc = DatasetService(base_dir=tmp_path)
    worker = ArticulatedTrainingWorker(dataset_service=ds_svc, base_dir=tmp_path)

    assert isinstance(worker.sheet_detector, SheetDetectorV2)
    assert isinstance(worker.frame_extractor.detector, SheetDetectorV2)
    assert worker.frame_extractor.detector == worker.sheet_detector


# 4. test_articulated_training_creates_templates
def test_articulated_training_creates_templates(tmp_path):
    """
    Verifica que el pipeline de entrenamiento cinemático V2 extraiga y guarde
    plantillas ArticulatedMotionTemplate con datos cinemáticos válidos.
    """
    templates_dir = tmp_path / "dataset" / "templates_v2"
    extractor = TemplateExtractorV2(base_templates_dir=templates_dir)
    cycle = _create_synthetic_walk_cycle()

    template = extractor.build_articulated_template("walk_down", [cycle])
    saved_path = extractor.save_template(template)

    assert saved_path.exists()
    assert saved_path.name == "walk_down.json"

    loaded = extractor.load_template("walk_down")
    assert loaded is not None
    assert loaded.animation_name == "walk_down"
    assert loaded.frame_count == 4
    assert len(loaded.frames) == 4


# 5. test_generate_v2_creates_16_animations
def test_generate_v2_creates_16_animations(tmp_path):
    """
    Verifica que generate_articulated_animations_v2 genere las 16 animaciones oficiales.
    """
    sprite_path = tmp_path / "hero.png"
    create_character_sprite_with_limbs(sprite_path)

    svc = AnimationService(base_dir=tmp_path)
    anims = svc.generate_articulated_animations_v2(sprite_path, "alex", "rbchef")

    assert len(anims) == 16
    for expected_row in OFFICIAL_ANIMATION_ROWS:
        assert expected_row in anims, f"Fila oficial '{expected_row}' no encontrada en resultado V2"


# 6. test_each_animation_has_4_frames
def test_each_animation_has_4_frames(tmp_path):
    """
    Verifica que cada una de las 16 animaciones V2 contenga exactamente 4 frames.
    """
    sprite_path = tmp_path / "hero.png"
    create_character_sprite_with_limbs(sprite_path)

    svc = AnimationService(base_dir=tmp_path)
    anims = svc.generate_articulated_animations_v2(sprite_path, "alex", "rbchef")

    for anim_name, anim in anims.items():
        assert len(anim.frames) == 4, f"La animación '{anim_name}' tiene {len(anim.frames)} frames en vez de 4."
        for idx, f in enumerate(anim.frames, start=1):
            assert f.index == idx


# 7. test_generated_frames_exist_on_disk
def test_generated_frames_exist_on_disk(tmp_path):
    """
    Verifica que los 64 frames generados se escriban en disco en:
    dataset/transferred_v2/<char>/<variant>/<anim>/<idx>.png
    """
    sprite_path = tmp_path / "hero.png"
    create_character_sprite_with_limbs(sprite_path)

    svc = AnimationService(base_dir=tmp_path)
    svc.generate_articulated_animations_v2(sprite_path, "alex", "rbchef")

    base_out = tmp_path / "dataset" / "transferred_v2" / "alex" / "rbchef"
    assert base_out.exists()

    total_pngs = list(base_out.rglob("*.png"))
    assert len(total_pngs) == 64, f"Se esperaban 64 archivos PNG en disco, se encontraron {len(total_pngs)}"


# 8. test_palette_guard_no_unexpected_colors
def test_palette_guard_no_unexpected_colors():
    """
    Verifica que PaletteGuard identifique colores ajenos y los proyecte
    hacia los colores válidos de la referencia sin introducir aberraciones.
    """
    ref_img = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    ref_img.putpixel((1, 1), (30, 40, 50, 255))
    ref_img.putpixel((2, 2), (200, 150, 100, 255))

    palette = PaletteGuard.extract_palette(ref_img)
    assert len(palette) == 2

    # Lienzo con color ajeno
    dirty = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    dirty.putpixel((1, 1), (30, 40, 50, 255))
    dirty.putpixel((3, 3), (255, 0, 255, 255))  # Alien magenta

    cleaned = PaletteGuard.clean_frame(dirty, palette)
    arr = np.array(cleaned)
    colors = set(tuple(c) for c in arr.reshape(-1, 4))
    colors.discard((0, 0, 0, 0))

    assert (255, 0, 255, 255) not in colors
    for col in colors:
        assert col in palette


# 9. test_source_dataset_not_modified
def test_source_dataset_not_modified():
    """
    Verifica que las operaciones V2 sobre un personaje real no modifiquen,
    renombren ni agreguen archivos dentro de la carpeta protegida approved.
    """
    master_alex_dir = Path("dataset/finished_characters/approved/personajes al 100%/alex").resolve()
    if not master_alex_dir.exists():
        pytest.skip("Dataset maestro de alex no encontrado en el entorno.")

    initial_hashes = _hash_directory(master_alex_dir)
    assert len(initial_hashes) > 0, "La carpeta de Alex no contiene archivos maestros."

    # Intentar operación que intente violar SourceDatasetGuard
    guard = SourceDatasetGuard(protected_root=master_alex_dir.parent)
    with pytest.raises(SourceDatasetWriteError):
        guard.assert_can_write(master_alex_dir / "illegal_file.txt", "prueba de intrusión")

    final_hashes = _hash_directory(master_alex_dir)
    assert initial_hashes == final_hashes, "ALERTA: Los archivos maestros fueron alterados!"


# 10. test_unity_export_v2_creates_sheet_and_metadata
def test_unity_export_v2_creates_sheet_and_metadata(tmp_path):
    """
    Verifica que la exportación de las 16 animaciones V2 cree un spritesheet consolidado
    y metadata.json con formato para motores de juego.
    """
    sprite_path = tmp_path / "hero.png"
    create_character_sprite_with_limbs(sprite_path)

    svc = AnimationService(base_dir=tmp_path)
    anims = svc.generate_articulated_animations_v2(sprite_path, "alex", "rbchef")

    export_svc = ExportService(output_dir=tmp_path / "exports")
    sheet_path, meta_path, metrics = export_svc.export_character_variant("alex", "rbchef", anims)

    assert sheet_path.exists()
    assert meta_path.exists()

    # Verificar JSON de metadata
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["total_frames"] == 64
    assert meta["columns"] == 4
    assert meta["rows"] == 16
    assert len(meta["animations"]) == 16


# Test de integridad de cabeza (Item 7)
def test_head_identity_integrity(tmp_path):
    """
    Comprueba que el recorte original de cabeza no sufra alteración cromática
    ni distorsión de rotación al transferir movimiento neutral.
    """
    sprite_path = tmp_path / "hero_head_test.png"
    create_character_sprite_with_limbs(sprite_path)
    img = Image.open(sprite_path).convert("RGBA")

    segmenter = BodyPartSegmenter()
    skel = Skeleton()
    for name in OFFICIAL_ANCHOR_NAMES:
        skel.set_anchor(name, 32, 48)
    skel.set_anchor("neck", 32, 36)
    skel.set_anchor("head", 32, 26)

    parts, part_images = segmenter.segment_frame(img, skeleton=skel, orientation="down")
    head_img = part_images["head"]

    transfer = MotionTransferV2(segmenter=segmenter)
    tpl = ArticulatedMotionTemplate(animation_name="idle_down", frame_count=1)

    frames = transfer.generate_animation_frames(img, tpl, skeleton=skel)
    assert len(frames) == 1
    gen_frame = frames[0]

    # El crop de la cabeza en el frame generado debe contener los píxeles originales
    hb = parts["head"].bbox
    gen_head_crop = gen_frame.crop((hb[0], hb[1], hb[2] + 1, hb[3] + 1))

    orig_arr = np.array(head_img)
    gen_arr = np.array(gen_head_crop)

    # Verificar que los píxeles visibles de la cabeza se preserven idénticos
    vis_mask = orig_arr[:, :, 3] > 0
    assert np.all(orig_arr[vis_mask] == gen_arr[vis_mask])


# ==============================================================================
# PRUEBAS FASE 1 - ENTRENAMIENTO ARTICULADO V2
# ==============================================================================

def test_articulated_template_contains_subparts(tmp_path):
    """
    CRÍTICO: Verifica que build_articulated_template preserve todas las 18 partes
    y subpartes anatómicas en el JSON final y no colapse solo a partes mayores:
    - left_upper_arm, left_forearm, left_hand
    - right_upper_arm, right_forearm, right_hand
    - left_thigh, left_lower_leg, left_foot
    - right_thigh, right_lower_leg, right_foot
    - head, torso, left_arm, right_arm, left_leg, right_leg
    """
    extractor = TemplateExtractorV2(base_templates_dir=tmp_path / "tpl")
    cycle1 = _create_synthetic_walk_cycle()
    cycle2 = _create_synthetic_walk_cycle()

    template = extractor.build_articulated_template("walk_down", [cycle1, cycle2])
    assert template.frame_count == 4

    required_subparts = [
        "left_upper_arm", "left_forearm", "left_hand",
        "right_upper_arm", "right_forearm", "right_hand",
        "left_thigh", "left_lower_leg", "left_foot",
        "right_thigh", "right_lower_leg", "right_foot",
        "head", "torso", "left_arm", "right_arm", "left_leg", "right_leg",
    ]

    for frame in template.frames:
        for part_name in required_subparts:
            assert part_name in frame.parts, f"Subparte faltante en frame {frame.frame_index}: {part_name}"
            part = frame.parts[part_name]
            assert hasattr(part, "dx")
            assert hasattr(part, "dy")
            assert hasattr(part, "dx_ratio")
            assert hasattr(part, "dy_ratio")
            assert hasattr(part, "angle_deg")

    # Guardar a disco y verificar estructura JSON
    saved_path = extractor.save_template(template)
    with open(saved_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frame1_parts = data["frames"][0]["parts"]
    for part_name in required_subparts:
        assert part_name in frame1_parts, f"Subparte {part_name} no encontrada en JSON serializado!"
        assert "dx_ratio" in frame1_parts[part_name]
        assert "dy_ratio" in frame1_parts[part_name]


def test_ratios_are_saved(tmp_path):
    """
    Verifica que las razones relativas de desplazamiento (dx_ratio, dy_ratio,
    root_dx_ratio, root_dy_ratio) se calculen, serialicen y deserialicen correctamente.
    """
    extractor = TemplateExtractorV2(base_templates_dir=tmp_path / "tpl")
    cycle = _create_synthetic_walk_cycle()

    template = extractor.build_articulated_template("walk_down", [cycle])
    f1 = template.get_frame(1)
    assert f1 is not None
    assert hasattr(f1, "root_dx_ratio")
    assert hasattr(f1, "root_dy_ratio")

    saved_path = extractor.save_template(template)
    loaded_template = extractor.load_template("walk_down")
    assert loaded_template is not None
    lf1 = loaded_template.get_frame(1)
    assert lf1 is not None
    assert isinstance(lf1.root_dx_ratio, float)
    assert isinstance(lf1.parts["left_upper_arm"].dx_ratio, float)


def test_orientation_specific_pose():
    """
    Verifica que PoseAnalyzerV2 maneje adecuadamente las siluetas por orientación:
    - left: extremidades derechas (ocluidas) deben tener baja confianza (~0.35).
    - right: extremidades izquierdas (ocluidas) deben tener baja confianza (~0.35).
    - down / up: extremidades simétricas con confianza alta (> 0.75).
    """
    from core.pose_analyzer_v2 import PoseAnalyzerV2
    analyzer = PoseAnalyzerV2()

    # Imagen de prueba sintética 64x96
    img = Image.new("RGBA", (64, 96), (0, 0, 0, 0))
    # Dibujar silueta simple en el centro
    arr = np.zeros((96, 64, 4), dtype=np.uint8)
    arr[16:80, 20:44] = [200, 150, 100, 255]
    img = Image.fromarray(arr, mode="RGBA")

    skel_left = analyzer.analyze_pose(img, orientation="left")
    assert skel_left.get_anchor("right_shoulder").confidence <= 0.40
    assert skel_left.get_anchor("right_elbow").confidence <= 0.40
    assert skel_left.get_anchor("left_shoulder").confidence >= 0.75

    skel_right = analyzer.analyze_pose(img, orientation="right")
    assert skel_right.get_anchor("left_shoulder").confidence <= 0.40
    assert skel_right.get_anchor("left_elbow").confidence <= 0.40
    assert skel_right.get_anchor("right_shoulder").confidence >= 0.75

    skel_down = analyzer.analyze_pose(img, orientation="down")
    assert skel_down.get_anchor("left_shoulder").confidence >= 0.75
    assert skel_down.get_anchor("right_shoulder").confidence >= 0.75


def test_low_confidence_excluded(tmp_path):
    """
    Verifica que secuencias con confianza < 0.60 sean marcadas como REVIEW_REQUIRED
    y no contaminen la agregación cuando existen secuencias limpias.
    """
    extractor = TemplateExtractorV2(base_templates_dir=tmp_path / "tpl")

    # Ciclo limpio (alta confianza 0.95)
    clean_cycle = _create_synthetic_walk_cycle()
    for skel in clean_cycle:
        for a in skel.anchors.values():
            a.confidence = 0.95

    # Ciclo ruidoso / de baja confianza (< 0.60) con desplazamientos absurdos (outlier contaminante)
    noisy_cycle = _create_synthetic_walk_cycle()
    for skel in noisy_cycle:
        for a in skel.anchors.values():
            a.confidence = 0.45  # < 0.60 REVIEW_REQUIRED
        # Desplazamiento distorsionado extremo
        skel.set_anchor("left_hand", 100, 100, confidence=0.45)

    template = extractor.build_articulated_template("walk_down", [clean_cycle, noisy_cycle])
    # La secuencia ruidosa debió ser excluida de la agregación primaria
    assert template.samples_used == 1
    # La mano izquierda no debió contaminarse con las coordenadas de la secuencia de baja confianza
    f1 = template.get_frame(1)
    assert abs(f1.parts["left_hand"].dx) < 20.0

