import json
from pathlib import Path
from PIL import Image
import numpy as np

from core.frame_extractor import FrameExtractor, OFFICIAL_ANIMATION_ROWS
from core.frame_normalizer import FrameNormalizer
from core.motion_analyzer import MotionAnalyzer
from core.template_extractor import TemplateExtractor
from core.template_library import TemplateLibrary
from core.template_matcher import TemplateMatcher
from core.motion_transfer import MotionTransferEngine
from core.spritesheet_builder import SpritesheetBuilder
from models.motion_template import MotionTemplate
from tests.conftest import create_dummy_png


def create_realistic_spritesheet(sheet_path: Path, cell_w: int = 64, cell_h: int = 48) -> Path:
    """Crea un spritesheet sintético de 4 columnas x 16 filas con canal alfa válido."""
    full_w = cell_w * 4
    full_h = cell_h * 16
    img = Image.new("RGBA", (full_w, full_h), (0, 0, 0, 0))

    pad_x = max(2, cell_w // 4)
    pad_y = max(2, cell_h // 4)
    box_w = max(4, cell_w // 2)
    box_h = max(4, cell_h // 2)

    # Dibujar un pequeño sprite centrado en cada celda
    for r in range(16):
        for c in range(4):
            x0 = c * cell_w + pad_x
            y0 = r * cell_h + pad_y
            for y in range(y0, y0 + box_h):
                for x in range(x0, x0 + box_w):
                    img.putpixel((x, y), (120, 180, 240, 255))

    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(sheet_path, "PNG")
    return sheet_path


def test_extract_64_frames(tmp_path: Path):
    """Verifica que FrameExtractor extraiga exactamente 64 frames organizados en 16 animaciones."""
    sheet_path = tmp_path / "movimientos_rnormal.png"
    create_realistic_spritesheet(sheet_path)

    extractor = FrameExtractor(output_base_dir=tmp_path / "extracted")
    anims = extractor.extract_from_sheet(sheet_path, "alex", "rnormal")

    total_frames = sum(len(a.frames) for a in anims.values())
    assert total_frames == 64, f"Se esperaban 64 frames, se extrajeron {total_frames}"


def test_16_animation_rows(tmp_path: Path):
    """Verifica que las 16 animaciones coincidan exactamente con la lista oficial."""
    sheet_path = tmp_path / "movimientos_rnormal.png"
    create_realistic_spritesheet(sheet_path)

    extractor = FrameExtractor(output_base_dir=tmp_path / "extracted")
    anims = extractor.extract_from_sheet(sheet_path, "alex", "rnormal")

    assert len(anims) == 16
    for row_name in OFFICIAL_ANIMATION_ROWS:
        assert row_name in anims, f"Falta la fila de animación oficial '{row_name}'"


def test_4_frames_per_animation(tmp_path: Path):
    """Verifica que cada fila de animación tenga exactamente 4 frames."""
    sheet_path = tmp_path / "movimientos_rbchef.png"
    create_realistic_spritesheet(sheet_path)

    extractor = FrameExtractor(output_base_dir=tmp_path / "extracted")
    anims = extractor.extract_from_sheet(sheet_path, "diego_vallenar", "rbchef")

    for anim_name, anim in anims.items():
        assert len(anim.frames) == 4, f"Animación '{anim_name}' tiene {len(anim.frames)} frames en vez de 4"
        for i, frame in enumerate(anim.frames):
            assert frame.image_path.exists()
            assert frame.index == i + 1


def test_alpha_preserved(tmp_path: Path):
    """Verifica que el canal alfa RGBA y la transparencia se preserven en todos los frames extraídos."""
    sheet_path = tmp_path / "movimientos_rnchef.png"
    create_realistic_spritesheet(sheet_path)

    extractor = FrameExtractor(output_base_dir=tmp_path / "extracted")
    anims = extractor.extract_from_sheet(sheet_path, "andrea", "rnchef")

    for anim in anims.values():
        for frame in anim.frames:
            with Image.open(frame.image_path) as img:
                assert img.mode == "RGBA"
                alpha = np.array(img.getchannel("A"))
                assert np.any(alpha == 0), "El frame no conserva píxeles transparentes"
                assert np.any(alpha > 0), "El frame está completamente vacío"


def test_normalization(tmp_path: Path):
    """Verifica que FrameNormalizer redimensione a 256x192 centrando el sprite pixel-perfect."""
    ref_path = tmp_path / "raw_frame.png"
    create_dummy_png(ref_path, width=40, height=70, color=(200, 100, 50, 255))

    normalizer = FrameNormalizer(canvas_width=256, canvas_height=192, baseline_offset_from_bottom=16)
    out_path = tmp_path / "norm_frame.png"
    norm_frame = normalizer.normalize_frame(ref_path, custom_output_path=out_path)

    assert out_path.exists()
    with Image.open(out_path) as img:
        assert img.size == (256, 192)
        assert img.mode == "RGBA"

    assert norm_frame.width == 256
    assert norm_frame.height == 192
    assert norm_frame.center_x == 128


def test_baseline(tmp_path: Path):
    """Verifica que la baseline de los pies se posicione consistentemente en target_baseline_y."""
    ref_path = tmp_path / "sprite.png"
    create_dummy_png(ref_path, width=32, height=48, color=(50, 150, 250, 255))

    normalizer = FrameNormalizer(canvas_width=256, canvas_height=192, baseline_offset_from_bottom=16)
    out_path = tmp_path / "baseline_test.png"
    norm_frame = normalizer.normalize_frame(ref_path, custom_output_path=out_path)

    # Con baseline_offset_from_bottom = 16, target_baseline_y = 192 - 16 = 176
    assert norm_frame.baseline_y == 176
    with Image.open(out_path) as img:
        alpha = np.array(img.getchannel("A"))
        coords = np.argwhere(alpha > 10)
        assert coords.max(axis=0)[0] == 176


def test_template_generation(tmp_path: Path):
    """Verifica que TemplateExtractor y TemplateLibrary generen y almacenen templates de movimiento con medianas."""
    motion_analyzer = MotionAnalyzer()
    template_extractor = TemplateExtractor()
    library = TemplateLibrary(templates_dir=tmp_path / "templates")

    # Crear 4 frames sintéticos para simular una animación
    frames = []
    for i in range(4):
        fp = tmp_path / f"walk_down_{i+1:02d}.png"
        create_dummy_png(fp, width=64, height=64)
        frames.append(fp)

    desc_alex = motion_analyzer.analyze_animation_motion(frames, "walk_down", "alex", "rnormal")
    desc_andrea = motion_analyzer.analyze_animation_motion(frames, "walk_down", "andrea", "rnormal")

    template = template_extractor.extract_template("walk_down", [desc_alex, desc_andrea])
    assert template.name == "walk_down"
    assert template.frame_count == 4
    assert len(template.frames) == 4

    saved_path = library.save_template(template)
    assert saved_path.exists()

    loaded = library.load_template("walk_down")
    assert loaded is not None
    assert loaded.name == "walk_down"


def test_template_matching(tmp_path: Path):
    """Verifica que TemplateMatcher compare la morfología de un personaje nuevo contra el dataset."""
    ref_alex = tmp_path / "alex_ref.png"
    create_dummy_png(ref_alex, width=48, height=80, color=(100, 100, 100, 255))

    ref_diego = tmp_path / "diego_ref.png"
    create_dummy_png(ref_diego, width=46, height=78, color=(110, 110, 110, 255))

    from models.character import Character
    from models.character_variant import CharacterVariant

    char_alex = Character(character_id="alex", display_name="Alex", is_approved=True)
    char_alex.variants["rnormal"] = CharacterVariant(variant="rnormal", reference_image=ref_alex)

    char_diego = Character(character_id="diego_vallenar", display_name="Diego Vallenar", is_approved=True)
    char_diego.variants["rnormal"] = CharacterVariant(variant="rnormal", reference_image=ref_diego)

    matcher = TemplateMatcher()
    target_new = tmp_path / "nuevo_char.png"
    create_dummy_png(target_new, width=47, height=79, color=(200, 50, 50, 255))

    matches = matcher.find_best_references(target_new, {"alex": char_alex, "diego_vallenar": char_diego})
    assert len(matches) == 2
    assert matches[0].similarity_score >= 0.80


def test_motion_transfer_preserves_identity(tmp_path: Path):
    """
    Verifica que MotionTransferEngine genere los 64 frames respetando la identidad
    visual del personaje original sin inventar píxeles extraños.
    """
    ref_path = tmp_path / "new_character_rbchef.png"
    create_dummy_png(ref_path, width=48, height=80, color=(255, 120, 80, 255))

    library = TemplateLibrary(templates_dir=tmp_path / "templates")
    templates = {}
    for anim in OFFICIAL_ANIMATION_ROWS:
        tmpl = library.load_template(anim) or MotionTemplate(name=anim, frame_count=4)
        templates[anim] = tmpl

    engine = MotionTransferEngine()
    out_dir = tmp_path / "transferred"
    generated_anims = engine.generate_full_64_frames(ref_path, templates, "nuevo_personaje", "rbchef", output_root=out_dir)

    assert len(generated_anims) == 16
    total_frames = sum(len(a.frames) for a in generated_anims.values())
    assert total_frames == 64

    # Verificar que el frame idle_down_01 existe y tiene formato 256x192
    f1 = generated_anims["idle_down"].frames[0]
    assert f1.image_path.exists()
    with Image.open(f1.image_path) as img:
        assert img.size == (256, 192)


def test_export_spritesheet(tmp_path: Path):
    """
    Verifica la exportación del spritesheet 4 columnas x 16 filas = 64 frames (1024x3072 px),
    la carpeta de frames individuales y los metadatos para Unity.
    """
    # Generar animaciones sintéticas
    from models.animation import Animation
    from models.frame import Frame

    anims = {}
    frames_pool = []
    for r, anim_name in enumerate(OFFICIAL_ANIMATION_ROWS):
        anim_frames = []
        for c in range(4):
            fp = tmp_path / f"export_frame_{anim_name}_{c+1:02d}.png"
            create_dummy_png(fp, width=256, height=192, color=(100 + r*5, 120 + c*10, 150, 255))
            f_obj = Frame(
                image_path=fp,
                index=c + 1,
                bbox=(50, 20, 200, 176),
                center_x=128,
                baseline_y=176,
                width=256,
                height=192,
                alpha_area=2000
            )
            anim_frames.append(f_obj)
        anims[anim_name] = Animation(name=anim_name, frames=anim_frames, row_index=r)

    builder = SpritesheetBuilder(columns=4, rows=16, cell_width=256, cell_height=192)
    export_out = tmp_path / "export_unity"
    sheet_p, meta_p, metrics = builder.build_spritesheet(anims, export_out)

    assert sheet_p.exists()
    assert meta_p.exists()

    # Verificar dimensiones del spritesheet final: 4*256 x 16*192 = 1024 x 3072
    with Image.open(sheet_p) as sheet_img:
        assert sheet_img.size == (1024, 3072)
        assert sheet_img.mode == "RGBA"

    # Verificar carpeta export/frames/ con 64 frames
    frames_dir = export_out / "frames"
    assert frames_dir.exists()
    assert len(list(frames_dir.glob("*.png"))) == 64

    # Verificar metadatos para Unity
    with open(meta_p, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    assert meta_data["columns"] == 4
    assert meta_data["rows"] == 16
    assert meta_data["cell_width"] == 256
    assert meta_data["cell_height"] == 192
    assert meta_data["total_frames"] == 64
    assert "idle_down" in meta_data["animations"]
    assert meta_data["animations"]["idle_down"]["row"] == 0
    assert meta_data["animations"]["idle_down"]["frames"] == 4
    assert "consistency_score" in meta_data
    assert metrics.overall_score >= 80.0


def test_acceptance_dataset_and_training_report(tmp_path: Path):
    """
    Criterio de Aceptación Oficial del Dataset:
    14 personajes x 3 variantes = 42 variantes
    42 spritesheets x 64 frames = 2688 frames extraídos
    Generación de las 16 plantillas aprendidas en animations/learned/
    """
    characters = [f"character_{i:02d}" for i in range(1, 15)]  # 14 personajes
    variants = ["rnormal", "rbchef", "rnchef"]                 # 3 variantes

    dataset_root = tmp_path / "dataset"
    approved_dir = dataset_root / "finished_characters" / "approved"
    approved_dir.mkdir(parents=True, exist_ok=True)

    for char in characters:
        char_dir = approved_dir / char
        char_dir.mkdir()
        for v in variants:
            # Referencia
            ref_path = char_dir / f"{char}_{v}.png"
            create_dummy_png(ref_path, width=48, height=64)
            # Spritesheet (64 frames)
            sheet_path = char_dir / f"movimientos_{v}.png"
            create_realistic_spritesheet(sheet_path, cell_w=32, cell_h=32)

    from services.dataset_service import DatasetService
    from services.training_service import TrainingService

    ds_service = DatasetService(base_dir=tmp_path)
    chars = ds_service.scan_approved()

    assert len(chars) == 14
    total_variants = sum(len(c.variants) for c in chars.values())
    assert total_variants == 42
    total_sheets = sum(c.spritesheet_count for c in chars.values())
    assert total_sheets == 42

    # Ejecutar pipeline síncrono de entrenamiento
    training_service = TrainingService(ds_service, base_dir=tmp_path)
    report = training_service.run_synchronous_training()

    assert report.characters_count == 14
    assert report.variants_count == 42
    assert report.spritesheets_count == 42
    assert report.frames_extracted == 2688  # 42 x 64 = 2688 frames
    assert report.animations_count == 16
    assert report.templates_generated == 16
    assert report.errors == 0

