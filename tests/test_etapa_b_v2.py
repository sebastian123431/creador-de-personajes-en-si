import json
import os
import sys
import pytest
from pathlib import Path
from PIL import Image
from PySide6.QtWidgets import QApplication

from core.annotation_manager import AnnotationManager
from core.anchor_tracker import AnchorTracker
from core.guard import SourceDatasetGuard, SourceDatasetWriteError
from core.pose_analyzer_v2 import PoseAnalyzerV2
from models.frame import Frame
from models.skeleton import Skeleton, Anchor, OFFICIAL_ANCHOR_NAMES
from ui.anchor_editor import InteractiveAnchorCanvas, AnchorEditorDialog
from tests.conftest import create_dummy_png


@pytest.fixture(scope="session")
def qapp():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_annotation_manager_paths_and_io(tmp_path: Path):
    """
    Verifica que AnnotationManager guarde y cargue anotaciones manuales en:
    dataset/annotations/<character_id>/<variant>/<animation>/<frame_index:02d>.json
    manteniendo la estructura JSON y las propiedades de los anchors.
    """
    annotations_dir = tmp_path / "annotations"
    manager = AnnotationManager(base_annotations_dir=annotations_dir)

    skel = Skeleton()
    skel.set_anchor("head", 32, 16, confidence=1.0, is_manual=True)
    skel.set_anchor("hip", 32, 48, confidence=0.85, is_manual=False)

    # Comprobar que inicialmente no existe
    assert not manager.has_annotation("diego_vallenar", "rnormal", "walk_down", 1)
    assert manager.load_annotation("diego_vallenar", "rnormal", "walk_down", 1) is None

    # Guardar anotación
    saved_path = manager.save_annotation("diego_vallenar", "rnormal", "walk_down", 1, skel)
    expected_path = annotations_dir / "diego_vallenar" / "rnormal" / "walk_down" / "01.json"
    assert saved_path == expected_path
    assert expected_path.exists()

    # Verificar contenido del archivo JSON
    with open(expected_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["character_id"] == "diego_vallenar"
    assert data["variant"] == "rnormal"
    assert data["animation"] == "walk_down"
    assert data["frame"] == 1
    assert "anchors" in data
    assert data["anchors"]["head"]["x"] == 32
    assert data["anchors"]["head"]["y"] == 16
    assert data["anchors"]["head"]["is_manual"] is True
    assert data["anchors"]["head"]["confidence"] == 1.0

    # Cargar y verificar reconstitución del esqueleto
    loaded_skel = manager.load_annotation("diego_vallenar", "rnormal", "walk_down", 1)
    assert loaded_skel is not None
    loaded_head = loaded_skel.get_anchor("head")
    assert loaded_head is not None
    assert loaded_head.x == 32
    assert loaded_head.y == 16
    assert loaded_head.is_manual is True
    assert loaded_head.confidence == 1.0


def test_annotation_manager_guard_protection(tmp_path: Path):
    """
    Verifica que SourceDatasetGuard impida estrictamente escribir anotaciones
    dentro del directorio protegido de personajes aprobados.
    """
    approved_dir = tmp_path / "finished_characters" / "approved" / "personajes al 100"
    approved_dir.mkdir(parents=True)
    guard = SourceDatasetGuard(protected_root=approved_dir)

    # Intentar configurar el directorio de anotaciones dentro del área protegida
    illegal_dir = approved_dir / "alex" / "annotations"
    manager = AnnotationManager(base_annotations_dir=illegal_dir, guard=guard)

    skel = Skeleton()
    skel.set_anchor("head", 30, 20, 1.0, is_manual=True)

    with pytest.raises(SourceDatasetWriteError):
        manager.save_annotation("alex", "rnormal", "walk_down", 1, skel)


def test_temporal_anchor_tracking_sequence(tmp_path: Path):
    """
    Verifica el rastreo coordinado de los 4 frames de una animación.
    Comprueba que el frame 1 sirva de base y los subsiguientes mantengan continuidad.
    """
    frames_paths = []
    for i in range(1, 5):
        fp = tmp_path / f"frame_{i:02d}.png"
        img = Image.new("RGBA", (64, 96), (0, 0, 0, 0))
        # Desplazar ligeramente la figura simulando caminata
        offset = (i - 1) * 2
        for y in range(20, 80):
            for x in range(20 + offset, 44 + offset):
                img.putpixel((x, y), (120, 160, 200, 255))
        img.save(fp, "PNG")
        frames_paths.append(fp)

    tracker = AnchorTracker(max_drift_px=14)

    # Simular una anotación manual en el frame 2 para la mano derecha
    manual_by_frame = {
        2: {"right_hand": {"x": 48, "y": 55}}
    }

    skeletons = tracker.track_animation_anchors(
        frames_paths,
        orientation="down",
        manual_annotations_by_frame=manual_by_frame
    )

    assert len(skeletons) == 4
    for skel in skeletons:
        assert skel.anchor_count == 18

    # Frame 2 debe tener right_hand manual
    f2_r_hand = skeletons[1].get_anchor("right_hand")
    assert f2_r_hand is not None
    assert f2_r_hand.x == 48
    assert f2_r_hand.y == 55
    assert f2_r_hand.is_manual is True
    assert f2_r_hand.confidence == 1.0


def test_anchor_tracker_drift_clamping():
    """
    Verifica que AnchorTracker aplique el clamp y penalice confianza
    cuando un anchor no manual experimente un salto brusco mayor a max_drift_px.
    """
    class MockPoseAnalyzer:
        def __init__(self):
            self.call_count = 0

        def analyze_pose(self, *args, **kwargs):
            self.call_count += 1
            skel = Skeleton()
            for name in OFFICIAL_ANCHOR_NAMES:
                skel.set_anchor(name, 50, 50, confidence=0.9, is_manual=False)

            if self.call_count == 2:
                # Provocar un salto extremo en 'left_foot' (distancia 50px > max_drift_px 10)
                skel.set_anchor("left_foot", 100, 50, confidence=0.9, is_manual=False)
            return skel

    mock_analyzer = MockPoseAnalyzer()
    tracker = AnchorTracker(pose_analyzer=mock_analyzer, max_drift_px=10)

    # Ejecutar con 2 frames dummy
    skeletons = tracker.track_animation_anchors(["dummy1.png", "dummy2.png"])
    assert len(skeletons) == 2

    f1_foot = skeletons[0].get_anchor("left_foot")
    f2_foot = skeletons[1].get_anchor("left_foot")

    assert f1_foot.x == 50
    # En f2 debe haberse clampeado a distancia max_drift_px (10px) desde f1 (50 + 10 = 60)
    assert f2_foot.x == 60
    assert f2_foot.confidence < 0.90  # Confianza penalizada por deriva


def test_interactive_anchor_canvas_coordinates(qapp, tmp_path: Path):
    """
    Verifica las transformaciones de coordenadas y la selección interactiva de anchors
    en InteractiveAnchorCanvas.
    """
    img_path = tmp_path / "canvas_test.png"
    create_dummy_png(img_path, width=48, height=80)

    canvas = InteractiveAnchorCanvas()
    skel = Skeleton()
    skel.set_anchor("head", 24, 12, 0.9)
    skel.set_anchor("hip", 24, 45, 0.9)

    canvas.set_data(img_path, skel, scale=4.0)

    # Test coordinate mapping
    sx, sy = canvas._screen_to_sprite_coords(96, 48)
    assert sx == 24
    assert sy == 12

    px, py = canvas._sprite_to_screen_coords(24, 12)
    assert px == 24 * 4.0 + 2.0
    assert py == 12 * 4.0 + 2.0

    # Simular movimiento manual de anchor
    moved_signals = []
    canvas.anchor_moved.connect(lambda name, x, y: moved_signals.append((name, x, y)))

    canvas.selected_anchor_name = "head"
    canvas.dragging = True
    # Mover a sprite (26, 14) -> screen (104, 56)
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtCore import QPointF, Qt
    event = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(106.0, 58.0),
        QPointF(106.0, 58.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier
    )
    canvas.mouseMoveEvent(event)

    assert len(moved_signals) == 1
    name, nx, ny = moved_signals[0]
    assert name == "head"
    assert nx == 26
    assert ny == 14

    head = canvas.skeleton.get_anchor("head")
    assert head.is_manual is True
    assert head.x == 26
    assert head.y == 14


def test_anchor_editor_dialog_init_and_save(qapp, tmp_path: Path, monkeypatch):
    """
    Verifica la apertura del diálogo AnchorEditorDialog y el flujo de guardado.
    """
    img_path = tmp_path / "char_edit.png"
    create_dummy_png(img_path, width=48, height=80)

    annotations_dir = tmp_path / "annotations"
    manager = AnnotationManager(base_annotations_dir=annotations_dir)

    skel = Skeleton()
    skel.set_anchor("head", 24, 10, confidence=0.8, is_manual=False)

    dialog = AnchorEditorDialog(
        image_path=img_path,
        skeleton=skel,
        character_id="amaro",
        variant="rbchef",
        animation="walk_up",
        frame_index=1,
        annotation_manager=manager
    )

    assert dialog is not None

    # Mockear el popup de información en modo prueba
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)

    # Simular guardado desde el botón de la interfaz
    dialog._on_save_clicked()

    # Verificar que se haya persistido en el filesystem
    assert manager.has_annotation("amaro", "rbchef", "walk_up", 1)
    saved_skel = manager.load_annotation("amaro", "rbchef", "walk_up", 1)
    assert saved_skel is not None
    assert saved_skel.get_anchor("head") is not None
    dialog.close()

