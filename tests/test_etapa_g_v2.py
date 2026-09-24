import json
import os
import sys
import pytest
from pathlib import Path
from PySide6.QtWidgets import QApplication

from models.animation import Animation
from models.frame import Frame
from services.dataset_service import DatasetService
from services.export_service import ExportService
from services.training_service import TrainingService, ArticulatedTrainingReport
from ui.compare_v1_v2_dialog import CompareV1V2Dialog
from ui.main_window import MainWindow
from tests.conftest import create_dummy_png


@pytest.fixture(scope="session")
def qapp():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_main_window_v2_actions_present(qapp, sample_dataset_dir: Path, tmp_path: Path):
    """
    Verifica que MainWindow inicialice correctamente los controles V2:
    TRAIN ARTICULATED (V2), GENERATE V2, COMPARAR V1 vs V2, EXPORT UNITY V2.
    """
    service = DatasetService(base_dir=tmp_path)
    service.import_external_directory_to_approved(sample_dataset_dir)

    window = MainWindow(dataset_service=service)
    assert window is not None

    # Verificar que los botones V2 existan y estén configurados
    assert window.train_v2_btn is not None
    assert "TRAIN ARTICULATED" in window.train_v2_btn.text()

    assert window.gen_v2_btn is not None
    assert "GENERATE V2" in window.gen_v2_btn.text()

    assert window.compare_v1_v2_btn is not None
    assert "COMPARAR V1 vs V2" in window.compare_v1_v2_btn.text()

    assert window.export_v2_btn is not None
    assert "EXPORT UNITY V2" in window.export_v2_btn.text()

    window.close()


def test_compare_v1_v2_dialog_init(qapp, tmp_path: Path):
    """
    Verifica que el diálogo interactivo de comparación V1 vs V2 se instancie,
    cargue animaciones y ejecute el timer de animación.
    """
    frame_p = tmp_path / "dummy_frame.png"
    create_dummy_png(frame_p, width=48, height=80)

    f1 = Frame(image_path=frame_p, index=1, bbox=(0, 0, 48, 80), center_x=24, baseline_y=79, width=48, height=80)
    f2 = Frame(image_path=frame_p, index=2, bbox=(0, 0, 48, 80), center_x=24, baseline_y=79, width=48, height=80)

    v1_anims = {"walk_down": Animation("walk_down", [f1, f2])}
    v2_anims = {"walk_down": Animation("walk_down", [f1, f2])}

    dlg = CompareV1V2Dialog(
        character_id="alex",
        variant="rnormal",
        v1_animations=v1_anims,
        v2_animations=v2_anims
    )

    assert dlg is not None
    assert dlg.canvas_v1.pixmap is not None
    assert dlg.canvas_v2.pixmap is not None
    assert dlg.timer.isActive()

    # Probar cambio de velocidad
    dlg._on_speed_changed(10)
    assert dlg.timer.interval() == 100

    dlg.close()


def test_export_service_unity_v2(tmp_path: Path):
    """
    Verifica que export_unity_package_v2 genere el spritesheet empaquetado
    y la metadata JSON enriquecida con información del rig V2 de 18 articulaciones.
    """
    frame_p = tmp_path / "frame.png"
    create_dummy_png(frame_p, width=32, height=48)

    frames = [
        Frame(image_path=frame_p, index=i, bbox=(0, 0, 32, 48), center_x=16, baseline_y=47, width=32, height=48)
        for i in range(1, 5)
    ]
    animations = {"walk_down": Animation("walk_down", frames)}

    export_service = ExportService(output_dir=tmp_path / "export_v2")
    sheet_p, meta_p, metrics = export_service.export_unity_package_v2(
        character_id="diego_vallenar",
        variant_name="rbchef",
        animations=animations
    )

    assert sheet_p.exists()
    assert meta_p.exists()
    assert "spritesheet_v2.png" in sheet_p.name
    assert "metadata_v2.json" in meta_p.name

    # Verificar metadata enriquecida
    with open(meta_p, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    assert meta_data["engine_version"] == "2.0-articulated"
    assert "rig" in meta_data
    assert meta_data["rig"]["type"] == "18_anchor_articulated"
    assert meta_data["rig"]["head_identity_locked"] is True
    assert "unity_importer_hints" in meta_data
    assert meta_data["unity_importer_hints"]["pixelsPerUnit"] == 16


def test_training_service_articulated_worker(sample_dataset_dir: Path, tmp_path: Path):
    """
    Verifica que run_synchronous_articulated_training escanee los personajes
    aprobados y devuelva un ArticulatedTrainingReport.
    """
    service = DatasetService(base_dir=tmp_path)
    service.import_external_directory_to_approved(sample_dataset_dir)

    training_service = TrainingService(dataset_service=service, base_dir=tmp_path)
    rep = training_service.run_synchronous_articulated_training()

    assert isinstance(rep, ArticulatedTrainingReport)
    assert rep.characters_count == 6
    summary = rep.summary_text()
    assert "REPORTE ENTRENAMIENTO ARTICULADO V2" in summary
