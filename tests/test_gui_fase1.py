import os
import sys
import pytest
from pathlib import Path
from PySide6.QtWidgets import QApplication

from services.dataset_service import DatasetService
from ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    # Modo offscreen para entornos de pruebas sin display activo
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_main_window_init(qapp, sample_dataset_dir: Path, tmp_path: Path):
    """Verifica que MainWindow inicialice, cargue personajes y configure paneles sin errores."""
    service = DatasetService(base_dir=tmp_path)
    service.import_external_directory_to_approved(sample_dataset_dir)

    window = MainWindow(dataset_service=service)
    assert window is not None
    assert window.windowTitle() == "Villa del Chef - Sprite Studio"

    # Verificar que el árbol esté poblado
    assert window.dataset_panel.tree.topLevelItemCount() == 1
    root_node = window.dataset_panel.tree.topLevelItem(0)
    assert root_node.childCount() == 6

    # Simular selección de un personaje en el árbol
    item = root_node.child(0)
    window.dataset_panel.tree.itemClicked.emit(item, 0)

    char = window.character_panel.current_character
    assert char is not None
    assert char.character_id in ["alex", "andrea", "diego_vallenar", "diego_serena", "andres_arica", "benja_bacaba"]

    # Simular cambio de variante a rbchef
    window.character_panel.radio_buttons["rbchef"].click()
    assert window.character_panel.current_variant_name == "rbchef"

    # Verificar que el preview canvas haya recibido la imagen
    assert window.preview_panel.canvas.pixmap is not None

    window.close()
