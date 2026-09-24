import logging
import sys
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from services.dataset_service import DatasetService
from ui.main_window import MainWindow


def setup_logging():
    """Configura logging estructurado para auditoría y depuración."""
    log_format = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    setup_logging()
    logger = logging.getLogger("SpriteStudio.App")
    logger.info("Iniciando Villa del Chef - Sprite Studio...")

    # Atributos de escalado DPI
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Villa del Chef - Sprite Studio")
    app.setOrganizationName("VillaDelChef")

    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "config" / "settings.json"

    dataset_service = DatasetService(base_dir=base_dir, config_path=config_path)

    window = MainWindow(dataset_service=dataset_service)
    window.show()

    logger.info("Aplicación iniciada exitosamente.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
