import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_training_logger(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configura y retorna el logger rotativo para entrenamiento V2.
    Escribe en logs/training_v2.log con maxBytes=5MB y 3 backups.
    """
    logger = logging.getLogger("SpriteStudio.TrainingV2")
    logger.setLevel(level)

    # Evitar duplicar handlers si ya se configuró previamente
    if logger.handlers:
        return logger

    target_path = Path(log_file or (Path.cwd() / "logs" / "training_v2.log")).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        target_path,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
