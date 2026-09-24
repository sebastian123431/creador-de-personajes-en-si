import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from core.frame_extractor import OFFICIAL_ANIMATION_ROWS
from models.motion_template import MotionTemplate, TemplateFrame

logger = logging.getLogger("SpriteStudio.TemplateLibrary")


class TemplateLibrary:
    """
    Repositorio de plantillas de movimiento aprendidas en animations/learned/.
    """

    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = Path(templates_dir or "animations/learned")
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.templates: Dict[str, MotionTemplate] = {}

    def save_template(self, template: MotionTemplate) -> Path:
        """Guarda un template en archivo JSON individual."""
        target_path = self.templates_dir / f"{template.name}.json"
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(template.to_dict(), f, indent=4, ensure_ascii=False)
        self.templates[template.name] = template
        logger.info(f"Template guardado: {target_path}")
        return target_path

    def load_template(self, animation_name: str) -> Optional[MotionTemplate]:
        """Carga un template específico desde disco."""
        target_path = self.templates_dir / f"{animation_name}.json"
        if not target_path.exists():
            return None

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            frames = [
                TemplateFrame(
                    index=fd.get("index", idx + 1),
                    body=fd.get("body", {"x": 0, "y": 0}),
                    head=fd.get("head", {"x": 0, "y": 0}),
                    left_foot=fd.get("left_foot", {"x": 0, "y": 0}),
                    right_foot=fd.get("right_foot", {"x": 0, "y": 0}),
                    bbox_scale=fd.get("bbox_scale", {"w": 1.0, "h": 1.0}),
                    anchors=fd.get("anchors", {}),
                )
                for idx, fd in enumerate(data.get("frames", []))
            ]

            tmpl = MotionTemplate(
                name=data.get("name", animation_name),
                frame_count=data.get("frame_count", len(frames)),
                loop=data.get("loop", True),
                samples_used=data.get("samples_used", 0),
                outliers_detected=data.get("outliers_detected", 0),
                frames=frames
            )
            self.templates[animation_name] = tmpl
            return tmpl
        except Exception as e:
            logger.error(f"Error cargando template {animation_name}: {e}")
            return None

    def load_all_templates(self) -> Dict[str, MotionTemplate]:
        """Carga todas las plantillas existentes."""
        self.templates.clear()
        for anim in OFFICIAL_ANIMATION_ROWS:
            self.load_template(anim)
        return self.templates

    def has_all_templates(self) -> bool:
        """Verifica si las 16 plantillas oficiales han sido aprendidas y guardadas."""
        return all((self.templates_dir / f"{anim}.json").exists() for anim in OFFICIAL_ANIMATION_ROWS)
