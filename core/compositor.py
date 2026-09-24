from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image


@dataclass
class PropTransform:
    pivot: Tuple[float, float] = (0.5, 0.5)  # centro relativo
    scale: float = 1.0
    dx: int = 0
    dy: int = 0
    rotation: float = 0.0


class PropCompositor:
    """
    Gestiona la composición y superposición de props (bowl, spoon, plate, box)
    en las animaciones que lo requieren (cook, pickup, serve).
    """

    def __init__(self, props_dir: Optional[Path] = None):
        self.props_dir = Path(props_dir or "assets/props")

    def get_prop_path(self, prop_name: str) -> Optional[Path]:
        p = self.props_dir / f"{prop_name}.png"
        return p if p.exists() else None

    def composite_prop(
        self,
        base_canvas: Image.Image,
        prop_name: str,
        hand_position: Tuple[int, int],
        transform: Optional[PropTransform] = None
    ) -> Image.Image:
        """
        Superpone un prop en la posición de anclaje (ej. manos) con escalado pixel-perfect (NEAREST).
        """
        prop_path = self.get_prop_path(prop_name)
        if not prop_path:
            return base_canvas

        with Image.open(prop_path) as prop_img:
            prop_img = prop_img.convert("RGBA")

            t = transform or PropTransform()
            if t.scale != 1.0:
                new_w = max(1, int(prop_img.width * t.scale))
                new_h = max(1, int(prop_img.height * t.scale))
                prop_img = prop_img.resize((new_w, new_h), resample=Image.Resampling.NEAREST)

            if t.rotation != 0.0:
                prop_img = prop_img.rotate(t.rotation, resample=Image.Resampling.NEAREST, expand=True)

            px = int(hand_position[0] - prop_img.width * t.pivot[0] + t.dx)
            py = int(hand_position[1] - prop_img.height * t.pivot[1] + t.dy)

            result = base_canvas.copy()
            result.paste(prop_img, (px, py), mask=prop_img)
            return result
