import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image


class FrameComparator:
    """
    Herramientas de comparación y análisis diferencial entre frames pixel art.
    """

    @staticmethod
    def compute_absdiff(image_a_path: Path, image_b_path: Path) -> Optional[np.ndarray]:
        """
        Calcula la diferencia absoluta (cv2.absdiff) entre dos frames.
        Retorna una imagen BGR/RGB con la visualización del diferencial.
        """
        if not image_a_path.exists() or not image_b_path.exists():
            return None

        # Cargar con OpenCV en formato RGBA o BGR
        img_a = cv2.imread(str(image_a_path), cv2.IMREAD_UNCHANGED)
        img_b = cv2.imread(str(image_b_path), cv2.IMREAD_UNCHANGED)

        if img_a is None or img_b is None:
            return None

        # Asegurar dimensiones iguales para la resta
        if img_a.shape != img_b.shape:
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_NEAREST)

        diff = cv2.absdiff(img_a, img_b)
        return diff

    @staticmethod
    def create_onion_skin_frame(
        current_img_path: Path,
        prev_img_path: Optional[Path] = None,
        next_img_path: Optional[Path] = None,
        ghost_opacity: float = 0.35
    ) -> Optional[Image.Image]:
        """
        Genera una vista previa Onion Skinning combinando:
        - Frame previo (con tinte azulado y opacidad reducida)
        - Frame actual (opacidad 100%)
        - Frame siguiente (con tinte rojizo/anaranjado y opacidad reducida)
        """
        if not current_img_path.exists():
            return None

        with Image.open(current_img_path) as cur:
            cur = cur.convert("RGBA")
            base = Image.new("RGBA", cur.size, (0, 0, 0, 0))

            # 1. Capa Previa (Azulada)
            if prev_img_path and prev_img_path.exists():
                with Image.open(prev_img_path) as prev:
                    prev = prev.convert("RGBA")
                    prev_arr = np.array(prev, dtype=np.float32)
                    # Tinte azul: atenuar R y G, resaltar B
                    prev_arr[:, :, 0] *= 0.2
                    prev_arr[:, :, 1] *= 0.5
                    prev_arr[:, :, 2] *= 1.0
                    prev_arr[:, :, 3] *= ghost_opacity
                    prev_tinted = Image.fromarray(np.clip(prev_arr, 0, 255).astype(np.uint8), "RGBA")
                    base.paste(prev_tinted, (0, 0), mask=prev_tinted)

            # 2. Capa Siguiente (Rojiza)
            if next_img_path and next_img_path.exists():
                with Image.open(next_img_path) as nxt:
                    nxt = nxt.convert("RGBA")
                    nxt_arr = np.array(nxt, dtype=np.float32)
                    # Tinte rojo: resaltar R, atenuar B
                    nxt_arr[:, :, 0] *= 1.0
                    nxt_arr[:, :, 1] *= 0.3
                    nxt_arr[:, :, 2] *= 0.2
                    nxt_arr[:, :, 3] *= ghost_opacity
                    nxt_tinted = Image.fromarray(np.clip(nxt_arr, 0, 255).astype(np.uint8), "RGBA")
                    base.paste(nxt_tinted, (0, 0), mask=nxt_tinted)

            # 3. Capa Actual
            base.paste(cur, (0, 0), mask=cur)
            return base
