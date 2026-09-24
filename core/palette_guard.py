import logging
from typing import Set, Tuple, Union
import numpy as np
from PIL import Image

logger = logging.getLogger("SpriteStudio.PaletteGuard")


class PaletteGuard:
    """
    Guardián cromático para preservación estricta de la paleta original del personaje.
    Evita la aparición de colores espurios, artefactos de interpolación o gradientes borrosos
    durante las transformaciones cinemáticas.
    """

    @staticmethod
    def extract_palette(image: Union[Image.Image, np.ndarray]) -> Set[Tuple[int, int, int, int]]:
        """
        Extrae el conjunto de colores únicos RGBA de los píxeles opacos (alfa > 15).
        """
        if isinstance(image, Image.Image):
            arr = np.array(image.convert("RGBA"))
        else:
            arr = image

        alpha = arr[:, :, 3]
        opaque_mask = alpha > 15
        opaque_pixels = arr[opaque_mask]

        if len(opaque_pixels) == 0:
            return set()

        unique_colors = np.unique(opaque_pixels, axis=0)
        return {tuple(c) for c in unique_colors}

    @staticmethod
    def validate_frame(frame: Image.Image, reference_palette: Set[Tuple[int, int, int, int]]) -> Tuple[bool, int]:
        """
        Verifica si todos los píxeles opacos del frame pertenecen a la paleta de referencia.
        Retorna (es_valido, cantidad_de_pixeles_invalidos).
        """
        frame_palette = PaletteGuard.extract_palette(frame)
        invalid_colors = frame_palette - reference_palette

        if not invalid_colors:
            return True, 0

        # Contar píxeles con colores inválidos
        arr = np.array(frame.convert("RGBA"))
        invalid_count = 0
        for col in invalid_colors:
            matches = np.all(arr == np.array(col), axis=-1)
            invalid_count += int(np.sum(matches))

        return False, invalid_count

    @staticmethod
    def clean_frame(frame: Image.Image, reference_palette: Set[Tuple[int, int, int, int]]) -> Image.Image:
        """
        Limpia el frame proyectando cualquier píxel que tenga un color ajeno a la paleta
        hacia el color más cercano de la paleta de referencia en distancia euclidiana RGB/RGBA.
        Elimina también cualquier semi-transparencia residual (alpha <= 15 -> 0, alpha > 15 -> 255 si opaco).
        """
        if not reference_palette:
            return frame

        arr = np.array(frame.convert("RGBA"))
        h, w = arr.shape[:2]

        alpha = arr[:, :, 3]
        opaque_mask = alpha > 15
        transparent_mask = ~opaque_mask

        # Limpiar fondo a transparencia pura (0, 0, 0, 0)
        arr[transparent_mask] = [0, 0, 0, 0]

        if not np.any(opaque_mask):
            return Image.fromarray(arr, mode="RGBA")

        palette_arr = np.array(list(reference_palette), dtype=np.int32)  # Shape: (K, 4)

        # Encontrar colores en el frame que no estén en la paleta
        current_opaque_colors = np.unique(arr[opaque_mask], axis=0)  # Shape: (U, 4)
        col_tuples = {tuple(c) for c in current_opaque_colors}
        foreign_colors = col_tuples - reference_palette

        if not foreign_colors:
            return Image.fromarray(arr, mode="RGBA")

        foreign_arr = np.array(list(foreign_colors), dtype=np.int32)  # Shape: (F, 4)

        # Para cada color extraño, calcular distancia euclidiana a cada color de la paleta
        # Distancia euclidiana en RGBA ponderada (RGB con mayor peso)
        diff = foreign_arr[:, np.newaxis, :] - palette_arr[np.newaxis, :, :]  # Shape: (F, K, 4)
        dist_sq = (
            diff[:, :, 0] ** 2 * 2 +
            diff[:, :, 1] ** 2 * 4 +
            diff[:, :, 2] ** 2 * 3 +
            diff[:, :, 3] ** 2 * 1
        )
        nearest_indices = np.argmin(dist_sq, axis=1)  # Shape: (F,)

        # Reemplazar colores extraños por los más cercanos
        for idx, f_col in enumerate(foreign_colors):
            closest_col = palette_arr[nearest_indices[idx]]
            matches = np.all(arr == np.array(f_col), axis=-1)
            arr[matches] = closest_col

        return Image.fromarray(arr, mode="RGBA")
