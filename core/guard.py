from pathlib import Path
from typing import Optional, Union


class SourceDatasetWriteError(Exception):
    """
    Excepción crítica lanzada cuando cualquier operación intenta escribir,
    modificar, crear archivos o borrar dentro de la carpeta protegida del dataset original.
    """
    pass


class SourceDatasetGuard:
    """
    Guardián estricto de inmutabilidad para el dataset de personajes aprobados.
    Garantiza que la carpeta original (ej. 'personajes al 100%') sea estrictamente READ-ONLY.
    """

    def __init__(self, protected_root: Optional[Union[str, Path]] = None):
        self._protected_root: Optional[Path] = None
        if protected_root:
            self.set_protected_root(protected_root)

    def set_protected_root(self, root: Union[str, Path]):
        self._protected_root = Path(root).resolve()

    @property
    def protected_root(self) -> Optional[Path]:
        return self._protected_root

    def is_inside_protected_root(self, target_path: Union[str, Path]) -> bool:
        """
        Determina si una ruta se encuentra dentro del árbol protegido del dataset original.
        """
        if not self._protected_root:
            return False

        try:
            resolved_target = Path(target_path).resolve()
            # Si resolved_target es igual o hijo de _protected_root
            resolved_target.relative_to(self._protected_root)
            return True
        except ValueError:
            return False

    def assert_can_write(self, target_path: Union[str, Path], operation_desc: str = "escritura"):
        """
        Lanza SourceDatasetWriteError si target_path apunta dentro del dataset protegido.
        """
        if self.is_inside_protected_root(target_path):
            raise SourceDatasetWriteError(
                f"VIOLACIÓN DE PROTECCIÓN: Intento de {operation_desc} en la ruta protegida '{target_path}'. "
                f"El dataset original ('{self._protected_root}') es estrictamente READ-ONLY. "
                f"Todos los archivos derivados deben escribirse fuera."
            )

    def count_files(self) -> int:
        """Retorna la cantidad total de archivos dentro del árbol protegido."""
        if not self._protected_root or not self._protected_root.is_dir():
            return 0
        return sum(1 for p in self._protected_root.rglob("*") if p.is_file())
