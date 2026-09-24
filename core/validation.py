import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image

from core.naming import OFFICIAL_VARIANTS
from models.character import Character

logger = logging.getLogger("SpriteStudio.Validation")


@dataclass
class ValidationIssue:
    severity: str  # "ERROR", "WARNING", "INFO"
    character_id: str
    variant: Optional[str]
    message: str
    file_path: Optional[Path] = None


@dataclass
class ValidationReport:
    total_characters: int = 0
    valid_characters: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "ERROR")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "WARNING")

    @property
    def is_valid(self) -> bool:
        return self.error_count == 0


class CharacterValidator:
    """
    Validador de assets de personajes y consistencia de variantes.
    """

    def validate_image_file(self, path: Path) -> List[str]:
        """Verifica que el archivo exista, sea legible y posea canal alfa."""
        issues = []
        if not path.exists():
            return [f"El archivo no existe: {path.name}"]

        try:
            with Image.open(path) as img:
                if img.format != "PNG":
                    issues.append(f"{path.name} no es PNG (formato detectado: {img.format})")
                if "A" not in img.getbands():
                    issues.append(f"{path.name} carece de canal alfa (transparencia).")
                if img.width == 0 or img.height == 0:
                    issues.append(f"{path.name} tiene dimensiones inválidas (0x0).")
        except Exception as e:
            issues.append(f"Error al abrir {path.name}: {str(e)}")

        return issues

    def validate_character(self, character: Character) -> List[ValidationIssue]:
        """Valida exhaustivamente un personaje individual y sus 3 variantes oficiales."""
        issues: List[ValidationIssue] = []

        # Verificar cada variante oficial
        for v_name in OFFICIAL_VARIANTS:
            variant = character.variants.get(v_name)
            if not variant:
                issues.append(ValidationIssue(
                    severity="ERROR",
                    character_id=character.character_id,
                    variant=v_name,
                    message=f"Falta la variante oficial obligatoria '{v_name}'"
                ))
                continue

            # Validar referencia
            if not variant.reference_image:
                issues.append(ValidationIssue(
                    severity="WARNING",
                    character_id=character.character_id,
                    variant=v_name,
                    message=f"Variante '{v_name}' no tiene imagen de referencia."
                ))
            else:
                img_errs = self.validate_image_file(variant.reference_image)
                for err in img_errs:
                    issues.append(ValidationIssue(
                        severity="ERROR",
                        character_id=character.character_id,
                        variant=v_name,
                        message=f"Referencia inválida: {err}",
                        file_path=variant.reference_image
                    ))

            # Validar spritesheet
            if not variant.spritesheet:
                issues.append(ValidationIssue(
                    severity="WARNING",
                    character_id=character.character_id,
                    variant=v_name,
                    message=f"Variante '{v_name}' no tiene spritesheet de movimiento."
                ))
            else:
                sheet_errs = self.validate_image_file(variant.spritesheet)
                for err in sheet_errs:
                    issues.append(ValidationIssue(
                        severity="ERROR",
                        character_id=character.character_id,
                        variant=v_name,
                        message=f"Spritesheet inválido: {err}",
                        file_path=variant.spritesheet
                    ))

        return issues

    def validate_dataset(self, characters: Dict[str, Character]) -> ValidationReport:
        """Valida una colección completa de personajes del dataset."""
        report = ValidationReport(total_characters=len(characters))

        for char_id, char in characters.items():
            char_issues = self.validate_character(char)
            report.issues.extend(char_issues)

            has_errors = any(i.severity == "ERROR" for i in char_issues)
            if not has_errors:
                report.valid_characters += 1

        logger.info(
            f"Validación de dataset concluida: {report.valid_characters}/{report.total_characters} válidos. "
            f"Errores: {report.error_count}, Advertencias: {report.warning_count}"
        )
        return report
