"""
Villa del Chef - Sprite Studio: CLI de Validación Cruzada V2 (Cross-Character Validation).
- 100% offline, local en CPU (Windows compatible).
- Crea o carga la partición Train / Validation (seed determinista).
- Entrena sobre personajes de entrenamiento.
- Transfiere movimiento y valida sobre personajes de validación.
- Genera muestras de auditoría visual (audit_samples/validation/).
- Produce reporte consolidado en JSON.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Agregar raíz del proyecto al sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.cross_character_validator import CrossCharacterValidator, DEFAULT_VALIDATION_ANIMATIONS
from core.dataset_splitter import DatasetSplitter
from core.guard import SourceDatasetGuard
from core.incremental_training_engine import IncrementalTrainingEngineV2
from services.dataset_service import DatasetService


def main():
    parser = argparse.ArgumentParser(
        description="Villa del Chef - Sprite Studio: Validación Cruzada entre Personajes V2"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para partición determinista (por defecto: 42)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.8,
        help="Ratio de personajes para entrenamiento (por defecto: 0.8)",
    )
    parser.add_argument(
        "--animations",
        type=str,
        default="walk_down,cook_down",
        help="Animaciones a validar separadas por coma (ej: walk_down,cook_down)",
    )
    parser.add_argument(
        "--max-val-chars",
        type=int,
        default=None,
        help="Límite opcional de personajes de validación a procesar",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Muestra información detallada de depuración en consola",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n" + "=" * 60)
    print("VILLA DEL CHEF - SPRITE STUDIO")
    print("INICIANDO VALIDACIÓN CRUZADA ENTRE PERSONAJES (V2)")
    print("=" * 60)

    # 1. Guardián de dataset maestro
    guard = SourceDatasetGuard(
        PROJECT_ROOT / "dataset" / "finished_characters" / "approved" / "personajes al 100%"
    )
    initial_count = guard.count_files()

    # 2. Partición Train / Validation
    dataset_service = DatasetService(base_dir=PROJECT_ROOT)
    splitter = DatasetSplitter(base_dir=PROJECT_ROOT, default_seed=args.seed, default_ratio=args.ratio)
    split = splitter.split_from_dataset(dataset_service=dataset_service, seed=args.seed, train_ratio=args.ratio, save_split=True)

    print(f"\n[SPLIT] Total personajes: {len(split.train_characters) + len(split.validation_characters)}")
    print(f"        Train ({len(split.train_characters)}): {', '.join(split.train_characters)}")
    print(f"        Validation ({len(split.validation_characters)}): {', '.join(split.validation_characters)}")

    # 3. Asegurar plantillas (entrenamiento incremental si faltan)
    target_anims = [a.strip() for a in args.animations.split(",") if a.strip()]
    templates_dir = PROJECT_ROOT / "dataset" / "templates_v2"
    missing_templates = [a for a in target_anims if not (templates_dir / f"{a}.json").is_file()]

    if missing_templates:
        print(f"\n[TRAINING] Generando plantillas faltantes: {missing_templates}...")
        engine = IncrementalTrainingEngineV2(base_dir=PROJECT_ROOT)
        engine.run_training(is_smoke=True)

    # 4. Ejecutar Cross-Character Validation
    print(f"\n[VALIDATION] Evaluando transferencia hacia personajes de validación...")
    validator = CrossCharacterValidator(base_dir=PROJECT_ROOT, guard=guard)
    summary = validator.run_cross_validation(
        split=split,
        animations=target_anims,
        max_validation_characters=args.max_val_chars,
    )

    # 5. Verificación final de integridad de fuentes
    final_count = guard.count_files()
    dataset_modified = (initial_count != final_count)

    print("\n" + "=" * 60)
    print("RESULTADO DE VALIDACIÓN CRUZADA V2:")
    print(f"Personajes de validación:    {summary['validation_characters_count']}")
    print(f"Animaciones evaluadas:       {', '.join(summary['animations_evaluated'])}")
    print(f"Muestras de auditoría (PNG): {summary['audit_samples_generated']}")
    print(f"Requieren revisión manual:   {summary['reviews_required']}")
    print(f"Errores:                     {summary['errors']}")
    print(f"Dataset maestro intacto:     {'SÍ (0 cambios)' if not dataset_modified else 'NO (ALERTA)'}")
    print("=" * 60 + "\n")

    if summary["errors"] > 0 or dataset_modified or summary["audit_samples_generated"] == 0:
        print("[FALLO] La validación cruzada no cumplió todos los criterios de aceptación.")
        sys.exit(1)
    else:
        print("[ÉXITO] Validación cruzada completada satisfactoriamente.")
        sys.exit(0)


if __name__ == "__main__":
    main()
