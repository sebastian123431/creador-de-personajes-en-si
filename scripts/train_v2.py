import argparse
import logging
import sys
from pathlib import Path

# Agregar raíz del proyecto al sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.incremental_training_engine import IncrementalTrainingEngineV2
from services.dataset_service import DatasetService


def main():
    parser = argparse.ArgumentParser(
        description="Villa del Chef - Sprite Studio: Motor de Entrenamiento Cinemático Articulado V2"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Ejecuta entrenamiento rápido de prueba (smoke training) con 3 personajes y 2 animaciones",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Ejecuta entrenamiento completo con todos los personajes aprobados y 16 animaciones",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Ignora la caché incremental y reconstruye todos los descriptores y plantillas",
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

    if args.full:
        print("\n" + "=" * 60)
        print("Full training not enabled until smoke validation passes.")
        print("=" * 60 + "\n")
        sys.exit(1)

    if not args.smoke:
        print("Uso: python scripts/train_v2.py --smoke [--force-rebuild] [--verbose]")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("VILLA DEL CHEF - SPRITE STUDIO")
    print("INICIANDO SMOKE TRAINING V2 (LOCAL & OFFLINE)")
    print("=" * 60)

    def on_progress(msg: str, cur: int, tot: int, details: dict):
        hits = details.get("cache_hits", 0)
        misses = details.get("cache_misses", 0)
        print(f"[{cur}/{tot}] {msg} (Hits: {hits}, Misses: {misses})")

    engine = IncrementalTrainingEngineV2(
        base_dir=PROJECT_ROOT,
        progress_callback=on_progress,
    )

    session = engine.run_training(
        is_smoke=True,
        force_rebuild=args.force_rebuild,
    )

    print("\n" + "=" * 60)
    print("RESULTADO DE SMOKE TRAINING V2:")
    print(f"Estado:              {session.status}")
    print(f"Personajes:          {session.characters_processed} / {session.characters_total}")
    print(f"Variantes:           {session.variants_processed} / {session.variants_total}")
    print(f"Frames procesados:   {session.frames_processed}")
    print(f"Frames desde caché:  {session.frames_cached}")
    print(f"Cache Hits:          {session.cache_hits}")
    print(f"Cache Misses:        {session.cache_misses}")
    print(f"Plantillas generadas:{session.templates_generated}")
    print(f"Errores:             {len(session.errors)}")
    print(f"Advertencias:        {len(session.warnings)}")
    print("=" * 60 + "\n")

    if session.status != "completed" or session.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
