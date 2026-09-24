import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from core.character_analyzer import CharacterAnalyzer, CharacterAnalysisResult
from models.character import Character

logger = logging.getLogger("SpriteStudio.TemplateMatcher")


@dataclass
class MatchResult:
    character_id: str
    display_name: str
    similarity_score: float  # 0.0 a 1.0
    matched_variant: str
    details: Dict[str, float]


class TemplateMatcher:
    """
    Compara un personaje nuevo con el dataset dorado de aprobados buscando
    afinidad estructural y anatómica (altura, proporción cabeza/cuerpo, ancho y silueta).
    NO se basa en el color de piel o vestimenta, sino en la MORFOLOGÍA.
    """

    def __init__(self, analyzer: Optional[CharacterAnalyzer] = None):
        self.analyzer = analyzer or CharacterAnalyzer()

    def compare_morphic_similarity(
        self,
        target_analysis: CharacterAnalysisResult,
        candidate_analysis: CharacterAnalysisResult
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calcula un puntaje compuesto ponderado entre 0.0 y 1.0:
        - Altura: 25%
        - Ancho: 15%
        - Ratio Cabeza: 30%
        - Ratio Cuerpo: 30%
        """
        # 1. Similitud de Altura
        h_ratio = min(target_analysis.sprite_height, candidate_analysis.sprite_height) / max(
            target_analysis.sprite_height, candidate_analysis.sprite_height, 1
        )

        # 2. Similitud de Ancho
        w_ratio = min(target_analysis.sprite_width, candidate_analysis.sprite_width) / max(
            target_analysis.sprite_width, candidate_analysis.sprite_width, 1
        )

        # 3. Similitud de Ratio de Cabeza
        head_diff = abs(target_analysis.head_ratio - candidate_analysis.head_ratio)
        head_sim = max(0.0, 1.0 - head_diff * 2.5)

        # 4. Similitud de Ratio de Cuerpo
        body_diff = abs(target_analysis.body_ratio - candidate_analysis.body_ratio)
        body_sim = max(0.0, 1.0 - body_diff * 2.5)

        # Ponderación morfológica
        score = (h_ratio * 0.25) + (w_ratio * 0.15) + (head_sim * 0.30) + (body_sim * 0.30)
        score = round(float(np.clip(score, 0.0, 1.0)), 2)

        details = {
            "height_sim": round(float(h_ratio), 2),
            "width_sim": round(float(w_ratio), 2),
            "head_sim": round(float(head_sim), 2),
            "body_sim": round(float(body_sim), 2),
        }
        return score, details

    def find_best_references(
        self,
        new_character_ref_path: Path,
        approved_characters: Dict[str, Character],
        preferred_variant: str = "rnormal",
        top_k: int = 5
    ) -> List[MatchResult]:
        """
        Analiza el nuevo personaje y busca los k personajes del dataset con mayor afinidad morfológica.
        """
        target_analysis = self.analyzer.analyze_sprite(new_character_ref_path)
        if not target_analysis:
            logger.error("No se pudo analizar la morfología del personaje entrante.")
            return []

        results: List[MatchResult] = []

        for char_id, char in approved_characters.items():
            # Buscar variante equivalente o la primera disponible
            candidate_variant = char.variants.get(preferred_variant)
            if not candidate_variant or not candidate_variant.has_reference:
                # Buscar cualquier variante que tenga referencia
                candidate_variant = next((v for v in char.variants.values() if v.has_reference), None)

            if not candidate_variant or not candidate_variant.reference_image:
                continue

            cand_analysis = self.analyzer.analyze_sprite(candidate_variant.reference_image)
            if not cand_analysis:
                continue

            score, details = self.compare_morphic_similarity(target_analysis, cand_analysis)
            results.append(MatchResult(
                character_id=char_id,
                display_name=char.display_name,
                similarity_score=score,
                matched_variant=candidate_variant.variant,
                details=details
            ))

        # Ordenar de mayor a menor afinidad
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:top_k]
