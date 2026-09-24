from core.naming import (
    OFFICIAL_VARIANTS,
    VARIANT_LABELS,
    sanitize_character_id,
    format_display_name,
    make_unique_key,
    parse_file_variant,
)
from core.alpha_analyzer import AlphaAnalyzer
from core.sheet_detector import SheetDetector, SheetDetectionResult, FrameRect
from core.frame_extractor import FrameExtractor, OFFICIAL_ANIMATION_ROWS
from core.bbox_detector import BoundingBoxDetector
from core.baseline_detector import BaselineDetector
from core.frame_normalizer import FrameNormalizer
from core.similarity import FrameComparator
from core.character_analyzer import CharacterAnalyzer, CharacterAnalysisResult
from core.pose_analyzer import PoseAnalyzer, PoseSkeleton, AnchorPoint
from core.pose_analyzer_v2 import PoseAnalyzerV2
from core.motion_analyzer import MotionAnalyzer, MovementDescriptor, FrameMotionDelta
from core.template_extractor import TemplateExtractor
from core.template_library import TemplateLibrary
from core.template_matcher import TemplateMatcher, MatchResult
from core.compositor import PropCompositor, PropTransform
from core.motion_transfer import MotionTransferEngine
from core.spritesheet_builder import SpritesheetBuilder, ConsistencyMetrics
from core.dataset_scanner import DatasetScanner, compute_file_sha256
from core.dataset_indexer import DatasetIndexer
from core.guard import SourceDatasetGuard, SourceDatasetWriteError
from core.validation import CharacterValidator, ValidationReport, ValidationIssue

__all__ = [
    "OFFICIAL_VARIANTS",
    "VARIANT_LABELS",
    "sanitize_character_id",
    "format_display_name",
    "make_unique_key",
    "parse_file_variant",
    "AlphaAnalyzer",
    "SheetDetector",
    "SheetDetectionResult",
    "FrameRect",
    "FrameExtractor",
    "OFFICIAL_ANIMATION_ROWS",
    "BoundingBoxDetector",
    "BaselineDetector",
    "FrameNormalizer",
    "FrameComparator",
    "CharacterAnalyzer",
    "CharacterAnalysisResult",
    "PoseAnalyzer",
    "PoseSkeleton",
    "AnchorPoint",
    "PoseAnalyzerV2",
    "MotionAnalyzer",
    "MovementDescriptor",
    "FrameMotionDelta",
    "TemplateExtractor",
    "TemplateLibrary",
    "TemplateMatcher",
    "MatchResult",
    "PropCompositor",
    "PropTransform",
    "MotionTransferEngine",
    "SpritesheetBuilder",
    "ConsistencyMetrics",
    "DatasetScanner",
    "compute_file_sha256",
    "DatasetIndexer",
    "SourceDatasetGuard",
    "SourceDatasetWriteError",
    "CharacterValidator",
    "ValidationReport",
    "ValidationIssue",
]
