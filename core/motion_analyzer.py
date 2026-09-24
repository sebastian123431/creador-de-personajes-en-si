from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

from core.bbox_detector import BoundingBoxDetector
from core.pose_analyzer import PoseAnalyzer, PoseSkeleton
from models.frame import Frame


@dataclass
class FrameMotionDelta:
    frame_index: int
    body_dx: int = 0
    body_dy: int = 0
    head_dx: int = 0
    head_dy: int = 0
    left_foot_dx: int = 0
    left_foot_dy: int = 0
    right_foot_dx: int = 0
    right_foot_dy: int = 0
    bbox_dw: int = 0
    bbox_dh: int = 0
    baseline_dy: int = 0
    anchors: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "body_dx": self.body_dx,
            "body_dy": self.body_dy,
            "head_dx": self.head_dx,
            "head_dy": self.head_dy,
            "left_foot_dx": self.left_foot_dx,
            "left_foot_dy": self.left_foot_dy,
            "right_foot_dx": self.right_foot_dx,
            "right_foot_dy": self.right_foot_dy,
            "bbox_dw": self.bbox_dw,
            "bbox_dh": self.bbox_dh,
            "baseline_dy": self.baseline_dy,
            "anchors": self.anchors,
        }


@dataclass
class MovementDescriptor:
    animation: str
    character_id: Optional[str] = None
    variant: Optional[str] = None
    frame_count: int = 4
    frames: List[FrameMotionDelta] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "animation": self.animation,
            "character_id": self.character_id,
            "variant": self.variant,
            "frame_count": len(self.frames),
            "frames": [f.to_dict() for f in self.frames],
        }


class MotionAnalyzer:
    """
    Analiza la cinemática y deltas de movimiento relativo entre los 4 frames
    de una secuencia de animación sin alterar la identidad del personaje.
    """

    def __init__(self, pose_analyzer: Optional[PoseAnalyzer] = None):
        self.pose_analyzer = pose_analyzer or PoseAnalyzer()

    def analyze_animation_motion(
        self,
        frames: List[Frame | Path],
        animation_name: str,
        character_id: Optional[str] = None,
        variant: Optional[str] = None
    ) -> MovementDescriptor:
        """
        Calcula los desplazamientos relativos tomando el Frame 1 como origen (dx=0, dy=0).
        """
        descriptor = MovementDescriptor(
            animation=animation_name,
            character_id=character_id,
            variant=variant,
            frame_count=len(frames)
        )

        if not frames:
            return descriptor

        skeletons: List[PoseSkeleton] = []
        bboxes: List[tuple] = []
        baselines: List[int] = []

        for f in frames:
            img_path = f.image_path if isinstance(f, Frame) else Path(f)
            skel = self.pose_analyzer.estimate_pose_anchors(img_path)
            skeletons.append(skel)

            with Image.open(img_path) as img:
                bb = BoundingBoxDetector.detect_sprite_bbox(img) or (0, 0, img.width, img.height)
                bboxes.append(bb)
                baselines.append(bb[3])

        # Frame 0 como referencia base
        base_skel = skeletons[0]
        base_bb = bboxes[0]
        base_bl = baselines[0]

        base_cx = (base_bb[0] + base_bb[2]) // 2
        base_cy = (base_bb[1] + base_bb[3]) // 2
        base_bw = base_bb[2] - base_bb[0]
        base_bh = base_bb[3] - base_bb[1]

        base_head = base_skel.anchors.get("head")
        base_lf = base_skel.anchors.get("left_foot")
        base_rf = base_skel.anchors.get("right_foot")

        for idx in range(len(frames)):
            cur_skel = skeletons[idx]
            cur_bb = bboxes[idx]
            cur_bl = baselines[idx]

            cur_cx = (cur_bb[0] + cur_bb[2]) // 2
            cur_cy = (cur_bb[1] + cur_bb[3]) // 2
            cur_bw = cur_bb[2] - cur_bb[0]
            cur_bh = cur_bb[3] - cur_bb[1]

            cur_head = cur_skel.anchors.get("head")
            cur_lf = cur_skel.anchors.get("left_foot")
            cur_rf = cur_skel.anchors.get("right_foot")

            delta = FrameMotionDelta(
                frame_index=idx + 1,
                body_dx=int(cur_cx - base_cx),
                body_dy=int(cur_cy - base_cy),
                head_dx=int((cur_head.x - base_head.x) if cur_head and base_head else 0),
                head_dy=int((cur_head.y - base_head.y) if cur_head and base_head else 0),
                left_foot_dx=int((cur_lf.x - base_lf.x) if cur_lf and base_lf else 0),
                left_foot_dy=int((cur_lf.y - base_lf.y) if cur_lf and base_lf else 0),
                right_foot_dx=int((cur_rf.x - base_rf.x) if cur_rf and base_rf else 0),
                right_foot_dy=int((cur_rf.y - base_rf.y) if cur_rf and base_rf else 0),
                bbox_dw=int(cur_bw - base_bw),
                bbox_dh=int(cur_bh - base_bh),
                baseline_dy=int(cur_bl - base_bl),
                anchors=cur_skel.to_dict()
            )
            descriptor.frames.append(delta)

        return descriptor
