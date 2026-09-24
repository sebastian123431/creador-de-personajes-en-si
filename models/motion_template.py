from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TemplateFrame:
    index: int
    body: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})
    head: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})
    left_foot: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})
    right_foot: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})
    bbox_scale: Dict[str, float] = field(default_factory=lambda: {"w": 1.0, "h": 1.0})
    anchors: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "body": self.body,
            "head": self.head,
            "left_foot": self.left_foot,
            "right_foot": self.right_foot,
            "bbox_scale": self.bbox_scale,
            "anchors": self.anchors,
        }


@dataclass
class MotionTemplate:
    name: str
    frame_count: int = 4
    loop: bool = True
    samples_used: int = 0
    outliers_detected: int = 0
    frames: List[TemplateFrame] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "frame_count": self.frame_count,
            "loop": self.loop,
            "samples_used": self.samples_used,
            "outliers_detected": self.outliers_detected,
            "frames": [f.to_dict() for f in self.frames],
        }
