#!/usr/bin/env python3
"""
Data models for the Agentic 2D→3D system.
"""

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
from pathlib import Path
import json
import numpy as np
from enum import Enum


class JobStatus(Enum):
    """Job processing status."""
    UPLOADED = "uploaded"
    SEGMENTING = "segmenting"
    SEGMENTED = "segmented"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"


class ObjectStatus(Enum):
    """Individual object processing status."""
    DETECTED = "detected"
    IMAGE_GENERATED = "image_generated"
    IMAGE_EDITED = "image_edited"
    GENERATING_3D = "generating_3d"
    COMPLETE_3D = "complete_3d"
    ERROR = "error"


@dataclass
class Asset3D:
    """3D asset metadata and paths."""
    ply_path: Optional[str] = None
    glb_path: Optional[str] = None
    preview_path: Optional[str] = None
    generation_seed: int = 42
    quality_scores: Dict[str, float] = field(default_factory=dict)
    generation_time: float = 0.0
    created_at: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Asset3D':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SegmentedObject:
    """Represents a segmented object from the image."""
    object_id: str
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    
    # File paths.
    mask_path: str
    masked_original_path: Optional[str] = None
    generated_image_path: Optional[str] = None
    edited_image_path: Optional[str] = None
    
    # Original mask data from Gemini (for passing back to image generation)
    original_mask_data: Optional[str] = None  # Base64 mask or other format
    
    # Processing metadata.
    edit_prompt: Optional[str] = None
    status: str = ObjectStatus.DETECTED.value
    asset_3d: Optional[Asset3D] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.asset_3d:
            data['asset_3d'] = self.asset_3d.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SegmentedObject':
        """Create from dictionary."""
        asset_data = data.pop('asset_3d', None)
        obj = cls(**data)
        if asset_data:
            obj.asset_3d = Asset3D.from_dict(asset_data)
        return obj
    
    def get_current_image_path(self) -> str:
        """Get the most recent image path (edited > generated > masked)."""
        if self.edited_image_path:
            return self.edited_image_path
        elif self.generated_image_path:
            return self.generated_image_path
        else:
            return self.masked_original_path or ""
    
    def has_3d_asset(self) -> bool:
        """Check if 3D asset has been generated."""
        return self.asset_3d is not None and self.asset_3d.glb_path is not None


@dataclass
class Job:
    """Represents a complete processing job."""
    job_id: str
    original_image_path: str  # Original uploaded image (for display)
    resized_image_path: str   # Resized image (for processing)
    overlay_mask_path: Optional[str] = None
    objects: List[SegmentedObject] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = JobStatus.UPLOADED.value
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['objects'] = [obj.to_dict() for obj in self.objects]
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Job':
        """Create from dictionary."""
        objects_data = data.pop('objects', [])
        job = cls(**data)
        job.objects = [SegmentedObject.from_dict(obj_data) for obj_data in objects_data]
        return job
    
    def save(self, base_dir: Path):
        """Save job metadata to disk."""
        job_dir = base_dir / self.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_path = job_dir / "job_metadata.json"
        self.updated_at = datetime.now().isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, base_dir: Path, job_id: str) -> Optional['Job']:
        """Load job from disk."""
        metadata_path = base_dir / job_id / "job_metadata.json"
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def get_object(self, object_id: str) -> Optional[SegmentedObject]:
        """Get object by ID."""
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None
    
    def update_object_status(self, object_id: str, status: ObjectStatus, error_message: Optional[str] = None):
        """Update status of a specific object."""
        obj = self.get_object(object_id)
        if obj:
            obj.status = status.value
            if error_message:
                obj.error_message = error_message
            self.updated_at = datetime.now().isoformat()
    
    def count_complete_3d(self) -> int:
        """Count how many objects have completed 3D generation."""
        return sum(1 for obj in self.objects if obj.has_3d_asset())
    
    def get_next_pending_object(self) -> Optional[SegmentedObject]:
        """Get next object pending 3D generation."""
        for obj in self.objects:
            if obj.status in [ObjectStatus.IMAGE_GENERATED.value, ObjectStatus.IMAGE_EDITED.value]:
                return obj
        return None


@dataclass
class GenerationQueue:
    """Queue for 3D generation jobs."""
    pending: List[Tuple[str, str]] = field(default_factory=list)  # (job_id, object_id)
    processing: Optional[Tuple[str, str]] = None
    completed: List[Tuple[str, str]] = field(default_factory=list)
    
    def add(self, job_id: str, object_id: str):
        """Add item to queue."""
        if (job_id, object_id) not in self.pending:
            self.pending.append((job_id, object_id))
    
    def add_batch(self, job_id: str, object_ids: List[str]):
        """Add multiple objects from a job."""
        for object_id in object_ids:
            self.add(job_id, object_id)
    
    def get_next(self) -> Optional[Tuple[str, str]]:
        """Get next item to process."""
        if self.pending and self.processing is None:
            self.processing = self.pending.pop(0)
            return self.processing
        return None
    
    def mark_complete(self, job_id: str, object_id: str):
        """Mark item as complete."""
        if self.processing == (job_id, object_id):
            self.completed.append(self.processing)
            self.processing = None
    
    def mark_error(self):
        """Mark current item as error and move to next."""
        if self.processing:
            self.completed.append(self.processing)
            self.processing = None
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self.pending) == 0 and self.processing is None
    
    def size(self) -> int:
        """Get queue size."""
        return len(self.pending) + (1 if self.processing else 0)
