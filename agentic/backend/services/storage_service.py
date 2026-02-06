#!/usr/bin/env python3
"""
Storage service for managing job files and directories.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
from PIL import Image
import numpy as np
from loguru import logger
import uuid
from rembg import remove as rembg_remove


class StorageService:
    """Manages file storage for jobs and objects."""
    
    def __init__(self, base_dir: str = "data/jobs"):
        """
        Initialize storage service.
        
        Args:
            base_dir: Base directory for storing job data
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Storage service initialized at: {self.base_dir}")
    
    def create_job_directory(self, job_id: Optional[str] = None) -> str:
        """
        Create directory structure for a new job.
        
        Args:
            job_id: Optional job ID (generates UUID if not provided)
            
        Returns:
            Job ID
        """
        if job_id is None:
            job_id = str(uuid.uuid4())
        
        job_dir = self.base_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories.
        (job_dir / "objects").mkdir(exist_ok=True)
        
        logger.info(f"Created job directory: {job_id}")
        return job_id
    
    def save_original_image(self, job_id: str, image: Image.Image) -> tuple[str, str]:
        """
        Save both original and resized images.
        
        Args:
            job_id: Job ID
            image: PIL Image (original uploaded)
            
        Returns:
            Tuple of (original_path, resized_path)
        """
        job_dir = self.base_dir / job_id
        
        # Save original as-is
        original_path = job_dir / "original.png"
        image.save(original_path)
        logger.info(f"Saved original image: {original_path} (size: {image.size})")
        
        # Create and save resized version for processing
        img_resized = image.copy()
        img_resized.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
        resized_path = job_dir / "image_resized.png"
        img_resized.save(resized_path)
        logger.info(f"Saved resized image: {resized_path} (size: {img_resized.size})")
        
        return str(original_path), str(resized_path)
    
    def save_overlay_mask(self, job_id: str, overlay_image: Image.Image) -> str:
        """
        Save overlay mask visualization.
        
        Args:
            job_id: Job ID
            overlay_image: PIL Image with masks overlay
            
        Returns:
            Path to saved image
        """
        overlay_path = self.base_dir / job_id / "overlay_masks.png"
        overlay_image.save(overlay_path)
        logger.info(f"Saved overlay mask: {overlay_path}")
        return str(overlay_path)
    
    def create_object_directory(self, job_id: str, object_id: str) -> Path:
        """
        Create directory for a specific object.
        
        Args:
            job_id: Job ID
            object_id: Object ID
            
        Returns:
            Path to object directory
        """
        object_dir = self.base_dir / job_id / "objects" / object_id
        object_dir.mkdir(parents=True, exist_ok=True)
        return object_dir
    
    def save_object_mask(self, job_id: str, object_id: str, mask: np.ndarray) -> str:
        """
        Save object segmentation mask.
        
        Args:
            job_id: Job ID
            object_id: Object ID
            mask: Binary mask array
            
        Returns:
            Path to saved mask
        """
        object_dir = self.create_object_directory(job_id, object_id)
        mask_path = object_dir / "mask.png"
        
        # Ensure mask is uint8.
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        
        Image.fromarray(mask).save(mask_path)
        logger.info(f"Saved object mask: {mask_path}")
        return str(mask_path)
    
    def save_masked_original(self, job_id: str, object_id: str, masked_image: Image.Image) -> str:
        """
        Save masked original image (object highlighted in original context).
        
        Args:
            job_id: Job ID
            object_id: Object ID
            masked_image: PIL Image
            
        Returns:
            Path to saved image
        """
        object_dir = self.create_object_directory(job_id, object_id)
        image_path = object_dir / "masked_original.png"
        masked_image.save(image_path)
        logger.info(f"Saved masked original: {image_path}")
        return str(image_path)
    
    def save_generated_image(self, job_id: str, object_id: str, image: Image.Image) -> str:
        """
        Save generated clean image and a no-background version using rembg.
        
        Saves:
        - generated.png: Original image from Gemini.
        - generated_nobg.png: Background removed with rembg (returned path).
        
        Args:
            job_id: Job ID
            object_id: Object ID
            image: PIL Image
            
        Returns:
            Path to the no-background image (generated_nobg.png).
        """
        object_dir = self.create_object_directory(job_id, object_id)
        original_path = object_dir / "generated.png"
        image.save(original_path)
        logger.info(f"Saved generated image XXXXXXX: {original_path}")
        
        nobg_path = object_dir / "generated_nobg.png"
        image_nobg = rembg_remove(image)
        image_nobg.save(nobg_path)
        logger.info(f"Saved no-background image: {nobg_path}")
        return str(nobg_path)
    
    def save_edited_image(self, job_id: str, object_id: str, image: Image.Image) -> str:
        """
        Save edited image.
        
        Args:
            job_id: Job ID
            object_id: Object ID
            image: PIL Image
            
        Returns:
            Path to the no-background image (edited_nobg.png).
        """
        object_dir = self.create_object_directory(job_id, object_id)
        image_path = object_dir / "edited.png"
        image.save(image_path)
        logger.info(f"Saved edited image: {image_path}")

        nobg_path = object_dir / "edited_nobg.png"
        image_nobg = rembg_remove(image)
        image_nobg.save(nobg_path)
        logger.info(f"Saved no-background edited image: {nobg_path}")
        return str(nobg_path)
    
    def get_assets_directory(self, job_id: str, object_id: str) -> Path:
        """
        Get (and create if needed) the assets directory for an object.
        
        Args:
            job_id: Job ID
            object_id: Object ID
            
        Returns:
            Path to assets directory
        """
        object_dir = self.create_object_directory(job_id, object_id)
        assets_dir = object_dir / "assets"
        assets_dir.mkdir(exist_ok=True)
        logger.info(f"Assets directory: {assets_dir}")
        return assets_dir
    
    def save_3d_assets(self, job_id: str, object_id: str, ply_path: str, glb_path: str) -> dict:
        """
        Copy 3D assets to object directory.
        
        Args:
            job_id: Job ID
            object_id: Object ID
            ply_path: Source PLY file path
            glb_path: Source GLB file path
            
        Returns:
            Dictionary of destination paths
        """
        object_dir = self.create_object_directory(job_id, object_id)
        assets_dir = object_dir / "assets"
        assets_dir.mkdir(exist_ok=True)
        
        paths = {}
        
        logger.info(f"Saving 3D assets for {job_id}/{object_id}")
        logger.info(f"  Source PLY: {ply_path} (exists: {os.path.exists(ply_path)})")
        logger.info(f"  Source GLB: {glb_path} (exists: {os.path.exists(glb_path)})")
        logger.info(f"  Destination: {assets_dir}")
        
        # Copy PLY.
        if ply_path and os.path.exists(ply_path):
            dest_ply = assets_dir / "model.ply"
            shutil.copy(ply_path, dest_ply)
            paths['ply'] = str(dest_ply)
            logger.info(f"✅ Saved PLY asset: {dest_ply}")
        else:
            logger.warning(f"PLY file not found or empty path: {ply_path}")
        
        # Copy GLB.
        if glb_path and os.path.exists(glb_path):
            dest_glb = assets_dir / "model.glb"
            shutil.copy(glb_path, dest_glb)
            paths['glb'] = str(dest_glb)
            logger.info(f"✅ Saved GLB asset: {dest_glb}")
        else:
            logger.warning(f"GLB file not found or empty path: {glb_path}")
        
        return paths
    
    def list_jobs(self) -> List[str]:
        """
        List all job IDs.
        
        Returns:
            List of job IDs
        """
        if not self.base_dir.exists():
            return []
        
        job_ids = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and (item / "job_metadata.json").exists():
                job_ids.append(item.name)
        
        return sorted(job_ids, reverse=True)  # Most recent first
    
    def delete_job(self, job_id: str):
        """
        Delete a job and all its data.
        
        Args:
            job_id: Job ID to delete
        """
        job_dir = self.base_dir / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir)
            logger.info(f"Deleted job: {job_id}")
