#!/usr/bin/env python3
"""
3D generation agent using SAM-3D-Objects.
"""

from PIL import Image
import numpy as np
from loguru import logger
import time
from datetime import datetime
from pathlib import Path

from backend.services.sam3d_service import SAM3DService
from backend.services.storage_service import StorageService
from backend.models import SegmentedObject, Asset3D, ObjectStatus


class Generation3DAgent:
    """Agent for generating 3D assets from 2D images."""
    
    def __init__(self, sam3d_service: SAM3DService, storage_service: StorageService):
        """
        Initialize 3D generation agent.
        
        Args:
            sam3d_service: SAM-3D service
            storage_service: Storage service
        """
        self.sam3d = sam3d_service
        self.storage = storage_service
        logger.info("3D Generation Agent initialized")
    
    def generate_3d_asset(
        self,
        job_id: str,
        obj: SegmentedObject,
        seed: int = 42
    ) -> Asset3D:
        """
        Generate 3D asset for an object.
        
        Args:
            job_id: Job ID
            obj: Segmented object
            seed: Random seed
            
        Returns:
            Asset3D with paths to generated files
        """
        logger.info(f"Generating 3D asset for {obj.object_id}: {obj.label}")
        obj.status = ObjectStatus.GENERATING_3D.value
        
        # Get current image.
        image_path = obj.get_current_image_path()
        if not image_path:
            raise ValueError(f"No image available for object {obj.object_id}")
        
        # Load image as PIL and convert to numpy array for SAM-3D.
        pil_image = Image.open(image_path)
        
        # Resize to 512x512 max for SAM-3D to reduce GPU memory usage
        # SAM-3D models take ~21GB, need to keep input images small
        max_size = 512
        if max(pil_image.size) > max_size:
            pil_image.thumbnail([max_size, max_size], Image.Resampling.LANCZOS)
            logger.info(f"Resized image to {pil_image.size} for SAM-3D to reduce memory usage")
        
        image = np.array(pil_image)
        
        # Close PIL image to free memory
        pil_image.close()
        del pil_image
        
        # Aggressively free GPU memory before starting generation
        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            logger.info("Cleared GPU cache before 3D generation")
        
        # Generate 3D asset.
        start_time = time.time()
        
        try:
            output = self.sam3d.generate_3d(
                image=image,  # Pass numpy array, not PIL Image
                mask=None,  # Don't use mask - use the clean generated image
                seed=seed
            )
            
            generation_time = time.time() - start_time
            logger.info(f"3D generation took {generation_time:.2f}s")
            
            # Save directly to permanent storage (no temp directory needed).
            output_dir = self.storage.get_assets_directory(job_id, obj.object_id)
            logger.info(f"Saving 3D assets directly to: {output_dir}")
            
            saved_paths = self.sam3d.save_outputs(output, output_dir)
            logger.info(f"Saved 3D assets: {saved_paths}")
            
            # Validate quality before deleting output.
            quality_scores = self.sam3d.validate_quality(output)
            
            # Explicitly delete output to free GPU memory
            del output
            
            # Create Asset3D record.
            asset = Asset3D(
                ply_path=saved_paths.get('ply'),
                glb_path=saved_paths.get('glb'),
                generation_seed=seed,
                quality_scores=quality_scores,
                generation_time=generation_time,
                created_at=datetime.now().isoformat()
            )
            
            # Update object.
            obj.asset_3d = asset
            obj.status = ObjectStatus.COMPLETE_3D.value
            
            logger.info(f"✅ 3D asset generated successfully")
            logger.info(f"   PLY: {asset.ply_path}")
            logger.info(f"   GLB: {asset.glb_path}")
            
            return asset
            
        except Exception as e:
            logger.error(f"3D generation failed: {e}")
            obj.status = ObjectStatus.ERROR.value
            obj.error_message = f"3D generation failed: {str(e)}"
            raise
    
    def regenerate_with_different_seed(
        self,
        job_id: str,
        obj: SegmentedObject,
        new_seed: int
    ) -> Asset3D:
        """
        Regenerate 3D asset with different seed.
        
        Args:
            job_id: Job ID
            obj: Segmented object
            new_seed: New random seed
            
        Returns:
            New Asset3D
        """
        logger.info(f"Regenerating 3D asset with new seed: {new_seed}")
        return self.generate_3d_asset(job_id, obj, seed=new_seed)
