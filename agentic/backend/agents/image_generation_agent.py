#!/usr/bin/env python3
"""
Image generation and editing agent using Gemini.
"""

from PIL import Image
from loguru import logger
from typing import List, Callable, Optional
import time
import numpy as np
import os

from backend.services.gemini_service import GeminiService
from backend.services.storage_service import StorageService
from backend.models import SegmentedObject, ObjectStatus


class ImageGenerationAgent:
    """Agent for generating and editing object images."""
    
    def __init__(self, gemini_service: GeminiService, storage_service: StorageService):
        """
        Initialize image generation agent.
        
        Args:
            gemini_service: Gemini API service
            storage_service: Storage service
        """
        self.gemini = gemini_service
        self.storage = storage_service
        logger.info("Image Generation Agent initialized")
    
    def generate_clean_image(
        self,
        job_id: str,
        obj: SegmentedObject,
        original_image: Image.Image,
        max_retries: int = 2
    ) -> str:
        """
        Generate clean image of object without background.
        
        Args:
            job_id: Job ID
            obj: Segmented object
            original_image: Original image
            max_retries: Number of retry attempts
            
        Returns:
            Path to generated image
        """
        logger.info(f"Generating clean image for {obj.object_id}: {obj.label}")
        
        # Prepare mask info for Gemini (following notebook approach)
        # Important: Include the original mask data from Gemini so it can accurately identify the object
        if obj.original_mask_data:
            try:
                # Parse the stored JSON string back to dict
                import json
                mask_dict = json.loads(obj.original_mask_data)
                logger.info(f"Using original mask data from Gemini (keys: {list(mask_dict.keys())})")
            except Exception as e:
                logger.warning(f"Failed to parse original_mask_data: {e}, falling back to basic mask_info")
                mask_dict = {
                    'label': obj.label,
                    'box_2d': list(obj.bbox),
                }
        else:
            # Fallback if no original mask data (shouldn't happen but be defensive)
            logger.warning(f"No original_mask_data for {obj.object_id}, using basic mask_info")
            mask_dict = {
                'label': obj.label,
                'box_2d': list(obj.bbox),
            }
        
        mask_info = mask_dict
        
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt}/{max_retries} for {obj.object_id}")
                    time.sleep(2 * attempt)  # Exponential backoff
                
                # Call Gemini to generate clean image (send full image + mask info).
                generated_image = self.gemini.generate_clean_image(
                    original_image, mask_info  # Send full image, not cropped
                )
                
                # Save generated image.
                image_path = self.storage.save_generated_image(
                    job_id, obj.object_id, generated_image
                )
                
                # Update object.
                obj.generated_image_path = image_path
                obj.status = ObjectStatus.IMAGE_GENERATED.value
                
                logger.info(f"✅ Clean image generated: {image_path}")
                return image_path
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed for {obj.object_id}: {e}")
                if attempt < max_retries:
                    continue
                else:
                    # All retries exhausted
                    logger.error(f"Failed to generate clean image after {max_retries + 1} attempts: {e}")
                    obj.status = ObjectStatus.ERROR.value
                    obj.error_message = f"Image generation failed: {str(e)}"
                    raise last_error
    
    def edit_image(
        self,
        job_id: str,
        obj: SegmentedObject,
        edit_prompt: str
    ) -> str:
        """
        Edit object image based on prompt.
        
        Args:
            job_id: Job ID
            obj: Segmented object
            edit_prompt: Text prompt for editing
            
        Returns:
            Path to edited image
        """
        logger.info(f"Editing image for {obj.object_id} with prompt: {edit_prompt}")
        
        # Get current image (generated or edited).
        current_image_path = obj.get_current_image_path()
        if not current_image_path:
            raise ValueError(f"No image available for object {obj.object_id}")
        
        current_image = Image.open(current_image_path)
        
        try:
            # Call Gemini to edit image.
            edited_image = self.gemini.edit_image(current_image, edit_prompt)
            
            # Save edited image.
            image_path = self.storage.save_edited_image(
                job_id, obj.object_id, edited_image
            )
            
            # Update object.
            obj.edited_image_path = image_path
            obj.edit_prompt = edit_prompt
            obj.status = ObjectStatus.IMAGE_EDITED.value
            
            logger.info(f"✅ Image edited: {image_path}")
            return image_path
            
        except Exception as e:
            logger.error(f"Failed to edit image: {e}")
            raise
    
    def batch_generate_clean_images(
        self,
        job_id: str,
        objects: list,
        original_image: Image.Image,
        progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
        max_workers: int = 1  # Changed to 1 for sequential processing
    ):
        """
        Generate clean images for all objects sequentially (one by one).
        
        Args:
            job_id: Job ID
            objects: List of SegmentedObject
            original_image: Original image
            progress_callback: Optional callback(completed, total, obj_id, image_path) for progress updates
            max_workers: Not used, kept for compatibility (always processes sequentially)
        """
        logger.info(f"Batch generating clean images for {len(objects)} objects (sequential processing)")
        
        # Process objects one by one sequentially
        for idx, obj in enumerate(objects):
            try:
                logger.info(f"Generating image {idx+1}/{len(objects)} for {obj.object_id}: {obj.label}")
                image_path = self.generate_clean_image(job_id, obj, original_image)
                
                # Call progress callback if provided.
                if progress_callback:
                    progress_callback(idx + 1, len(objects), obj.object_id, image_path)
                
                logger.info(f"Progress: {idx+1}/{len(objects)} images generated")
                
            except Exception as e:
                logger.error(f"Failed to generate image for {obj.object_id}: {e}")
                # Continue with next object even if this one failed
                if progress_callback:
                    progress_callback(idx + 1, len(objects), obj.object_id, None)
        
        logger.info("✅ Batch image generation complete")
